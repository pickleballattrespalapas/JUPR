"""Local incident acceptance tests and conservative uncertainty controls.

Uses in-memory service doubles, never Supabase or an HTTP origin. Actual SQL
constraints/transactions are independently exercised by operation_sql_breaks.mjs.
These tests do not emulate PostgREST retry behavior.
"""
from __future__ import annotations

import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from jupr_app.services import admin_live_ladder_operation_service as ledger
from jupr_app.services import admin_play_generator_service as generator
from jupr_app.services import admin_round_robin_hub_service as hub_service
from jupr_app.services.public_live_operation_service import PublicLiveConflictError


def _fixture_module(filename):
    path = Path(__file__).resolve().parents[1] / filename
    spec = importlib.util.spec_from_file_location(f"adversarial_fixture_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


hub = _fixture_module("test_admin_round_robin_hub_service.py")
ops = _fixture_module("test_admin_live_ladder_operation_service.py")


def trace(scenario, **values):
    print("AUDIT_TRACE " + json.dumps({"scenario": scenario, **values}, sort_keys=True, default=str))


def run_admin(db, mutate, *, key="audit-save-0001", entity="rr-a", version="version-1", payload=None):
    return ledger.run_durable_admin_operation(
        db, club_id="club-a", surface="play_generator", operation_type="save_round_scores",
        entity_id=entity, idempotency_key=key, expected_version=version,
        current_version=version, current_version_resolver=lambda: version,
        request_payload=payload or {"scores": [{"match_id": "m1", "score_a": 11, "score_b": 6}]},
        recovery={"entity_id": entity}, actor_email="audit@example.test",
        actor_role="administrator", source="local_adversarial_audit", mutate=mutate,
    )


class LeaseQuery(ops.FakeQuery):
    """Model the partial unique index; the SQL script tests the real constraint."""
    def execute(self):
        if self.table_name == ledger.OPERATION_TABLE and self.insert_payload is not None:
            incoming = self.insert_payload
            active = {"intent", "running", "mutated", "recovery_required"}
            columns = (("club_id", "surface", "entity_id", "expected_version")
                       if incoming["surface"] == "play_generator" else ("club_id", "surface", "expected_version"))
            for row in self.owner.storage[self.table_name]:
                if row["status"] in active and incoming["status"] in active and all(row[k] == incoming[k] for k in columns):
                    raise RuntimeError("duplicate key: live_ladder_admin_operations_active_version_unique")
        return super().execute()


class LeaseDB(ops.FakeSupabase):
    def table(self, name):
        return LeaseQuery(self, name)


def score_fixture(monkeypatch):
    source = hub.TransactionalHubDB()
    stored = hub.session(source, count=4)
    state = {"row": deepcopy(stored), "writes": 0}
    monkeypatch.setattr(generator, "_live_row", lambda *_args, **_kwargs: deepcopy(state["row"]))

    def persist(_db, *, before, event, expected_version, status=None):
        assert before["updated_at"] == expected_version
        state["writes"] += 1
        state["row"]["state"]["page_state"]["event"] = deepcopy(event)
        state["row"]["updated_at"] = "2026-10-23T01:00:01+00:00"
        state["row"]["version"] += 1
        return deepcopy(state["row"])

    monkeypatch.setattr(generator, "_persist_event", persist)
    matches = stored["state"]["page_state"]["event"]["rounds"][0]["matches"]
    scores = [{"match_id": m["id"], "score_a": 11, "score_b": 6} for m in matches]

    def save(db, score_rows):
        return generator.save_play_generator_round(
            db, club_id="club-a", session_key="rr-a", round_number=1,
            scores=score_rows, expected_version=stored["updated_at"],
            actor_email="audit@example.test", actor_role="administrator", source="local_adversarial_audit",
        )
    return state, scores, save


def test_noop_checkin_preserves_an_operator_snapshot():
    db = hub.TransactionalHubDB()
    hub.session(db)
    attached = hub.command(db, hub.create(db), "add_group", {"session_key": "rr-a"})
    state_before = deepcopy(db.storage[hub_service.HUB_TABLE][0]["state"])
    sessions_before = deepcopy(db.storage["live_sessions"])
    after = hub.command(db, attached, "process_checkins", key="audit-noop-checkins")
    assert db.storage[hub_service.HUB_TABLE][0]["state"] == state_before
    assert db.storage["live_sessions"] == sessions_before
    assert all(entry["patch"] == {} for entry in db.calls[-1]["p_sessions"])
    assert after["hub"]["version"] == attached["hub"]["version"]
    renamed = hub.command(db, attached, "rename_hub", {"title": "Staff's new title"}, "audit-still-current")
    assert renamed["hub"]["title"] == "Staff's new title"
    trace("noop-checkin-preserves-operator", old_version=attached["hub"]["version"],
          new_version=after["hub"]["version"], session_writes=0, state_changed=False)


def test_metadata_race_rejects_independent_roster_add_without_partial_write():
    db = hub.TransactionalHubDB()
    hub.session(db)
    attached = hub.command(db, hub.create(db), "add_group", {"session_key": "rr-a"})
    players_before = deepcopy(hub_service._event(hub.raw_session(db))["participants"])
    db.before_commit = lambda database: hub.command(
        database, attached, "rename_hub", {"title": "Concurrent title"}, "audit-race-winner")
    with pytest.raises(PublicLiveConflictError, match="changed"):
        hub.command(db, attached, "add_player", {"session_key": "rr-a", "name": "Waiting Guest"}, "audit-race-loser")
    assert hub_service._event(hub.raw_session(db))["participants"] == players_before
    assert db.storage[hub_service.HUB_TABLE][0]["title"] == "Concurrent title"
    assert not any(row["operation_key"] == "audit-race-loser" for row in db.storage[hub_service.OPERATION_TABLE])
    trace("metadata-versus-roster-race", winner="rename_hub", loser="add_player", partial_roster_write=False)


def test_hub_commit_then_lost_response_replays_once():
    class ResponseLossDB(hub.TransactionalHubDB):
        lose_response = False
        def _commit(self, params):
            result = super()._commit(params)
            if self.lose_response:
                self.lose_response = False
                raise TimeoutError("injected response loss AFTER atomic commit")
            return result
    db = ResponseLossDB()
    hub.session(db)
    attached = hub.command(db, hub.create(db), "add_group", {"session_key": "rr-a"})
    db.lose_response = True
    request = {"session_key": "rr-a", "name": "Response Lost Guest"}
    with pytest.raises(TimeoutError):
        hub.command(db, attached, "add_player", request, "audit-response-loss")
    committed = deepcopy(db.storage)
    replay = hub.command(db, attached, "add_player", request, "audit-response-loss")
    assert replay["replayed"] is True
    assert db.storage == committed
    assert sum(p["name"] == "Response Lost Guest" for p in hub_service._event(hub.raw_session(db))["participants"]) == 1
    trace("hub-response-loss-control", replayed=True, matching_players=1, extra_commit=False)


@pytest.mark.parametrize("corrected_entity", ["rr-a", "rr-b"])
def test_prewrite_validation_releases_same_and_other_session_leases(monkeypatch, corrected_entity):
    db = LeaseDB()
    state, valid_scores, save = score_fixture(monkeypatch)
    with pytest.raises(ledger.LiveLadderPrewriteRejectedError, match="Enter every score|Submit every match"):
        run_admin(db, lambda: save(db, []), payload={"scores": []})
    operation = db.storage[ledger.OPERATION_TABLE][0]
    assert state["writes"] == 0
    assert operation["status"] == "failed"
    assert any(message in operation["error_text"] for message in ("Enter every score", "Submit every match"))
    corrected_calls = []
    run_admin(db, lambda: corrected_calls.append(True) or {"ok": True},
              key="audit-corrected-key", entity=corrected_entity, payload={"scores": valid_scores})
    assert corrected_calls == [True]
    trace("validation-releases-lease", invalid_entity="rr-a", corrected_entity=corrected_entity,
          domain_writes=state["writes"], operation_status=operation["status"], corrected_request_blocked=False)


def test_domain_commit_without_any_receipt_stays_uncertain(monkeypatch):
    db = LeaseDB()
    state, valid_scores, save = score_fixture(monkeypatch)
    persist = generator._persist_event
    def commit_then_lose_response(*args, **kwargs):
        persist(*args, **kwargs)
        raise TimeoutError("injected transport response loss AFTER domain commit, BEFORE receipt")
    monkeypatch.setattr(generator, "_persist_event", commit_then_lose_response)
    with pytest.raises(ledger.LiveLadderUncertainError):
        run_admin(db, lambda: save(db, valid_scores))
    operation = db.storage[ledger.OPERATION_TABLE][0]
    assert state["writes"] == 1
    assert state["row"]["state"]["page_state"]["event"]["rounds"][0]["status"] == "saved"
    assert operation["status"] == "recovery_required"
    assert operation["result_json"] == {}
    with pytest.raises(ledger.LiveLadderUncertainError, match="incomplete and its outcome is uncertain"):
        run_admin(db, lambda: save(db, valid_scores))
    assert state["writes"] == 1
    trace("commit-response-loss-before-receipt", domain_writes=1, round_status="saved",
          operation_status=operation["status"], receipt_present=False, exact_retry_recovers=False)


def test_outer_completion_audit_failure_has_recoverable_receipt():
    db = LeaseDB()
    db.fail_audit_actions.add("complete_live_ladder_operation_admin")
    writes = []
    mutate = lambda: writes.append(True) or {"ok": True, "saved_version": "version-2"}
    with pytest.raises(ledger.LiveLadderUncertainError):
        run_admin(db, mutate)
    operation = db.storage[ledger.OPERATION_TABLE][0]
    assert operation["status"] == "recovery_required"
    assert operation["result_json"]["saved_version"] == "version-2"
    db.fail_audit_actions.clear()
    replay = run_admin(db, mutate)
    assert replay["idempotent_replay"] is True
    assert len(writes) == 1
    assert db.storage[ledger.OPERATION_TABLE][0]["status"] == "completed"
    trace("outer-audit-recovery-control", domain_writes=1, receipt_present=True, exact_retry_recovers=True)


def test_admin_retry_recovers_committed_hub_receipt():
    class CombinedDB(hub.TransactionalHubDB):
        def __init__(self):
            super().__init__()
            self.storage.update({ledger.OPERATION_TABLE: [], "admin_activity_log": []})
            self.fail_audit_actions = set()
            self.fail_operation_insert = False
            self.operation_insert_race_row = None
            self.lose_response = False
        def table(self, name):
            if name in {ledger.OPERATION_TABLE, "admin_activity_log"}:
                return ops.FakeQuery(self, name)
            return super().table(name)
        def _commit(self, params):
            result = super()._commit(params)
            if self.lose_response:
                self.lose_response = False
                raise TimeoutError("injected hub RPC response loss after score/court/receipt commit")
            return result
    db = CombinedDB()
    hub.session(db, count=4)
    attached = hub.command(db, hub.create(db), "add_group", {"session_key": "rr-a"})
    match = attached["workspace"]["eligible_queue"][0]
    hub.command(db, attached, "assign_next_court", {"session_key": "rr-a", "match_id": match["match_id"]})
    before = deepcopy(hub.raw_session(db))
    scores = [{"match_id": match["match_id"], "score_a": 11, "score_b": 6}]
    def mutate():
        return generator.save_play_generator_round(
            db, club_id="club-a", session_key="rr-a", round_number=1, scores=scores,
            expected_version=before["updated_at"], actor_email="audit@example.test",
            actor_role="administrator", source="local_adversarial_audit")
    db.lose_response = True
    recovered = run_admin(db, mutate, version=before["updated_at"], payload={"scores": scores})
    assert recovered["idempotent_replay"] is True
    committed_version = db.storage[hub_service.HUB_TABLE][0]["version"]
    receipts = [row for row in db.storage[hub_service.OPERATION_TABLE] if row["operation_key"].startswith("generator:")]
    assert len(receipts) == 1
    assert hub_service._event(hub.raw_session(db))["rounds"][0]["status"] == "saved"
    assert db.storage[hub_service.HUB_TABLE][0]["state"]["courts"][0]["current"] is None
    assert db.storage[ledger.OPERATION_TABLE][0]["result_json"]["session"]
    replay = run_admin(db, mutate, version=before["updated_at"], payload={"scores": scores})
    assert replay["idempotent_replay"] is True
    assert db.storage[hub_service.HUB_TABLE][0]["version"] == committed_version
    trace("admin-wrapper-recovers-hub-receipt", hub_receipts=1, score_saved=True,
          court_released=True, admin_result_present=True, exact_retry_recovers=True, duplicate_commit=False)


def test_stale_session_read_rejects_all_hub_patches():
    db = hub.TransactionalHubDB()
    hub.session(db, key="rr-a")
    hub.session(db, key="rr-b", offset=100)
    attached = hub.command(db, hub.create(db), "add_group", {"session_key": "rr-a"})
    attached = hub.command(db, attached, "add_group", {"session_key": "rr-b"})
    before_a = deepcopy(hub.raw_session(db, "rr-a"))
    before_hub = deepcopy(db.storage[hub_service.HUB_TABLE])
    db.before_commit = lambda database: hub.raw_session(database, "rr-b").update(version=99)
    with pytest.raises(PublicLiveConflictError):
        hub.command(db, attached, "add_player", {"session_key": "rr-a", "name": "Rejected Guest"}, "audit-stale-sibling")
    assert hub.raw_session(db, "rr-a") == before_a
    assert db.storage[hub_service.HUB_TABLE] == before_hub
    trace("stale-readset-control", sibling_version=99, target_mutated=False, hub_mutated=False)


class StandaloneReceiptDB(LeaseDB):
    lose_response = False
    hide_receipts = False
    def rpc(self, name, params):
        from generator_receipt_fake import commit_generator_rpc
        def execute():
            result = commit_generator_rpc(self.storage, name, params)
            if self.lose_response:
                self.lose_response = False
                raise TimeoutError("injected response loss after atomic receipt")
            return result
        return SimpleNamespace(execute=execute)
    def table(self, name):
        if name == generator.MUTATION_RECEIPT_TABLE and self.hide_receipts:
            raise TimeoutError("injected unavailable receipt lookup")
        return super().table(name)


def standalone_score(db):
    hub.session(db, count=4)
    before = deepcopy(hub.raw_session(db))
    scores = [{"match_id": m["id"], "score_a": 11, "score_b": 6}
              for m in before["state"]["page_state"]["event"]["rounds"][0]["matches"]]
    def mutate():
        return generator.save_play_generator_round(db, club_id="club-a", session_key="rr-a", round_number=1,
            scores=scores, expected_version=before["updated_at"], actor_email="audit@example.test",
            actor_role="administrator", source="test")
    return before, scores, mutate


def test_atomic_standalone_receipt_recovers_response_loss_after_later_edit():
    db = StandaloneReceiptDB()
    db.storage["live_sessions"] = []
    before, scores, mutate = standalone_score(db)
    db.lose_response = db.hide_receipts = True
    with pytest.raises(ledger.LiveLadderUncertainError):
        run_admin(db, mutate, version=before["updated_at"], payload={"scores": scores})
    assert hub.raw_session(db)["version"] == 2
    hub.raw_session(db).update(version=3, updated_at="2026-10-23T06:00:00+00:00", last_operation_key="later", title="Later title")
    db.hide_receipts = False
    replay = run_admin(db, lambda: pytest.fail("must not repeat mutation"),
                       version=before["updated_at"], payload={"scores": scores})
    assert replay["idempotent_replay"] is True
    assert replay["session"]["title"] == "Later title"
    assert hub.raw_session(db)["version"] == 3
    assert len(db.storage[generator.MUTATION_RECEIPT_TABLE]) == 1


def test_known_database_cas_rejection_releases_lease_without_receipt():
    db = StandaloneReceiptDB()
    db.storage["live_sessions"] = []
    before, scores, mutate = standalone_score(db)
    hub.raw_session(db)["updated_at"] = "2026-10-23T06:00:00+00:00"
    with pytest.raises(ledger.LiveLadderPrewriteConflictError):
        run_admin(db, mutate, version=before["updated_at"], payload={"scores": scores})
    assert db.storage[ledger.OPERATION_TABLE][0]["status"] == "failed"
    assert not db.storage.get(generator.MUTATION_RECEIPT_TABLE)


def test_unrelated_session_can_write_while_another_has_unknown_outcome():
    db = LeaseDB()
    with pytest.raises(ledger.LiveLadderUncertainError):
        run_admin(db, lambda: (_ for _ in ()).throw(TimeoutError("unknown write result")))
    assert db.storage[ledger.OPERATION_TABLE][0]["status"] == "recovery_required"
    assert run_admin(db, lambda: {"ok": True}, key="independent-session", entity="rr-b")["ok"] is True
    with pytest.raises(ledger.LiveLadderConflictError):
        run_admin(db, lambda: pytest.fail("same version is still leased"), key="same-session-new-key")


def test_transport_valueerror_after_write_attempt_keeps_uncertain_lease(monkeypatch):
    db = StandaloneReceiptDB()
    db.storage["live_sessions"] = []
    before, scores, mutate = standalone_score(db)
    def broken_rpc(*args, **kwargs):
        raise ValueError("response JSON decoding failed")
    monkeypatch.setattr(db, "rpc", broken_rpc)
    with pytest.raises(ledger.LiveLadderUncertainError):
        run_admin(db, mutate, version=before["updated_at"], payload={"scores": scores})
    assert db.storage[ledger.OPERATION_TABLE][0]["status"] == "recovery_required"
