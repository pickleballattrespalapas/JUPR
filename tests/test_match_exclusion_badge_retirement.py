from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.domain.gamification import match_exclusion_reconcile as reconcile
from jupr_app.services import match_exclusion_durability_service as service


class Query:
    def __init__(self, owner, table):
        self.owner, self.table = owner, table
        self.filters = []
        self.changes = None

    def select(self, columns):
        assert self.table == "badges"
        assert columns == "badge_id,state,is_active,eval_triggers,badge_status,badge_award_timing"
        return self

    def in_(self, key, values):
        assert key == "badge_id" and tuple(values) == reconcile.MATCH_EXCLUSION_BADGE_IDS
        return self

    def limit(self, count):
        assert count == len(reconcile.MATCH_EXCLUSION_BADGE_IDS)
        return self

    def update(self, changes):
        assert self.table == "match_exclusion_badge_progress"
        self.changes = changes
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def gt(self, key, value):
        assert (key, value) == ("lease_expires_at", "now")
        self.owner.expiry_fenced = True
        return self

    def execute(self):
        if self.table == "badges":
            return SimpleNamespace(data=[{"badge_id": "first_win", "state": "live", "is_active": True, "eval_triggers": ["match_updated"]}])
        row = self.owner.progress
        assert {key for key, _ in self.filters} == {"id", "operation_id", "club_id", "player_id", "status", "lease_token", "leased_by"}
        if self.owner.lose_lease or any(row.get(key) != value for key, value in self.filters):
            return SimpleNamespace(data=[])
        row.update(deepcopy(self.changes))
        return SimpleNamespace(data=[deepcopy(row)])


class Supabase:
    def __init__(self, *, lose_lease=False):
        self.tables, self.rpcs = [], []
        self.lose_lease, self.expiry_fenced = lose_lease, False
        self.progress = {"id": "progress", "operation_id": "operation", "club_id": "club", "player_id": 7, "status": "pending", "before_json": [{"evidence": "keep"}], "after_json": [{"evidence": "keep"}]}

    def table(self, name):
        assert name in {"badges", "match_exclusion_badge_progress"}, "Retirement must never access matches or stored awards"
        self.tables.append(name)
        return Query(self, name)

    def rpc(self, name, params):
        self.rpcs.append(name)
        if name == reconcile.MATCH_EXCLUSION_BADGE_CLAIM_RPC:
            if self.progress["status"] == "succeeded":
                result = {"ok": True, "claimed": False, "code": "BADGE_RECONCILE_COMPLETE"}
            else:
                self.progress.update(status="running", lease_token="lease", leased_by=params["p_worker_id"])
                result = {"ok": True, "claimed": True, "operation_id": "operation", "progress_id": "progress", "club_id": "club", "player_id": 7, "lease_token": "lease", "badge_ids": ["first_win"], "badge_contract_version": reconcile.MATCH_EXCLUSION_BADGE_CONTRACT_VERSION}
        elif name == reconcile.MATCH_EXCLUSION_BADGE_FAIL_RPC:
            result = {"ok": True, "status": "failed"}
        else:
            raise AssertionError("No award application RPC may run during retirement")
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=result))


def forbid_calculation(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("No history load or badge evaluation may run")
    for name in ("_load_reconciliation_context", "_canonical_evaluation_context", "_build_player_plan"):
        monkeypatch.setattr(reconcile, name, fail)


def test_retired_scope_lookup_reads_only_bounded_catalog(monkeypatch):
    forbid_calculation(monkeypatch)
    db = Supabase()
    assert reconcile.resolve_match_exclusion_badge_ids(db, club_id="club") == ["first_win"]
    assert db.tables == ["badges"]


def test_retirement_completes_owned_ledger_without_changing_awards(monkeypatch):
    forbid_calculation(monkeypatch)
    db = Supabase()
    result = reconcile.reconcile_match_exclusion_badges(db, club_id="club", operation_id="operation", player_ids=[7])
    assert result["ok"] is True and result["skipped_reason"] == "badges_retired"
    assert result["processed_player_ids"] == [7]
    assert result["inserted_count"] == result["updated_count"] == result["revoked_count"] == 0
    assert db.progress["status"] == "succeeded"
    assert db.progress["before_json"] == db.progress["after_json"] == [{"evidence": "keep"}]
    assert db.progress["lease_token"] is None and db.expiry_fenced
    assert db.tables == ["match_exclusion_badge_progress"]
    repeated = reconcile.reconcile_match_exclusion_badges(db, club_id="club", operation_id="operation", player_ids=[7])
    assert repeated["idempotent"] is True and repeated["processed_player_ids"] == []


def test_retirement_rejects_lost_lease_without_claiming_success(monkeypatch):
    forbid_calculation(monkeypatch)
    db = Supabase(lose_lease=True)
    with pytest.raises(RuntimeError, match="lease was lost"):
        reconcile.reconcile_match_exclusion_badges(db, club_id="club", operation_id="operation", player_ids=[7])
    assert db.progress["status"] == "running"
    assert db.rpcs[-1] == reconcile.MATCH_EXCLUSION_BADGE_FAIL_RPC


def test_empty_badge_plan_is_never_applied_when_retired():
    with pytest.raises(RuntimeError, match="retired"):
        reconcile.apply_match_exclusion_badge_plan(object(), operation_id="operation", progress_id="progress", club_id="club", lease_token="lease", worker_id="worker", actor_email="actor@example.test", desired_badges=[])


@pytest.mark.parametrize("recovery", [False, True])
def test_rating_replay_and_durable_recovery_finalize_with_retired_badges(monkeypatch, recovery):
    forbid_calculation(monkeypatch)
    db = Supabase()
    operation = {"id": "operation", "status": "recovery_required" if recovery else "pending_replay", "recovery_stage": "replay" if recovery else None, "replay_job_id": "replay", "affected_player_ids": [7]}
    replay_calls, final_calls = [], []
    def replay(*_args, **kwargs):
        replay_calls.append(kwargs)
        return {"job_status": "succeeded"}
    monkeypatch.setattr(service, "run_admin_replay_history", replay)
    monkeypatch.setattr(service, "_transition_after_replay", lambda *_args, **_kwargs: {**operation, "status": "pending_badge_reconcile", "recovery_stage": None})
    def finalize(*_args, **kwargs):
        assert db.progress["status"] == "succeeded"
        assert db.progress["result_json"]["skipped_reason"] == "badges_retired"
        final_calls.append(kwargs)
        operation["status"] = "succeeded"
        return {"operation_status": "succeeded"}
    monkeypatch.setattr(service, "_finalize_operation", finalize)
    monkeypatch.setattr(service, "_fetch_operation_row", lambda *_args, **_kwargs: dict(operation))
    result = service._complete_operation(db, operation=operation, club_id="club", operation_id="operation", actor_email="actor@example.test", actor_role="admin", source="test", retry_failed=recovery)
    assert result["operation_status"] == "succeeded"
    assert result["badge_reconcile"]["skipped_reason"] == "badges_retired"
    assert len(replay_calls) == len(final_calls) == 1
    assert replay_calls[0]["retry_failed"] is recovery
