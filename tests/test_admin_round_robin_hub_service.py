"""Service contracts against a transactional RPC double.

These tests exercise orchestration and generator integration. PostgreSQL locking,
RLS, trigger and constraint behavior is covered separately by the SQL tests.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from jupr_app.domain.adaptive_play_engine import create_generator_preview, generator_event_standings, start_generator_event
from jupr_app.services import admin_round_robin_hub_service as service
from jupr_app.services.public_live_operation_service import PublicLiveConflictError


class FakeConflict(Exception):
    code = "PT409"


class ReadQuery:
    def __init__(self, db, name):
        self.db, self.name = db, name
        self.filters, self.limit_count, self.order_by = [], None, None

    def select(self, *_args):
        return self

    def eq(self, key, value):
        self.filters.append(lambda row: str(row.get(key)) == str(value))
        return self

    def in_(self, key, values):
        self.filters.append(lambda row: row.get(key) in values)
        return self

    def order(self, key, desc=False):
        self.order_by = (key, desc)
        return self

    def limit(self, count):
        self.limit_count = count
        return self

    def execute(self):
        values = [deepcopy(row) for row in self.db.storage.setdefault(self.name, []) if all(test(row) for test in self.filters)]
        if self.order_by:
            key, reverse = self.order_by
            values.sort(key=lambda row: str(row.get(key) or ""), reverse=reverse)
        return SimpleNamespace(data=values[:self.limit_count])


class TransactionalHubDB:
    def __init__(self):
        self.storage = {"live_sessions": [], service.HUB_TABLE: [], service.OPERATION_TABLE: []}
        self.calls = []
        self.clock = datetime(2026, 10, 20, tzinfo=timezone.utc)
        self.before_commit = None

    def table(self, name):
        return ReadQuery(self, name)

    def rpc(self, name, params):
        assert name == "commit_round_robin_hub"
        self.calls.append(deepcopy(params))
        return SimpleNamespace(execute=lambda: self._commit(params))

    def _commit(self, p):
        if self.before_commit:
            action, self.before_commit = self.before_commit, None
            action(self)
        snapshot = deepcopy(self.storage)
        club, hub_id = p["p_club_id"], p["p_hub_id"]
        receipt = next((row for row in snapshot[service.OPERATION_TABLE]
                        if row["club_id"] == club and row["hub_id"] == hub_id and row["operation_key"] == p["p_operation_key"]), None)
        if receipt:
            if receipt["fingerprint"] != p["p_fingerprint"]:
                raise FakeConflict("operation fingerprint changed")
            return SimpleNamespace(data={**deepcopy(receipt["result"]), "replayed": True})
        hub = next((row for row in snapshot[service.HUB_TABLE] if row["club_id"] == club and row["id"] == hub_id), None)
        if (hub is None and p["p_expected_version"] != 0) or (hub is not None and hub["version"] != p["p_expected_version"]):
            raise FakeConflict("hub changed")
        old_members = {group["session_key"] for group in (hub or {}).get("state", {}).get("groups", [])}
        new_members = {group["session_key"] for group in p["p_state"].get("groups", [])}
        assert {entry["session_key"] for entry in p["p_sessions"]} == old_members | new_members
        read_rows = []
        for entry in p["p_sessions"]:
            session = next((row for row in snapshot["live_sessions"] if row["club_id"] == club and row["session_key"] == entry["session_key"]), None)
            if not session or session["updated_at"] != entry["expected_version"] or session.get("version", 1) != entry.get("expected_numeric_version", 1):
                raise FakeConflict("session changed")
            read_rows.append((entry, session))
        no_op = bool(hub and hub["state"] == p["p_state"] and hub["title"] == p["p_title"]
                      and hub["status"] == p["p_status"] and all(not entry["patch"] for entry, _ in read_rows))
        self.clock += timedelta(microseconds=1)
        now = self.clock.isoformat()
        for entry, session in read_rows:
            if entry["patch"]:
                numeric = int(session.get("version") or 1)
                session.update(deepcopy(entry["patch"]))
                session["updated_at"] = now
                session["version"] = entry["patch"].get("version", numeric + 1)
                if entry["patch"].get("last_operation_key"):
                    receipts = snapshot.setdefault("play_generator_mutation_receipts", [])
                    key = entry["patch"]["last_operation_key"]
                    receipt = {"club_id": club, "session_key": session["session_key"], "operation_key": key,
                               "request_fingerprint": entry["patch"]["last_request_fingerprint"], "committed_version": now}
                    prior = next((r for r in receipts if r["operation_key"] == key), None)
                    if prior and any(prior[k] != receipt[k] for k in ("club_id", "session_key", "request_fingerprint")):
                        raise FakeConflict("operation fingerprint changed")
                    if not prior:
                        receipts.append(receipt)
        if hub is None:
            hub = {"id": hub_id, "club_id": club, "version": 0, "created_at": now}
            snapshot[service.HUB_TABLE].append(hub)
        if not no_op:
            hub.update(title=p["p_title"], status=p["p_status"], state=deepcopy(p["p_state"]), updated_at=now, version=hub["version"] + 1)
        result = {"hub": deepcopy(hub), "sessions": [deepcopy(row) for _, row in read_rows], "replayed": False}
        snapshot[service.OPERATION_TABLE].append({"club_id": club, "hub_id": hub_id, "operation_key": p["p_operation_key"],
                                                "fingerprint": p["p_fingerprint"], "result": deepcopy(result)})
        self.storage = snapshot
        return SimpleNamespace(data=result)


def session(db, key="rr-a", *, count=8, club="club-a", mode="admin_play_generator", offset=0, scoring="scored"):
    preview = create_generator_preview(generator_kind="round_robin", play_format="doubles", title=key,
        participant_names=[f"Player {offset + i}" for i in range(1, count + 1)],
        player_ids=[offset + i for i in range(1, count + 1)], total_rounds=3, court_count=0,
        scoring_mode=scoring, rating_mode="unrated")
    event = start_generator_event(preview)
    row = {"club_id": club, "session_key": key, "title": key, "status": "active", "version": 1,
           "updated_at": "2026-10-19T22:00:00+00:00", "created_at": "2026-10-19T22:00:00+00:00",
           "created_by_email": "operator@example.test", "source": "play_generator", "expires_at": "2099-01-01T00:00:00+00:00",
           "state": {"mode": mode, "club_id": club, "generator_kind": "round_robin", "page_state": {"event": event}}}
    db.storage["live_sessions"].append(row)
    return deepcopy(row)


def create(db, key="create-test", *, club="club-a", courts=1):
    return service.create_round_robin_hub(db, club_id=club, title="Friday pickleball", actor_email="operator@example.test",
        courts=[{"id": f"c{i}", "label": f"Court {i}", "open": True} for i in range(1, courts + 1)], idempotency_key=key)


def command(db, hub, action, payload=None, key=None):
    return service.command_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"], action=action,
        payload=payload or {}, expected_version=hub["hub"]["version"], idempotency_key=key or f"{action}-{hub['hub']['version']}", actor_email="operator@example.test")


def raw_session(db, key="rr-a"):
    return next(row for row in db.storage["live_sessions"] if row["session_key"] == key)


def guest_identity_hub(db):
    """Two roster entries share a label, not a club profile."""
    for key, offset in (("rr-a", 0), ("rr-b", 10)):
        session(db, key=key, count=4, offset=offset)
        person = raw_session(db, key)["state"]["page_state"]["event"]["participants"][0]
        person["name"], person["player_id"] = "Alex", None
    hub = command(db, create(db, courts=2), "add_group", {"session_key": "rr-a"})
    return command(db, hub, "add_group", {"session_key": "rr-b"})


def test_identity_resolution_is_durable_and_exact_retry_does_not_repeat_it():
    db = TransactionalHubDB()
    hub = guest_identity_hub(db)
    assert len([match for match in hub["workspace"]["blocked_queue"] if match["round_number"] == 1]) == 2
    before_sessions = deepcopy(db.storage["live_sessions"])
    request = dict(club_id="club-a", hub_id=hub["hub"]["id"], action="resolve_identities",
        payload={"resolution": "all_different"}, expected_version=hub["hub"]["version"],
        idempotency_key="identity-different-once", actor_email="operator@example.test")
    result = service.command_round_robin_hub(db, **request)
    assert len(result["workspace"]["eligible_queue"]) == 2
    persisted = deepcopy(db.storage[service.HUB_TABLE][0])
    assert persisted["state"]["identity_version"] == 2
    assert persisted["state"]["identity_decisions"][0]["resolution"] == "different"
    assert db.storage["live_sessions"] == before_sessions
    call_count = len(db.calls)
    replay = service.command_round_robin_hub(db, **request)
    assert replay["replayed"] is True
    assert replay["hub"]["version"] == result["hub"]["version"]
    assert len(db.calls) == call_count
    assert db.storage[service.HUB_TABLE][0] == persisted
    with pytest.raises(PublicLiveConflictError, match="different change"):
        service.command_round_robin_hub(db, **{**request, "payload": {"resolution": "same"}})


def test_identity_resolution_read_set_rejects_concurrent_roster_change_atomically():
    db = TransactionalHubDB()
    hub = guest_identity_hub(db)
    before_hubs = deepcopy(db.storage[service.HUB_TABLE])
    before_receipts = deepcopy(db.storage[service.OPERATION_TABLE])

    def roster_changes(current):
        row = raw_session(current, "rr-b")
        row["updated_at"] = "2026-10-20T10:00:00+00:00"
        row["version"] += 1

    db.before_commit = roster_changes
    with pytest.raises(PublicLiveConflictError, match="changed"):
        command(db, hub, "resolve_identities", {"resolution": "all_different"})
    assert db.storage[service.HUB_TABLE] == before_hubs
    assert db.storage[service.OPERATION_TABLE] == before_receipts
    assert {entry["session_key"] for entry in db.calls[-1]["p_sessions"]} == {"rr-a", "rr-b"}
    refreshed = service.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])
    resolved = command(db, refreshed, "resolve_identities", {"resolution": "all_different"})
    assert len(resolved["workspace"]["eligible_queue"]) == 2


def test_create_and_command_response_loss_replays_without_duplicate_hubs_or_members():
    db = TransactionalHubDB()
    session(db)
    hub = create(db)
    assert create(db)["hub"]["id"] == hub["hub"]["id"]
    added = command(db, hub, "add_group", {"session_key": "rr-a"}, "add-once-key")
    replay = command(db, hub, "add_group", {"session_key": "rr-a"}, "add-once-key")
    assert replay["replayed"] is True
    assert replay["hub"]["version"] == added["hub"]["version"]
    assert len(replay["workspace"]["groups"]) == len(db.storage[service.HUB_TABLE]) == 1
    assert raw_session(db)["state"]["round_robin_hub_id"] == hub["hub"]["id"]


def test_changed_idempotency_payload_is_rejected_without_writes():
    db = TransactionalHubDB()
    hub = create(db)
    command(db, hub, "rename_hub", {"title": "First title"}, "same-key-123")
    before = deepcopy(db.storage)
    with pytest.raises(PublicLiveConflictError, match="different change"):
        command(db, hub, "rename_hub", {"title": "Second title"}, "same-key-123")
    assert db.storage == before


def test_partial_match_result_releases_court_and_counts_only_after_full_round():
    db = TransactionalHubDB()
    session(db)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    assert len(hub["workspace"]["eligible_queue"]) == 2
    first, second = hub["workspace"]["eligible_queue"]
    hub = command(db, hub, "assign_next_court", {"session_key": "rr-a", "match_id": first["match_id"]})
    hub = command(db, hub, "score_and_release", {"court_id": "c1", "session_key": "rr-a", "match_id": first["match_id"], "score_a": 11, "score_b": 5})
    assert hub["workspace"]["courts"][0]["current"] is None
    event = service._event(raw_session(db))
    assert event["rounds"][0]["status"] == "active"
    assert all(row["matches"] == 0 for row in generator_event_standings(event))
    assert {entry["round_number"] for entry in hub["workspace"]["eligible_queue"]} == {1}
    with pytest.raises(ValueError, match="current round"):
        command(db, hub, "advance_round", {"session_key": "rr-a"})
    hub = command(db, hub, "assign_next_court", {"session_key": "rr-a", "match_id": second["match_id"]})
    hub = command(db, hub, "score_and_release", {"court_id": "c1", "session_key": "rr-a", "match_id": second["match_id"], "score_a": 11, "score_b": 7})
    event = service._event(raw_session(db))
    assert event["rounds"][0]["status"] == "saved"
    assert all(row["matches"] == 1 for row in generator_event_standings(event))
    assert event["currentRoundNumber"] == 2
    assert {entry["round_number"] for entry in hub["workspace"]["eligible_queue"]} == {2}
    with pytest.raises(ValueError, match="current round"):
        command(db, hub, "advance_round", {"session_key": "rr-a"})


def test_last_score_and_next_round_replay_atomically_after_lost_response(monkeypatch):
    db = TransactionalHubDB()
    session(db, count=4)
    session(db, "rr-b", count=4, offset=20)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    hub = command(db, hub, "add_group", {"session_key": "rr-b"})
    matches = {match["session_key"]: match for match in hub["workspace"]["eligible_queue"]}
    hub = command(db, hub, "assign_next_court", matches["rr-a"])
    hub = command(db, hub, "reserve_game_for_court", {**matches["rr-b"], "court_id": "c1"})
    request = dict(club_id="club-a", hub_id=hub["hub"]["id"], action="score_and_release",
        payload={"court_id": "c1", "session_key": "rr-a", "match_id": matches["rr-a"]["match_id"], "score_a": 11, "score_b": 7},
        expected_version=hub["hub"]["version"], idempotency_key="final-score-response-lost", actor_email="operator@example.test")
    original_get = service.get_round_robin_hub

    def lose_response(*args, **kwargs):
        raise ConnectionError("response lost after commit")

    monkeypatch.setattr(service, "get_round_robin_hub", lose_response)
    calls_before = len(db.calls)
    with pytest.raises(ConnectionError, match="after commit"):
        service.command_round_robin_hub(db, **request)
    assert len(db.calls) == calls_before + 1
    persisted = deepcopy(db.storage)
    assert service._event(raw_session(db))["currentRoundNumber"] == 2
    assert service._event(raw_session(db))["rounds"][0]["status"] == "saved"
    assert service._event(raw_session(db, "rr-b"))["currentRoundNumber"] == 1
    assert db.storage[service.HUB_TABLE][0]["state"]["courts"][0]["current"]["session_key"] == "rr-b"
    monkeypatch.setattr(service, "get_round_robin_hub", original_get)
    replay = service.command_round_robin_hub(db, **request)
    assert replay["replayed"] is True
    assert len(db.calls) == calls_before + 1
    assert db.storage == persisted
    assert {match["round_number"] for match in replay["workspace"]["eligible_queue"]} == {2}


@pytest.mark.parametrize("locked_round_saved", [False, True])
def test_fill_catches_up_ready_groups_without_advancing_or_assigning_locked_results(locked_round_saved):
    from jupr_app.domain.adaptive_play_engine import save_generator_round

    db = TransactionalHubDB()
    session(db, count=4)
    session(db, "rr-b", count=4, offset=20)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    hub = command(db, hub, "add_group", {"session_key": "rr-b"})
    # Represent an existing hub left at a saved round before automatic progression.
    for key in (["rr-a", "rr-b"] if locked_round_saved else ["rr-b"]):
        row = raw_session(db, key)
        event = service._event(row)
        row["state"]["page_state"]["event"] = save_generator_round(event, round_number=1,
            scores=[{"match_id": event["rounds"][0]["matches"][0]["id"], "score_a": 11, "score_b": 7}])
    raw_session(db)["state"]["generator_submission"] = {"status": "pending"}
    locked_before = deepcopy(raw_session(db))
    result = command(db, hub, "auto_fill_courts", {"excluded_session_keys": []})
    assert raw_session(db) == locked_before
    assert result["workspace"]["courts"][0]["current"]["session_key"] == "rr-b"
    assert result["workspace"]["courts"][0]["current"]["round_number"] == 2
    assert service._event(raw_session(db, "rr-b"))["currentRoundNumber"] == 2
    assert next(item for item in db.calls[-1]["p_sessions"] if item["session_key"] == "rr-a")["patch"] == {}


def test_all_group_versions_are_checked_before_assigning_a_court():
    db = TransactionalHubDB()
    session(db)
    session(db, "rr-b", offset=20)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    hub = command(db, hub, "add_group", {"session_key": "rr-b"})
    candidate = next(match for match in hub["workspace"]["eligible_queue"] if match["session_key"] == "rr-a")
    db.before_commit = lambda db: raw_session(db, "rr-b").update(updated_at="2026-10-20T23:00:00+00:00")
    with pytest.raises(PublicLiveConflictError):
        command(db, hub, "assign_next_court", {"session_key": "rr-a", "match_id": candidate["match_id"]})
    current = service.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])
    assert current["workspace"]["courts"][0]["current"] is None
    assert {row["session_key"] for row in db.calls[-1]["p_sessions"]} == {"rr-a", "rr-b"}


def test_stale_operator_cannot_replace_a_new_court_assignment():
    db = TransactionalHubDB()
    session(db)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    first, second = hub["workspace"]["eligible_queue"]
    assigned = command(db, hub, "assign_next_court", {"session_key": "rr-a", "match_id": first["match_id"]})
    with pytest.raises(PublicLiveConflictError):
        command(db, hub, "assign_next_court", {"session_key": "rr-a", "match_id": second["match_id"]}, "different-op-key")
    assert service.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])["hub"] == assigned["hub"]


def test_cross_club_hubs_and_sessions_are_not_visible_or_attachable():
    db = TransactionalHubDB()
    session(db, club="club-b")
    hub = create(db)
    assert not service.list_round_robin_hubs(db, club_id="club-a")["sessions"]
    with pytest.raises(ValueError, match="no longer available"):
        command(db, hub, "add_group", {"session_key": "rr-a"})
    with pytest.raises(LookupError):
        service.get_round_robin_hub(db, club_id="club-b", hub_id=hub["hub"]["id"])


@pytest.mark.parametrize("state_patch", [{"generator_submission": {"status": "pending"}}, {"official_publish": {"published_match_ids": [1]}}])
def test_submitted_or_published_sessions_cannot_enter_hub(state_patch):
    db = TransactionalHubDB()
    session(db)
    raw_session(db)["state"].update(state_patch)
    hub = create(db)
    assert not service.list_round_robin_hubs(db, club_id="club-a")["sessions"]
    with pytest.raises(ValueError, match="not been submitted"):
        command(db, hub, "add_group", {"session_key": "rr-a"})


def test_complete_hub_rejects_later_commands_without_changing_court_state():
    db = TransactionalHubDB()
    hub = command(db, create(db), "finish_hub")
    before = deepcopy(db.storage)
    with pytest.raises(ValueError, match="complete"):
        command(db, hub, "rename_hub", {"title": "Unexpected restart"})
    assert db.storage == before


def test_snapshot_and_list_do_not_leak_private_generator_state():
    db = TransactionalHubDB()
    session(db)
    row = raw_session(db)
    row["state"]["generator_sharing"] = {"handoff_nonce": "private-handoff-nonce"}
    row["state"]["page_state"]["event"]["privateMoneyballLock"] = {"ratings": "private-frozen-ratings"}
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    rendered = str(hub) + str(service.list_round_robin_hubs(db, club_id="club-a"))
    assert "private-handoff-nonce" not in rendered
    assert "private-frozen-ratings" not in rendered


def test_original_generator_patch_is_atomic_with_court_release_and_public_cas_fields():
    from jupr_app.domain.adaptive_play_engine import save_generator_round
    db = TransactionalHubDB()
    session(db, count=4, mode="public_play_generator")
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    match = hub["workspace"]["eligible_queue"][0]
    command(db, hub, "assign_next_court", {"session_key": "rr-a", "match_id": match["match_id"]})
    before = deepcopy(raw_session(db))
    event = save_generator_round(service._event(before), round_number=1,
        scores=[{"match_id": match["match_id"], "score_a": 11, "score_b": 6}])
    state = deepcopy(before["state"])
    state["page_state"]["event"] = event
    patch = {"state": state, "version": before["version"] + 1, "last_operation_key": "public-operation-key",
             "last_request_fingerprint": "public-fingerprint", "expires_at": "2099-02-01T00:00:00+00:00"}
    updated = service.persist_linked_generator_patch(db, before=before, patch=patch, expected_version=before["updated_at"])
    assert updated["last_operation_key"] == "public-operation-key"
    assert updated["last_request_fingerprint"] == "public-fingerprint"
    assert updated["expires_at"] == patch["expires_at"]
    detail = service.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])
    assert detail["workspace"]["courts"][0]["current"] is None
    assert service._event(updated)["rounds"][0]["status"] == "saved"
    assert service._event(updated)["currentRoundNumber"] == 2
    assert detail["workspace"]["groups"][0]["completed_matches"] == 0
    assert detail["workspace"]["groups"][0]["current_round_number"] == 2
    replay = service.persist_linked_generator_patch(db, before=before, patch=patch, expected_version=before["updated_at"])
    assert replay["updated_at"] == updated["updated_at"]
    assert service._event(replay)["currentRoundNumber"] == 2


def test_no_partial_court_or_score_write_when_transaction_fails():
    db = TransactionalHubDB()
    session(db, count=4)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    match = hub["workspace"]["eligible_queue"][0]
    hub = command(db, hub, "assign_next_court", {"session_key": "rr-a", "match_id": match["match_id"]})
    before_hubs = deepcopy(db.storage[service.HUB_TABLE])
    before_receipts = deepcopy(db.storage[service.OPERATION_TABLE])
    db.before_commit = lambda db: raw_session(db).update(version=raw_session(db)["version"] + 1)
    with pytest.raises(PublicLiveConflictError):
        command(db, hub, "score_and_release", {"court_id": "c1", "session_key": "rr-a", "match_id": match["match_id"], "score_a": 11, "score_b": 5})
    stored = service._event(raw_session(db))["rounds"][0]["matches"]
    assert all(match.get("scoreA") is None for match in stored)
    assert service._event(raw_session(db))["currentRoundNumber"] == 1
    assert db.storage[service.HUB_TABLE] == before_hubs
    assert db.storage[service.OPERATION_TABLE] == before_receipts
    attempted = db.calls[-1]["p_sessions"][0]["patch"]["state"]["page_state"]["event"]
    assert attempted["currentRoundNumber"] == 2
    assert attempted["rounds"][0]["status"] == "saved"
    detail = service.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])
    assert detail["workspace"]["courts"][0]["current"]["match_id"] == match["match_id"]


def test_detaching_group_clears_membership_markers_without_changing_schedule():
    db = TransactionalHubDB()
    session(db)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    event_before = service._event(raw_session(db))
    hub = command(db, hub, "remove_group", {"session_key": "rr-a"})
    row = raw_session(db)
    assert "round_robin_hub_id" not in row["state"]
    event_after = service._event(row)
    assert "roundRobinHubId" not in event_after
    assert "hubAssignedMatchIds" not in event_after
    assert event_after["rounds"] == event_before["rounds"]
    assert not hub["workspace"]["groups"]


def test_generic_patch_rejects_stale_numeric_version_even_when_timestamp_matches():
    db = TransactionalHubDB()
    session(db)
    command(db, create(db), "add_group", {"session_key": "rr-a"})
    before = deepcopy(raw_session(db))
    raw_session(db)["version"] += 1
    with pytest.raises(PublicLiveConflictError):
        service.persist_linked_generator_patch(db, before=before, patch={"title": "Stale title"}, expected_version=before["updated_at"])
    assert raw_session(db)["title"] == "rr-a"


def test_closed_hub_allows_score_correction_but_cannot_restart_round_robin():
    from jupr_app.domain.adaptive_play_engine import advance_generator_event, complete_generator_event, save_generator_round
    db = TransactionalHubDB()
    session(db, count=4)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    match = hub["workspace"]["eligible_queue"][0]
    hub = command(db, hub, "assign_next_court", {"session_key": "rr-a", "match_id": match["match_id"]})
    hub = command(db, hub, "score_and_release", {"court_id": "c1", "session_key": "rr-a", "match_id": match["match_id"], "score_a": 11, "score_b": 5})
    before = deepcopy(raw_session(db))
    completed = complete_generator_event(service._event(before))
    state = deepcopy(before["state"])
    state["page_state"]["event"] = completed
    service.persist_linked_generator_patch(db, before=before, patch={"state": state, "status": "completed"}, expected_version=before["updated_at"])
    hub = service.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])
    assert hub["hub"]["status"] == "completed", "The last generator completion closes its hub atomically"
    assert db.calls[-1]["p_status"] == "completed"

    before = deepcopy(raw_session(db))
    corrected = save_generator_round(service._event(before), round_number=1,
        scores=[{"match_id": match["match_id"], "score_a": 11, "score_b": 7}])
    state = deepcopy(before["state"])
    state["page_state"]["event"] = corrected
    updated = service.persist_linked_generator_patch(db, before=before, patch={"state": state}, expected_version=before["updated_at"])
    assert service._event(updated)["rounds"][0]["matches"][0]["scoreB"] == 7

    before = deepcopy(raw_session(db))
    with pytest.raises(ValueError, match="active|complete"):
        advance_generator_event(service._event(before))
    restarted = service._event(before)
    restarted.update(status="active", completedAt=None)
    state = deepcopy(before["state"])
    state["page_state"]["event"] = restarted
    snapshot = deepcopy(db.storage)
    with pytest.raises(ValueError, match="complete|closed|new hub"):
        service.persist_linked_generator_patch(db, before=before, patch={"state": state, "status": "active"}, expected_version=before["updated_at"])
    assert db.storage == snapshot


@pytest.mark.parametrize("slot", ["current", "next"])
def test_original_generator_cannot_skip_round_with_assigned_or_reserved_game(slot):
    from jupr_app.domain.adaptive_play_engine import skip_generator_round
    db = TransactionalHubDB()
    session(db, count=4)
    session(db, key="rr-b", count=4, offset=10)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    hub = command(db, hub, "add_group", {"session_key": "rr-b"})
    matches = {entry["session_key"]: entry for entry in hub["workspace"]["eligible_queue"]}
    if slot == "next":
        hub = command(db, hub, "assign_next_court", matches["rr-b"])
        hub = command(db, hub, "reserve_game_for_court", {**matches["rr-a"], "court_id": "c1"})
    else:
        hub = command(db, hub, "assign_next_court", matches["rr-a"])
    before = deepcopy(raw_session(db))
    state = deepcopy(before["state"])
    # Domain rejects the real request; independently verify the service also
    # rejects a caller-supplied skipped candidate that bypasses that guard.
    with pytest.raises(ValueError, match="[Rr]elease|[Aa]ssigned"):
        skip_generator_round(service._event(before), round_number=1)
    candidate = service._event(before)
    candidate["rounds"][0]["status"] = "skipped"
    for match in candidate["rounds"][0]["matches"]:
        match.update(scoreA=None, scoreB=None, status="not_played")
    state["page_state"]["event"] = candidate
    snapshot = deepcopy(db.storage)
    with pytest.raises(ValueError, match="[Rr]elease|[Aa]ssigned"):
        service.persist_linked_generator_patch(db, before=before, patch={"state": state}, expected_version=before["updated_at"])
    assert db.storage == snapshot
    command(db, hub, "cancel_reservation" if slot == "next" else "release_game", {"court_id": "c1"})
    before = deepcopy(raw_session(db))
    state = deepcopy(before["state"])
    state["page_state"]["event"] = skip_generator_round(service._event(before), round_number=1)
    updated = service.persist_linked_generator_patch(db, before=before, patch={"state": state}, expected_version=before["updated_at"])
    assert service._event(updated)["rounds"][0]["status"] == "skipped"


def test_original_generator_cannot_reopen_skipped_round_after_advancing():
    from jupr_app.domain.adaptive_play_engine import reopen_generator_round, skip_generator_round
    db = TransactionalHubDB()
    session(db, count=4)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    before = deepcopy(raw_session(db))
    state = deepcopy(before["state"])
    state["page_state"]["event"] = skip_generator_round(service._event(before), round_number=1)
    service.persist_linked_generator_patch(db, before=before, patch={"state": state}, expected_version=before["updated_at"])
    hub = service.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])
    assert hub["workspace"]["groups"][0]["current_round_number"] == 2
    before = deepcopy(raw_session(db))
    state = deepcopy(before["state"])
    with pytest.raises(ValueError, match="earlier"):
        reopen_generator_round(service._event(before), round_number=1)
    state["page_state"]["event"]["rounds"][0]["status"] = "active"
    snapshot = deepcopy(db.storage)
    with pytest.raises(ValueError, match="order|earlier|current round"):
        service.persist_linked_generator_patch(db, before=before, patch={"state": state}, expected_version=before["updated_at"])
    assert db.storage == snapshot


def test_original_generator_cannot_skip_a_future_hub_round():
    from jupr_app.domain.adaptive_play_engine import skip_generator_round
    db = TransactionalHubDB()
    session(db, count=4)
    command(db, create(db), "add_group", {"session_key": "rr-a"})
    before = deepcopy(raw_session(db))
    state = deepcopy(before["state"])
    with pytest.raises(ValueError, match="current"):
        skip_generator_round(service._event(before), round_number=2)
    state["page_state"]["event"]["rounds"][1]["status"] = "skipped"
    snapshot = deepcopy(db.storage)
    with pytest.raises(ValueError, match="order"):
        service.persist_linked_generator_patch(db, before=before, patch={"state": state}, expected_version=before["updated_at"])
    assert db.storage == snapshot


def test_hub_add_guest_is_atomic_and_exact_retry_does_not_duplicate_roster_or_reset_wait():
    db = TransactionalHubDB()
    session(db, count=4)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    hub = command(db, hub, "auto_fill_courts")
    original_assignment = deepcopy(hub["workspace"]["courts"][0]["current"])
    before_version = raw_session(db)["version"]
    added = command(db, hub, "add_player", {"session_key": "rr-a", "name": "  New   Guest "}, "arrival-once-key")
    after = deepcopy(db.storage)
    replayed = command(db, hub, "add_player", {"session_key": "rr-a", "name": "  New   Guest "}, "arrival-once-key")
    assert replayed["replayed"] and db.storage == after
    assert raw_session(db)["version"] == before_version + 1
    assert added["hub"]["version"] == replayed["hub"]["version"]
    players = added["workspace"]["groups"][0]["players"]
    assert len(players) == 5 and players[-1]["name"] == "New Guest"
    assert players[-1]["identity"].startswith("guest:rr-a:")
    assert players[-1]["profile_id"] is None and players[-1]["active"]
    assert {key: added["workspace"]["courts"][0]["current"][key] for key in ("session_key", "match_id", "side_a", "side_b", "assigned_at")} == {key: original_assignment[key] for key in ("session_key", "match_id", "side_a", "side_b", "assigned_at")}
    event = service._event(raw_session(db))
    assert players[-1]["participant_id"] in event["rounds"][0]["byeParticipantIds"]


def test_hub_add_profile_uses_club_canonical_name_and_rejects_duplicates_and_other_clubs():
    db = TransactionalHubDB()
    session(db, count=4)
    db.storage["players"] = [{"id": 20, "club_id": "club-a", "name": "Canonical Player"},
                             {"id": 21, "club_id": "club-b", "name": "Private Other Club"}]
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    hub = command(db, hub, "add_player", {"session_key": "rr-a", "name": "Client alias", "player_id": 20})
    added = hub["workspace"]["groups"][0]["players"][-1]
    assert added["name"] == "Canonical Player"
    assert added["profile_id"] == 20 and added["identity"] == "player:20"
    for player_id, message in [(20, "already in"), (21, "not found in this club"), (999, "not found in this club")]:
        snapshot = deepcopy(db.storage)
        with pytest.raises(ValueError, match=message):
            command(db, hub, "add_player", {"session_key": "rr-a", "name": "Another alias", "player_id": player_id})
        assert db.storage == snapshot


@pytest.mark.parametrize("player_id", [True, 0, -1, 1.5, "20"])
def test_hub_add_profile_requires_positive_integer_identity(player_id):
    db = TransactionalHubDB()
    session(db, count=4)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    snapshot = deepcopy(db.storage)
    with pytest.raises(ValueError, match="valid club player"):
        command(db, hub, "add_player", {"session_key": "rr-a", "name": "Arrival", "player_id": player_id})
    assert db.storage == snapshot


def test_hub_add_player_rejects_unattached_and_locked_sessions_without_writes():
    db = TransactionalHubDB()
    session(db, count=4)
    session(db, "rr-b", count=4, offset=10)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    snapshot = deepcopy(db.storage)
    with pytest.raises(ValueError, match="attached to this hub"):
        command(db, hub, "add_player", {"session_key": "rr-b", "name": "Arrival"})
    assert db.storage == snapshot
    raw_session(db)["state"]["generator_submission"] = {"status": "pending"}
    hub = service.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])
    assert not hub["workspace"]["groups"][0]["can_add_player"]
    assert "locked" in hub["workspace"]["groups"][0]["add_player_unavailable_reason"]
    snapshot = deepcopy(db.storage)
    with pytest.raises(ValueError, match="locked"):
        command(db, hub, "add_player", {"session_key": "rr-a", "name": "Arrival"})
    assert db.storage == snapshot


def test_hub_add_player_rolls_back_if_another_group_changes_before_joint_commit():
    db = TransactionalHubDB()
    session(db, count=4)
    session(db, "rr-b", count=4, offset=10)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    hub = command(db, hub, "add_group", {"session_key": "rr-b"})
    target_before = deepcopy(raw_session(db))
    hub_before = deepcopy(db.storage[service.HUB_TABLE])
    db.before_commit = lambda current: raw_session(current, "rr-b").update(version=99)
    with pytest.raises(PublicLiveConflictError, match="changed"):
        command(db, hub, "add_player", {"session_key": "rr-a", "name": "Arrival"}, "arrival-conflict-key")
    assert raw_session(db) == target_before
    assert db.storage[service.HUB_TABLE] == hub_before


def test_remove_waiting_player_is_immediate_replay_safe_and_keeps_historical_record():
    db = TransactionalHubDB()
    session(db, count=8)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    participant = hub["workspace"]["groups"][0]["players"][0]
    payload = {"session_key": "rr-a", "participant_id": participant["participant_id"]}
    removed = command(db, hub, "remove_player", payload, "remove-waiting-once")
    snapshot = deepcopy(db.storage)
    replay = command(db, hub, "remove_player", payload, "remove-waiting-once")
    assert replay["replayed"] and db.storage == snapshot
    group = removed["workspace"]["groups"][0]
    old = next(p for p in group["players"] if p["participant_id"] == participant["participant_id"])
    assert len(group["players"]) == 8 and not old["active"] and not old["can_remove"]
    assert old["inactive_from_round"] == 1
    event = service._event(raw_session(db))
    assert participant["participant_id"] not in event["rosterByRound"]["1"]
    assert all(participant["participant_id"] not in m["sideA"] + m["sideB"] for m in event["rounds"][0]["matches"])
    with pytest.raises(ValueError, match="already left"):
        command(db, removed, "remove_player", payload)


def test_remove_playing_and_reserved_players_preserves_assignments_and_excludes_next_round():
    db = TransactionalHubDB()
    session(db, count=8)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    first, second = hub["workspace"]["eligible_queue"]
    hub = command(db, hub, "assign_game", {**first, "court_id": "c1"})
    hub = command(db, hub, "reserve_game_for_court", {**second, "court_id": "c1"})
    before_matches = deepcopy(service._event(raw_session(db))["rounds"][0]["matches"])
    before_courts = deepcopy(db.storage[service.HUB_TABLE][0]["state"]["courts"])
    departing = [first["side_a"][0]["id"], second["side_a"][0]["id"]]
    for pid in departing:
        hub = command(db, hub, "remove_player", {"session_key": "rr-a", "participant_id": pid})
        assert service._event(raw_session(db))["rounds"][0]["matches"] == before_matches
        assert db.storage[service.HUB_TABLE][0]["state"]["courts"] == before_courts
    for match in (first, second):
        hub = command(db, hub, "score_and_release", {"session_key": "rr-a", "court_id": "c1", "match_id": match["match_id"], "score_a": 11, "score_b": 6})
    event = service._event(raw_session(db))
    assert event["currentRoundNumber"] == 2
    assert not set(departing) & set(event["rosterByRound"]["2"])
    assert event["rounds"][0]["status"] == "saved"


@pytest.mark.parametrize("restriction", ["locked", "completed", "moneyball", "playoff"])
def test_remove_player_restrictions_match_group_availability_and_do_not_write(restriction):
    db = TransactionalHubDB()
    session(db, count=4)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    raw = raw_session(db)
    event = raw["state"]["page_state"]["event"]
    if restriction == "locked":
        raw["state"]["generator_submission"] = {"status": "pending"}
    elif restriction == "completed":
        raw["status"] = event["status"] = "completed"
    elif restriction == "moneyball":
        event["eventFormat"] = "moneyball_expectations"
    else:
        event["playoff"] = {"format": "top_eight"}
    hub = service.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])
    group = hub["workspace"]["groups"][0]
    assert not group["can_remove_player"] and group["remove_player_unavailable_reason"]
    snapshot = deepcopy(db.storage)
    with pytest.raises(ValueError):
        command(db, hub, "remove_player", {"session_key": "rr-a", "participant_id": event["participants"][0]["id"]})
    assert db.storage == snapshot


def test_finish_selected_group_keeps_other_group_running_and_last_finish_closes_atomically():
    db = TransactionalHubDB()
    session(db, count=4)
    session(db, key="rr-b", count=4, offset=20)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    hub = command(db, hub, "add_group", {"session_key": "rr-b"})
    other = deepcopy(raw_session(db, "rr-b"))
    first = command(db, hub, "finish_group", {"session_key": "rr-a"})
    assert first["hub"]["status"] == "active"
    assert raw_session(db)["status"] == "completed"
    assert raw_session(db, "rr-b") == other
    assert {m["session_key"] for m in first["workspace"]["eligible_queue"]} == {"rr-b"}
    assert len(first["workspace"]["groups"]) == 2
    final = command(db, first, "finish_group", {"session_key": "rr-b"}, key="finish-final-once")
    assert final["hub"]["status"] == "completed"
    assert all(row["status"] == "completed" for row in db.storage["live_sessions"])
    assert db.calls[-1]["p_status"] == "completed"
    assert {entry["session_key"] for entry in db.calls[-1]["p_sessions"]} == {"rr-a", "rr-b"}
    assert all(not court["current"] and not court["next"] for court in final["workspace"]["courts"])
    before = deepcopy(db.storage)
    calls = len(db.calls)
    replay = command(db, first, "finish_group", {"session_key": "rr-b"}, key="finish-final-once")
    assert replay["replayed"] is True
    assert replay["hub"] == final["hub"]
    assert db.storage == before and len(db.calls) == calls


def test_finish_request_preserves_partial_scores_and_drains_before_last_hub_closes():
    db = TransactionalHubDB()
    session(db, count=8)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    first, second = [m for m in hub["workspace"]["eligible_queue"] if m["round_number"] == 1]
    hub = command(db, hub, "assign_next_court", first)
    hub = command(db, hub, "score_and_release", {**first, "court_id": "c1", "score_a": 11, "score_b": 5})
    hub = command(db, hub, "finish_group", {"session_key": "rr-a"})
    group = hub["workspace"]["groups"][0]
    assert group["finish_requested"] and group["finish_after_round"] == 1
    assert hub["hub"]["status"] == raw_session(db)["status"] == "active"
    assert all(row["matches"] == 0 for row in generator_event_standings(service._event(raw_session(db))))
    assert {m["round_number"] for m in hub["workspace"]["eligible_queue"]} == {1}
    hub = command(db, hub, "assign_next_court", second)
    hub = command(db, hub, "score_and_release", {**second, "court_id": "c1", "score_a": 11, "score_b": 7})
    assert hub["hub"]["status"] == raw_session(db)["status"] == "completed"
    assert not hub["workspace"]["eligible_queue"]
    event = service._event(raw_session(db))
    assert event["rounds"][0]["status"] == "saved"
    assert all(row["matches"] == 1 for row in generator_event_standings(event))
    assert {m["id"]: m["scoreB"] for m in event["rounds"][0]["matches"]} == {first["match_id"]: 5, second["match_id"]: 7}
    assert db.calls[-1]["p_status"] == "completed"


def test_finishing_last_group_rejects_concurrent_session_change_without_partial_close():
    db = TransactionalHubDB()
    session(db, count=4)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    before_hub = deepcopy(db.storage[service.HUB_TABLE])
    before_operations = deepcopy(db.storage[service.OPERATION_TABLE])

    def change_session(current):
        raw_session(current)["updated_at"] = "2026-10-20T23:00:00+00:00"
        raw_session(current)["version"] += 1

    db.before_commit = change_session
    with pytest.raises(PublicLiveConflictError):
        command(db, hub, "finish_group", {"session_key": "rr-a"})
    assert db.storage[service.HUB_TABLE] == before_hub
    assert db.storage[service.OPERATION_TABLE] == before_operations
    assert raw_session(db)["status"] == "active"


def test_finish_group_respects_submission_lock_and_route_allows_command():
    from services.api.admin_round_robin_hub_routes import HubCommandRequest
    request = HubCommandRequest(action="finish_group", payload={"session_key": "rr-a"},
        expected_version=2, idempotency_key="finish-locked-group")
    assert request.action == "finish_group"
    db = TransactionalHubDB()
    session(db, count=4)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    raw_session(db)["state"]["generator_submission"] = {"status": "pending"}
    before = deepcopy(db.storage)
    with pytest.raises(ValueError, match="locked"):
        command(db, hub, "finish_group", {"session_key": "rr-a"})
    detail = service.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])
    assert detail["workspace"]["groups"][0]["can_finish"] is False
    assert db.storage == before


def test_rolling_commands_and_linked_sharing_do_not_replan_submitted_siblings():
    db = TransactionalHubDB()
    session(db, count=4)
    session(db, key="rr-b", count=5, offset=20)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    hub = command(db, hub, "add_group", {"session_key": "rr-b"})
    locked = raw_session(db, "rr-b")
    locked["state"]["generator_submission"] = {"status": "pending"}
    locked_event = locked["state"]["page_state"]["event"]
    locked_event["rounds"] = locked_event["rounds"][:1]
    locked_event.pop("hubRollingEnabled", None)
    before_locked = deepcopy(locked)
    identity = hub["workspace"]["groups"][0]["players"][0]["identity"]
    hub = command(db, hub, "pause_player", {"identity": identity})
    assert raw_session(db, "rr-b") == before_locked
    before = deepcopy(raw_session(db))
    shared_state = deepcopy(before["state"])
    shared_state["generator_sharing"] = {"enabled": True}
    service.persist_linked_generator_patch(db, before=before, patch={"state": shared_state}, expected_version=before["updated_at"])
    assert raw_session(db, "rr-b") == before_locked
    assert raw_session(db)["state"]["generator_sharing"]["enabled"] is True
