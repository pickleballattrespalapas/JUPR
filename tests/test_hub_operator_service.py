"""Atomic orchestration for planned-round operator controls."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from tests.test_admin_round_robin_hub_service import (
    TransactionalHubDB, ReadQuery, command, create, raw_session, session,
)
from jupr_app.services import admin_round_robin_hub_service as hub_service
from jupr_app.services import admin_play_generator_service as generator_service
from jupr_app.services.public_live_operation_service import PublicLiveConflictError


def attached(count=9):
    db = TransactionalHubDB()
    session(db, count=count)
    return db, command(db, create(db), "add_group", {"session_key": "rr-a"})


def test_exact_lineup_and_batch_departure_use_one_receipt_and_replay_without_replanning():
    db, hub = attached()
    event = raw_session(db)["state"]["page_state"]["event"]
    match = event["rounds"][0]["matches"][0]
    side_a, side_b = match["sideA"], match["sideB"]
    payload = {"session_key": "rr-a", "match_id": match["id"],
        "side_a": [side_a[0], side_b[0]], "side_b": [side_a[1], side_b[1]]}
    request = dict(club_id="club-a", hub_id=hub["hub"]["id"], action="set_match_lineup",
        payload=payload, expected_version=hub["hub"]["version"],
        idempotency_key="exact-lineup-once", actor_email="operator@example.test")
    hub = hub_service.command_round_robin_hub(db, **request)
    after = deepcopy(db.storage)
    replay = hub_service.command_round_robin_hub(db, **request)
    assert replay["replayed"] is True and db.storage == after
    changed = raw_session(db)["state"]["page_state"]["event"]["rounds"][0]["matches"][0]
    assert changed["sideA"] == payload["side_a"] and changed["sideB"] == payload["side_b"]
    assert changed["hubPlanLocked"] is True
    departure = {**request, "action": "remove_players", "payload": {
        "session_key": "rr-a", "participant_ids": side_a},
        "expected_version": hub["hub"]["version"], "idempotency_key": "depart-together-once"}
    hub_service.command_round_robin_hub(db, **departure)
    after = deepcopy(db.storage)
    assert hub_service.command_round_robin_hub(db, **departure)["replayed"] is True
    assert db.storage == after


@pytest.mark.parametrize("action", ["set_match_lineup", "remove_players"])
def test_new_roster_controls_cannot_modify_submitted_sessions(action):
    db, hub = attached()
    event = raw_session(db)["state"]["page_state"]["event"]
    match = event["rounds"][0]["matches"][0]
    payload = ({"session_key": "rr-a", "match_id": match["id"],
        "side_a": match["sideA"], "side_b": match["sideB"]} if action == "set_match_lineup" else
        {"session_key": "rr-a", "participant_ids": [match["sideA"][0]]})
    raw_session(db)["state"]["generator_submission"] = {"status": "pending"}
    before = deepcopy(db.storage)
    with pytest.raises(ValueError, match="locked"):
        command(db, hub, action, payload)
    assert db.storage == before


def test_new_lineup_rejects_concurrent_session_change_without_partial_write():
    db, hub = attached()
    match = raw_session(db)["state"]["page_state"]["event"]["rounds"][0]["matches"][0]
    before_hub = deepcopy(db.storage[hub_service.HUB_TABLE])
    before_event = deepcopy(raw_session(db)["state"])
    def concurrent(current):
        raw_session(current)["version"] += 1
        raw_session(current)["updated_at"] = "2026-10-28T03:00:00+00:00"
    db.before_commit = concurrent
    with pytest.raises(PublicLiveConflictError):
        command(db, hub, "set_match_lineup", {"session_key": "rr-a", "match_id": match["id"],
            "side_a": match["sideB"], "side_b": match["sideA"]})
    assert db.storage[hub_service.HUB_TABLE] == before_hub
    assert raw_session(db)["state"] == before_event


def test_hub_session_payload_adds_provisional_view_without_changing_saved_standings():
    db, _ = attached(count=8)
    row = raw_session(db)
    match = row["state"]["page_state"]["event"]["rounds"][0]["matches"][0]
    match.update(scoreA=11, scoreB=7, status="scored")
    before = deepcopy(row)
    result = generator_service._session_payload(row)
    assert sum(s["matches"] for s in result["standings"]) == 0
    assert sum(s["matches"] for s in result["provisional_standings"]) == 4
    assert result["standings_view"]["final"] is False
    assert row == before
    standalone = deepcopy(row)
    standalone["state"].pop("round_robin_hub_id")
    assert "provisional_standings" not in generator_service._session_payload(standalone)


class InsertQuery(ReadQuery):
    def insert(self, value):
        self.insertion = deepcopy(value)
        return self

    def execute(self):
        if hasattr(self, "insertion"):
            value = {"version": 1, **self.insertion}
            self.db.storage[self.name].append(deepcopy(value))
            return SimpleNamespace(data=[value])
        return super().execute()


class ExtraDB(TransactionalHubDB):
    def table(self, name):
        return InsertQuery(self, name)


def extra_fixture():
    db = ExtraDB()
    session(db, count=7)
    session(db, key="rr-b", count=7, offset=20)
    hub = command(db, create(db, courts=2), "add_group", {"session_key": "rr-a"})
    hub = command(db, hub, "add_group", {"session_key": "rr-b"})
    refs = []
    for key in ("rr-a", "rr-b"):
        event = raw_session(db, key)["state"]["page_state"]["event"]
        refs.extend({"session_key": key, "participant_id": pid}
            for pid in event["rounds"][0]["byeParticipantIds"][:2])
    return db, hub, {"side_a": [refs[0], refs[2]], "side_b": [refs[1], refs[3]], "standings_policy": "exhibition"}


def test_extra_game_attach_is_replay_safe_and_keeps_source_memberships():
    db, hub, payload = extra_fixture()
    originals = {key: deepcopy(raw_session(db, key)["state"]["page_state"]["event"])
        for key in ("rr-a", "rr-b")}
    request = dict(club_id="club-a", hub_id=hub["hub"]["id"], action="create_extra_game", payload=payload,
        expected_version=hub["hub"]["version"], idempotency_key="cross-group-once", actor_email="operator@example.test")
    result = hub_service.command_round_robin_hub(db, **request)
    assert len(result["workspace"]["groups"]) == 3
    assert len(db.storage["live_sessions"]) == 3
    after = deepcopy(db.storage)
    assert hub_service.command_round_robin_hub(db, **request)["replayed"] is True
    assert db.storage == after
    for key, original in originals.items():
        assert raw_session(db, key)["state"]["page_state"]["event"] == original
    assert {entry["session_key"] for entry in db.calls[-1]["p_sessions"]} == {r["session_key"] for r in db.storage["live_sessions"]}


def test_exhibition_cannot_publish_after_it_is_detached_and_cannot_change_format():
    db, hub, payload = extra_fixture()
    command(db, hub, "create_extra_game", payload)
    row = next(r for r in db.storage["live_sessions"] if r["session_key"] not in {"rr-a", "rr-b"})
    detached = deepcopy(row)
    detached["state"].pop("round_robin_hub_id", None)
    with pytest.raises(ValueError, match="exhibitions"):
        generator_service._require_hub_ready_for_publication(detached)
    event = deepcopy(row["state"]["page_state"]["event"])
    event["participants"][0]["name"] = "Replacement"
    with pytest.raises(ValueError, match="original players"):
        generator_service._persist_event(db, before=row, event=event, expected_version=row["updated_at"])


@pytest.mark.parametrize("change", ["rating", "teams", "metadata", "roster", "rounds"])
def test_exhibition_policy_also_guards_direct_linked_generator_patches(change):
    db, hub, payload = extra_fixture()
    command(db, hub, "create_extra_game", payload)
    row = next(r for r in db.storage["live_sessions"] if r["session_key"] not in {"rr-a", "rr-b"})
    state = deepcopy(row["state"])
    event = state["page_state"]["event"]
    if change == "rating": event["ratingMode"] = "rated"
    if change == "teams": event["rounds"][0]["matches"][0]["sideA"].reverse()
    if change == "metadata": event.pop("hubSupplemental")
    if change == "roster": event["rosterByRound"]["1"].pop()
    if change == "rounds": event["rounds"].append({"number": 2, "matches": []})
    before = deepcopy(db.storage)
    with pytest.raises(ValueError, match="original players"):
        hub_service.persist_linked_generator_patch(db, before=row, patch={"state": state}, expected_version=row["updated_at"])
    assert db.storage == before


def test_exhibition_detail_editor_rejects_rated_before_attempting_a_write():
    from jupr_app.services.generator_session_details_service import _details
    with pytest.raises(ValueError, match="cannot be rated"):
        _details({"hubSupplemental": {"standingsPolicy": "exhibition"}, "scoringMode": "scored"},
            title="Extra game", rating_mode="rated")


def test_failed_attach_keeps_unattachable_draft_and_original_hub_unchanged():
    db, hub, payload = extra_fixture()
    before_hub = deepcopy(db.storage[hub_service.HUB_TABLE])
    def concurrent(current):
        raw_session(current, "rr-a")["version"] += 1
    db.before_commit = concurrent
    with pytest.raises(PublicLiveConflictError):
        command(db, hub, "create_extra_game", payload, "interrupted-extra")
    assert db.storage[hub_service.HUB_TABLE] == before_hub
    draft = next(r for r in db.storage["live_sessions"] if r["session_key"] not in {"rr-a", "rr-b"})
    assert draft["state"]["page_state"]["event"]["hubSupplementalDraft"] is True
    listed = hub_service.list_round_robin_hubs(db, club_id="club-a")
    assert draft["session_key"] not in {r["session_key"] for r in listed["sessions"]}
    generator_list = generator_service.list_play_generator_sessions(db, club_id="club-a")
    assert draft["session_key"] not in {r["session_key"] for r in generator_list["sessions"]}
    with pytest.raises(ValueError, match="saved extra-game"):
        generator_service._live_row(db, club_id="club-a", session_key=draft["session_key"])
    with pytest.raises(ValueError, match="saved extra-game"):
        command(db, hub, "add_group", {"session_key": draft["session_key"]})
