"""Arrival-driven hub: empty creation through durable intake and rolling play."""
from copy import deepcopy

import pytest

from jupr_app.domain.adaptive_play_engine import create_generator_preview, start_generator_event
from jupr_app.services import round_robin_hub_registration_service as registration
from jupr_app.services import admin_round_robin_hub_service as hubs
from tests.test_round_robin_hub_registration import CheckinDB
from tests.test_admin_round_robin_hub_service import create, command, session, raw_session


def empty_hub(fmt="doubles"):
    db = CheckinDB()
    session(db, count=4)
    event = start_generator_event(create_generator_preview(generator_kind="round_robin", play_format=fmt,
        title="3.0–3.5", participant_names=[], _hub_registration=True,
        doubles_court_count=1 if fmt == "doubles_singles" else 0,
        singles_court_count=1 if fmt == "doubles_singles" else 0))
    raw_session(db)["state"]["page_state"]["event"] = event
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    return db, command(db, hub, "set_registration", {"open": True})


def arrive(db, hub, name):
    registration.submit_hub_checkin(db, hub_id=hub["hub"]["id"], session_key="rr-a", name=name)
    return hubs.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])


@pytest.mark.parametrize("fmt,minimum", [("doubles", 4), ("singles", 2), ("doubles_singles", 2)])
def test_empty_group_generates_first_game_on_threshold(fmt, minimum):
    db, hub = empty_hub(fmt)
    assert hub["workspace"]["groups"][0]["players"] == []
    assert not hub["workspace"]["eligible_queue"]
    for number in range(1, minimum + 1):
        hub = arrive(db, hub, f"Guest {number}")
        assert len(hub["workspace"]["groups"][0]["players"]) == number
        assert bool(hub["workspace"]["eligible_queue"]) == (number >= minimum)
        assert hub["registration"]["pending_count"] == 0
    assert all(c["participant_id"] for c in hub["registration"]["checkins"])
    assert not any(c["current"] for c in hub["workspace"]["courts"]), "Staff dispatch ready games"


def test_arrivals_keep_live_game_and_continue_after_scoring():
    db, hub = empty_hub()
    for n in range(1, 5): hub = arrive(db, hub, f"Guest {n}")
    hub = command(db, hub, "auto_fill_courts")
    playing = next(c for c in hub["workspace"]["courts"] if c["current"])
    before = deepcopy(playing["current"])
    for n in range(5, 9): hub = arrive(db, hub, f"Guest {n}")
    after = next(c["current"] for c in hub["workspace"]["courts"] if c["id"] == playing["id"])
    for field in ("match_id", "side_a", "side_b", "score_a", "score_b"):
        assert after[field] == before[field]
    assert hub["workspace"]["eligible_queue"], "Four additional arrivals can play while first game is on court"
    hub = command(db, hub, "score_and_release", {"court_id": playing["id"],
        "session_key": before["session_key"], "match_id": before["match_id"], "score_a": 11, "score_b": 7})
    assert hub["workspace"]["eligible_queue"]
    assert hub["hub"]["status"] == "active"
    event = raw_session(db)["state"]["page_state"]["event"]
    assert any(m.get("scoreA") == 11 for r in event["rounds"] for m in r["matches"])


def test_duplicate_retry_does_not_add_twice_or_change_version():
    db, hub = empty_hub()
    hub = arrive(db, hub, "  Guest   One ")
    version = hub["hub"]["version"]
    again = arrive(db, hub, "guest one")
    assert again["hub"]["version"] == version
    assert len(again["workspace"]["groups"][0]["players"]) == 1
    assert len(again["registration"]["checkins"]) == 1


def test_existing_name_requires_review_and_is_not_retried_forever():
    db, hub = empty_hub()
    hub = command(db, hub, "add_player", {"session_key": "rr-a", "name": "Guest One"})
    hub = arrive(db, hub, "Guest One")
    assert len(hub["workspace"]["groups"][0]["players"]) == 1
    assert hub["registration"]["pending_count"] == 1
    assert hub["registration"]["automatic_pending_count"] == 0
    assert "already on the roster" in hub["registration"]["checkins"][0]["review_reason"]


def test_public_admission_retries_a_concurrent_court_change():
    db, hub = empty_hub()
    def concurrent(db):
        db.storage[hubs.HUB_TABLE][0]["version"] += 1
    db.before_commit = concurrent
    hub = arrive(db, hub, "Guest One")
    assert hub["registration"]["pending_count"] == 0
    assert len(hub["workspace"]["groups"][0]["players"]) == 1


def test_saved_intake_recovers_through_admin_command(monkeypatch):
    db, hub = empty_hub()
    with monkeypatch.context() as patch:
        patch.setattr(registration, "_process_saved_checkins", lambda *args, **kwargs: None)
        hub = arrive(db, hub, "Guest One")
    assert hub["registration"]["automatic_pending_count"] == 1
    hub = command(db, hub, "process_checkins")
    assert hub["registration"]["automatic_pending_count"] == 0
    assert len(hub["workspace"]["groups"][0]["players"]) == 1


def test_empty_session_can_finish_without_games():
    db, hub = empty_hub()
    hub = command(db, hub, "finish_group", {"session_key": "rr-a"})
    assert hub["hub"]["status"] == "completed"


@pytest.mark.parametrize("override", [{}, {"generator_kind": "ladder", "_hub_registration": True},
    {"event_format": "moneyball_expectations", "_hub_registration": True}])
def test_empty_creation_is_only_supported_for_registration_round_robins(override):
    with pytest.raises(ValueError):
        create_generator_preview(**{"generator_kind": "round_robin", "play_format": "doubles",
            "title": "No players", "participant_names": [], **override})


@pytest.mark.parametrize("hub_club,hub_status,allowed", [("club-a", "active", True),
    ("other-club", "active", False), ("club-a", "completed", False)])
def test_empty_session_creation_is_bound_to_active_club_hub(hub_club, hub_status, allowed):
    from tests.test_admin_play_generator_service import FakeSupabase
    from jupr_app.services.admin_play_generator_service import create_play_generator_session
    db = FakeSupabase()
    db.storage[hubs.HUB_TABLE] = [{"id": "hub-a", "club_id": hub_club, "status": hub_status}]
    args = dict(club_id="club-a", generator_kind="round_robin", play_format="doubles", title="3.5",
        participant_names=[], player_ids=[], total_rounds=1, court_count=0, preview_fingerprint=None,
        actor_email="staff@example.invalid", actor_role="administrator", source="test", registration_hub_id="hub-a")
    if allowed:
        result = create_play_generator_session(db, **args)["session"]
        assert result["event"]["participants"] == []
        assert result["event"]["hubRegistrationTargetId"] == "hub-a"
    else:
        with pytest.raises((ValueError, LookupError)):
            create_play_generator_session(db, **args)
        assert not db.storage["live_sessions"]
