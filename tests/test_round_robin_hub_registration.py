from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from jupr_app.services import round_robin_hub_registration_service as registration
from jupr_app.services import admin_round_robin_hub_service as hubs
from jupr_app.services.public_live_operation_service import PublicLiveConflictError
from services.api import public_round_robin_hub_routes as routes
from tests.test_admin_round_robin_hub_service import TransactionalHubDB, create, command, session, raw_session


class CheckinDB(TransactionalHubDB):
    """Intake persistence double. SQL suite exercises the real constraints/locks."""
    def rpc(self, name, params):
        if name != "submit_round_robin_hub_checkin":
            return super().rpc(name, params)
        def execute():
            row = next(r for r in self.storage[hubs.HUB_TABLE] if r["id"] == params["p_hub_id"])
            checkins = self.storage.setdefault(registration.CHECKIN_TABLE, [])
            if not any(r["hub_id"] == row["id"] and r["session_key"] == params["p_session_key"] and r["name_key"] == params["p_name_key"] for r in checkins):
                checkins.append({"id": str(uuid4()), "club_id": row["club_id"], "hub_id": row["id"], "session_key": params["p_session_key"],
                                 "name": params["p_name"], "name_key": params["p_name_key"], "created_at": "2026-10-21T18:00:00Z"})
            return SimpleNamespace(data={"ok": True, "private_field": "must not escape"})
        return SimpleNamespace(execute=execute)


@pytest.fixture
def event():
    db = CheckinDB()
    session(db, count=4)
    detail = command(db, create(db), "add_group", {"session_key": "rr-a"})
    return db, command(db, detail, "set_registration", {"open": True})


def checkin(db, hub, name="New Player"):
    response = registration.submit_hub_checkin(db, hub_id=hub["hub"]["id"], session_key="rr-a", name=name)
    assert set(response) == {"ok", "name", "message"}
    return db.storage[registration.CHECKIN_TABLE][-1]


def test_public_view_is_narrow_and_closed_by_default():
    db = CheckinDB(); session(db)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    value = registration.public_hub_registration(db, hub_id=hub["hub"]["id"])
    assert value["groups"] == [] and not value["event"]["registration_open"]
    command(db, hub, "set_registration", {"open": True})
    value = registration.public_hub_registration(db, hub_id=hub["hub"]["id"])
    assert set(value) == {"ok", "event", "groups"}
    assert value["groups"] == [{"session_key": "rr-a", "title": "rr-a"}]
    text = str(value)
    assert "operator@" not in text and "participants" not in text and "version" not in text


@pytest.mark.parametrize("change", ["finish", "submit", "completed"])
def test_unavailable_groups_not_offered(event, change):
    db, hub = event
    row = raw_session(db)
    if change == "finish": row["state"]["page_state"]["event"]["hubFinishAfterRound"] = 1
    if change == "submit": row["state"]["generator_submission"] = {"id": "submission"}
    if change == "completed": row["status"] = "completed"
    assert registration.public_hub_registration(db, hub_id=hub["hub"]["id"])["groups"] == []


def test_normalized_duplicate_and_inbox_does_not_change_court_version(event):
    db, hub = event
    original = deepcopy(db.storage[hubs.HUB_TABLE])
    checkin(db, hub, "  New   Player  "); checkin(db, hub, "new player")
    assert len(db.storage[registration.CHECKIN_TABLE]) == 1
    assert db.storage[hubs.HUB_TABLE] == original
    refreshed = hubs.get_round_robin_hub(db, club_id="club-a", hub_id=hub["hub"]["id"])
    assert refreshed["registration"]["pending_count"] == 1
    assert refreshed["registration"]["checkins"][0]["name"] == "New Player"


def test_add_is_atomic_and_exact_retry_preserves_assigned_games(event):
    db, hub = event
    hub = command(db, hub, "auto_fill_courts")
    courts = deepcopy(hub["workspace"]["courts"])
    c = checkin(db, hub)
    payload = {"checkin_id": c["id"], "disposition": "add", "name": c["name"]}
    saved = command(db, hub, "review_checkin", payload, "checkin-add-once")
    retry = command(db, hub, "review_checkin", payload, "checkin-add-once")
    for before, after in zip(courts, saved["workspace"]["courts"]):
        assert before["id"] == after["id"]
        for field in ("session_key", "match_id", "side_a", "side_b", "score_a", "score_b"):
            assert before["current"].get(field) == after["current"].get(field)
    assert len(saved["workspace"]["groups"][0]["players"]) == 5
    assert saved["registration"]["pending_count"] == 0
    assert retry["replayed"] and retry["hub"]["version"] == saved["hub"]["version"]
    with pytest.raises(ValueError, match="already"):
        command(db, saved, "review_checkin", payload)


def test_stale_hub_rolls_back_decision_and_roster(event):
    db, hub = event; c = checkin(db, hub)
    def conflict(current):
        current.storage[hubs.HUB_TABLE][0]["version"] += 1
    db.before_commit = conflict
    before = deepcopy(db.storage["live_sessions"])
    with pytest.raises(PublicLiveConflictError):
        command(db, hub, "review_checkin", {"checkin_id": c["id"], "disposition": "add"})
    assert db.storage["live_sessions"] == before
    assert not db.storage[hubs.HUB_TABLE][0]["state"].get("checkin_decisions")


def test_matching_existing_player_does_not_duplicate_or_resume(event):
    db, hub = event
    p = hub["workspace"]["groups"][0]["players"][0]
    hub = command(db, hub, "pause_player", {"identity": p["identity"]})
    c = checkin(db, hub, p["name"])
    saved = command(db, hub, "review_checkin", {"checkin_id": c["id"], "disposition": "existing", "participant_id": p["participant_id"]})
    group = saved["workspace"]["groups"][0]
    assert len(group["players"]) == 4 and group["players"][0]["paused"]
    assert saved["registration"]["checkins"][0]["participant_id"] == p["participant_id"]


def test_dismiss_restore_and_cross_event_review(event):
    db, hub = event; c = checkin(db, hub)
    dismissed = command(db, hub, "review_checkin", {"checkin_id": c["id"], "disposition": "dismiss"})
    assert dismissed["registration"]["checkins"][0]["status"] == "dismissed"
    restored = command(db, dismissed, "review_checkin", {"checkin_id": c["id"], "disposition": "restore"})
    assert restored["registration"]["pending_count"] == 1
    other = create(db, key="other-club-event", club="club-b")
    with pytest.raises(ValueError, match="not found"):
        hubs.command_round_robin_hub(db, club_id="club-b", hub_id=other["hub"]["id"], action="review_checkin",
            payload={"checkin_id": c["id"], "disposition": "dismiss"}, expected_version=1, idempotency_key="cross-club-checkin", actor_email="other@example.test")


def test_same_name_requires_explicit_match_and_removed_group_cannot_add(event):
    db, hub = event; c = checkin(db, hub, "Player 1")
    with pytest.raises(ValueError, match="already on the roster"):
        command(db, hub, "review_checkin", {"checkin_id": c["id"], "disposition": "add"})
    removed = command(db, hub, "remove_group", {"session_key": "rr-a"})
    with pytest.raises(ValueError, match="no longer accepting"):
        command(db, removed, "review_checkin", {"checkin_id": c["id"], "disposition": "add"})


@pytest.mark.parametrize("name", [" ", "A", "a" * 121, "Name\x00Here", "Name\u200bHidden"])
def test_invalid_names(name):
    with pytest.raises(ValueError): registration.clean_checkin_name(name)


def test_public_routes_no_auth_narrow_input_and_write_gate(event, monkeypatch):
    db, hub = event
    monkeypatch.setattr(routes, "require_public_intake_or_403", lambda: None)
    app = FastAPI(); routes.install_public_round_robin_hub_routes(app, get_supabase_client=lambda: db)
    client = TestClient(app); base = f"/public/round-robin-hubs/{hub['hub']['id']}"
    response = client.get(f"{base}/registration")
    assert response.status_code == 200 and response.headers["cache-control"] == "no-store"
    assert client.post(f"{base}/checkins", json={"name": "New Player", "session_key": "rr-a"}).status_code == 200
    assert client.post(f"{base}/checkins", json={"name": "New Player", "session_key": "rr-a", "player_id": 99}).status_code == 422
    assert client.get(f"/public/round-robin-hubs/{uuid4()}/registration").status_code == 404
    def closed(): raise HTTPException(status_code=403, detail="Writes closed")
    monkeypatch.setattr(routes, "require_public_intake_or_403", closed)
    assert client.post(f"{base}/checkins", json={"name": "Other Player", "session_key": "rr-a"}).status_code == 403
    assert client.get(f"{base}/registration").status_code == 200
