from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from services.api import admin_play_generator_routes as generator_routes
from services.api import admin_round_robin_hub_routes as routes
from jupr_app.services.admin_play_generator_service import build_play_generator_status
from tests.test_admin_round_robin_hub_service import TransactionalHubDB, command, create, session, guest_identity_hub


@pytest.fixture
def api(monkeypatch):
    db = TransactionalHubDB()

    def authenticate(value):
        if value != "Bearer club-a-operator":
            raise HTTPException(status_code=401, detail="Authentication required")
        return SimpleNamespace(email="operator@example.test", user_id="staff-a")

    monkeypatch.setattr(generator_routes, "authenticate_bearer", authenticate)
    monkeypatch.setattr(generator_routes, "resolve_admin_role", lambda **kwargs: SimpleNamespace(
        role="club_owner" if kwargs["club_id"] == "club-a" else "viewer"))
    monkeypatch.setattr(generator_routes, "write_admin_activity_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(routes, "_require_write_gate", lambda _club_id: None)
    app = FastAPI()
    routes.install_admin_round_robin_hub_routes(app, get_supabase_client=lambda: db)
    return TestClient(app), db


HEADERS = {"Authorization": "Bearer club-a-operator"}
BASE = "/admin/clubs/club-a/round-robin-hubs"


def create_body():
    return {"title": "Friday night", "courts": [{"id": "c1", "label": "Court 1"}], "idempotency_key": "create-hub-once"}


@pytest.fixture
def production_api(monkeypatch):
    """Exercise the real club-scoped gate using only a local RPC double."""
    for key, value in {
        "JUPR_ENV": "production",
        "FLY_APP_NAME": "juprleagues-api",
        "SUPABASE_URL": "https://dnoockbwfenunhcibwfn.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "local-test-service-role",
        "JUPR_PRODUCTION_WRITE_POLICY": "enabled",
        "JUPR_STAGING_WRITE_WAVE": "none",
        "JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE": "1",
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_JUPR_LIVE_WRITES": "0",
    }.items():
        monkeypatch.setenv(key, value)
    db = TransactionalHubDB()

    def authenticate(value):
        if value != HEADERS["Authorization"]:
            raise HTTPException(status_code=401, detail="Authentication required")
        return SimpleNamespace(email="operator@example.test", user_id="staff-a")

    monkeypatch.setattr(generator_routes, "authenticate_bearer", authenticate)
    # This operator has both club roles, so cross-club denial tests the write
    # policy itself rather than being masked by a role denial.
    monkeypatch.setattr(generator_routes, "resolve_admin_role", lambda **_kwargs: SimpleNamespace(role="club_owner"))
    monkeypatch.setattr(generator_routes, "write_admin_activity_log", lambda *_args, **_kwargs: None)
    app = FastAPI()
    routes.install_admin_round_robin_hub_routes(app, get_supabase_client=lambda: db)
    return TestClient(app), db


def test_production_tres_club_gate_allows_authenticated_create_and_command(production_api):
    client, db = production_api
    base = "/admin/clubs/tres_palapas/round-robin-hubs"
    assert build_play_generator_status(db, club_id="tres_palapas")["writes_enabled"] is True
    assert client.post(base, json=create_body()).status_code == 401
    assert not db.storage["round_robin_hubs"]
    created = client.post(base, headers=HEADERS, json=create_body())
    assert created.status_code == 200
    hub = created.json()["hub"]
    path = f"{base}/{hub['id']}/commands"
    body = {"action": "rename_hub", "payload": {"title": "Tres club night"},
        "expected_version": hub["version"], "idempotency_key": "rename-production-hub"}
    before = deepcopy(db.storage)
    assert client.post(path, json=body).status_code == 401
    assert db.storage == before
    changed = client.post(path, headers=HEADERS, json=body)
    assert changed.status_code == 200
    assert changed.json()["hub"]["title"] == "Tres club night"
    assert changed.json()["hub"]["version"] == hub["version"] + 1


@pytest.mark.parametrize("club_id,disabled_flag", [
    ("another_club", None),
    ("tres_palapas", "JUPR_PRODUCTION_WRITE_POLICY"),
    ("tres_palapas", "JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE"),
])
def test_production_gate_denies_other_clubs_and_disabled_writes_but_keeps_reads(
    production_api, monkeypatch, club_id, disabled_flag,
):
    client, db = production_api
    hub = create(db, club=club_id)["hub"]
    if disabled_flag:
        monkeypatch.setenv(disabled_flag, "0")
    base = f"/admin/clubs/{club_id}/round-robin-hubs"
    assert build_play_generator_status(db, club_id=club_id)["writes_enabled"] is False
    before = deepcopy(db.storage)
    assert client.get(base, headers=HEADERS).status_code == 200
    assert client.get(f"{base}/{hub['id']}", headers=HEADERS).status_code == 200
    assert client.post(base, headers=HEADERS, json=create_body()).status_code == 403
    assert client.post(f"{base}/{hub['id']}/commands", headers=HEADERS, json={
        "action": "rename_hub", "payload": {"title": "Blocked change"},
        "expected_version": hub["version"], "idempotency_key": "blocked-production-command",
    }).status_code == 403
    assert db.storage == before


def test_all_hub_reads_and_writes_require_staff_authentication(api):
    client, db = api
    hub = create(db)
    before = deepcopy(db.storage)
    assert client.get(BASE).status_code == 401
    assert client.get(f"{BASE}/{hub['hub']['id']}").status_code == 401
    assert client.post(BASE, json=create_body()).status_code == 401
    assert client.post(f"{BASE}/{hub['hub']['id']}/commands", json={
        "action": "finish_hub", "payload": {}, "expected_version": 1, "idempotency_key": "finish-once-key",
    }).status_code == 401
    assert db.storage == before


def test_club_role_is_resolved_for_requested_scope_before_any_data_access(api):
    client, db = api
    other = create(db, club="club-b")
    before = deepcopy(db.storage)
    assert client.get("/admin/clubs/club-b/round-robin-hubs", headers=HEADERS).status_code == 403
    assert client.get(f"/admin/clubs/club-b/round-robin-hubs/{other['hub']['id']}", headers=HEADERS).status_code == 403
    assert client.post("/admin/clubs/club-b/round-robin-hubs", headers=HEADERS, json=create_body()).status_code == 403
    assert db.storage == before


def test_wrong_club_hub_id_is_not_disclosed(api):
    client, db = api
    other = create(db, club="club-b")
    response = client.get(f"{BASE}/{other['hub']['id']}", headers=HEADERS)
    assert response.status_code == 404
    assert "club-b" not in response.text


def test_create_normalizes_labels_and_retries_same_operation(api):
    client, db = api
    body = create_body()
    body["title"] = "  Friday   night  "
    body["courts"][0]["label"] = "  Court   1  "
    first = client.post(BASE, headers=HEADERS, json=body)
    second = client.post(BASE, headers=HEADERS, json=body)
    assert first.status_code == second.status_code == 200
    assert first.json()["hub"]["id"] == second.json()["hub"]["id"]
    assert first.json()["hub"]["title"] == "Friday night"
    assert first.json()["workspace"]["courts"][0]["label"] == "Court 1"
    assert len(db.storage["round_robin_hubs"]) == 1


@pytest.mark.parametrize("change", [
    {"title": "   "}, {"courts": []}, {"courts": [{"id": "c1", "label": "  "}]},
    {"courts": [{"id": "c1", "label": "Court 1", "current": {"match_id": "injected"}}]},
    {"idempotency_key": "x"}, {"unexpected": True},
])
def test_invalid_setup_payload_cannot_create_hub(api, change):
    client, db = api
    response = client.post(BASE, headers=HEADERS, json={**create_body(), **change})
    assert response.status_code == 422
    assert not db.storage["round_robin_hubs"]


def test_commands_have_strict_action_version_and_identity_envelope(api):
    client, db = api
    hub = create(db)
    before = deepcopy(db.storage)
    path = f"{BASE}/{hub['hub']['id']}/commands"
    body = {"action": "finish_hub", "payload": {}, "expected_version": 1, "idempotency_key": "command-once"}
    for change in ({"action": "raw_sql"}, {"expected_version": 0}, {"actor_email": "owner@example.test"}, {"payload": {"data": "x" * 20001}}):
        assert client.post(path, headers=HEADERS, json={**body, **change}).status_code == 422
    assert db.storage == before


def test_stale_command_is_409_and_invalid_match_is_400(api):
    client, db = api
    session(db)
    hub = create(db)
    attached = command(db, hub, "add_group", {"session_key": "rr-a"})
    path = f"{BASE}/{hub['hub']['id']}/commands"
    before = deepcopy(db.storage)
    response = client.post(path, headers=HEADERS, json={
        "action": "assign_next_court", "payload": {"session_key": "rr-a", "match_id": "missing"},
        "expected_version": hub["hub"]["version"], "idempotency_key": "stale-assignment",
    })
    assert response.status_code == 409
    response = client.post(path, headers=HEADERS, json={
        "action": "assign_next_court", "payload": {"session_key": "rr-a", "match_id": "missing"},
        "expected_version": attached["hub"]["version"], "idempotency_key": "invalid-assignment",
    })
    assert response.status_code == 400
    assert db.storage == before


def test_read_remains_available_when_write_gate_closed(api, monkeypatch):
    client, db = api
    hub = create(db)

    def closed(_club_id):
        raise HTTPException(status_code=403, detail="Staging writes are closed")

    monkeypatch.setattr(routes, "_require_write_gate", closed)
    assert client.get(BASE, headers=HEADERS).status_code == 200
    assert client.get(f"{BASE}/{hub['hub']['id']}", headers=HEADERS).status_code == 200
    before = deepcopy(db.storage)
    assert client.post(BASE, headers=HEADERS, json=create_body()).status_code == 403
    assert db.storage == before


def test_pause_requires_no_reason_and_stops_match_eligibility(api):
    client, db = api
    session(db)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    player = hub["workspace"]["players"][0]
    response = client.post(f"{BASE}/{hub['hub']['id']}/commands", headers=HEADERS, json={
        "action": "pause_player", "payload": {"identity": player["identity"]},
        "expected_version": hub["hub"]["version"], "idempotency_key": "pause-no-reason",
    })
    assert response.status_code == 200
    result = response.json()
    assert next(row for row in result["workspace"]["players"] if row["identity"] == player["identity"])["paused"]
    assert any("A player is paused." in match["blockers"] for match in result["workspace"]["blocked_queue"])


def test_unknown_uuid_returns_not_found(api):
    client, _ = api
    assert client.get(f"{BASE}/{uuid4()}", headers=HEADERS).status_code == 404


def test_tournament_style_auto_fill_action_is_reachable(api):
    client, db = api
    session(db)
    hub = command(db, create(db, courts=2), "add_group", {"session_key": "rr-a"})
    response = client.post(f"{BASE}/{hub['hub']['id']}/commands", headers=HEADERS, json={
        "action": "auto_fill_courts", "payload": {}, "expected_version": hub["hub"]["version"], "idempotency_key": "auto-fill-once",
    })
    assert response.status_code == 200
    assert all(court["current"] is not None for court in response.json()["workspace"]["courts"])


def test_identity_batch_resolution_uses_authenticated_cas_and_exact_retry(api):
    client, db = api
    hub = guest_identity_hub(db)
    path = f"{BASE}/{hub['hub']['id']}/commands"
    body = {"action": "resolve_identities", "payload": {"resolution": "all_different"},
        "expected_version": hub["hub"]["version"], "idempotency_key": "identity-batch-once"}
    before = deepcopy(db.storage)
    assert client.post(path, json=body).status_code == 401
    assert db.storage == before
    first = client.post(path, headers=HEADERS, json=body)
    assert first.status_code == 200
    assert len(first.json()["workspace"]["eligible_queue"]) == 2
    second = client.post(path, headers=HEADERS, json=body)
    assert second.status_code == 200
    assert second.json()["replayed"] is True
    assert second.json()["hub"]["version"] == first.json()["hub"]["version"]
    assert client.post(path, headers=HEADERS, json={**body, "idempotency_key": "stale-resolution"}).status_code == 409


def test_add_player_command_returns_group_roster_and_replays_once(api):
    client, db = api
    session(db, count=4)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    body = {"action": "add_player", "payload": {"session_key": "rr-a", "name": "New arrival"},
            "expected_version": hub["hub"]["version"], "idempotency_key": "add-player-api-once"}
    path = f"{BASE}/{hub['hub']['id']}/commands"
    first = client.post(path, headers=HEADERS, json=body)
    retry = client.post(path, headers=HEADERS, json=body)
    assert first.status_code == retry.status_code == 200
    assert retry.json()["replayed"]
    group = first.json()["workspace"]["groups"][0]
    assert group["can_add_player"] and len(group["players"]) == 5
    assert group["players"][-1]["name"] == "New arrival"
    assert first.json()["hub"]["version"] == retry.json()["hub"]["version"]


@pytest.mark.parametrize("payload", [{"session_key": "rr-a", "name": "  "},
    {"session_key": "rr-a", "name": "x" * 161}, {"session_key": "rr-a", "name": ["not a name"]},
    {"session_key": "rr-a", "name": "Guest", "player_id": True}])
def test_add_player_invalid_payload_is_rejected_without_writes(api, payload):
    client, db = api
    session(db, count=4)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    snapshot = deepcopy(db.storage)
    response = client.post(f"{BASE}/{hub['hub']['id']}/commands", headers=HEADERS, json={
        "action": "add_player", "payload": payload, "expected_version": hub["hub"]["version"],
        "idempotency_key": "invalid-add-player"})
    assert response.status_code == 422
    assert db.storage == snapshot


def test_remove_player_command_is_scoped_and_replays_without_reason(api):
    client, db = api
    session(db, count=8)
    hub = command(db, create(db), "add_group", {"session_key": "rr-a"})
    pid = hub["workspace"]["groups"][0]["players"][0]["participant_id"]
    path = f"{BASE}/{hub['hub']['id']}/commands"
    body = {"action": "remove_player", "payload": {"session_key": "rr-a", "participant_id": pid},
        "expected_version": hub["hub"]["version"], "idempotency_key": "remove-player-api-once"}
    snapshot = deepcopy(db.storage)
    invalid = client.post(path, headers=HEADERS, json={**body, "payload": {"session_key": "other-group", "participant_id": pid}})
    assert invalid.status_code == 400 and db.storage == snapshot
    missing = client.post(path, headers=HEADERS, json={**body, "payload": {"session_key": "rr-a", "participant_id": "unknown"}})
    assert missing.status_code == 400 and db.storage == snapshot
    first = client.post(path, headers=HEADERS, json=body)
    replay = client.post(path, headers=HEADERS, json=body)
    assert first.status_code == replay.status_code == 200 and replay.json()["replayed"]
    player = next(p for p in first.json()["workspace"]["groups"][0]["players"] if p["participant_id"] == pid)
    assert not player["active"] and not player["can_remove"]
