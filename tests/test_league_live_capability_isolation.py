from __future__ import annotations

from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import pytest

from tests.conftest import require_api_dependency
from tests.test_admin_match_uploader_service import FakeSupabase, fake_load_data, fake_storage

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from jupr_app.services import admin_match_uploader_service as uploader
from services.api.admin_league_manager_routes import install_admin_league_manager_routes
from services.api.admin_match_uploader_routes import install_admin_match_uploader_routes


LIVE = "/admin/clubs/club/league-manager/live"
UPLOADER = "/admin/clubs/club/match-uploader"
HEADERS = {"Authorization": "Bearer local"}
PREVIEW = {"courts": [{"court": 1, "format_type": "4-Player", "player_names": ["Alex", "Blair", "Casey", "Devon"]}]}


@pytest.fixture
def live_client(monkeypatch):
    for key, value in {
        "JUPR_ENV": "staging",
        "JUPR_PRODUCTION_WRITE_POLICY": "disabled",
        "JUPR_STAGING_WRITE_WAVE": "open",
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES": "1",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER": "1",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN": "1",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT": "1",
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER": "0",
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_PREVIEW": "0",
        "SUPABASE_SERVICE_ROLE_KEY": "server-only",
    }.items():
        monkeypatch.setenv(key, value)
    supabase = FakeSupabase(fake_storage())

    def authenticate(authorization):
        if authorization != HEADERS["Authorization"]:
            raise HTTPException(status_code=401, detail="missing bearer token")
        return SimpleNamespace(email="admin@example.com", user_id="admin-1")

    for module in ("admin_league_manager_routes", "admin_match_uploader_routes"):
        monkeypatch.setattr(f"services.api.{module}.authenticate_bearer", authenticate)
        monkeypatch.setattr(f"services.api.{module}.resolve_admin_role", lambda **_kwargs: SimpleNamespace(role="club_owner"))
    app = FastAPI()
    install_admin_league_manager_routes(app, get_supabase_client=lambda: supabase)
    install_admin_match_uploader_routes(app, get_supabase_client=lambda: supabase)
    return TestClient(app), supabase


def _player_payload():
    players = [{"name": "New Player", "starting_jupr": 3.25}]
    return {
        "players": players,
        "reviewed_fingerprint": uploader.match_uploader_player_batch_fingerprint(players),
        "idempotency_key": "live-setup-players-001",
        "confirmation_text": "CREATE PLAYERS",
    }


def test_live_preview_and_status_are_independent_of_uploader(live_client):
    client, _ = live_client
    status = client.get(f"{LIVE}/status").json()
    assert status["enabled"] is True
    assert status["player_create_enabled"] is True
    assert status["round_robin_format_options"] == ["4-Player", "5-Player"]
    assert status["round_robin_preview_endpoint"].endswith("/live/round-robin/preview")
    response = client.post(f"{LIVE}/round-robin/preview", headers=HEADERS, json=PREVIEW)
    assert response.status_code == 200, response.text
    assert response.json()["match_count"] == 3
    for path, payload in (("round-robin/preview", PREVIEW), ("players", _player_payload()), ("batch", {"matches": [], "idempotency_key": "denied-direct-001"})):
        response = client.post(f"{UPLOADER}/{path}", headers=HEADERS, json={**payload, "source": "next_league_live_round_publish", "allow_league_live": True})
        assert response.status_code == 403, response.text


def test_live_player_creation_and_recovery_use_their_own_club_scoped_workflow(live_client):
    client, supabase = live_client
    payload = _player_payload()
    response = client.post(f"{LIVE}/players", headers=HEADERS, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["created_count"] == 1
    assert response.json()["players"][0]["rating"] == 1300
    operation = supabase.storage["admin_guarded_operations"][0]
    assert operation["workflow"] == "league_live_player_batch"
    assert operation["club_id"] == "club"
    key = payload["idempotency_key"]
    assert client.get(f"{LIVE}/player-operations/{key}", headers=HEADERS).json()["status"] == "completed"
    reconcile = client.post(f"{LIVE}/player-operations/{key}/reconcile", headers=HEADERS, json={"confirmation_text": "RECONCILE PLAYER BATCH"})
    assert reconcile.status_code == 200, reconcile.text
    retry = client.post(f"{LIVE}/players", headers=HEADERS, json=payload)
    assert retry.status_code == 200, retry.text
    assert len([player for player in supabase.storage["players"] if player["name"] == "New Player"]) == 1
    foreign = client.get(f"{LIVE.replace('/club/', '/other-club/')}/player-operations/{key}", headers=HEADERS)
    assert foreign.status_code == 404
    operation["workflow"] = "match_uploader_player_batch"
    assert client.get(f"{LIVE}/player-operations/{key}", headers=HEADERS).status_code == 404
    assert client.post(f"{LIVE}/player-operations/{key}/reconcile", headers=HEADERS, json={"confirmation_text": "RECONCILE PLAYER BATCH"}).status_code == 400


@pytest.mark.parametrize("path,payload", [("round-robin/preview", PREVIEW), ("players", _player_payload())])
def test_live_setup_authenticates_and_checks_club_permission(live_client, monkeypatch, path, payload):
    client, supabase = live_client
    assert client.post(f"{LIVE}/{path}", json=payload).status_code == 401
    seen = []

    def denied(**kwargs):
        seen.append(kwargs["club_id"])
        return SimpleNamespace(role="viewer")

    monkeypatch.setattr("services.api.admin_league_manager_routes.resolve_admin_role", denied)
    assert client.post(f"{LIVE}/{path}", headers=HEADERS, json=payload).status_code == 403
    assert seen == ["club"]
    assert len(supabase.storage["players"]) == 4


def test_live_preview_cannot_resolve_another_clubs_players(live_client):
    client, supabase = live_client
    supabase.storage["players"][0]["club_id"] = "other-club"
    response = client.post(f"{LIVE}/round-robin/preview", headers=HEADERS, json=PREVIEW)
    assert response.status_code == 200, response.text
    assert response.json()["missing_players"] == ["Alex"]
    assert response.json()["match_count"] == 0


@pytest.mark.parametrize("flag", ["JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN"])
def test_disabled_live_domain_cannot_be_bypassed_by_uploader_flag(live_client, monkeypatch, flag):
    client, _ = live_client
    monkeypatch.setenv(flag, "0")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    assert client.post(f"{LIVE}/round-robin/preview", headers=HEADERS, json=PREVIEW).status_code == 403
    assert client.post(f"{LIVE}/players", headers=HEADERS, json=_player_payload()).status_code == 403


@pytest.mark.parametrize("flag,value", [("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT", "0"), ("JUPR_ENV", "production")])
def test_live_player_writes_require_live_publish_capability(live_client, monkeypatch, flag, value):
    client, _ = live_client
    monkeypatch.setenv(flag, value)
    assert client.post(f"{LIVE}/players", headers=HEADERS, json=_player_payload()).status_code == 403
    assert client.get(f"{LIVE}/status").json()["player_create_enabled"] is False
    assert client.post(f"{LIVE}/round-robin/preview", headers=HEADERS, json=PREVIEW).status_code == 200


def test_live_setup_requires_server_key_and_staging_write_gate(live_client, monkeypatch):
    client, _ = live_client
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY")
    assert client.post(f"{LIVE}/round-robin/preview", headers=HEADERS, json=PREVIEW).status_code == 503
    status = client.get(f"{LIVE}/status").json()
    assert status["round_robin_preview_endpoint"] is None
    assert status["player_create_enabled"] is False
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES", "0")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    assert client.post(f"{LIVE}/players", headers=HEADERS, json=_player_payload()).status_code == 403
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "league-live-submit")
    response = client.post(f"{LIVE}/players", headers=HEADERS, json=_player_payload())
    assert response.status_code == 200, response.text


def test_live_preview_rejects_unsupported_court_sizes(live_client):
    client, _ = live_client
    payload = {"courts": [{**PREVIEW["courts"][0], "format_type": "6-Player"}]}
    response = client.post(f"{LIVE}/round-robin/preview", headers=HEADERS, json=payload)
    assert response.status_code == 400
    assert "4-Player and 5-Player" in response.json()["detail"]


@pytest.mark.parametrize("context", [{}, {"context_type": "event", "context_name": "Not Live"}, {"context_type": "league_live_session", "context_id": "not-a-uuid"}])
def test_live_internal_publish_rejects_unscoped_rows_before_any_write(live_client, monkeypatch, context):
    _, supabase = live_client
    valid = {"date": "2026-10-16", "league": "Open", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 8, "context_type": "league_live_session", "context_id": str(uuid5(NAMESPACE_URL, "live-round"))}
    invalid = {**{key: value for key, value in valid.items() if not key.startswith("context_")}, **context}
    with pytest.raises((PermissionError, ValueError)):
        uploader.submit_admin_match_uploader_batch(supabase, club_id="club", matches=[valid, invalid], actor_email="admin@example.com", actor_role="club_owner", idempotency_key="live-invalid-batch-001", allow_league_live_context=True)
    assert supabase.storage["events"] == []
    assert len(supabase.storage["matches"]) == 1


def test_internal_live_batch_uses_real_shared_service_with_uploader_disabled(live_client, monkeypatch):
    _, supabase = live_client
    captured = []
    monkeypatch.setattr(uploader, "load_data", fake_load_data)
    monkeypatch.setattr(uploader, "submit_atomic_direct_matches", lambda _supabase, **kwargs: captured.append(kwargs) or {"ok": True, "committed": True})
    match = {"date": "2026-10-16", "league": "Open", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 8, "context_type": "league_live_session", "context_id": str(uuid5(NAMESPACE_URL, "live-round"))}
    kwargs = dict(club_id="club", matches=[match], actor_email="admin@example.com", actor_role="club_owner", idempotency_key="live-valid-batch-001")
    assert uploader.submit_admin_match_uploader_batch(supabase, **kwargs, allow_league_live_context=True)["ok"] is True
    assert captured[0]["matches"][0]["context_type"] == "league_live_session"
    with pytest.raises(PermissionError, match="Uploader is disabled"):
        uploader.submit_admin_match_uploader_batch(supabase, **kwargs, source="next_league_live_round_publish")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT", "0")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    with pytest.raises(PermissionError, match="publish is disabled"):
        uploader.submit_admin_match_uploader_batch(supabase, **kwargs, allow_league_live_context=True)
