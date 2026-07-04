from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_league_manager_service import FakeSupabase, fake_storage

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def _install_auth(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="owner@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_league_manager_status_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = TestClient(app).get("/admin/clubs/club/league-manager/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["leagues_endpoint"] is None


def test_league_manager_list_disabled_before_auth(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", raising=False)
    called = {"auth": False}

    def fake_auth(*_args, **_kwargs):
        called["auth"] = True
        raise AssertionError("auth should not run while league manager flag is disabled")

    monkeypatch.setattr("services.api.admin_league_manager_routes.authenticate_bearer", fake_auth)

    response = TestClient(app).get("/admin/clubs/club/league-manager/leagues")

    assert response.status_code == 403
    assert called == {"auth": False}


def test_league_manager_status_enabled_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(fake_storage()))

    response = TestClient(app).get("/admin/clubs/club/league-manager/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["leagues_endpoint"] == "/admin/clubs/{club_id}/league-manager/leagues"
    assert payload["league_count"] == 2


def test_league_manager_list_and_detail_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(fake_storage()))
    _install_auth(monkeypatch)

    client = TestClient(app)
    list_response = client.get("/admin/clubs/club/league-manager/leagues", headers={"Authorization": "Bearer local"})
    detail_response = client.get("/admin/clubs/club/league-manager/leagues/Open", headers={"Authorization": "Bearer local"})

    assert list_response.status_code == 200
    assert list_response.json()["count"] == 2
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["league"]["league_name"] == "Open"
    assert len(detail["schedule_preview"]) == 3
    assert detail["standings"][0]["player_name"] == "Alex"
