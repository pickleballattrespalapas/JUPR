from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_badge_diagnostics import badge_tables

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def _install_super_env(monkeypatch, supabase) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_badge_diagnostics_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_badge_diagnostics_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="super_admin"),
    )


def test_admin_badge_status_exposes_repair_endpoints(monkeypatch):
    supabase = FakeSupabase(badge_tables())
    _install_super_env(monkeypatch, supabase)

    response = TestClient(app).get("/admin/clubs/club/badges/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready_for_badge_diagnostics_and_repair"
    assert payload["recompute_endpoint"].endswith("/badges/recompute")
    assert payload["confirmation_text"]["revoke"] == "REVOKE BADGE"


def test_admin_badge_recompute_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(badge_tables())
    _install_super_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/badges/recompute",
        headers={"Authorization": "Bearer local"},
        json={"mode": "dry-run", "player_id": 1, "badge_id": "high_roller", "confirmation_text": "RUN"},
    )

    assert response.status_code == 400
    assert "RECOMPUTE BADGES" in response.json()["detail"]


def test_admin_badge_recompute_route_invokes_domain(monkeypatch):
    supabase = FakeSupabase(badge_tables())
    _install_super_env(monkeypatch, supabase)

    def fake_recompute(*_args, **kwargs):
        assert kwargs["mode"] == "dry-run"
        assert kwargs["player_id"] == 1
        return {"ok": True, "mode": "badge_recompute", "recompute_mode": "dry-run", "summary": {"new_awards_count": 0}}

    monkeypatch.setattr("services.api.admin_badge_diagnostics_routes.run_admin_badge_recompute", fake_recompute)

    response = TestClient(app).post(
        "/admin/clubs/club/badges/recompute",
        headers={"Authorization": "Bearer local"},
        json={"mode": "dry-run", "player_id": 1, "badge_id": "high_roller", "confirmation_text": "RECOMPUTE BADGES"},
    )

    assert response.status_code == 200
    assert response.json()["recompute_mode"] == "dry-run"


def test_admin_badge_revoke_updates_rows_and_audit(monkeypatch):
    tables = badge_tables()
    supabase = FakeSupabase(tables)
    _install_super_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/badges/revoke",
        headers={"Authorization": "Bearer local"},
        json={"player_badge_id": "pb1", "revoke_reason": "duplicate award", "confirmation_text": "REVOKE BADGE"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["revoked_count"] == 1
    assert tables["player_badges"][0]["revoked_at"]
    assert tables["player_badges"][0]["revoke_reason"] == "duplicate award"
    assert tables["admin_activity_log"]
