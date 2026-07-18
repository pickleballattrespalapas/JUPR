from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def badge_tables() -> dict[str, list[dict]]:
    return {
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "rating": 1500, "wins": 10, "losses": 2, "matches_played": 12, "active": True},
            {"club_id": "club", "id": 2, "name": "Blair", "rating": 1300, "wins": 5, "losses": 5, "matches_played": 10, "active": True},
        ],
        "badges": [
            {
                "badge_id": "high_roller",
                "name": "High Roller",
                "state": "live",
                "state_changed_at": None,
                "state_change_reason": None,
            },
        ],
        "player_badges": [
            {"id": "pb1", "club_id": "club", "player_id": 1, "badge_id": "high_roller", "context_id": "season", "revoked_at": None},
        ],
        "admin_activity_log": [],
    }


def _install_env(monkeypatch, supabase) -> None:
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
        lambda **_kwargs: SimpleNamespace(role="club_owner", assigned=True, source="admin_role_assignments"),
    )


def test_admin_badge_diagnostics_status_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS", raising=False)
    response = TestClient(app).get("/admin/clubs/club/badges/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["options_endpoint"] is None


def test_admin_badge_diagnostics_options_contract(monkeypatch):
    supabase = FakeSupabase(badge_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/badges/options",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["player_count"] == 2
    high_roller = next(row for row in payload["badges"] if row["badge_id"] == "high_roller")
    assert high_roller["state"] == "live"
    assert high_roller["definition_found"] is True


def test_admin_badge_debug_route_contract(monkeypatch):
    supabase = FakeSupabase(badge_tables())
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr(
        "services.api.admin_badge_diagnostics_routes.build_admin_badge_debug",
        lambda *_args, **_kwargs: {"ok": True, "mode": "badge_debug", "report": {"candidates": [{"badge_id": "high_roller"}], "errors": []}},
    )

    response = TestClient(app).get(
        "/admin/clubs/club/badges/debug?player_id=1&badge_id=high_roller",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "badge_debug"
    assert payload["report"]["candidates"][0]["badge_id"] == "high_roller"


def test_admin_badge_audit_route_contract(monkeypatch):
    supabase = FakeSupabase(badge_tables())
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr(
        "services.api.admin_badge_diagnostics_routes.build_admin_badge_audit",
        lambda *_args, **_kwargs: {"ok": True, "mode": "badge_audit", "report": {"counts": {"missing_exact_count": 0}, "per_badge_summary": []}},
    )

    response = TestClient(app).get(
        "/admin/clubs/club/badges/audit?badge_id=high_roller&include_revoked=true",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "badge_audit"
    assert payload["report"]["counts"]["missing_exact_count"] == 0


def test_badge_definition_state_requires_manage_roles(monkeypatch):
    tables = badge_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/badges/high_roller/state",
        headers={"Authorization": "Bearer local"},
        json={
            "expected_state": "live",
            "target_state": "frozen",
            "reason": "Review",
            "confirmation_text": "UPDATE BADGE STATE",
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "insufficient permission"
    assert tables["badges"][0]["state"] == "live"
    assert tables["admin_activity_log"][-1]["after_json"]["required_permission"] == "manage_roles"
