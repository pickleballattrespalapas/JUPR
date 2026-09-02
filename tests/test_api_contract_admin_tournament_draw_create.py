from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def tournament_tables():
    return {
        "tournaments": [
            {
                "club_id": "club",
                "id": "tour_1",
                "name": "Spring Classic",
                "status": "PUBLISHED",
                "created_at": "2026-03-01T00:00:00Z",
            }
        ],
        "tournament_event_options": [
            {
                "id": "event_1",
                "tournament_id": "tour_1",
                "registration_day_id": "day_1",
                "event_family_label": "Gender Doubles",
                "division_name": "3.5",
                "event_format_default": "round_robin",
                "scoring_default": "one_game_to_11",
                "status": "open",
                "enabled": True,
                "sort_order": 1,
            }
        ],
        "tournament_event_draws": [],
        "admin_activity_log": [],
    }


def _install_auth(monkeypatch):
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_tournament_create_draw_contract(monkeypatch):
    tables = tournament_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws",
        headers={"Authorization": "Bearer local"},
        json={
            "event_option_id": "event_1",
            "name": "3.5 Friday Draw",
            "confirmation_text": "CREATE DRAW",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_draw_create"
    assert payload["draw"]["name"] == "3.5 Friday Draw"
    assert payload["draw"]["registration_day_id"] == "day_1"
    assert payload["draw"]["event_option_id"] == "event_1"
    assert tables["tournament_event_draws"][0]["status"] == "draft"
    assert tables["admin_activity_log"][0]["action_type"] == "create_tournament_event_draw_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_create_draw_refuses_disabled_or_cancelled_event(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    _install_auth(monkeypatch)

    for event_state in (
        {"enabled": False, "status": "open"},
        {"enabled": True, "status": "cancelled"},
        {"enabled": True, "status": "inactive"},
        {"enabled": True, "status": "archived"},
    ):
        tables = tournament_tables()
        tables["tournament_event_options"][0].update(event_state)
        supabase = FakeSupabase(tables)
        monkeypatch.setattr(
            "services.api.main.create_client",
            lambda _url, _credential, current=supabase: current,
        )

        response = TestClient(app).post(
            "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws",
            headers={"Authorization": "Bearer local"},
            json={
                "event_option_id": "event_1",
                "name": "Cancelled 3.0 Draw",
                "confirmation_text": "CREATE DRAW",
            },
        )

        assert response.status_code == 400
        assert "disabled or cancelled event" in response.json()["detail"]
        assert tables["tournament_event_draws"] == []
        assert tables["admin_activity_log"] == []
