from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def league_manager_tables() -> dict[str, list[dict]]:
    return {
        "leagues_metadata": [
            {
                "club_id": "club",
                "league_name": "Tuesday Ladder",
                "is_active": True,
                "status": "active",
                "k_factor": 32,
                "min_games": 3,
                "schedule_config": {},
                "court_board_defaults": {},
                "rules_config": {},
                "awards_config": {},
                "event_tags": {},
            }
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "active": True},
            {"club_id": "club", "id": 2, "name": "Blair", "active": True},
        ],
        "league_ratings": [
            {"club_id": "club", "id": 1, "player_id": 1, "league_name": "Tuesday Ladder", "rating": 1500, "wins": 2, "losses": 1, "matches_played": 3, "is_active": True},
        ],
        "admin_activity_log": [],
    }


def _install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_league_manager_settings_update_contract(monkeypatch):
    tables = league_manager_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={
            "status": "active",
            "k_factor": 28,
            "min_games": 4,
            "schedule_config": {"start_date": "2026-01-05", "weekday": 0, "weeks": 2, "time_start": "18:00", "time_end": "20:00"},
            "court_board_defaults": {"max_used_courts": 4, "players_per_court": "4"},
            "rules_config": {"format": "ladder"},
            "awards_config": {"top_performer": True},
            "event_tags": {"season": "winter"},
            "confirmation_text": "SAVE LEAGUE",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "league_manager_settings_update"
    assert payload["league"]["k_factor"] == 28
    assert payload["detail"]["schedule_preview"][0]["date"] == "2026-01-05"
    assert tables["leagues_metadata"][0]["min_games"] == 4
    assert tables["leagues_metadata"][0]["court_board_defaults"]["max_used_courts"] == 4
    assert tables["admin_activity_log"][0]["action_type"] == "update_league_manager_settings_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_league_manager_settings_update_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(league_manager_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={"k_factor": 28, "confirmation_text": "SAVE"},
    )

    assert response.status_code == 400
    assert "SAVE LEAGUE" in response.json()["detail"]
