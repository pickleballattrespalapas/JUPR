from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def tournament_game_tables():
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
        "tournament_event_draws": [
            {
                "id": "draw_1",
                "tournament_id": "tour_1",
                "registration_day_id": "day_1",
                "event_option_id": "event_1",
                "name": "3.5 Draw",
                "status": "draft",
            }
        ],
        "tournament_teams": [
            {"id": "team_1", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 1, "player1_id": 1, "player2_id": 2},
            {"id": "team_2", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 2, "player1_id": 3, "player2_id": 4},
            {"id": "team_3", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 3, "player1_id": 5, "player2_id": 6},
            {"id": "team_4", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 4, "player1_id": 7, "player2_id": 8},
        ],
        "tournament_games": [],
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


def test_admin_tournament_round_robin_generation_contract(monkeypatch):
    tables = tournament_game_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/games/round-robin",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "GENERATE GAMES"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_round_robin_generate"
    assert payload["draw_id"] == "draw_1"
    assert payload["game_count"] == 6
    assert len(tables["tournament_games"]) == 6
    assert {row["draw_id"] for row in tables["tournament_games"]} == {"draw_1"}
    assert tables["tournament_games"][0]["stage"] == "ROUND_ROBIN"
    assert tables["admin_activity_log"][0]["action_type"] == "generate_tournament_round_robin_games_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_round_robin_generation_blocks_existing_games(monkeypatch):
    tables = tournament_game_tables()
    tables["tournament_games"].append({"id": "game_existing", "tournament_id": "tour_1", "draw_id": "draw_1"})
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/games/round-robin",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "GENERATE GAMES"},
    )

    assert response.status_code == 400
    assert "already has games" in response.json()["detail"]
