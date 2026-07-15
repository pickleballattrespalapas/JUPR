from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def singles_tournament_tables() -> dict[str, list[dict]]:
    return {
        "tournaments": [{"club_id": "club", "id": "tour_1", "name": "Spring Classic", "status": "PUBLISHED", "start_date": "2026-04-10"}],
        "tournament_event_options": [{"id": "event_1", "tournament_id": "tour_1", "event_family_label": "Singles", "division_name": "4.0"}],
        "tournament_event_draws": [{"id": "draw_1", "tournament_id": "tour_1", "event_option_id": "event_1", "name": "Singles 4.0"}],
        "tournament_teams": [
            {"id": "team_1", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 1, "player1_id": 1, "player2_id": None},
            {"id": "team_2", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 2, "player1_id": 2, "player2_id": None},
        ],
        "tournament_games": [
            {
                "id": "game_1",
                "tournament_id": "tour_1",
                "draw_id": "draw_1",
                "stage": "PLAYOFF",
                "playoff_round": "Final",
                "team_a_id": "team_1",
                "team_b_id": "team_2",
                "score_a": 11,
                "score_b": 7,
                "winner_team_id": "team_1",
                "loser_team_id": "team_2",
                "finalized_at": "2026-04-10T17:00:00Z",
            }
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "rating": 1200, "singles_rating": 1200, "singles_wins": 0, "singles_losses": 0, "singles_matches_played": 0},
            {"club_id": "club", "id": 2, "name": "Blair", "rating": 1200, "singles_rating": 1200, "singles_wins": 0, "singles_losses": 0, "singles_matches_played": 0},
        ],
        "league_ratings": [],
        "leagues_metadata": [],
        "matches": [],
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


def _install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)


def test_admin_tournament_publish_singles_matches_contract(monkeypatch):
    tables = singles_tournament_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    captured: dict[str, object] = {}

    def fake_singles_process(match_list, **_kwargs):
        captured["match_list"] = match_list
        return {"inserted": len(match_list), "match_format": "singles"}

    monkeypatch.setattr("jupr_app.services.admin_tournament_match_publish_service.process_singles_matches", fake_singles_process)
    monkeypatch.setattr("jupr_app.services.admin_tournament_match_publish_service.process_matches", lambda *_args, **_kwargs: {"inserted": 0})

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH MATCHES", "playoff_winner_bonus_elo": 4},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["singles_match_count"] == 1
    assert payload["doubles_match_count"] == 0
    match_payload = captured["match_list"][0]
    assert match_payload["match_format"] == "singles"
    assert match_payload["match_type"] == "Tournament Singles"
    assert match_payload["t1_p1"] == 1
    assert match_payload["t2_p1"] == 2
    assert "t1_p2" not in match_payload
    assert match_payload["winner_bonus_elo"] == 4


def test_admin_tournament_publish_blocks_mixed_singles_doubles(monkeypatch):
    tables = singles_tournament_tables()
    tables["tournament_teams"][1]["player2_id"] = 3
    tables["players"].append({"club_id": "club", "id": 3, "name": "Casey", "rating": 1200, "singles_rating": 1200})
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH MATCHES"},
    )

    assert response.status_code == 400
    assert "either two singles teams or two doubles teams" in response.json()["detail"]
