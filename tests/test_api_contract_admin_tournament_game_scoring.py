from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def tournament_scoring_tables():
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
        "tournament_games": [
            {
                "id": "game_1",
                "tournament_id": "tour_1",
                "draw_id": "draw_1",
                "stage": "ROUND_ROBIN",
                "rr_round_number": 1,
                "rr_slot_number": 1,
                "team_a_id": "team_1",
                "team_b_id": "team_2",
            }
        ],
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


def test_admin_tournament_round_robin_score_contract(monkeypatch):
    tables = tournament_scoring_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/games/game_1/score",
        headers={"Authorization": "Bearer local"},
        json={"score_a": 11, "score_b": 7, "confirmation_text": "SAVE SCORE"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_round_robin_score"
    assert payload["game"]["score_a"] == 11
    assert payload["game"]["score_b"] == 7
    assert payload["game"]["winner_team_id"] == "team_1"
    assert payload["game"]["loser_team_id"] == "team_2"
    assert tables["tournament_games"][0]["winner_team_id"] == "team_1"
    assert tables["admin_activity_log"][0]["action_type"] == "score_tournament_round_robin_game_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_round_robin_score_blocks_ties(monkeypatch):
    tables = tournament_scoring_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/games/game_1/score",
        headers={"Authorization": "Bearer local"},
        json={"score_a": 9, "score_b": 9, "confirmation_text": "SAVE SCORE"},
    )

    assert response.status_code == 400
    assert "Ties" in response.json()["detail"]


def test_admin_tournament_round_robin_score_requires_confirmation(monkeypatch):
    tables = tournament_scoring_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).patch(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/games/game_1/score",
        headers={"Authorization": "Bearer local"},
        json={"score_a": 11, "score_b": 7, "confirmation_text": "SAVE"},
    )

    assert response.status_code == 400
    assert "SAVE SCORE" in response.json()["detail"]
