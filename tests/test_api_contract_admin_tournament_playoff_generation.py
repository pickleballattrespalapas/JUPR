from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def tournament_playoff_tables(*, complete=True, existing_playoff=False):
    games = [
        {"id": "rr_1", "tournament_id": "tour_1", "draw_id": "draw_1", "stage": "ROUND_ROBIN", "team_a_id": "team_1", "team_b_id": "team_2", "score_a": 11, "score_b": 7, "winner_team_id": "team_1"},
        {"id": "rr_2", "tournament_id": "tour_1", "draw_id": "draw_1", "stage": "ROUND_ROBIN", "team_a_id": "team_1", "team_b_id": "team_3", "score_a": 11, "score_b": 8, "winner_team_id": "team_1"},
        {"id": "rr_3", "tournament_id": "tour_1", "draw_id": "draw_1", "stage": "ROUND_ROBIN", "team_a_id": "team_1", "team_b_id": "team_4", "score_a": 11, "score_b": 9, "winner_team_id": "team_1"},
        {"id": "rr_4", "tournament_id": "tour_1", "draw_id": "draw_1", "stage": "ROUND_ROBIN", "team_a_id": "team_2", "team_b_id": "team_3", "score_a": 11, "score_b": 9, "winner_team_id": "team_2"},
        {"id": "rr_5", "tournament_id": "tour_1", "draw_id": "draw_1", "stage": "ROUND_ROBIN", "team_a_id": "team_2", "team_b_id": "team_4", "score_a": 11, "score_b": 8, "winner_team_id": "team_2"},
        {"id": "rr_6", "tournament_id": "tour_1", "draw_id": "draw_1", "stage": "ROUND_ROBIN", "team_a_id": "team_3", "team_b_id": "team_4", "score_a": 11, "score_b": 6, "winner_team_id": "team_3"},
    ]
    if not complete:
        games[0].pop("winner_team_id")
    if existing_playoff:
        games.append({"id": "p_existing", "tournament_id": "tour_1", "draw_id": "draw_1", "stage": "PLAYOFF", "playoff_game_code": "P1"})
    return {
        "tournaments": [{"club_id": "club", "id": "tour_1", "name": "Spring Classic", "status": "PUBLISHED"}],
        "tournament_event_draws": [{"id": "draw_1", "tournament_id": "tour_1", "registration_day_id": "day_1", "event_option_id": "event_1", "name": "3.5 Draw"}],
        "tournament_teams": [
            {"id": "team_1", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 1},
            {"id": "team_2", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 2},
            {"id": "team_3", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 3},
            {"id": "team_4", "tournament_id": "tour_1", "draw_id": "draw_1", "team_number": 4},
        ],
        "tournament_games": games,
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


def test_admin_tournament_playoff_generation_contract(monkeypatch):
    tables = tournament_playoff_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/games/playoffs",
        headers={"Authorization": "Bearer local"},
        json={"advance_count": 4, "confirmation_text": "GENERATE PLAYOFFS"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_playoff_generate"
    assert payload["advance_count"] == 4
    assert payload["game_count"] == 4
    playoff_rows = [row for row in tables["tournament_games"] if row.get("stage") == "PLAYOFF"]
    assert [row["playoff_game_code"] for row in playoff_rows] == ["P1", "P2", "P3", "P4"]
    assert tables["admin_activity_log"][0]["action_type"] == "generate_tournament_playoff_games_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_playoff_generation_requires_complete_rr(monkeypatch):
    tables = tournament_playoff_tables(complete=False)
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/games/playoffs",
        headers={"Authorization": "Bearer local"},
        json={"advance_count": 4, "confirmation_text": "GENERATE PLAYOFFS"},
    )

    assert response.status_code == 400
    assert "round-robin games must be scored" in response.json()["detail"]


def test_admin_tournament_playoff_generation_blocks_existing_playoffs(monkeypatch):
    tables = tournament_playoff_tables(existing_playoff=True)
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/games/playoffs",
        headers={"Authorization": "Bearer local"},
        json={"advance_count": 4, "confirmation_text": "GENERATE PLAYOFFS"},
    )

    assert response.status_code == 400
    assert "already has playoff games" in response.json()["detail"]
