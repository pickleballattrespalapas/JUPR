from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def match_publish_tables() -> dict[str, list[dict]]:
    return {
        "tournaments": [
            {
                "club_id": "club",
                "id": "tour_1",
                "name": "Spring Classic",
                "status": "PUBLISHED",
                "start_date": "2026-04-10",
                "created_at": "2026-03-01T00:00:00Z",
            }
        ],
        "tournament_event_options": [
            {
                "id": "event_1",
                "tournament_id": "tour_1",
                "registration_day_id": "day_1",
                "event_family_label": "Men's Doubles",
                "division_name": "4.0",
            }
        ],
        "tournament_event_draws": [
            {
                "id": "draw_1",
                "tournament_id": "tour_1",
                "registration_day_id": "day_1",
                "event_option_id": "event_1",
                "name": "Men's Doubles 4.0",
                "status": "draft",
                "updated_at": "2026-04-10T16:00:00Z",
            }
        ],
        "tournament_teams": [
            {
                "id": "team_1",
                "tournament_id": "tour_1",
                "draw_id": "draw_1",
                "team_number": 1,
                "player1_id": 1,
                "player2_id": 2,
                "updated_at": "2026-04-10T16:00:00Z",
            },
            {
                "id": "team_2",
                "tournament_id": "tour_1",
                "draw_id": "draw_1",
                "team_number": 2,
                "player1_id": 3,
                "player2_id": 4,
                "updated_at": "2026-04-10T16:00:00Z",
            },
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
                "score_a": 11,
                "score_b": 8,
                "winner_team_id": "team_1",
                "loser_team_id": "team_2",
                "finalized_at": "2026-04-10T17:00:00Z",
                "updated_at": "2026-04-10T17:00:00Z",
            }
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 2, "name": "Blair", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 3, "name": "Casey", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 4, "name": "Devon", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
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


def test_admin_tournament_publish_matches_contract(monkeypatch):
    tables = match_publish_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    captured: dict[str, object] = {}

    def fake_process_matches(match_list, **kwargs):
        captured["match_list"] = match_list
        captured["club_id"] = kwargs.get("club_id")
        return {"inserted": len(match_list), "badge_summary": {"mode": "test"}, "player_update_queue": {"mode": "test"}}

    monkeypatch.setattr("jupr_app.services.admin_tournament_match_publish_service.process_matches", fake_process_matches)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH MATCHES"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_official_matches_publish"
    assert payload["match_count"] == 1
    match_payload = captured["match_list"][0]
    assert match_payload["match_type"] == "Tournament"
    assert match_payload["tournament_id"] == "tour_1"
    assert match_payload["tournament_game_id"] == "game_1"
    assert match_payload["t1_p1"] == 1
    assert match_payload["t1_p2"] == 2
    assert match_payload["t2_p1"] == 3
    assert match_payload["t2_p2"] == 4
    assert match_payload["score_t1"] == 11
    assert match_payload["score_t2"] == 8
    assert "Spring Classic" in match_payload["league"]
    assert tables["admin_activity_log"][0]["action_type"] == "publish_tournament_games_to_matches_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_publish_matches_applies_semifinal_bonus_to_winner_only_payload(monkeypatch):
    tables = match_publish_tables()
    tables["tournament_games"][0].update({"stage": "PLAYOFF", "playoff_round": "SF", "playoff_game_code": "P1"})
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    captured: dict[str, object] = {}

    def fake_process_matches(match_list, **kwargs):
        captured["match_list"] = match_list
        return {"inserted": len(match_list), "winner_bonus_summary": {"match_count": 1, "player_elo_total": 12.0}}

    monkeypatch.setattr("jupr_app.services.admin_tournament_match_publish_service.process_matches", fake_process_matches)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH MATCHES", "playoff_winner_bonus_elo": 6},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["playoff_winner_bonus_elo"] == 6.0
    assert payload["bonus_match_count"] == 1
    match_payload = captured["match_list"][0]
    assert match_payload["winner_bonus_elo"] == 6.0
    assert match_payload["rating_bonus_elo"] == 6.0
    assert match_payload["winner_bonus_reason"] == "tournament_semifinal_winner_bonus"
    assert tables["admin_activity_log"][0]["after_json"]["bonus_tournament_game_ids"] == ["game_1"]


def test_admin_tournament_publish_matches_blocks_duplicate(monkeypatch):
    tables = match_publish_tables()
    tables["matches"] = [{"club_id": "club", "tournament_id": "tour_1", "tournament_game_id": "game_1"}]
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr("jupr_app.services.admin_tournament_match_publish_service.process_matches", lambda *_args, **_kwargs: {"inserted": 1})

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH MATCHES"},
    )

    assert response.status_code == 400
    assert "already published" in response.json()["detail"]


def test_admin_tournament_publish_matches_requires_confirmation(monkeypatch):
    tables = match_publish_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH"},
    )

    assert response.status_code == 400
    assert "PUBLISH MATCHES" in response.json()["detail"]


def test_admin_tournament_publish_matches_requires_doubles(monkeypatch):
    tables = match_publish_tables()
    tables["tournament_teams"][0]["player2_id"] = None
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    monkeypatch.setattr("jupr_app.services.admin_tournament_match_publish_service.process_matches", lambda *_args, **_kwargs: {"inserted": 1})

    response = TestClient(app).post(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/matches/publish",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "PUBLISH MATCHES"},
    )

    assert response.status_code == 400
    assert "doubles teams" in response.json()["detail"]
