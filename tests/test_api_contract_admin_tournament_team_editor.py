from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def tournament_team_tables():
    return {
        "tournaments": [
            {
                "club_id": "club",
                "id": "tour_1",
                "name": "Spring Classic",
                "status": "PUBLISHED",
                "start_date": "2026-04-10",
                "end_date": "2026-04-12",
                "created_at": "2026-03-01T00:00:00Z",
            }
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex"},
            {"club_id": "club", "id": 2, "name": "Blair"},
            {"club_id": "club", "id": 3, "name": "Casey"},
            {"club_id": "club", "id": 4, "name": "Devon"},
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
            {
                "id": "old_team",
                "tournament_id": "tour_1",
                "draw_id": "draw_1",
                "registration_day_id": "day_1",
                "event_option_id": "event_1",
                "team_number": 1,
                "player1_id": 1,
                "player2_id": 2,
                "seed": 1,
                "source": "MANUAL",
            }
        ],
        "tournament_games": [],
        "tournament_podium": [],
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


def test_admin_tournament_draw_team_replace_contract(monkeypatch):
    tables = tournament_team_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).put(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams",
        headers={"Authorization": "Bearer local"},
        json={
            "teams": [
                {"team_number": 1, "player1_id": 1, "player2_id": 2, "seed": 1, "notes": "Top seed"},
                {"team_number": 2, "player1_id": 3, "player2_id": 4, "seed": 2, "notes": "Second seed"},
            ],
            "confirmation_text": "SAVE TEAMS",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "tournament_draw_team_replace"
    assert payload["draw_id"] == "draw_1"
    assert payload["updated_count"] == 2
    assert [row["team_number"] for row in payload["teams"]] == [1, 2]
    assert [row["player1_id"] for row in tables["tournament_teams"]] == [1, 3]
    assert {row["draw_id"] for row in tables["tournament_teams"]} == {"draw_1"}
    assert tables["admin_activity_log"][0]["action_type"] == "replace_tournament_draw_teams_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_tournament_ops_snapshot_includes_player_options(monkeypatch):
    tables = tournament_team_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).get(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/ops?draw_id=draw_1",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["teams"] == 1
    assert payload["players"][0]["name"] == "Alex"
    assert payload["teams"][0]["draw_id"] == "draw_1"


def test_admin_tournament_draw_team_replace_blocks_existing_games(monkeypatch):
    tables = tournament_team_tables()
    tables["tournament_games"].append({"id": "game_1", "tournament_id": "tour_1", "draw_id": "draw_1"})
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    _install_auth(monkeypatch)

    response = TestClient(app).put(
        "/admin/clubs/club/tournaments/admin/tournaments/tour_1/draws/draw_1/teams",
        headers={"Authorization": "Bearer local"},
        json={
            "teams": [{"team_number": 1, "player1_id": 1, "player2_id": 2}],
            "confirmation_text": "SAVE TEAMS",
        },
    )

    assert response.status_code == 400
    assert "already has games" in response.json()["detail"]
