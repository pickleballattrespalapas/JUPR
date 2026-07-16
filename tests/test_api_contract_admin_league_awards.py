from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def _storage() -> dict[str, list[dict]]:
    return {
        "leagues_metadata": [
            {
                "club_id": "club",
                "league_name": "Open",
                "is_active": True,
                "status": "active",
                "min_games": 2,
                "awards_config": {"default_min_games": 2, "default_depth": 1},
            }
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex"},
            {"club_id": "club", "id": 2, "name": "Blair"},
            {"club_id": "club", "id": 3, "name": "Casey"},
        ],
        "league_ratings": [
            {"club_id": "club", "player_id": 1, "league_name": "Open", "rating": 1600, "starting_rating": 1400, "wins": 5, "losses": 1, "matches_played": 6, "is_active": True},
            {"club_id": "club", "player_id": 2, "league_name": "Open", "rating": 1500, "starting_rating": 1500, "wins": 4, "losses": 2, "matches_played": 6, "is_active": True},
            {"club_id": "club", "player_id": 3, "league_name": "Open", "rating": 1300, "starting_rating": 1320, "wins": 1, "losses": 5, "matches_played": 6, "is_active": True},
        ],
        "admin_activity_log": [],
    }


def _install(monkeypatch, supabase) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="owner@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_league_awards_preview_contract(monkeypatch):
    supabase = FakeSupabase(_storage())
    _install(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/league-manager/leagues/Open/awards/preview",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "league_awards_preview"
    assert payload["award_count"] >= 1
    assert any(row["player_name"] == "Alex" for row in payload["awards"])


def test_admin_league_awards_close_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(_storage())
    _install(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues/Open/awards/close",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "CLOSE", "award_badges": False},
    )

    assert response.status_code == 400
    assert "CLOSE LEAGUE" in response.json()["detail"]


def test_admin_league_awards_close_updates_metadata_and_audits(monkeypatch):
    tables = _storage()
    supabase = FakeSupabase(tables)
    _install(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues/Open/awards/close",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "CLOSE LEAGUE", "award_badges": False},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["league"]["status"] == "ended"
    assert tables["leagues_metadata"][0]["is_active"] is False
    assert tables["leagues_metadata"][0]["end_awards"]["top_performers"]
    assert tables["admin_activity_log"][-1]["action_type"] == "close_league_award_top_performers_admin"
