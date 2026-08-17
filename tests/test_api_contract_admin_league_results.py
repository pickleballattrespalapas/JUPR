from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.conftest import require_api_dependency
from tests.test_public_league_results_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture
def historical_results_client(monkeypatch) -> TestClient:
    supabase = FakeSupabase(
        {
            "clubs": [
                {
                    "id": "club",
                    "slug": "tres-palapas",
                    "name": "Tres Palapas",
                }
            ],
            "leagues_metadata": [
                {
                    "id": "archived-id",
                    "club_id": "club",
                    "league_name": "Archived League",
                    "league_type": "Individual",
                    "is_active": False,
                    "status": "archived",
                    "min_games": 1,
                    "k_factor": 24,
                    "schedule_config": {"weeks": 1},
                }
            ],
            "players": [
                {
                    "id": player_id,
                    "club_id": "club",
                    "name": name,
                    "rating": 1500,
                    "active": False,
                    "inactive_at": "2026-01-01T00:00:00Z",
                }
                for player_id, name in enumerate(
                    ["Alex", "Blair", "Casey", "Devon"],
                    start=1,
                )
            ],
            "league_ratings": [
                {
                    "club_id": "club",
                    "player_id": 1,
                    "league_name": "Archived League",
                    "rating": 1600,
                    "starting_rating": 1500,
                    "wins": 1,
                    "losses": 0,
                    "matches_played": 1,
                    "is_active": False,
                }
            ],
            "matches": [
                {
                    "id": 10,
                    "club_id": "club",
                    "date": "2025-01-01T00:00:00Z",
                    "league": "Archived League",
                    "match_type": "Live Match",
                    "week_tag": "Week 1",
                    "t1_p1": 1,
                    "t1_p2": 2,
                    "t2_p1": 3,
                    "t2_p2": 4,
                    "score_t1": 11,
                    "score_t2": 7,
                    "deleted_at": None,
                }
            ],
        }
    )
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr(
        "services.api.main.create_client",
        lambda _url, _credential: supabase,
    )

    def authenticate(authorization: str | None):
        if authorization != "Bearer local-admin":
            raise HTTPException(status_code=401, detail="authentication required")
        return SimpleNamespace(email="admin@example.com", user_id="user-1")

    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.authenticate_bearer",
        authenticate,
    )
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )
    return TestClient(app)


def test_archived_results_are_not_available_from_public_route(
    historical_results_client: TestClient,
) -> None:
    response = historical_results_client.get(
        "/clubs/tres-palapas/league-results",
        params={"league_name": "Archived League"},
    )

    assert response.status_code == 200
    assert response.json()["selected_league"] is None
    assert response.json()["standings"] == []


def test_admin_historical_results_route_requires_bearer_authentication(
    historical_results_client: TestClient,
) -> None:
    path = "/admin/clubs/club/league-manager/leagues/Archived%20League/results"

    denied = historical_results_client.get(path)
    allowed = historical_results_client.get(
        path,
        headers={"Authorization": "Bearer local-admin"},
    )

    assert denied.status_code == 401
    assert allowed.status_code == 200
    payload = allowed.json()
    assert payload["selected_league"] == "Archived League"
    assert payload["league_status"] == "archived"
    assert payload["publicly_visible"] is False
    assert payload["standings"][0]["player_name"] == "Alex"
