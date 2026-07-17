from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.services.admin_league_manager_create_service import create_admin_league_manager_draft
from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def league_create_tables() -> dict[str, list[dict]]:
    return {
        "leagues_metadata": [
            {
                "club_id": "club",
                "league_name": "Existing League",
                "description": "Already here",
                "is_active": True,
                "status": "active",
                "k_factor": 32,
                "min_games": 4,
                "event_tags": {},
            }
        ],
        "players": [],
        "league_ratings": [],
        "admin_activity_log": [],
    }


def test_create_league_draft_service(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = league_create_tables()
    result = create_admin_league_manager_draft(
        FakeSupabase(tables),
        club_id="club",
        league_name="Summer Ladder",
        description="Tuesday club league",
        min_games=6,
        k_factor=28,
        actor_email="owner@example.com",
        actor_role="club_owner",
        confirmation_text="CREATE LEAGUE",
    )

    assert result["ok"] is True
    assert result["mode"] == "league_manager_draft_create"
    assert result["created"] is True
    assert result["league"]["league_name"] == "Summer Ladder"
    assert result["league"]["description"] == "Tuesday club league"
    assert result["league"]["status"] == "draft"
    assert tables["leagues_metadata"][-1]["is_active"] is False
    assert tables["leagues_metadata"][-1]["event_tags"] == {"skill_levels": [], "date_tags": []}
    assert tables["admin_activity_log"][0]["action_type"] == "create_league_manager_draft_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_create_league_draft_rejects_case_insensitive_duplicate(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = league_create_tables()

    with pytest.raises(ValueError, match="already exists"):
        create_admin_league_manager_draft(
            FakeSupabase(tables),
            club_id="club",
            league_name=" existing league ",
            description="",
            min_games=6,
            k_factor=32,
            actor_email="owner@example.com",
            actor_role="club_owner",
            confirmation_text="CREATE LEAGUE",
        )

    assert len(tables["leagues_metadata"]) == 1
    assert tables["admin_activity_log"] == []


def _install_api(monkeypatch, supabase: FakeSupabase) -> None:
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


def test_create_league_draft_api_contract(monkeypatch) -> None:
    tables = league_create_tables()
    _install_api(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues",
        headers={"Authorization": "Bearer local"},
        json={
            "league_name": "Summer Ladder",
            "description": "Tuesday club league",
            "min_games": 6,
            "k_factor": 28,
            "confirmation_text": "CREATE LEAGUE",
        },
    )

    assert response.status_code == 200
    assert response.json()["league"]["league_name"] == "Summer Ladder"
    assert len(tables["leagues_metadata"]) == 2


def test_create_league_draft_api_requires_confirmation(monkeypatch) -> None:
    tables = league_create_tables()
    _install_api(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues",
        headers={"Authorization": "Bearer local"},
        json={"league_name": "Summer Ladder", "confirmation_text": "CREATE"},
    )

    assert response.status_code == 400
    assert "CREATE LEAGUE" in response.json()["detail"]
    assert len(tables["leagues_metadata"]) == 1
