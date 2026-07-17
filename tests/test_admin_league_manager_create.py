from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.domain.admin_activity_log import ActivityLogWriteResult
from jupr_app.services.admin_league_manager_create_service import (
    create_admin_league_manager_draft,
    duplicate_admin_league_manager_draft,
)
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
                "schedule_config": {"weekday": 1, "weeks": 8},
                "court_board_defaults": {"total_courts": 4},
                "rules_config": {"overview": {"league_type": "Ladder"}},
                "awards_config": {"default_depth": 1},
                "event_tags": {"skill_levels": ["Open"], "date_tags": ["2026"]},
                "started_at": "2026-01-01T00:00:00Z",
                "ended_at": None,
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


def test_duplicate_league_draft_copies_configuration_without_history(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = league_create_tables()

    result = duplicate_admin_league_manager_draft(
        FakeSupabase(tables),
        club_id="club",
        source_league_name="Existing League",
        target_league_name="Existing League Fall",
        actor_email="owner@example.com",
        actor_role="club_owner",
        confirmation_text="DUPLICATE LEAGUE",
    )

    created = tables["leagues_metadata"][-1]
    assert result["ok"] is True
    assert result["mode"] == "league_manager_draft_duplicate"
    assert result["roster_copied"] is False
    assert result["league"]["league_name"] == "Existing League Fall"
    assert created["status"] == "draft"
    assert created["is_active"] is False
    assert created["schedule_config"] == {"weekday": 1, "weeks": 8}
    assert created["court_board_defaults"] == {"total_courts": 4}
    assert created["rules_config"] == {"overview": {"league_type": "Ladder"}}
    assert created["awards_config"] == {"default_depth": 1}
    assert "started_at" not in created
    assert "ended_at" not in created
    assert tables["admin_activity_log"][-1]["action_type"] == "duplicate_league_manager_draft_admin"


def test_duplicate_league_draft_required_audit_failure_removes_new_draft(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_create_service.write_admin_activity_log",
        lambda *_args, **_kwargs: ActivityLogWriteResult(ok=False, warning="audit unavailable"),
    )
    tables = league_create_tables()

    with pytest.raises(RuntimeError, match="audit log write required"):
        duplicate_admin_league_manager_draft(
            FakeSupabase(tables),
            club_id="club",
            source_league_name="Existing League",
            target_league_name="Unaudited Draft",
            actor_email="owner@example.com",
            actor_role="club_owner",
            confirmation_text="DUPLICATE LEAGUE",
        )

    assert [row["league_name"] for row in tables["leagues_metadata"]] == ["Existing League"]


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


def test_duplicate_league_draft_api_contract(monkeypatch) -> None:
    tables = league_create_tables()
    _install_api(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues/Existing%20League/duplicate",
        headers={"Authorization": "Bearer local"},
        json={
            "target_league_name": "Existing League Fall",
            "confirmation_text": "DUPLICATE LEAGUE",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["league"]["league_name"] == "Existing League Fall"
    assert payload["source_league_name"] == "Existing League"
    assert payload["roster_copied"] is False
    assert len(tables["leagues_metadata"]) == 2


def test_duplicate_league_draft_api_requires_confirmation(monkeypatch) -> None:
    tables = league_create_tables()
    _install_api(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues/Existing%20League/duplicate",
        headers={"Authorization": "Bearer local"},
        json={"target_league_name": "Existing League Fall", "confirmation_text": "DUPLICATE"},
    )

    assert response.status_code == 400
    assert "DUPLICATE LEAGUE" in response.json()["detail"]
    assert len(tables["leagues_metadata"]) == 1


def test_duplicate_league_draft_api_rejects_missing_token_without_write(monkeypatch) -> None:
    tables = league_create_tables()
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(tables))

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues/Existing%20League/duplicate",
        json={"target_league_name": "Existing League Fall", "confirmation_text": "DUPLICATE"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "missing bearer token"
    assert len(tables["leagues_metadata"]) == 1
