from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.domain.admin_activity_log import ActivityLogWriteResult
from jupr_app.services.admin_league_manager_lifecycle_service import (
    transition_admin_league_manager_lifecycle,
)
from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def lifecycle_tables(*, status: str = "draft", is_active: bool = False) -> dict[str, list[dict]]:
    return {
        "leagues_metadata": [
            {
                "club_id": "club",
                "league_name": "Tuesday Ladder",
                "description": "Tuesday night",
                "status": status,
                "is_active": is_active,
                "k_factor": 32,
                "min_games": 4,
                "schedule_config": {},
                "court_board_defaults": {},
                "rules_config": {},
                "awards_config": {},
                "event_tags": {},
            }
        ],
        "players": [],
        "league_ratings": [],
        "admin_activity_log": [],
    }


def transition(supabase: FakeSupabase, action: str, confirmation_text: str) -> dict:
    return transition_admin_league_manager_lifecycle(
        supabase,
        club_id="club",
        league_name="Tuesday Ladder",
        action=action,
        actor_email="owner@example.com",
        actor_role="club_owner",
        confirmation_text=confirmation_text,
    )


def test_lifecycle_transition_sequence_is_guarded_and_audited(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = lifecycle_tables()
    supabase = FakeSupabase(tables)

    started = transition(supabase, "start", "START LEAGUE")
    started_at = tables["leagues_metadata"][0]["started_at"]
    paused = transition(supabase, "pause", "PAUSE LEAGUE")
    resumed = transition(supabase, "resume", "RESUME LEAGUE")
    ended = transition(supabase, "end", "END LEAGUE")
    ended_at = tables["leagues_metadata"][0]["ended_at"]
    archived = transition(supabase, "archive", "ARCHIVE LEAGUE")

    assert started["previous_status"] == "draft"
    assert started["new_status"] == "active"
    assert paused["new_status"] == "paused"
    assert paused["detail"]["league"]["status"] == "paused"
    assert resumed["new_status"] == "active"
    assert tables["leagues_metadata"][0]["started_at"] == started_at
    assert ended["new_status"] == "ended"
    assert tables["leagues_metadata"][0]["ended_by"] == "owner@example.com"
    assert archived["new_status"] == "archived"
    assert tables["leagues_metadata"][0]["ended_at"] == ended_at
    assert tables["leagues_metadata"][0]["is_active"] is False
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "start_league_manager_admin",
        "pause_league_manager_admin",
        "resume_league_manager_admin",
        "end_league_manager_admin",
        "archive_league_manager_admin",
    ]
    assert all(row["flagged_for_review"] is True for row in tables["admin_activity_log"])


def test_lifecycle_rejects_illegal_transition_without_write(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = lifecycle_tables()

    with pytest.raises(ValueError, match="Cannot pause a draft league"):
        transition(FakeSupabase(tables), "pause", "PAUSE LEAGUE")

    assert tables["leagues_metadata"][0]["status"] == "draft"
    assert tables["admin_activity_log"] == []


def test_lifecycle_archive_cannot_bypass_persisted_awards_mint(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = lifecycle_tables(status="ended", is_active=False)
    tables["leagues_metadata"][0]["end_awards"] = {
        "workflow": {
            "version": 2,
            "status": "mint_failed",
            "mint": {"status": "failed", "attempt_count": 1},
        }
    }

    with pytest.raises(ValueError, match="verify the persisted League Awards mint"):
        transition(FakeSupabase(tables), "archive", "ARCHIVE LEAGUE")

    assert tables["leagues_metadata"][0]["status"] == "ended"


def test_lifecycle_requires_action_specific_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = lifecycle_tables()

    with pytest.raises(ValueError, match="START LEAGUE"):
        transition(FakeSupabase(tables), "start", "START")

    assert tables["leagues_metadata"][0]["status"] == "draft"


def test_lifecycle_rejects_inconsistent_status_flag_without_write(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = lifecycle_tables(status="draft", is_active=True)

    with pytest.raises(ValueError, match="lifecycle state is inconsistent"):
        transition(FakeSupabase(tables), "start", "START LEAGUE")

    assert tables["leagues_metadata"][0]["status"] == "draft"
    assert tables["admin_activity_log"] == []


def test_required_audit_failure_rolls_back_lifecycle_transition(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_lifecycle_service.write_admin_activity_log",
        lambda *_args, **_kwargs: ActivityLogWriteResult(ok=False, warning="audit unavailable"),
    )
    tables = lifecycle_tables(status="active", is_active=True)

    with pytest.raises(RuntimeError, match="audit log write required"):
        transition(FakeSupabase(tables), "pause", "PAUSE LEAGUE")

    assert tables["leagues_metadata"][0]["status"] == "active"
    assert tables["leagues_metadata"][0]["is_active"] is True


def test_required_audit_exception_rolls_back_lifecycle_transition(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")

    def raise_audit(*_args, **_kwargs):
        raise RuntimeError("audit transport unavailable")

    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_lifecycle_service.write_admin_activity_log",
        raise_audit,
    )
    tables = lifecycle_tables(status="active", is_active=True)

    with pytest.raises(RuntimeError, match="audit transport unavailable"):
        transition(FakeSupabase(tables), "pause", "PAUSE LEAGUE")

    assert tables["leagues_metadata"][0]["status"] == "active"
    assert tables["leagues_metadata"][0]["is_active"] is True


def install_authorized_api(monkeypatch, supabase: FakeSupabase) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="owner@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_lifecycle_api_contract(monkeypatch) -> None:
    tables = lifecycle_tables()
    install_authorized_api(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/lifecycle",
        headers={"Authorization": "Bearer local"},
        json={"action": "start", "confirmation_text": "START LEAGUE"},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "league_manager_lifecycle_transition"
    assert response.json()["new_status"] == "active"
    assert tables["leagues_metadata"][0]["is_active"] is True


def test_lifecycle_api_rejects_missing_token_without_write(monkeypatch) -> None:
    tables = lifecycle_tables()
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(tables))

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/lifecycle",
        json={"action": "start", "confirmation_text": "NOT VALID"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "missing bearer token"
    assert tables["leagues_metadata"][0]["status"] == "draft"
    assert tables["admin_activity_log"] == []
