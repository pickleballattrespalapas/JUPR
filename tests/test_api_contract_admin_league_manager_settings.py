from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.domain.admin_activity_log import ActivityLogWriteResult
from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def league_manager_tables() -> dict[str, list[dict]]:
    return {
        "leagues_metadata": [
            {
                "club_id": "club",
                "league_name": "Tuesday Ladder",
                "is_active": True,
                "status": "active",
                "k_factor": 32,
                "min_games": 3,
                "schedule_config": {},
                "court_board_defaults": {},
                "rules_config": {},
                "awards_config": {},
                "event_tags": {},
            }
        ],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "active": True},
            {"club_id": "club", "id": 2, "name": "Blair", "active": True},
        ],
        "league_ratings": [
            {"club_id": "club", "id": 1, "player_id": 1, "league_name": "Tuesday Ladder", "rating": 1500, "wins": 2, "losses": 1, "matches_played": 3, "is_active": True},
        ],
        "admin_activity_log": [],
    }


def _install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_league_manager_settings_update_contract(monkeypatch):
    tables = league_manager_tables()
    tables["leagues_metadata"][0].update({"status": "draft", "is_active": False})
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={
            "description": "Tuesday evening ladder",
            "k_factor": 28,
            "min_games": 4,
            "schedule_config": {"start_date": "2026-01-05", "weekday": 0, "weeks": 2, "time_start": "18:00", "time_end": "20:00"},
            "court_board_defaults": {"max_used_courts": 4, "players_per_court": "4"},
            "rules_config": {"format": "ladder"},
            "awards_config": {"top_performer": True},
            "event_tags": {"season": "winter"},
            "confirmation_text": "SAVE LEAGUE",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "league_manager_settings_update"
    assert payload["league"]["k_factor"] == 28
    assert payload["detail"]["schedule_preview"][0]["date"] == "2026-01-05"
    assert tables["leagues_metadata"][0]["min_games"] == 4
    assert tables["leagues_metadata"][0]["description"] == "Tuesday evening ladder"
    assert tables["leagues_metadata"][0]["court_board_defaults"]["max_used_courts"] == 4
    assert tables["admin_activity_log"][0]["action_type"] == "update_league_manager_settings_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_league_manager_structured_draft_settings_are_normalized_and_preserve_extensions(monkeypatch):
    tables = league_manager_tables()
    tables["leagues_metadata"][0].update(
        {
            "status": "draft",
            "is_active": False,
            "schedule_config": {"extension_mode": "keep"},
            "court_board_defaults": {"extension_court": True},
            "rules_config": {"legacy_format": "keep"},
            "awards_config": {"legacy_award": True},
            "event_tags": {"skill_levels": ["3.5"], "date_tags": ["Old"]},
        }
    )
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={
            "description": "<b>Structured</b> draft",
            "k_factor": 24,
            "min_games": 6,
            "schedule_config": {
                "extension_mode": "keep",
                "start_date": "2026-01-05",
                "weeks": 8,
                "end_date": "",
                "weekday": 0,
                "time_start": "18:15",
                "time_end": "20:45",
                "timezone": "America/Chicago",
                "blackout_dates": ["2026-01-19", "2026-01-19"],
                "session_capacity": 32,
            },
            "court_board_defaults": {
                "extension_court": True,
                "total_courts": 6,
                "court_identifiers": ["1", "2", "2"],
                "max_used_courts": 4,
                "players_per_court": "5",
                "rotation_mode": "queue",
                "game_format_points": 15,
                "game_format_time": 20,
            },
            "rules_config": {
                "legacy_format": "keep",
                "overview": {"league_type": "Ladder", "divisions": ["Open", "Open", "Advanced"], "summary": "Weekly play"},
                "competition": {
                    "scoring_rules": "Win by two",
                    "match_format": "Round robin",
                    "tie_break_rules": "Point differential",
                    "dispute_window": "24 hours",
                    "dispute_policy": "Captains",
                },
            },
            "awards_config": {
                "legacy_award": True,
                "default_min_games": 6,
                "default_depth": 3,
                "categories": {
                    "highest_rating": {"enabled": True, "min_games": 6, "depth": 3},
                    "most_improved": {"enabled": False, "min_games": 4, "depth": 1},
                },
            },
            "confirmation_text": "SAVE LEAGUE",
        },
    )

    assert response.status_code == 200
    saved = tables["leagues_metadata"][0]
    assert saved["description"] == "bStructured/b draft"
    assert saved["schedule_config"]["extension_mode"] == "keep"
    assert saved["schedule_config"]["blackout_dates"] == ["2026-01-19"]
    assert saved["court_board_defaults"]["extension_court"] is True
    assert saved["court_board_defaults"]["court_identifiers"] == ["1", "2"]
    assert saved["rules_config"]["legacy_format"] == "keep"
    assert saved["rules_config"]["overview"]["divisions"] == ["Open", "Advanced"]
    assert saved["awards_config"]["legacy_award"] is True
    assert saved["awards_config"]["categories"]["most_improved"]["enabled"] is False
    assert saved["event_tags"]["skill_levels"] == ["3.5"]
    assert "January 2026" in saved["event_tags"]["date_tags"]
    assert tables["admin_activity_log"][0]["after_json"]["patch"]["schedule_config"]["timezone"] == "America/Chicago"


def test_admin_league_manager_structured_match_and_operation_rules_drive_saved_draft(monkeypatch):
    tables = league_manager_tables()
    tables["leagues_metadata"][0].update({"status": "draft", "is_active": False})
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={
            "rules_config": {
                "overview": {"league_format": "ladder"},
                "competition": {
                    "scoring_profile": "standard_pickleball",
                    "match_structure": {"kind": "fixed_games", "games": 2},
                    "standings_tiebreak": "wins_then_point_differential",
                    "correction_window": "until_next_round",
                    "score_submission_policy": "rostered_player_or_admin",
                    "playoff_format": "single_elimination",
                },
                "operation": {
                    "session_mode": "live_court_board",
                    "move_up_count": 1,
                    "move_down_count": 2,
                },
            },
            "confirmation_text": "SAVE LEAGUE",
        },
    )

    assert response.status_code == 200
    rules = tables["leagues_metadata"][0]["rules_config"]
    assert rules["overview"]["league_format"] == "ladder"
    assert rules["competition"]["scoring_profile"] == "standard_pickleball"
    assert rules["competition"]["match_structure"] == {
        "kind": "fixed_games",
        "games": 2,
        "result_counting": "each_game",
        "completion": "all_games",
    }
    assert rules["competition"]["score_submission_policy"] == "rostered_player_or_admin"
    assert rules["operation"] == {
        "session_mode": "live_court_board",
        "move_up_count": 1,
        "move_down_count": 2,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schedule_config", {"weekday": 7}, "weekday must be at most 6"),
        ("schedule_config", {"start_date": "01/05/2026"}, "YYYY-MM-DD"),
        ("schedule_config", {"time_start": "20:00", "time_end": "18:00"}, "time_end must be after"),
        ("schedule_config", {"timezone": "America/Chicago\nBAD"}, "unsupported characters"),
        ("court_board_defaults", {"total_courts": 3, "max_used_courts": 4}, "cannot exceed total_courts"),
        ("court_board_defaults", {"players_per_court": "9"}, "whole number from 2 through 8"),
        ("rules_config", {"overview": {"divisions": "Open"}}, "divisions must be a list"),
        ("rules_config", {"competition": {"match_structure": {"kind": "best_of", "games": 4}}}, "odd game count"),
        ("rules_config", {"overview": {"league_format": "ladder"}, "operation": {"session_mode": "self_scheduled"}}, "Ladder leagues need scheduled rounds"),
        ("awards_config", {"default_depth": 2}, "must be 1 or 3"),
        ("awards_config", {"categories": {"most_wins": {"enabled": "yes"}}}, "must be true or false"),
        ("rules_config", {"extension": "x" * 5001}, "longer than 5000"),
    ],
)
def test_admin_league_manager_structured_settings_reject_invalid_config(monkeypatch, field, value, message):
    tables = league_manager_tables()
    tables["leagues_metadata"][0].update({"status": "draft", "is_active": False})
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={field: value, "confirmation_text": "SAVE LEAGUE"},
    )

    assert response.status_code == 400
    assert message in response.json()["detail"]
    assert tables["leagues_metadata"][0][field] == {}
    assert tables["admin_activity_log"] == []


def test_admin_league_manager_schedule_preview_is_read_only(monkeypatch):
    tables = league_manager_tables()
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/schedule/preview",
        headers={"Authorization": "Bearer local"},
        json={
            "schedule_config": {
                "start_date": "2026-01-05",
                "weekday": 0,
                "weeks": 3,
                "time_start": "18:00",
                "time_end": "20:00",
                "timezone": "America/Chicago",
                "blackout_dates": ["2026-01-12"],
            }
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "league_manager_schedule_preview"
    assert len(payload["schedule_preview"]) == 2
    assert payload["schedule_preview"][1]["date"] == "2026-01-19"
    assert payload["schedule_ics"].count("BEGIN:VEVENT") == 2
    assert tables["leagues_metadata"][0]["schedule_config"] == {}
    assert tables["admin_activity_log"] == []


def test_admin_league_manager_schedule_preview_requires_authentication(monkeypatch):
    tables = league_manager_tables()
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(tables))

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder/schedule/preview",
        json={"schedule_config": {"start_date": "2026-01-05", "weekday": 0, "weeks": 1}},
    )

    assert response.status_code == 401
    assert tables["leagues_metadata"][0]["schedule_config"] == {}
    assert tables["admin_activity_log"] == []


@pytest.mark.parametrize("status", ["active", "paused"])
def test_admin_league_manager_settings_allows_description_only_while_running(monkeypatch, status):
    tables = league_manager_tables()
    tables["leagues_metadata"][0].update({"status": status, "is_active": status == "active"})
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={"description": f"Safe {status} description", "confirmation_text": "SAVE LEAGUE"},
    )

    assert response.status_code == 200
    assert response.json()["league"]["description"] == f"Safe {status} description"
    assert tables["leagues_metadata"][0]["k_factor"] == 32
    assert tables["admin_activity_log"][0]["after_json"]["edit_policy_status"] == status


def test_admin_league_manager_settings_rejects_inconsistent_lifecycle_state(monkeypatch):
    tables = league_manager_tables()
    tables["leagues_metadata"][0].update({"status": "active", "is_active": False})
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={"description": "Should not save", "confirmation_text": "SAVE LEAGUE"},
    )

    assert response.status_code == 400
    assert "lifecycle state is inconsistent" in response.json()["detail"]
    assert "description" not in tables["leagues_metadata"][0]
    assert tables["admin_activity_log"] == []


@pytest.mark.parametrize("status", ["active", "paused"])
def test_admin_league_manager_settings_blocks_configuration_while_running(monkeypatch, status):
    tables = league_manager_tables()
    tables["leagues_metadata"][0].update({"status": status, "is_active": status == "active"})
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={"k_factor": 28, "confirmation_text": "SAVE LEAGUE"},
    )

    assert response.status_code == 400
    assert f"Only description can be edited while a league is {status}" in response.json()["detail"]
    assert tables["leagues_metadata"][0]["k_factor"] == 32
    assert tables["admin_activity_log"] == []


@pytest.mark.parametrize("status", ["ended", "archived"])
def test_admin_league_manager_settings_are_read_only_after_close(monkeypatch, status):
    tables = league_manager_tables()
    tables["leagues_metadata"][0].update({"status": status, "is_active": False})
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={"description": "Should not save", "confirmation_text": "SAVE LEAGUE"},
    )

    assert response.status_code == 400
    assert f"read-only after a league is {status}" in response.json()["detail"]
    assert "description" not in tables["leagues_metadata"][0]
    assert tables["admin_activity_log"] == []


def test_admin_league_manager_settings_update_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(league_manager_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={"k_factor": 28, "confirmation_text": "SAVE"},
    )

    assert response.status_code == 400
    assert "SAVE LEAGUE" in response.json()["detail"]


def test_admin_league_manager_settings_update_rejects_status_bypass(monkeypatch):
    tables = league_manager_tables()
    _install_env(monkeypatch, FakeSupabase(tables))

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={"status": "ended", "confirmation_text": "SAVE LEAGUE"},
    )

    assert response.status_code == 400
    assert "lifecycle action" in response.json()["detail"]
    assert tables["leagues_metadata"][0]["status"] == "active"


def test_admin_league_manager_settings_required_audit_failure_rolls_back(monkeypatch):
    tables = league_manager_tables()
    tables["leagues_metadata"][0].update({"status": "draft", "is_active": False})
    _install_env(monkeypatch, FakeSupabase(tables))
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_update_service.write_admin_activity_log",
        lambda *_args, **_kwargs: ActivityLogWriteResult(ok=False, warning="audit unavailable"),
    )

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={"k_factor": 28, "confirmation_text": "SAVE LEAGUE"},
    )

    assert response.status_code == 500
    assert "audit log write required" in response.json()["detail"]
    assert tables["leagues_metadata"][0]["k_factor"] == 32


def test_admin_league_manager_settings_rejects_stale_status(monkeypatch):
    tables = league_manager_tables()
    _install_env(monkeypatch, FakeSupabase(tables))
    before = {**tables["leagues_metadata"][0], "status": "draft", "is_active": False}
    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_update_service._fetch_league_meta",
        lambda *_args, **_kwargs: before,
    )

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Tuesday%20Ladder",
        headers={"Authorization": "Bearer local"},
        json={"k_factor": 28, "confirmation_text": "SAVE LEAGUE"},
    )

    assert response.status_code == 400
    assert "changed before this save completed" in response.json()["detail"]
    assert tables["leagues_metadata"][0]["k_factor"] == 32
    assert tables["admin_activity_log"] == []


def test_admin_league_manager_settings_update_does_not_create_missing_league(monkeypatch):
    tables = league_manager_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/league-manager/leagues/Unknown",
        headers={"Authorization": "Bearer local"},
        json={"k_factor": 28, "confirmation_text": "SAVE LEAGUE"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "league not found"
    assert len(tables["leagues_metadata"]) == 1
