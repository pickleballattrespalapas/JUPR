from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import uuid

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase


require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.admin_tournament_day_live_routes import (
    install_admin_tournament_day_live_routes,
)


ROOT = Path(__file__).resolve().parents[1]
ROUTE = "services/api/admin_tournament_day_live_routes.py"
INSTALLER = "services/api/admin_operations_routes.py"
DAY_PREFIX = (
    "/admin/clubs/club-1/tournament-live/tournaments/tour-1/"
    "days/day-1"
)


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _request(action: str, *, payload: dict | None = None) -> dict:
    return {
        "action": action,
        "client_idempotency_key": str(uuid.uuid4()),
        "confirmation_text": {
            "activate_day": "ACTIVATE DAY",
            "activate_draw": "ACTIVATE DRAW",
            "pause_draw": "PAUSE DRAW",
            "resume_draw": "RESUME DRAW",
            "auto_fill_courts": "AUTO FILL COURTS",
            "assign_next_court": "ASSIGN NEXT OPEN COURT",
            "assign_game_to_court": "ASSIGN GAME TO COURT",
            "reserve_game_for_court": "WAIT FOR SELECTED COURT",
            "requeue_game": "RETURN GAME TO QUEUE",
            "move_game_to_court": "MOVE GAME TO COURT",
            "score_and_release": "SAVE SCORE AND RELEASE COURT",
            "correct_completed_score": "CORRECT COMPLETED SCORE",
            "record_non_played_result": "RECORD NON-PLAYED RESULT",
            "generate_playoffs": "GENERATE PLAYOFFS",
            "close_day": "CLOSE TOURNAMENT DAY",
        }[action],
        "expected": {
            "day_run_version": 0,
            "state_fingerprint": "a" * 64,
        },
        "payload": dict(payload or {}),
    }


def _install_api(monkeypatch, supabase) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-live")
    monkeypatch.setenv(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES", "1"
    )
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(
            email="admin@example.com", user_id="user-1"
        ),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def _client(monkeypatch, supabase) -> TestClient:
    _install_api(monkeypatch, supabase)
    test_app = FastAPI()
    install_admin_tournament_day_live_routes(
        test_app, get_supabase_client=lambda: supabase
    )
    return TestClient(test_app)


def test_day_live_routes_use_one_day_scoped_nested_command_envelope() -> None:
    source = _read(ROUTE)
    installer = _read(INSTALLER)

    snapshot_path = (
        '"/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/'
        'days/{day_id}/snapshot"'
    )
    command_path = (
        '"/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/'
        'days/{day_id}/commands"'
    )
    reconcile_path = (
        '"/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/'
        'days/{day_id}/operations/{operation_key}/reconcile"'
    )
    assert "@app.get(" in source and snapshot_path in source
    assert "@app.post(" in source and command_path in source
    assert reconcile_path in source
    assert "class AdminTournamentDay" in source
    assert "expected:" in source
    assert "day_run_version" in source
    assert "state_fingerprint" in source
    assert "draw_version" in source
    assert "game_version" in source
    assert "court_version" in source
    assert "target_court_version" in source
    assert "queue_version" in source
    assert "payload:" in source
    assert "draw_ids" not in source
    assert "draw_id" in source
    assert "advance_count" in source
    assert "game_id" in source
    assert "court_id" in source
    assert "score_a" in source and "score_b" in source
    assert "non_playing_team_id" in source
    assert "winner_team_id" not in source
    assert '"generate_playoffs"' in source
    for action in (
        "assign_next_court",
        "assign_game_to_court",
        "reserve_game_for_court",
        "requeue_game",
        "move_game_to_court",
    ):
        assert f'"{action}"' in source
    assert "client_idempotency_key" in source
    assert "execute_admin_tournament_day_live_command" in source
    assert "build_admin_tournament_day_live_snapshot" in source
    assert "registration_day_id=day_id" in source
    assert "def install_admin_tournament_day_live_routes" in source
    assert "install_admin_tournament_day_live_routes" in installer
    assert "RECONCILE DAY OPERATIONS" in _read(
        "jupr_app/services/admin_tournament_day_live_service.py"
    )


def test_day_live_route_keeps_read_score_and_schedule_permissions_distinct() -> None:
    source = _read(ROUTE)

    assert "_resolve_tournament_role_or_403" in source
    assert "PERMISSION_ENTER_SCORES" in source
    assert "PERMISSION_MANAGE_TOURNAMENTS" in source
    assert '"score_and_release",\n            "correct_completed_score",\n            "record_non_played_result"' in source
    compact = "".join(source.split())
    assert "required_permissions=(PERMISSION_ENTER_SCORES,)" in compact
    assert "required_permissions=(PERMISSION_MANAGE_TOURNAMENTS,)" in compact
    assert "require_all=True" in source
    assert "is_admin_tournament_admin_enabled" in source


def test_day_live_snapshot_route_passes_exact_club_tournament_and_day(
    monkeypatch,
) -> None:
    tables = {"admin_activity_log": []}
    supabase = FakeSupabase(tables)
    calls: list[dict] = []

    def fake_snapshot(_supabase, **scope):
        calls.append(scope)
        return {
            "ok": True,
            "mode": "tournament_day_live",
            "tournament": {"id": scope["tournament_id"]},
            "day_scope": {
                "selected_day_id": scope["registration_day_id"],
                "selected_day": {"id": scope["registration_day_id"]},
                "available_days": [{"id": scope["registration_day_id"]}],
            },
            "day_run": {
                "id": "",
                "registration_day_id": scope["registration_day_id"],
                "state": "DRAFT",
                "version": "0",
                "updated_at": None,
            },
            "state_fingerprint": "a" * 64,
            "queue_version": "0",
            "summary": {},
            "draws": [],
            "courts": [],
            "games": [],
            "eligible_queue": [],
            "held_games": [],
            "blocked_games": [],
            "operations": [],
            "readiness": {
                "activate_day": {"ready": False, "blockers": ["DAY_NOT_ACTIVE"]},
                "auto_fill_courts": {"ready": False, "blockers": ["DAY_NOT_ACTIVE"]},
            },
            "runtime": {},
            "warnings": [],
        }

    monkeypatch.setattr(
        "services.api.admin_tournament_day_live_routes.build_admin_tournament_day_live_snapshot",
        fake_snapshot,
    )
    response = _client(monkeypatch, supabase).get(
        f"{DAY_PREFIX}/snapshot", headers={"Authorization": "Bearer local"}
    )

    assert response.status_code == 200, response.text
    assert calls == [
        {
            "club_id": "club-1",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
        }
    ]
    assert response.json()["day_scope"]["selected_day_id"] == "day-1"


def test_day_live_command_route_forwards_nested_reviewed_request(
    monkeypatch,
) -> None:
    tables = {"admin_activity_log": []}
    supabase = FakeSupabase(tables)
    calls: list[dict] = []
    request = _request("activate_day", payload={})

    def fake_execute(_supabase, **kwargs):
        calls.append(kwargs)
        return {
            "command": {
                "action": kwargs["request"]["action"],
                "registration_day_id": kwargs["registration_day_id"],
            },
            "operation": {"status": "completed", "idempotent_replay": False},
            "snapshot": {
                "ok": True,
                "mode": "tournament_day_live",
                "state_fingerprint": "b" * 64,
            },
        }

    monkeypatch.setattr(
        "services.api.admin_tournament_day_live_routes.execute_admin_tournament_day_live_command",
        fake_execute,
    )
    response = _client(monkeypatch, supabase).post(
        f"{DAY_PREFIX}/commands",
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert response.status_code == 200, response.text
    assert len(calls) == 1
    call = calls[0]
    assert call["club_id"] == "club-1"
    assert call["tournament_id"] == "tour-1"
    assert call["registration_day_id"] == "day-1"
    assert call["request"]["action"] == request["action"]
    assert str(call["request"]["client_idempotency_key"]) == request[
        "client_idempotency_key"
    ]
    assert call["request"]["confirmation_text"] == request["confirmation_text"]
    assert call["request"]["expected"] == request["expected"]
    assert call["request"]["payload"] == request["payload"]
    assert call["actor_email"] == "admin@example.com"
    assert call["actor_role"] == "club_owner"
    assert response.json()["command"]["action"] == "activate_day"
    assert response.json()["operation"]["status"] == "completed"


def test_day_live_request_rejects_flat_or_incomplete_command_payload(
    monkeypatch,
) -> None:
    tables = {"admin_activity_log": []}
    supabase = FakeSupabase(tables)
    client = _client(monkeypatch, supabase)

    flat = client.post(
        f"{DAY_PREFIX}/commands",
        headers={"Authorization": "Bearer local"},
        json={
            "command": "activate_day",
            "idempotency_key": str(uuid.uuid4()),
            "expected_state_fingerprint": "a" * 64,
            "draw_ids": ["draw-a"],
        },
    )
    missing_expected = client.post(
        f"{DAY_PREFIX}/commands",
        headers={"Authorization": "Bearer local"},
        json={
            "action": "activate_day",
            "client_idempotency_key": str(uuid.uuid4()),
            "confirmation_text": "ACTIVATE DAY",
            "payload": {},
        },
    )

    assert flat.status_code == 422
    assert missing_expected.status_code == 422


def test_day_live_route_accepts_exact_completed_score_correction_envelope(
    monkeypatch,
) -> None:
    supabase = FakeSupabase({"admin_activity_log": []})
    calls: list[dict] = []

    def fake_execute(_supabase, **kwargs):
        calls.append(kwargs["request"])
        return {
            "command": {"action": kwargs["request"]["action"]},
            "operation": {"status": "completed"},
            "snapshot": {"ok": True, "state_fingerprint": "b" * 64},
        }

    monkeypatch.setattr(
        "services.api.admin_tournament_day_live_routes.execute_admin_tournament_day_live_command",
        fake_execute,
    )
    request = _request(
        "correct_completed_score",
        payload={"game_id": "game-a", "score_a": 6, "score_b": 11},
    )
    request["expected"].update(
        {"queue_version": 4, "draw_version": 3, "game_version": "game-a-v1"}
    )

    response = _client(monkeypatch, supabase).post(
        f"{DAY_PREFIX}/commands",
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert response.status_code == 200, response.text
    assert calls == [request]
