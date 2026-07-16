from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def league_live_tables() -> dict[str, list[dict]]:
    return {
        "league_live_sessions": [],
        "league_live_rounds": [],
        "league_live_courts": [],
        "admin_activity_log": [],
    }


def _install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_admin_league_live_create_and_detail_contract(monkeypatch):
    tables = league_live_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/league-manager/live-sessions",
        headers={"Authorization": "Bearer local"},
        json={
            "league_name": "Tuesday Ladder",
            "week_tag": "Week 1",
            "total_rounds": 3,
            "current_round": 1,
            "roster": [{"player_id": 1, "player_name": "Alex"}],
            "courts": [{"court_number": 1, "format_type": "4-player", "player_names": ["Alex", "Blair", "Casey", "Devon"]}],
            "confirmation_text": "CREATE LIVE SESSION",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "league_live_session_create"
    session_id = payload["session"]["id"]
    assert session_id
    assert tables["league_live_sessions"][0]["league_name"] == "Tuesday Ladder"
    assert tables["league_live_courts"][0]["court_number"] == 1
    assert tables["admin_activity_log"][0]["action_type"] == "create_league_live_session_admin"

    detail = TestClient(app).get(
        f"/admin/clubs/club/league-manager/live-sessions/{session_id}",
        headers={"Authorization": "Bearer local"},
    )
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["session"]["id"] == session_id
    assert detail_payload["courts"][0]["player_names"] == ["Alex", "Blair", "Casey", "Devon"]


def test_admin_league_live_round_save_requires_confirmation(monkeypatch):
    tables = league_live_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    created = TestClient(app).post(
        "/admin/clubs/club/league-manager/live-sessions",
        headers={"Authorization": "Bearer local"},
        json={
            "league_name": "Tuesday Ladder",
            "week_tag": "Week 1",
            "total_rounds": 2,
            "current_round": 1,
            "confirmation_text": "CREATE LIVE SESSION",
        },
    )
    session_id = created.json()["session"]["id"]

    response = TestClient(app).post(
        f"/admin/clubs/club/league-manager/live-sessions/{session_id}/rounds/1",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "SAVE", "matches": []},
    )

    assert response.status_code == 400
    assert "SAVE ROUND" in response.json()["detail"]


def test_admin_league_live_round_save_advances_session(monkeypatch):
    tables = league_live_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    created = TestClient(app).post(
        "/admin/clubs/club/league-manager/live-sessions",
        headers={"Authorization": "Bearer local"},
        json={
            "league_name": "Tuesday Ladder",
            "week_tag": "Week 1",
            "total_rounds": 2,
            "current_round": 1,
            "courts": [{"court_number": 1, "player_names": ["Alex", "Blair", "Casey", "Devon"]}],
            "confirmation_text": "CREATE LIVE SESSION",
        },
    )
    session_id = created.json()["session"]["id"]

    response = TestClient(app).post(
        f"/admin/clubs/club/league-manager/live-sessions/{session_id}/rounds/1",
        headers={"Authorization": "Bearer local"},
        json={
            "round_label": "Round 1",
            "match_date": "2026-01-15",
            "matches": [{"t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 8}],
            "submitted_match_count": 1,
            "courts": [{"court_number": 1, "player_names": ["Alex", "Blair", "Casey", "Devon"]}],
            "advance_after_save": True,
            "confirmation_text": "SAVE ROUND",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["round"]["status"] == "submitted"
    assert payload["session"]["current_round"] == 2
    assert tables["league_live_rounds"][0]["submitted_match_count"] == 1
    assert tables["admin_activity_log"][-1]["action_type"] == "save_league_live_round_admin"
