from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


ROSTER = [
    {"player_id": 1, "player_name": "Alex", "rating": 1400},
    {"player_id": 2, "player_name": "Blair", "rating": 1380},
    {"player_id": 3, "player_name": "Casey", "rating": 1360},
    {"player_id": 4, "player_name": "Devon", "rating": 1340},
]
COURTS = [
    {
        "court_number": 1,
        "format_type": "4-Player",
        "player_names": ["Alex", "Blair", "Casey", "Devon"],
    }
]
MATCHES = [
    {
        "court": 1,
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "score_t1": 11,
        "score_t2": 8,
    }
]


def league_live_tables() -> dict[str, list[dict]]:
    return {
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "rating": 1400, "is_active": True},
            {"club_id": "club", "id": 2, "name": "Blair", "rating": 1380, "is_active": True},
            {"club_id": "club", "id": 3, "name": "Casey", "rating": 1360, "is_active": True},
            {"club_id": "club", "id": 4, "name": "Devon", "rating": 1340, "is_active": True},
        ],
        "league_live_sessions": [],
        "league_live_rounds": [],
        "league_live_courts": [],
        "admin_activity_log": [],
    }


def _install_env(monkeypatch, supabase, *, service_role: bool = True):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    if service_role:
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")
    else:
        monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_league_manager_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def _create(client: TestClient, *, total_rounds: int = 2):
    return client.post(
        "/admin/clubs/club/league-manager/live-sessions",
        headers={"Authorization": "Bearer local"},
        json={
            "league_name": "Tuesday Ladder",
            "week_tag": "Week 1",
            "total_rounds": total_rounds,
            "current_round": 1,
            "roster": ROSTER,
            "courts": COURTS,
            "idempotency_key": "create-live-session-test",
            "confirmation_text": "CREATE LIVE SESSION",
        },
    )


def _plan(client: TestClient, session: dict):
    return client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1/plan",
        headers={"Authorization": "Bearer local"},
        json={
            "expected_updated_at": session["updated_at"],
            "matches": MATCHES,
            "courts": COURTS,
        },
    )


def test_admin_league_live_requires_server_only_supabase_key(monkeypatch):
    supabase = FakeSupabase(league_live_tables())
    _install_env(monkeypatch, supabase, service_role=False)

    response = TestClient(app).get(
        "/admin/clubs/club/league-manager/live-sessions",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 503
    assert "SUPABASE_SERVICE_ROLE_KEY" in response.json()["detail"]


def test_admin_league_live_status_fails_closed_without_server_only_key(monkeypatch):
    supabase = FakeSupabase(league_live_tables())
    _install_env(monkeypatch, supabase, service_role=False)
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT", "1")

    response = TestClient(app).get("/admin/clubs/club/league-manager/live/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["status"] == "service_role_required"
    assert payload["service_role_configured"] is False
    assert payload["submit_enabled"] is False
    for endpoint_key in (
        "sessions_endpoint",
        "roster_suggestion_endpoint",
        "round_plan_endpoint",
        "round_submit_endpoint",
        "round_reconcile_endpoint",
        "round_compensate_endpoint",
        "guest_endpoint",
        "export_endpoint",
    ):
        assert payload[endpoint_key] is None


def test_admin_league_live_status_is_configuration_only(monkeypatch):
    class NoDataPlaneSupabase:
        def table(self, _name):
            raise AssertionError("status endpoint must not query private League Live tables")

    _install_env(monkeypatch, NoDataPlaneSupabase())

    response = TestClient(app).get("/admin/clubs/club/league-manager/live/status")

    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert response.json()["movement_authority"] == "python_fastapi"
    assert "session_count" not in response.json()


def test_admin_league_live_disabled_status_attests_submit_gate_closed(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only")

    for manager_enabled, domain_enabled, expected_status in (
        ("0", "0", "guarded_off"),
        ("1", "0", "streamlit_fallback"),
    ):
        monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", manager_enabled)
        monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN", domain_enabled)

        response = TestClient(app).get("/admin/clubs/club/league-manager/live/status")

        assert response.status_code == 200
        payload = response.json()
        assert payload["enabled"] is False
        assert payload["status"] == expected_status
        assert payload["submit_enabled"] is False
        for endpoint_key in (
            "round_submit_endpoint",
            "round_reconcile_endpoint",
            "round_compensate_endpoint",
            "guest_endpoint",
            "export_endpoint",
        ):
            assert payload[endpoint_key] is None


def test_admin_league_live_private_storage_failure_is_fail_closed(monkeypatch):
    class FailingSupabase:
        def table(self, _name):
            raise RuntimeError("storage unavailable")

    supabase = FailingSupabase()
    _install_env(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/league-manager/live-sessions",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 503
    assert "private storage" in response.json()["detail"]


def test_admin_league_live_roster_suggestion_verifies_club_players(monkeypatch):
    supabase = FakeSupabase(league_live_tables())
    _install_env(monkeypatch, supabase)
    client = TestClient(app)

    response = client.post(
        "/admin/clubs/club/league-manager/live/roster-suggestion",
        headers={"Authorization": "Bearer local"},
        json={"roster": ROSTER, "round_number": 1},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "league_live_roster_suggestion"
    assert response.json()["court_sizes"] == [4]
    assert response.json()["bench"] == []

    denied = client.post(
        "/admin/clubs/club/league-manager/live/roster-suggestion",
        headers={"Authorization": "Bearer local"},
        json={"roster": [*ROSTER, {"player_id": 99, "player_name": "Wrong Club"}]},
    )
    assert denied.status_code == 400
    assert "outside this club" in denied.json()["detail"]


def test_admin_league_live_create_and_detail_contract(monkeypatch):
    tables = league_live_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    client = TestClient(app)

    response = _create(client, total_rounds=3)

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "league_live_session_create"
    session_id = payload["session"]["id"]
    assert session_id
    assert payload["session"]["updated_at"]
    assert tables["league_live_sessions"][0]["league_name"] == "Tuesday Ladder"
    assert tables["league_live_courts"][0]["court_number"] == 1
    assert any(
        row["action_type"] == "create_league_live_session_admin"
        for row in tables["admin_activity_log"]
    )

    detail = client.get(
        f"/admin/clubs/club/league-manager/live-sessions/{session_id}",
        headers={"Authorization": "Bearer local"},
    )
    assert detail.status_code == 200
    detail_payload = detail.json()
    assert detail_payload["session"]["id"] == session_id
    assert detail_payload["courts"][0]["player_names"] == ["Alex", "Blair", "Casey", "Devon"]


def test_admin_league_live_create_exact_retry_replays_without_second_session(monkeypatch):
    tables = league_live_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    client = TestClient(app)

    first = _create(client, total_rounds=3)
    replay = _create(client, total_rounds=3)

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json()["idempotent"] is True
    assert replay.json()["session"]["id"] == first.json()["session"]["id"]
    assert len(tables["league_live_sessions"]) == 1


def test_admin_league_live_create_reconciles_authoritative_session_after_lost_completion(monkeypatch):
    tables = league_live_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    from services.api import admin_league_manager_routes as routes

    original_update = routes.update_guarded_operation
    lost_completion = {"raised": False}

    def fail_first_completion(*args, **kwargs):
        if kwargs.get("status") == "completed" and not lost_completion["raised"]:
            lost_completion["raised"] = True
            raise RuntimeError("response lost while completing ledger")
        return original_update(*args, **kwargs)

    monkeypatch.setattr(routes, "update_guarded_operation", fail_first_completion)
    client = TestClient(app)

    created = _create(client, total_rounds=3)

    assert created.status_code == 409
    assert created.json()["detail"]["code"] == "RECOVERY_REQUIRED"
    assert len(tables["league_live_sessions"]) == 1
    operation = tables["admin_guarded_operations"][0]
    assert operation["status"] == "recovery_required"

    reconciled = client.post(
        "/admin/clubs/club/league-manager/live-operations/create-live-session-test/reconcile",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "RECONCILE LIVE SESSION", "source": "test"},
    )

    assert reconciled.status_code == 200
    assert reconciled.json()["reconciled"] is True
    assert reconciled.json()["session"]["id"] == tables["league_live_sessions"][0]["id"]
    assert operation["status"] == "completed"
    assert any(row["action_type"] == "reconcile_league_live_session_create" for row in tables["admin_activity_log"])


def test_admin_league_live_create_rejects_same_key_for_changed_request(monkeypatch):
    tables = league_live_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    client = TestClient(app)

    first = _create(client, total_rounds=2)
    changed = _create(client, total_rounds=3)

    assert first.status_code == 200
    assert changed.status_code == 400
    assert "different request" in str(changed.json()["detail"])
    assert len(tables["league_live_sessions"]) == 1


def test_admin_league_live_create_fails_closed_when_intent_cannot_persist(monkeypatch):
    tables = league_live_tables()
    tables["__failed_insert_tables__"] = {"admin_activity_log"}
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = _create(TestClient(app), total_rounds=2)

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "DURABLE_INTENT_UNAVAILABLE"
    assert tables["league_live_sessions"] == []
    assert tables["league_live_courts"] == []


def test_admin_league_live_snapshot_rejects_stale_state(monkeypatch):
    tables = league_live_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    client = TestClient(app)
    session = _create(client).json()["session"]

    response = client.patch(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/snapshot",
        headers={"Authorization": "Bearer local"},
        json={
            "expected_updated_at": "stale-version",
            "roster": ROSTER,
            "courts": COURTS,
            "confirmation_text": "SAVE SESSION",
        },
    )

    assert response.status_code == 409
    assert "another browser" in response.json()["detail"]


def test_admin_league_live_round_save_requires_confirmation(monkeypatch):
    tables = league_live_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    client = TestClient(app)
    session = _create(client).json()["session"]

    response = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1",
        headers={"Authorization": "Bearer local"},
        json={
            "expected_updated_at": session["updated_at"],
            "expected_operation_key": "a" * 64,
            "confirmation_text": "SAVE",
            "matches": MATCHES,
            "courts": COURTS,
        },
    )

    assert response.status_code == 400
    assert "SAVE ROUND" in response.json()["detail"]


def test_admin_league_live_python_plan_save_and_idempotent_retry(monkeypatch):
    tables = league_live_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    client = TestClient(app)
    session = _create(client).json()["session"]
    planned = _plan(client, session)

    assert planned.status_code == 200
    plan = planned.json()
    assert plan["movement"]["authority"] == "python_fastapi"
    assert len(plan["operation_key"]) == 64

    request = {
        "round_label": "Round 1",
        "match_date": "2026-01-15",
        "matches": MATCHES,
        "submitted_match_count": 1,
        "courts": COURTS,
        "advance_after_save": True,
        "expected_updated_at": session["updated_at"],
        "expected_operation_key": plan["operation_key"],
        "confirmation_text": "SAVE ROUND",
    }
    response = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1",
        headers={"Authorization": "Bearer local"},
        json=request,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["idempotent_replay"] is False
    assert payload["round"]["status"] == "submitted"
    assert payload["round"]["operation_key"] == plan["operation_key"]
    assert payload["session"]["current_round"] == 2
    assert tables["league_live_rounds"][0]["submitted_match_count"] == 1
    assert tables["admin_activity_log"][-1]["action_type"] == "save_league_live_round_admin"

    retried = client.post(
        f"/admin/clubs/club/league-manager/live-sessions/{session['id']}/rounds/1",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert retried.status_code == 200
    assert retried.json()["idempotent_replay"] is True
    assert len(tables["league_live_rounds"]) == 1
