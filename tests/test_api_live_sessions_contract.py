from __future__ import annotations

import logging

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("supabase")

from fastapi.testclient import TestClient

from jupr_app.domain.live_beta_engine import create_round_robin_event, update_round_robin_score
from services.api.main import _is_public_registration_path, app
from tests.test_public_live_write_service import FakeSupabase as FakeWriteSupabase


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, table_name: str, rows: list[dict]):
        self.table_name = table_name
        self.rows = rows
        self.filters: dict[str, object] = {}
        self.row_limit: int | None = None
        self.select_expr = ""

    def select(self, expr="*", *_args, **_kwargs):
        self.select_expr = str(expr or "*")
        return self

    def eq(self, key, value):
        self.filters[str(key)] = value
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.row_limit = int(value)
        return self

    def execute(self):
        if self.table_name != "live_sessions":
            return FakeResponse([])
        rows = list(self.rows)
        for key, value in self.filters.items():
            rows = [row for row in rows if row.get(key) == value]
        if self.row_limit is not None:
            rows = rows[: self.row_limit]
        if self.select_expr and self.select_expr != "*":
            selected_keys = [key.strip() for key in self.select_expr.split(",") if key.strip()]
            rows = [{key: row.get(key) for key in selected_keys if key in row} for row in rows]
        return FakeResponse(rows)


class FakeSupabase:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def table(self, table_name):
        return FakeQuery(str(table_name), self.rows)


class FakeUnavailableLiveSessionsQuery:
    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        raise Exception("Could not find the 'state' column of 'live_sessions' in the schema cache")


class FakeUnavailableLiveSessionsSupabase:
    def table(self, table_name):
        if str(table_name) == "live_sessions":
            return FakeUnavailableLiveSessionsQuery()
        return FakeQuery(str(table_name), [])


class FakeLegacyLiveSessionsQuery(FakeQuery):
    def execute(self):
        if "version" in self.select_expr or "completed_at" in self.select_expr:
            raise Exception("Could not find the 'version' column of 'live_sessions' in the schema cache")
        return super().execute()


class FakeLegacyLiveSessionsSupabase:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def table(self, table_name):
        return FakeLegacyLiveSessionsQuery(str(table_name), self.rows)


class FakeUnavailableOperationLedgerQuery(FakeQuery):
    def execute(self):
        raise RuntimeError("relation public_live_operations does not exist")


class FakeUnavailableOperationLedgerSupabase(FakeSupabase):
    def table(self, table_name):
        if str(table_name) == "public_live_operations":
            return FakeUnavailableOperationLedgerQuery(str(table_name), [])
        return super().table(table_name)


def _row() -> dict:
    event = create_round_robin_event(
        name="API Live Test",
        participant_names=["Amy", "Brooke", "Chris", "Dana"],
    )
    first_match_id = event["rounds"][0]["matches"][0]["id"]
    update_round_robin_score(event, first_match_id, 11, 9)
    return {
        "club_id": "club-1",
        "session_key": "public-session",
        "title": "API Live Test",
        "status": "active",
        "created_at": "2099-07-02T10:00:00+00:00",
        "updated_at": "2099-07-02T10:05:00+00:00",
        "last_seen_at": "2099-07-02T10:05:00+00:00",
        "expires_at": "2099-07-02T20:00:00+00:00",
        "state": {
            "version": 1,
            "session_key": "public-session",
            "event_name": event.get("name"),
            "event_type": event.get("type"),
            "page_state": {"event": event},
            "widget_state": {"admin_only": "hidden"},
        },
    }


def _patch_club(monkeypatch):
    monkeypatch.setattr(
        "services.api.main.get_club",
        lambda club_slug: {
            "club_id": "club-1",
            "club_slug": club_slug,
            "club_name": "Test Club",
        },
    )


def _patch_service_role(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role")


@pytest.fixture
def client(monkeypatch):
    rows = [_row(), {**_row(), "session_key": "abandoned", "status": "abandoned"}]
    _patch_service_role(monkeypatch)
    _patch_club(monkeypatch)
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: FakeSupabase(rows))
    return TestClient(app)


def test_live_sessions_list_returns_sanitized_public_summaries_with_event_metadata(client):
    response = client.get("/clubs/test-club/live-sessions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "test-club", "name": "Test Club"}
    assert [row["session_key"] for row in payload["sessions"]] == ["public-session"]
    assert "state" not in payload["sessions"][0]
    assert payload["sessions"][0]["has_event"] is True
    assert payload["sessions"][0]["event_type"] == "round_robin"


def test_live_session_detail_returns_public_scoreboard_shape(client):
    response = client.get("/clubs/test-club/live-sessions/public-session")

    assert response.status_code == 200
    session = response.json()["session"]
    assert session["title"] == "API Live Test"
    assert session["rounds"][0]["matches"][0]["score_a"] == 11
    assert session["standings"]
    assert "state" not in session
    assert "page_state" not in session
    assert "widget_state" not in session


def test_abandoned_live_session_detail_404s(client):
    response = client.get("/clubs/test-club/live-sessions/abandoned")

    assert response.status_code == 404


def test_live_sessions_requires_service_role_for_private_projection(monkeypatch):
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    _patch_club(monkeypatch)
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: FakeSupabase([]))

    response = TestClient(app).get("/clubs/test-club/live-sessions")

    assert response.status_code == 503
    assert response.json()["detail"] == "Live sessions are temporarily unavailable. Please try again later."
    assert "SUPABASE_SERVICE_ROLE_KEY" not in response.json()["detail"]


def test_live_sessions_list_reports_schema_unavailable(monkeypatch):
    _patch_service_role(monkeypatch)
    _patch_club(monkeypatch)
    monkeypatch.setattr(
        "services.api.main.get_supabase_client",
        lambda: FakeUnavailableLiveSessionsSupabase(),
    )

    response = TestClient(app).get("/clubs/test-club/live-sessions")

    assert response.status_code == 503
    assert response.json()["detail"] == "Live sessions are temporarily unavailable. Please try again later."
    assert "live_sessions" not in response.json()["detail"]


def test_live_session_detail_reports_schema_unavailable(monkeypatch):
    _patch_service_role(monkeypatch)
    _patch_club(monkeypatch)
    monkeypatch.setattr(
        "services.api.main.get_supabase_client",
        lambda: FakeUnavailableLiveSessionsSupabase(),
    )

    response = TestClient(app).get("/clubs/test-club/live-sessions/public-session")

    assert response.status_code == 503


def test_live_session_health_keeps_safe_reason_and_logs_query_failure(monkeypatch, caplog):
    _patch_service_role(monkeypatch)
    monkeypatch.setattr(
        "services.api.main.get_supabase_client",
        lambda: FakeUnavailableLiveSessionsSupabase(),
    )

    with caplog.at_level(logging.ERROR, logger="services.api.main"):
        response = TestClient(app).get("/health/live-sessions")

    payload = response.json()
    assert response.status_code == 200
    assert payload["reason"] == "live_sessions_query_failed"
    assert payload["detail"] == "The live-session data check failed. Review the API logs."
    assert "state" not in payload["detail"]
    assert "Could not find the 'state' column" in caplog.text


def test_live_session_health_distinguishes_durability_schema(monkeypatch, caplog):
    _patch_service_role(monkeypatch)
    monkeypatch.setattr(
        "services.api.main.get_supabase_client",
        lambda: FakeLegacyLiveSessionsSupabase([_row()]),
    )

    with caplog.at_level(logging.WARNING, logger="services.api.main"):
        response = TestClient(app).get("/health/live-sessions")

    payload = response.json()
    assert response.status_code == 200
    assert payload["reason"] == "durability_schema_missing"
    assert payload["live_sessions_query_ok"] is True
    assert payload["durability_schema_ready"] is False
    assert payload["detail"] == "Apply the public live durability migration before enabling writes."
    assert "Could not find the 'version' column" in caplog.text


def test_live_session_health_distinguishes_operation_ledger_failure(monkeypatch, caplog):
    _patch_service_role(monkeypatch)
    monkeypatch.setattr(
        "services.api.main.get_supabase_client",
        lambda: FakeUnavailableOperationLedgerSupabase([_row()]),
    )

    with caplog.at_level(logging.ERROR, logger="services.api.main"):
        response = TestClient(app).get("/health/live-sessions")

    payload = response.json()
    assert response.status_code == 200
    assert payload["reason"] == "operation_ledger_query_failed"
    assert payload["live_sessions_query_ok"] is True
    assert payload["operation_ledger_query_ok"] is False
    assert "public_live_operations" not in payload["detail"]
    assert "relation public_live_operations does not exist" in caplog.text


def test_public_registration_error_scope_only_wraps_public_routes():
    assert _is_public_registration_path("/clubs/test/tournament-registration")
    assert _is_public_registration_path("/clubs/test/tournament-registration/edit")
    assert _is_public_registration_path("/clubs/test/tournament-roster")
    assert not _is_public_registration_path("/admin/clubs/test/tournament-registration")


def test_live_session_reads_fall_back_view_only_before_durability_migration(monkeypatch):
    _patch_service_role(monkeypatch)
    _patch_club(monkeypatch)
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_PUBLIC_LIVE_WRITES", "1")
    monkeypatch.setattr(
        "services.api.main.get_supabase_client",
        lambda: FakeLegacyLiveSessionsSupabase([_row()]),
    )
    legacy_client = TestClient(app)

    listed = legacy_client.get("/clubs/test-club/live-sessions")
    detail = legacy_client.get("/clubs/test-club/live-sessions/public-session")

    assert listed.status_code == 200
    assert listed.json()["write_enabled"] is False
    assert detail.status_code == 200
    assert detail.json()["session"]["version"] == 1


@pytest.fixture
def write_client(monkeypatch):
    supabase = FakeWriteSupabase()
    _patch_service_role(monkeypatch)
    _patch_club(monkeypatch)
    monkeypatch.setenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "api-public-live-token-secret-long-enough")
    monkeypatch.setenv("JUPR_PUBLIC_LIVE_RATE_LIMIT_SECRET", "api-public-live-rate-secret-long-enough")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "public-live")
    monkeypatch.setenv("JUPR_ENABLE_PUBLIC_LIVE_WRITES", "1")
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: supabase)
    return TestClient(app), supabase


def _create_payload(key: str) -> dict:
    return {
        "event_name": "API Write Test",
        "event_type": "round_robin",
        "participant_names": ["Amy", "Brooke", "Chris", "Dana"],
        "live_mode": "quick",
        "idempotency_key": key,
    }


def test_public_live_write_api_token_stale_and_projection_contract(write_client):
    client, _supabase = write_client
    created_response = client.post(
        "/clubs/test-club/live-sessions",
        headers={"fly-client-ip": "203.0.113.8"},
        json=_create_payload("api-create-key-0001"),
    )
    assert created_response.status_code == 200
    created = created_response.json()
    token = created["edit_token"]
    session = created["session"]
    assert token
    assert "edit_token" not in session

    public_response = client.get(f"/clubs/test-club/live-sessions/{session['session_key']}")
    assert public_response.status_code == 200
    assert token not in public_response.text

    first_match = session["rounds"][0]["matches"][0]
    wrong = client.patch(
        f"/clubs/test-club/live-sessions/{session['session_key']}/scores",
        headers={"fly-client-ip": "203.0.113.8"},
        json={
            "edit_token": "wrong-token",
            "expected_version": 1,
            "idempotency_key": "api-score-wrong-0001",
            "scores": [{"match_id": first_match["id"], "score_a": 11, "score_b": 7}],
        },
    )
    assert wrong.status_code == 403

    saved = client.patch(
        f"/clubs/test-club/live-sessions/{session['session_key']}/scores",
        headers={"fly-client-ip": "203.0.113.8"},
        json={
            "edit_token": token,
            "expected_version": 1,
            "idempotency_key": "api-score-save-0001",
            "scores": [{"match_id": first_match["id"], "score_a": 11, "score_b": 7}],
        },
    )
    assert saved.status_code == 200
    assert saved.json()["session"]["version"] == 2

    stale = client.patch(
        f"/clubs/test-club/live-sessions/{session['session_key']}/scores",
        headers={"fly-client-ip": "203.0.113.8"},
        json={
            "edit_token": token,
            "expected_version": 1,
            "idempotency_key": "api-score-stale-0001",
            "scores": [{"match_id": first_match["id"], "score_a": 11, "score_b": 8}],
        },
    )
    assert stale.status_code == 409


def test_public_live_write_api_validates_idempotency_key(write_client):
    client, supabase = write_client
    response = client.post(
        "/clubs/test-club/live-sessions",
        headers={"fly-client-ip": "203.0.113.9"},
        json=_create_payload("short"),
    )
    assert response.status_code == 422
    assert supabase.tables["live_sessions"] == []


def test_public_live_write_api_bounds_public_collections_before_service_work(write_client):
    client, supabase = write_client
    payload = _create_payload("api-bounded-roster-0001")
    payload["participant_names"] = [f"Player {index}" for index in range(21)]
    response = client.post(
        "/clubs/test-club/live-sessions",
        headers={"fly-client-ip": "203.0.113.12"},
        json=payload,
    )
    assert response.status_code == 422
    assert supabase.tables["live_sessions"] == []
    assert supabase.tables["public_live_operations"] == []


def test_public_live_write_api_maps_durable_rate_limit(write_client, monkeypatch):
    client, supabase = write_client
    monkeypatch.setenv("JUPR_PUBLIC_LIVE_CREATE_LIMIT_PER_HOUR", "1")
    first = client.post(
        "/clubs/test-club/live-sessions",
        headers={"fly-client-ip": "203.0.113.10"},
        json=_create_payload("api-rate-create-0001"),
    )
    second = client.post(
        "/clubs/test-club/live-sessions",
        headers={"fly-client-ip": "203.0.113.10"},
        json=_create_payload("api-rate-create-0002"),
    )
    assert first.status_code == 200
    assert second.status_code == 429
    assert len(supabase.tables["live_sessions"]) == 1


def test_public_live_rate_scope_keeps_vercel_visitors_separate(write_client, monkeypatch):
    client, supabase = write_client
    monkeypatch.setenv("JUPR_PUBLIC_LIVE_CREATE_LIMIT_PER_HOUR", "1")
    shared_fly_headers = {"fly-client-ip": "198.51.100.7"}

    first = client.post(
        "/clubs/test-club/live-sessions",
        headers={**shared_fly_headers, "x-vercel-forwarded-for": "203.0.113.21"},
        json=_create_payload("api-vercel-visitor-one"),
    )
    second = client.post(
        "/clubs/test-club/live-sessions",
        headers={**shared_fly_headers, "x-vercel-forwarded-for": "203.0.113.22"},
        json=_create_payload("api-vercel-visitor-two"),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert len({row["requester_hash"] for row in supabase.tables["public_live_operations"]}) == 2


def test_public_live_write_api_needs_separate_production_gate(write_client, monkeypatch):
    client, supabase = write_client
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_ENABLE_PUBLIC_LIVE_WRITES", "1")
    monkeypatch.delenv("JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION", raising=False)
    response = client.post(
        "/clubs/test-club/live-sessions",
        headers={"fly-client-ip": "203.0.113.11"},
        json=_create_payload("api-prod-gate-0001"),
    )
    assert response.status_code == 403
    assert supabase.tables["live_sessions"] == []


def test_public_live_write_gate_fails_closed_for_unknown_runtime(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_PUBLIC_LIVE_WRITES", "1")
    monkeypatch.setenv("JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION", "1")
    monkeypatch.setenv("JUPR_ENV", "stagin")

    from services.api.main import is_public_live_write_enabled

    assert is_public_live_write_enabled() is False

    monkeypatch.delenv("JUPR_ENV")
    assert is_public_live_write_enabled() is False
