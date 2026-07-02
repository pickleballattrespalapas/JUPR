from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("supabase")

from fastapi.testclient import TestClient

from jupr_app.domain.live_beta_engine import create_round_robin_event, update_round_robin_score
from services.api.main import app


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, table_name: str, rows: list[dict]):
        self.table_name = table_name
        self.rows = rows
        self.filters: dict[str, object] = {}
        self.row_limit: int | None = None

    def select(self, *_args, **_kwargs):
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
        raise Exception("Could not find the 'source' column of 'live_sessions' in the schema cache")


class FakeUnavailableLiveSessionsSupabase:
    def table(self, table_name):
        if str(table_name) == "live_sessions":
            return FakeUnavailableLiveSessionsQuery()
        return FakeQuery(str(table_name), [])


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
        "created_at": "2026-07-02T10:00:00+00:00",
        "updated_at": "2026-07-02T10:05:00+00:00",
        "last_seen_at": "2026-07-02T10:05:00+00:00",
        "expires_at": "2026-07-02T20:00:00+00:00",
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


@pytest.fixture
def client(monkeypatch):
    rows = [_row(), {**_row(), "session_key": "abandoned", "status": "abandoned"}]
    _patch_club(monkeypatch)
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: FakeSupabase(rows))
    return TestClient(app)


def test_live_sessions_list_returns_public_summaries(client):
    response = client.get("/clubs/test-club/live-sessions")

    assert response.status_code == 200
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "test-club", "name": "Test Club"}
    assert [row["session_key"] for row in payload["sessions"]] == ["public-session"]
    assert "state" not in payload["sessions"][0]


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


def test_live_sessions_list_degrades_to_empty_when_schema_is_unavailable(monkeypatch):
    _patch_club(monkeypatch)
    monkeypatch.setattr(
        "services.api.main.get_supabase_client",
        lambda: FakeUnavailableLiveSessionsSupabase(),
    )

    response = TestClient(app).get("/clubs/test-club/live-sessions")

    assert response.status_code == 200
    assert response.json()["sessions"] == []


def test_live_session_detail_404s_when_schema_is_unavailable(monkeypatch):
    _patch_club(monkeypatch)
    monkeypatch.setattr(
        "services.api.main.get_supabase_client",
        lambda: FakeUnavailableLiveSessionsSupabase(),
    )

    response = TestClient(app).get("/clubs/test-club/live-sessions/public-session")

    assert response.status_code == 404
