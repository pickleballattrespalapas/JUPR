from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


class FakeQuery:
    def __init__(self, rows):
        self._rows = list(rows)
        self._filters: dict[str, object] = {}
        self._limit: int | None = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self._filters[key] = value
        return self

    def limit(self, value):
        self._limit = int(value)
        return self

    def range(self, start, end):
        self.page_bounds = (start, end)
        return self

    def order(self, *_args, **_kwargs):
        return self

    def execute(self):
        rows = list(self._rows)
        for key, expected in self._filters.items():
            rows = [row for row in rows if row.get(key) == expected]
        if self._limit is not None:
            rows = rows[: self._limit]
        if hasattr(self, "page_bounds"):
            rows = rows[self.page_bounds[0]:self.page_bounds[1] + 1]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, tables):
        self._tables = tables

    def table(self, name):
        return FakeQuery(self._tables.get(name, []))


@pytest.fixture
def client(monkeypatch):
    tables = {
        "clubs": [{"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas", "admin_notes": "private"}],
        "badges": [
            {
                "badge_id": "participant",
                "name": "Participant",
                "category": "Participation",
                "prestige": 10,
                "requirements": "Play a recorded match",
                "description_md": "Earn your first reel.",
                "state": "live",
                "is_active": True,
                "internal_rule_sql": "private",
            },
            {
                "badge_id": "old_unused",
                "name": "Old Unused",
                "category": "Old",
                "prestige": 1,
                "state": "deprecated",
                "is_active": False,
            },
        ],
        "player_badges": [
            {"club_id": "club-1", "player_id": 1, "badge_id": "participant", "earned_at": "2026-01-03T00:00:00Z", "raw_eval_payload": "private"},
            {"club_id": "club-1", "player_id": 2, "badge_id": "participant", "earned_at": "2026-01-02T00:00:00Z"},
        ],
        "players": [
            {"id": 1, "club_id": "club-1", "name": "Alex", "active": True, "private_email": "hidden"},
            {"id": 2, "club_id": "club-1", "name": "Blair", "active": True},
        ],
    }
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(tables))
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    return TestClient(app)


def test_public_badge_codex_contract(client):
    response = client.get("/clubs/tres-palapas/badges")

    assert response.status_code == 200
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert payload["summary"]["badge_count"] == 1
    assert payload["sections"][0]["name"] == "Participation"
    assert [bucket["name"] for bucket in payload["catalog_buckets"]] == [
        "Live Now",
        "Seasonal / League Close",
        "Manual / Curated",
        "Tracked / Disabled",
    ]

    badge = payload["sections"][0]["badges"][0]
    assert badge["badge_id"] == "participant"
    assert badge["earners_count"] == 2
    assert badge["catalog_bucket"] == "Live Now"
    assert badge["badge_scope"] == "lifetime"
    assert badge["requirements"] == "Play 1 recorded match (lifetime)."
    assert badge["recent_earners"][0]["player_name"] == "Alex"
    assert "internal_rule_sql" not in badge
    assert "raw_eval_payload" not in badge["recent_earners"][0]
    assert "admin_notes" not in payload["club"]
    assert payload["trophy_room"][0]["player_name"] == "Alex"
    assert "private_email" not in str(payload["trophy_room"])


def test_public_badge_earners_contract(client):
    response = client.get("/clubs/tres-palapas/badges/participant/earners?offset=0&limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["badge_id"] == "participant"
    assert payload["total"] == 2
    assert payload["earners"] == [{"player_id": 1, "player_name": "Alex", "earned_at": "2026-01-03T00:00:00Z"}]
    assert payload["badge"]["catalog_bucket"] == "Live Now"
    assert "private_email" not in payload["earners"][0]


def test_public_badge_earners_rejects_unknown_badges(client):
    response = client.get("/clubs/tres-palapas/badges/not-a-real-badge/earners")

    assert response.status_code == 400
    assert response.json()["detail"] == "badge not found"
