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

    def execute(self):
        rows = list(self._rows)
        for key, expected in self._filters.items():
            rows = [row for row in rows if row.get(key) == expected]
        if self._limit is not None:
            rows = rows[: self._limit]
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
        "ladder_settings": [{"club_id": "club-1", "challenge_range": 7, "accept_window_hours": 48, "play_window_days": 7, "cooldown_hours": 72, "protected_hours": 72, "pass_hold_hours": 72}],
        "players": [
            {"id": 1, "club_id": "club-1", "name": "Alex", "rating": 1700, "active": True, "private_email": "hidden"},
            {"id": 2, "club_id": "club-1", "name": "Blair", "rating": 1600, "active": True},
            {"id": 3, "club_id": "club-1", "name": "Casey", "rating": 1500, "active": True},
        ],
        "ladder_roster": [
            {"id": 10, "club_id": "club-1", "player_id": 1, "tier_id": "PREM", "rank": 1, "is_active": True, "notes": "private"},
            {"id": 11, "club_id": "club-1", "player_id": 2, "tier_id": "PREM", "rank": 2, "is_active": True},
            {"id": 12, "club_id": "club-1", "player_id": 3, "tier_id": "ADV", "rank": 1, "is_active": True},
        ],
        "ladder_player_flags": [],
        "ladder_challenges": [
            {
                "id": 20,
                "club_id": "club-1",
                "challenger_id": 2,
                "defender_id": 1,
                "tier_id": "PREM",
                "status": "PENDING_ACCEPTANCE",
                "created_at": "2099-01-01T00:00:00Z",
                "accept_by": "2099-01-03T00:00:00Z",
                "accepted_at": None,
                "play_by": None,
                "completed_at": None,
                "winner_id": None,
                "ledger_ref": "private ledger",
                "challenger_contact": "private@example.com",
            }
        ],
        "ladder_pass_usage": [],
    }
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(tables))
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    return TestClient(app)


def test_public_challenge_ladder_contract(client):
    response = client.get("/clubs/tres-palapas/challenge-ladder")

    assert response.status_code == 200
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert payload["settings"]["challenge_range"] == 7
    assert payload["summary"]["active_player_count"] == 3
    assert payload["tiers"][0]["tier_id"] == "PREM"
    assert payload["tiers"][0]["players"][0]["player_name"] == "Alex"
    assert payload["tiers"][0]["players"][0]["status"] == "Locked"

    pending = next(section for section in payload["challenge_sections"] if section["name"] == "Pending Acceptance")
    assert pending["challenges"][0]["defender"] == {"player_id": 1, "player_name": "Alex"}

    assert "admin_notes" not in payload["club"]
    assert "notes" not in payload["tiers"][0]["players"][0]
    assert "private_email" not in payload["tiers"][0]["players"][0]
    assert "ledger_ref" not in pending["challenges"][0]
    assert "challenger_contact" not in pending["challenges"][0]
