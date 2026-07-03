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

    def in_(self, key, values):
        self._filters[key] = set(values)
        return self

    def limit(self, value):
        self._limit = int(value)
        return self

    def execute(self):
        rows = list(self._rows)
        for key, expected in self._filters.items():
            if isinstance(expected, set):
                rows = [row for row in rows if row.get(key) in expected]
            else:
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
        "clubs": [
            {
                "id": "club-1",
                "slug": "tres-palapas",
                "name": "Tres Palapas",
                "admin_notes": "private",
            }
        ],
        "players": [
            {"id": 1, "club_id": "club-1", "name": "Alex", "rating": 1600, "wins": 1, "losses": 0, "matches_played": 1, "active": True},
            {"id": 2, "club_id": "club-1", "name": "Blair", "rating": 1400, "wins": 0, "losses": 1, "matches_played": 1, "active": True},
            {"id": 3, "club_id": "club-1", "name": "Casey", "rating": 1200, "wins": 0, "losses": 1, "matches_played": 1, "active": True},
            {"id": 4, "club_id": "club-1", "name": "Devon", "rating": 1200, "wins": 0, "losses": 1, "matches_played": 1, "active": True},
        ],
        "league_ratings": [
            {"player_id": 1, "club_id": "club-1", "league_name": "Open", "rating": 1500, "is_active": True},
            {"player_id": 2, "club_id": "club-1", "league_name": "Open", "rating": 1450, "is_active": True},
            {"player_id": 3, "club_id": "club-1", "league_name": "Open", "rating": 1300, "is_active": True},
            {"player_id": 4, "club_id": "club-1", "league_name": "Open", "rating": 1250, "is_active": True},
        ],
        "leagues_metadata": [
            {"club_id": "club-1", "league_name": "Open", "is_active": True, "status": "active", "k_factor": 24},
        ],
    }
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _key: FakeSupabase(tables))
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "fake-anon-key")
    return TestClient(app)


def test_public_match_explorer_context_contract(client):
    response = client.get("/clubs/tres-palapas/match-explorer")

    assert response.status_code == 200
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert payload["contexts"] == ["OVERALL", "Open"]
    assert "admin_notes" not in payload["club"]


def test_public_match_explorer_preview_contract(client):
    response = client.get(
        "/clubs/tres-palapas/match-explorer/preview",
        params={
            "me": 1,
            "partner": 2,
            "opp1": 3,
            "opp2": 4,
            "context": "Open",
            "score_you": 11,
            "score_opp": 7,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    preview = payload["preview"]
    assert preview["context"] == {"name": "Open", "k_factor": 24}
    assert preview["teams"]["you"]["players"][0]["name"] == "Alex"
    assert preview["rating_delta"]["you_team_elo"] > 0

    player_payload = preview["teams"]["you"]["players"][0]
    assert set(player_payload) == {"id", "name", "overall_rating", "overall_jupr", "context_rating", "context_jupr"}
    assert "email" not in player_payload
    assert "admin_notes" not in payload["club"]


def test_public_match_explorer_preview_rejects_duplicate_players(client):
    response = client.get(
        "/clubs/tres-palapas/match-explorer/preview",
        params={"me": 1, "partner": 1, "opp1": 3, "opp2": 4},
    )

    assert response.status_code == 400
    assert "four different players" in response.json()["detail"]
