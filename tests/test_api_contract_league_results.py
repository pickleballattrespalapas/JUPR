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

    def is_(self, key, value):
        assert value is None
        self._filters[key] = None
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
        "players": [
            {"id": 1, "club_id": "club-1", "name": "Alex", "rating": 1600, "active": True, "private_email": "hidden"},
            {"id": 2, "club_id": "club-1", "name": "Blair", "rating": 1500, "active": True},
            {"id": 3, "club_id": "club-1", "name": "Casey", "rating": 1400, "active": True},
            {"id": 4, "club_id": "club-1", "name": "Devon", "rating": 1300, "active": True},
        ],
        "leagues_metadata": [{"club_id": "club-1", "league_name": "Open", "is_active": True, "status": "active", "min_games": 4, "k_factor": 24}],
        "league_ratings": [
            {"club_id": "club-1", "player_id": 1, "league_name": "Open", "rating": 1640, "starting_rating": 1600, "wins": 3, "losses": 1, "matches_played": 4, "is_active": True, "admin_notes": "private"},
            {"club_id": "club-1", "player_id": 2, "league_name": "Open", "rating": 1500, "starting_rating": 1500, "wins": 2, "losses": 2, "matches_played": 4, "is_active": True},
        ],
        "matches": [
            {
                "id": 10,
                "club_id": "club-1",
                "date": "2026-01-01T00:00:00Z",
                "league": "Open",
                "match_type": "Live Match",
                "week_tag": "Week 1",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
                "t1_p1_r": 1600,
                "t1_p1_r_end": 1610,
                "t1_p2_r": 1500,
                "t1_p2_r_end": 1510,
                "t2_p1_r": 1400,
                "t2_p1_r_end": 1390,
                "t2_p2_r": 1300,
                "t2_p2_r_end": 1290,
                "admin_flag": "secret",
            }
        ],
    }
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(tables))
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    return TestClient(app)


def test_public_league_results_contract(client):
    response = client.get("/clubs/tres-palapas/league-results?league_name=Open")

    assert response.status_code == 200
    payload = response.json()
    assert payload["club"] == {"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas"}
    assert payload["selected_league"] == "Open"
    assert payload["past_leagues"] == []
    assert payload["league"] == {
        "name": "Open",
        "league_type": "Individual",
        "match_format": "doubles",
        "min_games": 4,
        "k_factor": 24,
        "start_week": None,
        "end_week": None,
        "num_weeks": None,
    }
    assert payload["standings"][0]["player_name"] == "Alex"
    assert payload["weeks"] == [{"week_num": 1, "week_label": "Week 1", "has_results": True}]
    assert payload["selected_week"] == 1
    assert payload["weekly_results"]
    assert payload["cumulative"]
    alex_season = next(
        row for row in payload["cumulative"] if row["player_name"] == "Alex"
    )
    assert (
        alex_season["games"],
        alex_season["wins"],
        alex_season["losses"],
    ) == (4, 3, 1)
    assert payload["weekly_results"][0]["rank"]
    assert payload["weekly_results"][0]["rank_delta"] is None
    assert payload["weekly_highlights"]["scope"] == "week"
    assert payload["season_highlights"]["scope"] == "season"
    assert payload["players"]
    assert payload["player_summary"]
    assert (
        payload["player_summary"]["games"],
        payload["player_summary"]["wins"],
        payload["player_summary"]["losses"],
    ) == (4, 3, 1)
    assert payload["recent_matches"]
    assert payload["award_progress"]["award_count"] == len(
        payload["award_progress"]["awards"]
    )

    assert "admin_notes" not in payload["club"]
    assert "admin_notes" not in payload["standings"][0]
    assert "admin_flag" not in payload["weekly_results"][0]
    assert "private_email" not in payload["standings"][0]


def test_public_league_results_defaults_to_first_available_league(client):
    response = client.get("/clubs/tres-palapas/league-results")

    assert response.status_code == 200
    assert response.json()["selected_league"] == "Open"


def test_public_league_results_deep_link_selectors_are_server_authoritative(client):
    response = client.get(
        "/clubs/tres-palapas/league-results",
        params={"league_name": "Open", "week": 1, "player": 2, "weekly_min_games": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_week"] == 1
    assert payload["selected_player_id"] == 2
    assert payload["player_summary"]["player_name"] == "Blair"
    assert (
        payload["player_summary"]["games"],
        payload["player_summary"]["wins"],
        payload["player_summary"]["losses"],
    ) == (4, 2, 2)
    assert payload["weekly_highlights"]["min_games"] == 1
    assert payload["recent_matches"][0]["partner"]["player_name"] == "Alex"


def test_public_league_results_rejects_invalid_qualification(client):
    response = client.get(
        "/clubs/tres-palapas/league-results?weekly_min_games=0"
    )

    assert response.status_code == 422
