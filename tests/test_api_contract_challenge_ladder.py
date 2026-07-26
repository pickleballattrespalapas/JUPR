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
        "clubs": [{"id": "club-1", "slug": "tres-palapas", "name": "Tres Palapas", "admin_notes": "private"}],
        "ladder_settings": [{"club_id": "club-1", "challenge_range": 7, "accept_window_hours": 48, "play_window_days": 7, "cooldown_hours": 72, "protected_hours": 72, "pass_hold_hours": 72}],
        "players": [
            {"id": 1, "club_id": "club-1", "name": "Alex", "rating": 1700, "active": True, "private_email": "hidden"},
            {"id": 2, "club_id": "club-1", "name": "Blair", "rating": 1600, "active": True},
            {"id": 3, "club_id": "club-1", "name": "Casey", "rating": 1500, "active": True},
            {"id": 4, "club_id": "club-1", "name": "Dana", "rating": 1480, "active": False},
            {"id": 5, "club_id": "club-1", "name": "Eli", "rating": 1460, "active": False},
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
                "challenger_rank_at_create": 2,
                "defender_rank_at_create": 1,
                "tier_id": "PREM",
                "status": "PENDING_ACCEPTANCE",
                "created_at": "2099-01-01T00:00:00Z",
                "accept_by": "2099-01-03T00:00:00Z",
                "accepted_at": None,
                "play_by": None,
                "completed_at": None,
                "winner_id": None,
                "public_result_json": None,
                "ledger_ref": "private ledger",
                "challenger_contact": "private@example.com",
            },
            {
                "id": 21,
                "club_id": "club-1",
                "challenger_id": 3,
                "defender_id": 1,
                "challenger_rank_at_create": 2,
                "defender_rank_at_create": 1,
                "tier_id": "ADV",
                "status": "COMPLETED",
                "created_at": "2026-01-01T00:00:00Z",
                "accept_by": None,
                "accepted_at": "2026-01-02T00:00:00Z",
                "play_by": None,
                "completed_at": "2026-01-05T00:00:00Z",
                "winner_id": 3,
                "resolution_notes": "private imported note",
                "public_result_json": {
                    "version": 1,
                    "match_ids": {"a": 501, "b": 502},
                    "rank_change": {
                        "swapped": True,
                        "challenger": {"player_id": 3, "before": 2, "after": 1},
                        "defender": {"player_id": 1, "before": 1, "after": 2},
                    },
                },
            }
        ],
        "ladder_pass_usage": [],
        "matches": [
            {
                "id": 501,
                "club_id": "club-1",
                "date": "2026-01-05",
                "context_type": "challenge_ladder",
                "context_id": "private-context-a",
                "deleted_at": None,
                "t1_p1": 3,
                "t1_p2": 4,
                "t2_p1": 1,
                "t2_p2": 5,
                "score_t1": 22,
                "score_t2": 15,
                "t1_p1_r": 1500,
                "t1_p1_r_end": 1510,
                "t1_p2_r": 1480,
                "t1_p2_r_end": 1490,
                "t2_p1_r": 1700,
                "t2_p1_r_end": 1690,
                "t2_p2_r": 1460,
                "t2_p2_r_end": 1450,
            },
            {
                "id": 502,
                "club_id": "club-1",
                "date": "2026-01-05",
                "context_type": "challenge_ladder",
                "context_id": "private-context-b",
                "deleted_at": None,
                "t1_p1": 3,
                "t1_p2": 5,
                "t2_p1": 1,
                "t2_p2": 4,
                "score_t1": 22,
                "score_t2": 16,
                "t1_p1_r": 1510,
                "t1_p1_r_end": 1520,
                "t1_p2_r": 1450,
                "t1_p2_r_end": 1460,
                "t2_p1_r": 1690,
                "t2_p1_r_end": 1680,
                "t2_p2_r": 1490,
                "t2_p2_r_end": 1480,
            },
        ],
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
    assert payload["tiers"][0]["players"][0]["eligibility"]["authority"] == "python"
    assert payload["eligibility_authority"] == "python"
    assert len(payload["status_legend"]) == 8
    assert any(section["title"] == "Swing Partner Swap format" for section in payload["rulebook"])

    pending = next(section for section in payload["challenge_sections"] if section["name"] == "Pending Acceptance")
    assert pending["challenges"][0]["defender"] == {
        "player_id": 1,
        "player_name": "Alex",
        "rank_at_create": 1,
        "current_rank": 1,
        "current_rating_jupr": 4.25,
    }
    recent = next(section for section in payload["challenge_sections"] if section["name"] == "Recently Completed")
    assert recent["challenges"][0]["completed_at"] == "2026-01-05T00:00:00Z"
    assert recent["challenges"][0]["challenger"] == {
        "player_id": 3,
        "player_name": "Casey",
        "rank_at_create": 2,
        "current_rank": 1,
        "current_rating_jupr": 3.75,
    }
    assert recent["challenges"][0]["winner"] == recent["challenges"][0]["challenger"]
    details = recent["challenges"][0]["result_details"]
    assert details["rank_change"]["challenger"]["before"] == 2
    assert details["rank_change"]["challenger"]["after"] == 1
    assert details["matches"][0]["match_id"] == 501
    assert details["matches"][0]["score_challenger_team"] == 22
    assert details["matches"][0]["challenger_partner"] == {
        "player_id": 4,
        "player_name": "Dana",
    }
    assert details["matches"][0]["rating_changes"][0]["before_jupr"] == 3.75
    assert details["matches"][0]["rating_changes"][0]["after_jupr"] == 3.775
    assert "context_id" not in str(details)

    assert "admin_notes" not in payload["club"]
    assert "notes" not in payload["tiers"][0]["players"][0]
    assert "private_email" not in payload["tiers"][0]["players"][0]
    assert "ledger_ref" not in pending["challenges"][0]
    assert "challenger_contact" not in pending["challenges"][0]
    assert "resolution_notes" not in recent["challenges"][0]
    assert "private" not in str(payload).lower()
