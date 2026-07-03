from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.services.public_match_explorer_service import (
    build_public_match_explorer_preview,
    get_public_match_explorer_contexts,
)


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


def fake_supabase() -> FakeSupabase:
    return FakeSupabase(
        {
            "players": [
                {"id": 1, "club_id": "club", "name": "Alex", "rating": 1600, "wins": 1, "losses": 0, "matches_played": 1, "active": True},
                {"id": 2, "club_id": "club", "name": "Blair", "rating": 1400, "wins": 0, "losses": 1, "matches_played": 1, "active": True},
                {"id": 3, "club_id": "club", "name": "Casey", "rating": 1200, "wins": 0, "losses": 1, "matches_played": 1, "active": True},
                {"id": 4, "club_id": "club", "name": "Devon", "rating": 1200, "wins": 0, "losses": 1, "matches_played": 1, "active": True},
            ],
            "league_ratings": [
                {"player_id": 1, "club_id": "club", "league_name": "Open", "rating": 1500, "is_active": True},
                {"player_id": 2, "club_id": "club", "league_name": "Open", "rating": 1450, "is_active": True},
                {"player_id": 3, "club_id": "club", "league_name": "Open", "rating": 1300, "is_active": True},
                {"player_id": 4, "club_id": "club", "league_name": "Open", "rating": 1250, "is_active": True},
            ],
            "leagues_metadata": [
                {"club_id": "club", "league_name": "Open", "is_active": True, "status": "active", "k_factor": 24},
                {"club_id": "club", "league_name": "Archived", "is_active": False, "status": "archived", "k_factor": 24},
            ],
        }
    )


def test_public_match_explorer_contexts_include_overall_and_active_leagues() -> None:
    contexts = get_public_match_explorer_contexts(fake_supabase(), club_id="club")

    assert contexts == ["OVERALL", "Open"]


def test_public_match_explorer_preview_is_public_safe_and_read_only() -> None:
    preview = build_public_match_explorer_preview(
        fake_supabase(),
        club_id="club",
        me=1,
        partner=2,
        opp1=3,
        opp2=4,
        context_name="Open",
        score_you=11,
        score_opp=7,
    )

    assert preview["context"] == {"name": "Open", "k_factor": 24}
    assert preview["teams"]["you"]["players"][0]["name"] == "Alex"
    assert preview["teams"]["you"]["average_rating"] == 1475
    assert preview["teams"]["opponents"]["average_rating"] == 1275
    assert 0 <= preview["expected"]["you"] <= 1
    assert preview["rating_delta"]["you_team_elo"] > 0

    player_payload = preview["teams"]["you"]["players"][0]
    assert set(player_payload) == {"id", "name", "overall_rating", "overall_jupr", "context_rating", "context_jupr"}


def test_public_match_explorer_rejects_duplicate_players() -> None:
    with pytest.raises(ValueError, match="four different players"):
        build_public_match_explorer_preview(
            fake_supabase(),
            club_id="club",
            me=1,
            partner=1,
            opp1=3,
            opp2=4,
        )
