from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.match_processing import process_matches


class _ReadQuery:
    def __init__(self, rows: list[dict[str, object]]):
        self._rows = rows
        self._filters: list[tuple[str, object]] = []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, field: str, value: object):
        self._filters.append((field, value))
        return self

    def in_(self, field: str, values: list[object]):
        allowed = set(values)
        self._rows = [row for row in self._rows if row.get(field) in allowed]
        return self

    def execute(self):
        rows = list(self._rows)
        for field, value in self._filters:
            rows = [row for row in rows if str(row.get(field)) == str(value)]
        return SimpleNamespace(data=rows)


class _ReadOnlySupabase:
    def __init__(self, players: list[dict[str, object]]):
        self._tables = {
            "players": players,
            "league_ratings": [],
        }

    def table(self, table_name: str):
        return _ReadQuery(list(self._tables.get(table_name, [])))


def _players() -> list[dict[str, object]]:
    return [
        {
            "id": player_id,
            "club_id": "club-1",
            "rating": 1200.0,
            "wins": 0,
            "losses": 0,
            "matches_played": 0,
            "last_game_at": None,
            "inactive_at": None,
            "active": True,
        }
        for player_id in range(1, 5)
    ]


def test_doubles_tournament_label_does_not_create_managed_league_updates():
    players = _players()
    synthetic_tournament_label = "Tournament · August Open · 3.5 Doubles"

    result = process_matches(
        [
            {
                "date": "2026-08-17T18:00:00Z",
                "league": synthetic_tournament_label,
                "week_tag": "Final",
                "match_type": "Tournament",
                "match_format": "doubles",
                "context_type": "tournament_game",
                "context_id": "game-123",
                "tournament_id": "tournament-123",
                "tournament_game_id": "game-123",
                "rating_scope": "",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 8,
            }
        ],
        supabase=_ReadOnlySupabase(players),
        club_id="club-1",
        name_to_id={},
        df_players_all=pd.DataFrame(players),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        build_write_plan_only=True,
    )

    plan = result["write_plan"]
    assert len(plan["player_updates"]) == 4
    assert plan["league_rating_updates"] == []
    assert plan["league_metadata_expectations"] == []
    assert len(plan["match_rows"]) == 1
    assert plan["match_rows"][0]["league"] == synthetic_tournament_label
    assert plan["match_rows"][0]["match_type"] == "Tournament"
    assert plan["match_rows"][0]["tournament_game_id"] == "game-123"
