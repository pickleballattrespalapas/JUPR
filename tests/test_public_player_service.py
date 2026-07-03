from __future__ import annotations

from jupr_app.services.public_player_service import (
    get_public_match_detail,
    get_public_matches,
    get_public_player_profile,
    get_public_players,
)


class FakeResponse:
    def __init__(self, data):
        self.data = data


class FakeQuery:
    def __init__(self, table_name: str, rows_by_table: dict[str, list[dict]]):
        self.table_name = table_name
        self.rows_by_table = rows_by_table
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
        rows = list(self.rows_by_table.get(self.table_name, []))
        for key, value in self.filters.items():
            rows = [row for row in rows if str(row.get(key)) == str(value)]
        if self.row_limit is not None:
            rows = rows[: self.row_limit]
        return FakeResponse(rows)


class FakeSupabase:
    def __init__(self):
        self.rows_by_table = {
            "players": [
                {"id": 1, "club_id": "club-1", "name": "Alex", "rating": 1600, "wins": 3, "losses": 1, "matches_played": 4, "active": True},
                {"id": 2, "club_id": "club-1", "name": "Blair", "rating": 1500, "wins": 2, "losses": 2, "matches_played": 4, "active": True},
                {"id": 3, "club_id": "club-1", "name": "Casey", "rating": 1490, "wins": 1, "losses": 2, "matches_played": 3, "active": True},
                {"id": 4, "club_id": "club-1", "name": "Devon", "rating": 1480, "wins": 1, "losses": 2, "matches_played": 3, "active": True},
            ],
            "league_ratings": [
                {"id": 11, "club_id": "club-1", "player_id": 1, "league_name": "Open", "rating": 1600, "wins": 3, "losses": 1, "matches_played": 4, "is_active": True}
            ],
            "matches": [
                {
                    "id": 99,
                    "club_id": "club-1",
                    "date": "2026-07-02",
                    "league": "Open",
                    "t1_p1": 1,
                    "t1_p2": 2,
                    "t2_p1": 3,
                    "t2_p2": 4,
                    "score_t1": 11,
                    "score_t2": 8,
                    "elo_delta": 4.5,
                    "t1_p1_r": 1595,
                    "t1_p1_r_end": 1600,
                    "t1_p2_r": 1495,
                    "t1_p2_r_end": 1500,
                    "t2_p1_r": 1495,
                    "t2_p1_r_end": 1490,
                    "t2_p2_r": 1485,
                    "t2_p2_r_end": 1480,
                }
            ],
        }

    def table(self, table_name):
        return FakeQuery(str(table_name), self.rows_by_table)


def test_public_players_are_sanitized_and_searchable():
    rows = get_public_players(FakeSupabase(), club_id="club-1", search="alex")

    assert len(rows) == 1
    assert rows[0]["name"] == "Alex"
    assert rows[0]["rating"] == 1600.0
    assert "email" not in rows[0]


def test_public_player_profile_includes_leagues_and_recent_matches():
    profile = get_public_player_profile(FakeSupabase(), club_id="club-1", player_id=1)

    assert profile is not None
    assert profile["player"]["name"] == "Alex"
    assert profile["league_ratings"][0]["league_name"] == "Open"
    assert profile["recent_matches"][0]["team_1"][0]["name"] == "Alex"
    assert profile["recent_matches"][0]["score_t1"] == 11


def test_public_matches_include_linkable_public_players():
    matches = get_public_matches(FakeSupabase(), club_id="club-1")

    assert len(matches) == 1
    assert matches[0]["team_1"][0] == {"id": 1, "name": "Alex"}
    assert matches[0]["team_2"][1] == {"id": 4, "name": "Devon"}
    assert "t1_p1_r" not in matches[0]


def test_public_match_detail_includes_rating_snapshot_without_raw_columns():
    detail = get_public_match_detail(FakeSupabase(), club_id="club-1", match_id=99)

    assert detail is not None
    assert detail["id"] == 99
    assert detail["rating_snapshot"]["team_1"][0]["start_rating"] == 1595.0
    assert detail["rating_snapshot"]["team_1"][0]["end_rating"] == 1600.0
    assert "t1_p1_r" not in detail
