import pandas as pd
import pytest

from jupr_app.domain.match_processing import process_matches


class _Result:
    def __init__(self, data=None, error=None):
        self.data = data
        self.error = error


class _Query:
    def __init__(self, supabase, table_name):
        self.supabase = supabase
        self.table_name = table_name
        self.payload = None
        self.operation = "select"
        self.filters = []
        self.row_limit = None

    def select(self, _columns):
        self.operation = "select"
        return self

    def insert(self, payload):
        self.operation = "insert"
        self.payload = payload
        return self

    def update(self, payload):
        self.operation = "update"
        self.payload = payload
        return self

    def eq(self, column, value):
        self.filters.append((column, lambda actual: actual == value))
        return self

    def is_(self, column, value):
        value = None if value == "null" else value
        self.filters.append((column, lambda actual: actual is value))
        return self

    def in_(self, column, values):
        self.filters.append((column, lambda actual: actual in values))
        return self

    def limit(self, count):
        self.row_limit = count
        return self

    def execute(self):
        return self.supabase.execute(self)


class _FakeSupabase:
    def __init__(self, *, matches_data, matches_error=None):
        self.matches_data = matches_data
        self.matches_error = matches_error
        self.inserted_matches = []
        self.players = _players_df().to_dict("records")

    def table(self, table_name):
        return _Query(self, table_name)

    def execute(self, query):
        if query.table_name == "matches" and query.operation == "insert":
            self.inserted_matches.extend(query.payload or [])
            return _Result(data=self.matches_data, error=self.matches_error)
        if query.table_name == "players":
            rows = [row for row in self.players
                    if all(predicate(row.get(column)) for column, predicate in query.filters)]
            if query.row_limit is not None:
                rows = rows[:query.row_limit]
            if query.operation == "update":
                for row in rows:
                    row.update(query.payload)
            return _Result(data=[dict(row) for row in rows])
        return _Result(data=[])


def _players_df():
    return pd.DataFrame(
        [
            {"id": 1, "club_id": "club-1", "name": "Player 1", "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
            {"id": 2, "club_id": "club-1", "name": "Player 2", "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
            {"id": 3, "club_id": "club-1", "name": "Player 3", "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
            {"id": 4, "club_id": "club-1", "name": "Player 4", "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
        ]
    )


def _match_payload():
    return {
        "league": "Valentine",
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "score_t1": 21,
        "score_t2": 19,
        "tournament_game_id": "game-123",
        "match_type": "PopUp",
        "is_popup": True,
    }


def test_process_matches_raises_when_match_insert_returns_no_data():
    supabase = _FakeSupabase(matches_data=None, matches_error={"message": "insert failed"})

    with pytest.raises(RuntimeError, match="Failed to insert match rows"):
        process_matches(
            [_match_payload()],
            supabase=supabase,
            club_id="club-1",
            name_to_id={},
            df_players_all=_players_df(),
            df_leagues=pd.DataFrame(),
            df_meta=pd.DataFrame(),
        )


def test_process_matches_inserts_rows_when_match_insert_succeeds():
    supabase = _FakeSupabase(matches_data=[{"id": "m1"}])

    result = process_matches(
        [_match_payload()],
        supabase=supabase,
        club_id="club-1",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )

    assert result["inserted"] == 1
    assert len(supabase.inserted_matches) == 1


def test_legacy_synthetic_tournament_label_remains_unmanaged():
    supabase = _FakeSupabase(matches_data=[{"id": "m1"}])

    result = process_matches(
        [
            {
                **_match_payload(),
                "league": "Tournament Pool A",
                "match_type": "Tournament",
                "is_popup": False,
                "rating_scope": "unrated",
            }
        ],
        supabase=supabase,
        club_id="club-1",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame([{"league_name": "Different Managed League"}]),
    )

    assert result["inserted"] == 1
    assert len(supabase.inserted_matches) == 1
    for player in supabase.players:
        assert player["last_game_at"] == supabase.inserted_matches[0]["date"]
        assert (player["rating"], player["wins"], player["losses"], player["matches_played"]) == (1200, 0, 0, 0)
        assert player["active"] is True and player["inactive_at"] is None


def test_process_matches_winner_bonus_adds_to_winners_without_extra_loser_penalty():
    base_supabase = _FakeSupabase(matches_data=[{"id": "base"}])
    bonus_supabase = _FakeSupabase(matches_data=[{"id": "bonus"}])

    process_matches(
        [_match_payload()],
        supabase=base_supabase,
        club_id="club-1",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )
    bonus_result = process_matches(
        [{**_match_payload(), "winner_bonus_elo": 5.0, "winner_bonus_reason": "tournament_semifinal_winner_bonus"}],
        supabase=bonus_supabase,
        club_id="club-1",
        name_to_id={},
        df_players_all=_players_df(),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
    )

    base_row = base_supabase.inserted_matches[0]
    bonus_row = bonus_supabase.inserted_matches[0]
    assert round(float(bonus_row["t1_p1_r_end"]) - float(base_row["t1_p1_r_end"]), 6) == 5.0
    assert round(float(bonus_row["t1_p2_r_end"]) - float(base_row["t1_p2_r_end"]), 6) == 5.0
    assert round(float(bonus_row["t2_p1_r_end"]) - float(base_row["t2_p1_r_end"]), 6) == 0.0
    assert round(float(bonus_row["t2_p2_r_end"]) - float(base_row["t2_p2_r_end"]), 6) == 0.0
    assert bonus_row["rating_bonus_elo"] == 5.0
    assert bonus_row["rating_bonus_reason"] == "tournament_semifinal_winner_bonus"
    assert bonus_result["winner_bonus_summary"]["match_count"] == 1
    assert bonus_result["winner_bonus_summary"]["player_elo_total"] == 10.0
