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

    def insert(self, payload):
        self.payload = payload
        return self

    def update(self, payload):
        self.payload = payload
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def execute(self):
        return self.supabase.execute(self.table_name, self.payload)


class _FakeSupabase:
    def __init__(self, *, matches_data, matches_error=None):
        self.matches_data = matches_data
        self.matches_error = matches_error
        self.inserted_matches = []

    def table(self, table_name):
        return _Query(self, table_name)

    def execute(self, table_name, payload):
        if table_name == "matches":
            self.inserted_matches.extend(payload or [])
            return _Result(data=self.matches_data, error=self.matches_error)
        if table_name == "players":
            return _Result(data=[{"ok": True}])
        return _Result(data=[])


def _players_df():
    return pd.DataFrame(
        [
            {"id": 1, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
            {"id": 2, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
            {"id": 3, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
            {"id": 4, "rating": 1200.0, "wins": 0, "losses": 0, "matches_played": 0},
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
