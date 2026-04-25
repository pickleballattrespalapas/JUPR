import math

from jupr_app.domain import player_rating_series as prs


class _DummyResp:
    def __init__(self, data):
        self.data = data


class _DummyQuery:
    def __init__(self, rows, missing_columns, select_log):
        self._rows = rows
        self._missing_columns = set(missing_columns)
        self._select_log = select_log
        self._start = 0
        self._end = None
        self._selected_cols = []

    def select(self, cols):
        self._selected_cols = [c.strip() for c in str(cols).split(",") if c.strip()]
        self._select_log.append(self._selected_cols)
        missing = [c for c in self._selected_cols if c in self._missing_columns]
        if missing:
            raise Exception(f"column does not exist: {missing[0]}")
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def or_(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, n):
        self._start = 0
        self._end = int(n) - 1
        return self

    def range(self, start, end):
        self._start = int(start)
        self._end = int(end)
        return self

    def execute(self):
        if self._end is None:
            data = self._rows[self._start :]
        else:
            data = self._rows[self._start : self._end + 1]
        projected = []
        for row in data:
            projected.append({k: row.get(k) for k in self._selected_cols if k in row})
        return _DummyResp(projected)


class _DummySupabase:
    def __init__(self, rows, missing_columns=()):
        self._rows = rows
        self._missing_columns = set(missing_columns)
        self.select_log = []

    def table(self, _name):
        return _DummyQuery(self._rows, self._missing_columns, self.select_log)


_REQUIRED_ROW = {
    "id": 1,
    "date": "2026-01-05T12:00:00Z",
    "league": "Overall",
    "match_type": "league",
    "score_t1": 11,
    "score_t2": 7,
    "t1_p1": 1,
    "t1_p2": 2,
    "t2_p1": 3,
    "t2_p2": 4,
    "elo_delta": 4,
}


_SNAPSHOT_FIELDS = {
    "t1_p1_r": 1200.0,
    "t1_p1_r_end": 1204.0,
    "t1_p2_r": 1100.0,
    "t1_p2_r_end": 1104.0,
    "t2_p1_r": 1000.0,
    "t2_p1_r_end": 996.0,
    "t2_p2_r": 1000.0,
    "t2_p2_r_end": 996.0,
}


def test_missing_context_columns_do_not_crash():
    rows = [{**_REQUIRED_ROW, **_SNAPSHOT_FIELDS}]
    supabase = _DummySupabase(rows, missing_columns={"context_id", "context_type"})

    df = prs.build_player_overall_rating_series(supabase, "club-1", 1, limit=50)

    assert len(df) == 1
    assert "context_id" in df.columns
    assert "context_type" in df.columns
    assert df.loc[0, "context_id"] is None
    assert df.loc[0, "context_type"] == ""
    assert any("context_id" in attempt for attempt in supabase.select_log)
    assert any("context_id" not in attempt for attempt in supabase.select_log)


def test_missing_tournament_id_does_not_crash():
    rows = [{**_REQUIRED_ROW, **_SNAPSHOT_FIELDS, "context_id": 99, "context_type": "ladder"}]
    supabase = _DummySupabase(rows, missing_columns={"tournament_id"})

    df = prs.build_player_overall_rating_series(supabase, "club-1", 1, limit=50)

    assert len(df) == 1
    assert "tournament_id" in df.columns
    assert df.loc[0, "tournament_id"] is None


def test_missing_snapshot_columns_falls_back_to_base_selects():
    rows = [{**_REQUIRED_ROW, "context_id": 7, "context_type": "round_robin", "tournament_id": 88}]
    supabase = _DummySupabase(
        rows,
        missing_columns={
            "t1_p1_r",
            "t1_p1_r_end",
            "t1_p2_r",
            "t1_p2_r_end",
            "t2_p1_r",
            "t2_p1_r_end",
            "t2_p2_r",
            "t2_p2_r_end",
        },
    )

    df = prs.build_player_overall_rating_series(supabase, "club-1", 1, limit=50)

    assert len(df) == 1
    assert math.isclose(df.loc[0, "Overall Δ"], 0.01)
    assert math.isnan(df.loc[0, "Overall After"])
    assert any("t1_p1_r" in attempt for attempt in supabase.select_log)
    assert any("t1_p1_r" not in attempt for attempt in supabase.select_log)


def test_base_match_data_only_still_produces_rating_series():
    rows = [dict(_REQUIRED_ROW)]
    supabase = _DummySupabase(
        rows,
        missing_columns={
            "context_id",
            "context_type",
            "tournament_id",
            "t1_p1_r",
            "t1_p1_r_end",
            "t1_p2_r",
            "t1_p2_r_end",
            "t2_p1_r",
            "t2_p1_r_end",
            "t2_p2_r",
            "t2_p2_r_end",
        },
    )

    df = prs.build_player_overall_rating_series(supabase, "club-1", 1, limit=50)

    assert len(df) == 1
    assert df.loc[0, "Score"] == "11-7"
    assert df.loc[0, "Result"] == "WIN"
    assert any(attempt == [
        "id",
        "date",
        "league",
        "match_type",
        "score_t1",
        "score_t2",
        "t1_p1",
        "t1_p2",
        "t2_p1",
        "t2_p2",
        "elo_delta",
    ] for attempt in supabase.select_log)


def test_partial_rows_return_clean_dataframe_without_crashing():
    rows = [
        {
            "id": 1,
            "date": None,
            "league": "Overall",
            "match_type": "league",
            "score_t1": 11,
            "score_t2": 9,
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "elo_delta": 4,
        }
    ]
    supabase = _DummySupabase(rows)

    df = prs.build_player_overall_rating_series(supabase, "club-1", 1, limit=50)

    assert df.empty
