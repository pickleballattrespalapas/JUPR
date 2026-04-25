from pathlib import Path

from jupr_app.domain import player_rating_series as prs


class _DummyResp:
    def __init__(self, data):
        self.data = data


class _DummyQuery:
    def __init__(self, rows):
        self._rows = rows
        self._start = 0
        self._end = None

    def select(self, _cols):
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
        return _DummyResp(data)


class _DummySupabase:
    def __init__(self, rows):
        self._rows = rows

    def table(self, _name):
        return _DummyQuery(self._rows)


def test_build_player_overall_rating_series_supports_paginated_all_matches():
    rows = []
    for idx in range(2505, 0, -1):
        rows.append(
            {
                "id": idx,
                "date": f"2026-01-{(idx % 28) + 1:02d}T12:00:00Z",
                "league": "Overall",
                "match_type": "league",
                "score_t1": 11,
                "score_t2": 7,
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "elo_delta": 4,
                "t1_p1_r": 400.0 + idx,
                "t1_p1_r_end": 404.0 + idx,
            }
        )

    supabase = _DummySupabase(rows)

    df_all = prs.build_player_overall_rating_series(supabase, "club-1", 1, limit=None)
    assert len(df_all) == 2505

    df_recent = prs.build_player_overall_rating_series(supabase, "club-1", 1, limit=60)
    assert len(df_recent) == 60


def test_players_page_has_profile_match_scope_controls_and_default_recent_limit():
    contents = Path("jupr_app/ui/pages/players.py").read_text(encoding="utf-8")

    assert 'options=["Recent 60", "All matches"]' in contents
    assert 'selected_scope == "All matches"' in contents
    assert 'build_player_overall_rating_series(_supabase, club_id, pid, limit=60)' in contents
    assert 'build_player_overall_rating_series(_supabase, club_id, pid, limit=None)' in contents
    assert 'Showing recent {shown_matches} matches for faster loading.' in contents
    assert 'Showing all {shown_matches} matches.' in contents
