from jupr_app.ui.pages.player_editor import _plan_league_rating_merge


class _Resp:
    def __init__(self, data):
        self.data = data


class _Query:
    def __init__(self, rows):
        self._rows = rows
        self._filters = {}

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, col, val):
        self._filters[col] = val
        return self

    def execute(self):
        data = [
            r for r in self._rows
            if all(r.get(k) == v for k, v in self._filters.items())
        ]
        return _Resp(data)


class _Supabase:
    def __init__(self, league_rows):
        self.league_rows = league_rows

    def table(self, name):
        assert name == "league_ratings"
        return _Query(self.league_rows)


def test_plan_league_rating_merge_moves_when_target_missing_league():
    sb = _Supabase([
        {"id": 1, "club_id": "c1", "player_id": 10, "league_name": "Monday"},
    ])

    plan = _plan_league_rating_merge(sb, "c1", 10, 20)

    assert plan["move_ids"] == [1]
    assert plan["delete_ids"] == []
    assert plan["conflicts"] == []


def test_plan_league_rating_merge_deletes_when_target_has_same_league():
    sb = _Supabase([
        {"id": 1, "club_id": "c1", "player_id": 10, "league_name": "Monday"},
        {"id": 2, "club_id": "c1", "player_id": 20, "league_name": "Monday"},
    ])

    plan = _plan_league_rating_merge(sb, "c1", 10, 20)

    assert plan["move_ids"] == []
    assert plan["delete_ids"] == [1]
    assert plan["conflicts"] == ["Monday"]


def test_plan_league_rating_merge_noop_when_source_has_no_rows():
    sb = _Supabase([
        {"id": 2, "club_id": "c1", "player_id": 20, "league_name": "Monday"},
    ])

    plan = _plan_league_rating_merge(sb, "c1", 10, 20)

    assert plan["src_rows"] == []
    assert plan["move_ids"] == []
    assert plan["delete_ids"] == []
    assert plan["conflicts"] == []
