from __future__ import annotations

from jupr_app.data.load import load_data


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeTableQuery:
    def __init__(self, supabase, table_name: str):
        self.supabase = supabase
        self.table_name = table_name
        self.filters: list[tuple[str, str, object]] = []

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, column: str, value):
        self.filters.append(("eq", column, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        self.supabase.executed.append((self.table_name, list(self.filters)))
        return _FakeResponse(self.supabase.data.get(self.table_name, []))


class _FakeSupabase:
    def __init__(self):
        self.data = {
            "players": [],
            "league_ratings": [],
            "matches": [],
            "leagues_metadata": [{"id": 1, "club_id": "club123", "league_name": "Open"}],
            "badges": [],
            "player_badges": [],
        }
        self.executed: list[tuple[str, list[tuple[str, str, object]]]] = []

    def table(self, name: str):
        return _FakeTableQuery(self, name)


def test_load_data_reads_leagues_metadata_scoped_by_club(monkeypatch):
    monkeypatch.setenv("JUPR_SKIP_BADGE_SCHEMA_PREFLIGHT", "1")
    supabase = _FakeSupabase()

    (
        _df_players_all,
        _df_players_active,
        _df_leagues,
        _df_matches,
        df_meta,
        _df_badges,
        _df_player_badges,
        _name_to_id,
        _id_to_name,
        _schema_degraded,
        _schema_degraded_reason,
    ) = load_data(supabase, "club123", match_limit=5)

    meta_queries = [q for q in supabase.executed if q[0] == "leagues_metadata"]
    assert len(meta_queries) == 1
    assert ("eq", "club_id", "club123") in meta_queries[0][1]
    assert not df_meta.empty
    assert "league_name" in df_meta.columns
