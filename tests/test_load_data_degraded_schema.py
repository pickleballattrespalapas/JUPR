from __future__ import annotations

import pytest
from postgrest.exceptions import APIError

from jupr_app.data.load import load_data


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeTableQuery:
    def __init__(self, supabase, table_name: str):
        self.supabase = supabase
        self.table_name = table_name
        self._select = ""

    def select(self, cols):
        self._select = cols
        return self

    def eq(self, *args, **kwargs):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def execute(self):
        if (
            self.table_name == "player_badges"
            and self.supabase.raise_missing_columns
            and "awarded_by" in self._select
        ):
            raise APIError(
                {
                    "code": "42703",
                    "message": "column player_badges.awarded_by does not exist",
                }
            )
        return _FakeResponse(self.supabase.data.get(self.table_name, []))


class _FakeSupabase:
    def __init__(self):
        self.data = {
            "players": [],
            "league_ratings": [],
            "matches": [],
            "leagues_metadata": [],
            "badges": [],
            "player_badges": [],
        }
        self.raise_missing_columns = True

    def table(self, name: str):
        return _FakeTableQuery(self, name)


def test_load_data_fails_on_missing_player_badges_columns(monkeypatch):
    monkeypatch.setenv("JUPR_SKIP_BADGE_SCHEMA_PREFLIGHT", "1")
    supabase = _FakeSupabase()

    with pytest.raises(APIError):
        load_data(supabase, "club123", match_limit=5)

