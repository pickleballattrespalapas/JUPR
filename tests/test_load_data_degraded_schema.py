from __future__ import annotations

import pandas as pd
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


def test_load_data_degrades_player_badges_schema():
    supabase = _FakeSupabase()

    (
        df_players_all,
        df_players_active,
        df_leagues,
        df_matches,
        df_meta,
        df_badges,
        df_player_badges,
        name_to_id,
        id_to_name,
        schema_degraded,
        schema_degraded_reason,
    ) = load_data(supabase, "club123", match_limit=5)

    assert schema_degraded is True
    assert schema_degraded_reason is not None
    assert "migrations/20260625_badge_recompute_runs.sql" in schema_degraded_reason
    assert "migrations/20260630_player_badges_revocation.sql" in schema_degraded_reason
    assert isinstance(df_player_badges, pd.DataFrame)
    for col in [
        "awarded_by",
        "rule_version",
        "eval_run_id",
        "revoked_at",
        "revoked_by",
        "revoke_reason",
    ]:
        assert col in df_player_badges.columns
