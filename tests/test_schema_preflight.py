from __future__ import annotations

import os

import pytest

from jupr_app.data.schema_preflight import ensure_badge_schema_preflight


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeColumnsQuery:
    def __init__(self, supabase):
        self.supabase = supabase
        self.filters = {}

    def select(self, _cols):
        return self

    def eq(self, key, value):
        self.filters[key] = value
        return self

    def execute(self):
        if (
            self.filters.get("table_schema") == "public"
            and self.filters.get("table_name") == "player_badges"
        ):
            return _FakeResponse([{"column_name": col} for col in self.supabase.columns])
        return _FakeResponse([])


class _FakeSchema:
    def __init__(self, supabase):
        self.supabase = supabase

    def table(self, _name):
        return _FakeColumnsQuery(self.supabase)


class _FakeSupabase:
    def __init__(self, columns):
        self.columns = columns

    def schema(self, _name):
        return _FakeSchema(self)


def test_preflight_passes_with_required_columns():
    supabase = _FakeSupabase(
        [
            "awarded_by",
            "rule_version",
            "eval_run_id",
            "revoked_at",
            "revoked_by",
            "revoke_reason",
        ]
    )
    assert ensure_badge_schema_preflight(supabase) is True


def test_preflight_raises_when_columns_missing(monkeypatch):
    monkeypatch.delenv("JUPR_SKIP_DB_PREFLIGHT", raising=False)
    supabase = _FakeSupabase(["awarded_by"])
    with pytest.raises(RuntimeError) as excinfo:
        ensure_badge_schema_preflight(supabase)
    assert "migrations/20260625_badge_recompute_runs.sql" in str(excinfo.value)
    assert "migrations/20260630_player_badges_revocation.sql" in str(excinfo.value)
