from __future__ import annotations

import pytest
from postgrest.exceptions import APIError

from jupr_app.data.schema_preflight import (
    REQUIRED_SCHEMA_VERSION,
    ensure_app_write_invariants,
    ensure_badge_schema_preflight,
)


class _FakeResponse:
    def __init__(self, data):
        self.data = data


class _FakeRpcQuery:
    def __init__(self, supabase, name: str):
        self.supabase = supabase
        self.name = name

    def execute(self):
        if self.name == "assert_app_invariants" and not self.supabase.rpc_invariants_ok:
            raise APIError({"code": "P0001", "message": "missing required unique indexes"})
        return _FakeResponse([])


class _FakeTableQuery:
    def __init__(self, supabase, table_name: str):
        self.supabase = supabase
        self.table_name = table_name
        self.selected = None

    def select(self, cols):
        self.selected = cols
        return self

    def in_(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def limit(self, _):
        return self

    def execute(self):
        if self.table_name not in self.supabase.tables:
            raise APIError(
                {
                    "code": "PGRST205",
                    "message": f"Could not find the table '{self.table_name}'",
                }
            )
        if self.table_name == "schema_version":
            return _FakeResponse([{"version": self.supabase.schema_version}])
        if self.selected:
            column = self.selected.split(",")[0].strip()
            if column not in self.supabase.tables[self.table_name]:
                raise APIError(
                    {
                        "code": "PGRST204",
                        "message": f"column {self.table_name}.{column} does not exist",
                    }
                )
        return _FakeResponse([{}])


class _FakeSupabase:
    def __init__(self, tables, *, rpc_invariants_ok: bool = True):
        self.tables = tables
        self.schema_version = REQUIRED_SCHEMA_VERSION
        self.rpc_invariants_ok = rpc_invariants_ok

    def table(self, name: str):
        return _FakeTableQuery(self, name)

    def rpc(self, name: str, _payload: dict):
        return _FakeRpcQuery(self, name)


def test_preflight_passes_with_required_columns():
    supabase = _FakeSupabase(
        {
            "schema_version": {"version"},
            "player_badges": {
                "awarded_by",
                "rule_version",
                "eval_run_id",
                "revoked_at",
                "revoked_by",
                "revoke_reason",
            },
            "badge_eval_runs": {"id"},
        }
    )
    assert ensure_badge_schema_preflight(supabase) is True


def test_preflight_raises_when_schema_version_mismatch(monkeypatch):
    monkeypatch.delenv("JUPR_SKIP_BADGE_SCHEMA_PREFLIGHT", raising=False)
    supabase = _FakeSupabase(
        {
            "schema_version": {"version"},
            "player_badges": {
                "awarded_by",
                "rule_version",
                "eval_run_id",
                "revoked_at",
                "revoked_by",
                "revoke_reason",
            },
            "badge_eval_runs": {"id"},
        }
    )
    supabase.schema_version = "old_version"

    with pytest.raises(RuntimeError) as excinfo:
        ensure_badge_schema_preflight(supabase)

    message = str(excinfo.value)
    assert "Database schema version mismatch" in message
    assert REQUIRED_SCHEMA_VERSION in message
    assert "old_version" in message


def test_preflight_raises_when_write_indexes_are_missing(monkeypatch):
    monkeypatch.delenv("JUPR_SKIP_BADGE_SCHEMA_PREFLIGHT", raising=False)
    supabase = _FakeSupabase(
        {
            "schema_version": {"version"},
            "player_badges": {
                "awarded_by",
                "rule_version",
                "eval_run_id",
                "revoked_at",
                "revoked_by",
                "revoke_reason",
            },
            "badge_eval_runs": {"id"},
        },
        rpc_invariants_ok=False,
    )
    with pytest.raises(RuntimeError) as excinfo:
        ensure_badge_schema_preflight(supabase)
    assert "Database write preflight failed" in str(excinfo.value)


def test_preflight_raises_when_columns_missing(monkeypatch):
    monkeypatch.delenv("JUPR_SKIP_BADGE_SCHEMA_PREFLIGHT", raising=False)
    supabase = _FakeSupabase({"schema_version": {"version"}, "player_badges": {"awarded_by"}})
    with pytest.raises(RuntimeError) as excinfo:
        ensure_badge_schema_preflight(supabase)
    assert "migrations/20260625_badge_recompute_runs.sql" in str(excinfo.value)
    assert "migrations/20260630_player_badges_revocation.sql" in str(excinfo.value)
    assert "enforce_uniques_and_preflight" in str(excinfo.value)
    assert "badge_eval_runs" in str(excinfo.value)


def test_ensure_app_write_invariants_raises_for_missing_unique_indexes():
    supabase = _FakeSupabase({"schema_version": {"version"}}, rpc_invariants_ok=False)
    with pytest.raises(RuntimeError) as excinfo:
        ensure_app_write_invariants(supabase)
    assert "Database write preflight failed" in str(excinfo.value)
