from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jupr_app.domain.admin.roles import PERMISSION_VIEW_AUDIT_LOG
from jupr_app.services.admin_guarded_write_service import (
    GuardedWriteRecoveryRequired,
    begin_guarded_operation,
    require_staging_service_role_write,
    update_guarded_operation,
)
from jupr_app.ui.admin_page_permissions import ADMIN_PAGE_PERMISSION_MATRIX


MIGRATION = Path("supabase/migrations/20260719204500_admin_diagnostics_guarded_operations.sql")
GUIDE = Path("apps/web/app/admin/guide/page.tsx")


class GuardedQuery:
    def __init__(self, storage: dict[str, list[dict]], table: str):
        self.storage = storage
        self.table = table
        self.filters: list[tuple[str, object]] = []
        self.limit_value: int | None = None
        self.insert_payload: dict | None = None
        self.update_payload: dict | None = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = dict(payload)
        return self

    def update(self, payload):
        self.update_payload = dict(payload)
        return self

    def execute(self):
        rows = self.storage.setdefault(self.table, [])
        if self.insert_payload is not None:
            row = {"id": f"{self.table}-{len(rows) + 1}", **self.insert_payload}
            rows.append(row)
            return SimpleNamespace(data=[dict(row)])
        scoped = [
            row
            for row in rows
            if all(str(row.get(key)) == str(value) for key, value in self.filters)
        ]
        if self.update_payload is not None:
            for row in scoped:
                row.update(self.update_payload)
            return SimpleNamespace(data=[dict(row) for row in scoped])
        if self.limit_value is not None:
            scoped = scoped[: self.limit_value]
        return SimpleNamespace(data=[dict(row) for row in scoped])


class GuardedSupabase:
    def __init__(self):
        self.storage = {"admin_activity_log": [], "admin_guarded_operations": []}

    def table(self, name):
        return GuardedQuery(self.storage, name)


def _begin(supabase: GuardedSupabase, *, key: str, payload: dict):
    return begin_guarded_operation(
        supabase,
        club_id="club",
        workflow="test_workflow",
        action="test_action",
        operation_key=key,
        request_payload=payload,
        actor_email="owner@example.com",
        actor_role="super_admin",
        source="test_guardrails",
        before_json={"before": True},
    )


def test_guarded_operation_replays_only_exact_completed_request() -> None:
    supabase = GuardedSupabase()
    operation, idempotent = _begin(supabase, key="operation-exact-1", payload={"value": 1})
    assert idempotent is False
    assert supabase.storage["admin_activity_log"][0]["action_type"] == "test_action_intent"

    with pytest.raises(GuardedWriteRecoveryRequired, match="Reconcile or recover"):
        _begin(supabase, key="operation-exact-1", payload={"value": 1})

    update_guarded_operation(
        supabase,
        operation_id=operation["id"],
        status="completed",
        result_json={"ok": True, "operation_key": "operation-exact-1"},
    )
    replay, idempotent = _begin(supabase, key="operation-exact-1", payload={"value": 1})
    assert idempotent is True
    assert replay["status"] == "completed"

    with pytest.raises(ValueError, match="different request"):
        _begin(supabase, key="operation-exact-1", payload={"value": 2})


def test_guarded_ledger_update_failure_is_explicit_recovery_required() -> None:
    supabase = GuardedSupabase()
    operation, _ = _begin(supabase, key="operation-ledger-1", payload={"value": 1})
    supabase.storage["admin_guarded_operations"] = []

    with pytest.raises(GuardedWriteRecoveryRequired) as caught:
        update_guarded_operation(
            supabase,
            operation_id=operation["id"],
            operation_key="operation-ledger-1",
            status="completed",
            result_json={"ok": True},
        )

    assert caught.value.operation_key == "operation-ledger-1"
    assert "ledger is stale" in str(caught.value)


def test_write_preflight_fails_before_touching_database_outside_staging(monkeypatch) -> None:
    touched: list[str] = []

    class NeverTouch:
        def table(self, name):
            touched.append(name)
            raise AssertionError("database should not be touched")

    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-secret")
    with pytest.raises(PermissionError, match="staging-only"):
        require_staging_service_role_write(NeverTouch(), workflow="Guard test")
    assert touched == []

    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FastAPI only"):
        require_staging_service_role_write(NeverTouch(), workflow="Guard test")
    assert touched == []


def test_guarded_operation_migration_is_server_only_and_atomic() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()
    assert "enable row level security" in sql
    assert "force row level security" in sql
    assert "revoke all on table public.admin_guarded_operations from public, anon, authenticated" in sql
    assert "grant select, insert, update on table public.admin_guarded_operations to service_role" in sql
    assert "grant execute on function public.apply_match_canonical_patches_atomic" in sql
    assert "to service_role" in sql
    assert "security definer" in sql
    assert "for update" in sql
    assert "pg_advisory_xact_lock" in sql
    assert sql.index("update public.matches") < sql.index("insert into public.admin_activity_log")
    assert "status = 'completed'" in sql
    assert "grant" not in "\n".join(
        line for line in sql.splitlines() if "authenticated" in line and "revoke" not in line
    )


def test_admin_diagnostics_pages_are_read_gated_and_guide_is_exact() -> None:
    assert ADMIN_PAGE_PERMISSION_MATRIX["badge_debug"] == (PERMISSION_VIEW_AUDIT_LOG,)
    assert ADMIN_PAGE_PERMISSION_MATRIX["badge_audit"] == (PERMISSION_VIEW_AUDIT_LOG,)
    assert ADMIN_PAGE_PERMISSION_MATRIX["match_canonical_audit"] == (PERMISSION_VIEW_AUDIT_LOG,)
    assert ADMIN_PAGE_PERMISSION_MATRIX["admin_tools"] == (PERMISSION_VIEW_AUDIT_LOG,)

    guide = GUIDE.read_text(encoding="utf-8")
    for phrase in (
        "UPDATE BADGE STATE",
        "RECOMPUTE BADGES",
        "REVOKE BADGE",
        "APPLY NORMALIZE",
        "APPROVE SOCIAL SUBMISSION",
        "REJECT SOCIAL SUBMISSION",
        "SAVE ROLE",
        "REVOKE ROLE",
        "PROCESS BADGE QUEUE",
        "DRAIN BADGE QUEUE",
        "RUN BADGE RECOMPUTE",
        "BACKFILL TOURNAMENT MATCHES",
        "RECOVER TOURNAMENT BACKFILL",
    ):
        assert phrase in guide
    for permission in ("view_audit_log", "manage_matches", "manage_roles", "run_replay"):
        assert permission in guide
    assert "Migrated route safety contracts" in guide
    assert "Streamlit" in guide
