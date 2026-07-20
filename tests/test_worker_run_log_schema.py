import re
from pathlib import Path


MIGRATION = Path("supabase/migrations/20260720123402_baseline_worker_run_log.sql")


def _sql() -> str:
    return re.sub(r"\s+", " ", MIGRATION.read_text(encoding="utf-8").lower()).strip()


def test_worker_run_log_forward_baseline_is_canonical_and_idempotent():
    sql = _sql()

    assert MIGRATION.parent.as_posix() == "supabase/migrations"
    assert MIGRATION.name.startswith("20260720123402_")
    assert "create table if not exists public.worker_run_log" in sql
    assert "create index if not exists worker_run_log_club_created_idx" in sql


def test_worker_run_log_forward_baseline_is_service_role_only():
    sql = _sql()

    assert "alter table public.worker_run_log enable row level security" in sql
    assert "revoke all on table public.worker_run_log from public, anon, authenticated" in sql
    assert "grant all privileges on table public.worker_run_log to service_role" in sql
    assert "notify pgrst, 'reload schema'" in sql
