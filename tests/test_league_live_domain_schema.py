from pathlib import Path


MIGRATION = Path("supabase/migrations/20260719182921_league_live_domain_contract.sql")


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_league_live_contract_uses_canonical_supabase_migration() -> None:
    assert MIGRATION.parent.as_posix() == "supabase/migrations"
    sql = _sql()
    for table in ("league_live_sessions", "league_live_rounds", "league_live_courts"):
        assert f"create table if not exists public.{table}" in sql


def test_league_live_tables_are_private_service_role_only() -> None:
    sql = _sql()
    for table in ("league_live_sessions", "league_live_rounds", "league_live_courts"):
        assert f"alter table public.{table} enable row level security" in sql
        assert f"revoke all on table public.{table} from public, anon, authenticated" in sql
        assert f"grant select, insert, update, delete on table public.{table} to service_role" in sql
        assert f"grant select on table public.{table} to anon" not in sql
        assert f"grant select on table public.{table} to authenticated" not in sql


def test_league_live_rounds_have_durable_idempotency_and_schema_reload() -> None:
    sql = _sql()
    assert "add column if not exists operation_key text" in sql
    assert "create unique index if not exists idx_league_live_rounds_session_operation" in sql
    assert "where operation_key is not null and operation_key <> ''" in sql
    assert "notify pgrst, 'reload schema'" in sql
