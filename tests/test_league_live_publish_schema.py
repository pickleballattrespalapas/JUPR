from pathlib import Path


MIGRATION = Path("supabase/migrations/20260719190954_league_live_publish_reconciliation.sql")


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_publish_coordination_uses_canonical_supabase_migration() -> None:
    sql = _sql()
    assert "create table if not exists public.league_live_publish_operations" in sql
    assert "create table if not exists public.league_live_guest_players" in sql
    assert "request_fingerprint text not null" in sql
    assert "constraint league_live_publish_operations_round_unique unique (session_id, round_number)" in sql
    assert "constraint league_live_publish_operations_idempotency_unique unique (club_id, idempotency_key)" in sql
    assert "recovery_required" in sql
    assert "compensated" in sql


def test_publish_coordination_is_private_service_role_only() -> None:
    sql = _sql()
    for table in ("league_live_publish_operations", "league_live_guest_players"):
        assert f"alter table public.{table} enable row level security" in sql
        assert f"revoke all on table public.{table} from public, anon, authenticated" in sql
        assert f"grant select, insert, update, delete on table public.{table} to service_role" in sql
        assert f"grant select on table public.{table} to authenticated" not in sql


def test_match_publish_context_is_unique_and_schema_reloads() -> None:
    sql = _sql()
    assert "create unique index if not exists idx_matches_league_live_publish_context" in sql
    assert "where context_type = 'league_live_session' and context_id is not null" in sql
    assert "notify pgrst, 'reload schema'" in sql
