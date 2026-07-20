from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase" / "migrations" / "20260719193000_admin_player_merge_transactions.sql"


def test_player_merge_migration_is_atomic_recoverable_and_server_only() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "create table if not exists public.admin_player_merge_operations" in sql
    assert "merged_pending_replay" in sql
    assert "replay_verified" in sql
    assert "compensated" in sql
    assert "create or replace function public.server_merge_player_accounts" in sql
    assert "create or replace function public.server_compensate_player_merge" in sql
    assert "create or replace function public.server_verify_player_merge_replay" in sql
    assert sql.count("security invoker") == 3
    assert "player_merge_stale_preview" in sql
    assert "player_merge_match_collision" in sql
    assert "v_source.inactive_at is not null" in sql
    assert "v_target.inactive_at is not null" in sql
    assert "player_merge_compensation_stale" in sql
    assert sql.count("- 'updated_at'") >= 2
    assert "freeze every surviving row" in sql
    assert sql.count("lock table public.replay_jobs in share mode") == 2
    assert "player_merge_replay_in_progress" in sql
    assert sql.count("in ('pending', 'running')") == 2
    assert sql.count("for update;") >= 9
    assert "all (full system reset)" in sql
    assert "revoke all on table public.admin_player_merge_operations from public, anon, authenticated" in sql
    assert sql.count("revoke execute on function") == 3
    assert sql.count("grant execute on function") == 3
    assert "to service_role" in sql


def test_merge_operation_audit_is_in_the_same_postgres_functions() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert sql.count("insert into public.admin_activity_log") == 3
    assert "merge_player_editor_players_admin" in sql
    assert "compensate_player_editor_merge_admin" in sql
    assert "verify_player_editor_merge_replay_admin" in sql
