from pathlib import Path


MIGRATION = Path("supabase/migrations/20260719172000_replay_job_idempotency.sql")


def test_atomic_match_edit_rpc_is_service_role_only_and_transactional() -> None:
    source = MIGRATION.read_text(encoding="utf-8").lower()

    assert "create table if not exists public.match_edit_operations" in source
    assert "create or replace function public.apply_match_log_patches_atomic" in source
    assert "pg_advisory_xact_lock" in source
    assert "for update" in source
    assert "security definer" in source
    assert "revoke all on function public.apply_match_log_patches_atomic" in source
    assert "grant execute on function public.apply_match_log_patches_atomic" in source
    assert "to service_role" in source


def test_atomic_rpc_couples_edit_audit_and_pending_replay_job() -> None:
    source = MIGRATION.read_text(encoding="utf-8").lower()

    assert "insert into public.replay_jobs" in source
    assert "insert into public.match_edit_operations" in source
    assert "insert into public.admin_activity_log" in source
    assert "pending_replay" in source
    assert "recovery_required" in source
    assert "idempotency_key" in source
    assert "notes = case when v_patch ? 'notes'" in source
