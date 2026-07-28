from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/20260727210000_match_log_club_recovery_mutex.sql"
)


def _source() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_match_log_recovery_mutex_serializes_every_authoritative_claim() -> None:
    source = _source()

    assert "create or replace function public.assert_match_log_recovery_guard_atomic(" in source
    assert "pg_advisory_xact_lock" in source
    assert "jupr:replay-club:" in source
    assert "jupr_match_log_recovery_lock_ambiguous" in source
    assert "jupr_match_log_recovery_locked" in source
    assert source.count("perform public.assert_match_log_recovery_guard_atomic(") >= 5


def test_match_log_mutation_rpcs_are_wrapped_without_expanding_permissions() -> None:
    source = _source()

    for rpc in (
        "apply_match_log_patches_atomic",
        "apply_match_exclusions_atomic",
    ):
        assert f"alter function public.{rpc}(" in source
        assert f"rename to {rpc}_unguarded" in source
        assert f"create or replace function public.{rpc}(" in source
        assert f"revoke all on function public.{rpc}_unguarded(" in source
        assert f"grant execute on function public.{rpc}(" in source

    assert "to service_role;" in source
    assert "to anon;" not in source
    assert "to authenticated;" not in source


def test_recovery_ledgers_and_replay_claim_fail_closed_on_bypass() -> None:
    source = _source()

    for trigger in (
        "trg_match_edit_operation_recovery_guard",
        "trg_match_exclusion_operation_recovery_guard",
        "trg_match_log_duplicate_resolution_recovery_guard",
        "trg_match_log_replay_claim_recovery_guard",
    ):
        assert f"create trigger {trigger}" in source

    assert "before insert on public.match_edit_operations" in source
    assert "before insert on public.match_exclusion_operations" in source
    assert "before insert or update on public.admin_match_log_duplicate_resolutions" in source
    assert "before update of status on public.replay_jobs" in source
    assert "new.status = 'running'" in source
    assert "replay_job.status = 'running'" in source
    assert "replay_job.status = 'pending'" not in source


def test_existing_ambiguous_recovery_state_blocks_migration() -> None:
    source = _source()

    assert "having pg_catalog.count(*) > 1" in source
    assert "jupr_match_log_recovery_migration_conflict" in source
    assert "notify pgrst, 'reload schema'" in source
