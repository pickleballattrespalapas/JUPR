from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / "supabase/migrations/20261108003000_tournament_terminal_completion_and_schedule_recovery.sql"


def _source() -> str:
    return MIGRATION.read_text(encoding="utf-8")


def test_terminal_receipt_and_recovery_rpcs_are_service_role_only() -> None:
    source = _source()

    assert "create table if not exists public.tournament_lifecycle_receipts" in source
    assert "enable row level security" in source
    assert "force row level security" in source
    assert "security definer" not in source.lower()
    assert source.lower().count("security invoker") >= 8
    assert source.count("set search_path = ''") >= 8
    for function_name in (
        "admin_tournament_completion_snapshot",
        "admin_transition_tournament_terminal_status_cas",
        "admin_rebuild_tournament_round_robin_games_cas",
        "admin_reconcile_tournament_round_robin_games_cas",
        "admin_cancel_empty_tournament_draw_cas",
        "admin_cancel_empty_tournament_event_cas",
    ):
        assert f"revoke all on function public.{function_name}" in source
        assert f"grant execute on function public.{function_name}" in source


def test_completion_is_public_terminal_state_and_archive_is_later_visibility_state() -> None:
    source = _source()

    assert "(action = 'complete' and from_status = 'ACTIVE' and to_status = 'COMPLETED')" in source
    assert "action = 'archive' and from_status = 'COMPLETED' and to_status = 'ARCHIVED'" in source
    assert "action = 'unarchive' and from_status = 'ARCHIVED' and to_status = 'COMPLETED'" in source
    assert "insert into public.tournament_lifecycle_receipts" in source
    assert "set status = v_to_status" in source
    assert "JUPR_TOURNAMENT_CLOSEOUT_SNAPSHOT_STALE" in source
    assert "tournament_lifecycle_receipt_operation_unique unique (operation_key)" in source
    assert "tournament_lifecycle_receipt_evidence_unique" not in source
    assert "operation.excluded_match_ids" in source
    assert "operation.targets_json" in source
    assert "tournament_match.tournament_id::text = p_tournament_id" in source
    assert "operation.replay_job_id = job.id" in source
    assert "idx_tournament_lifecycle_receipts_tournament_fk" in source
    assert (
        "operation.status in ('pending_replay', 'pending_badge_reconcile', 'recovery_required')"
        in source
    )


def test_completion_whitelists_active_and_rechecks_receipt_after_serialization() -> None:
    source = _source()

    assert "if v_from_status <> 'ACTIVE'" in source
    operation_lock = source.index("for update;", source.index("from public.tournament_admin_operations"))
    receipt_reads = [
        index
        for index in range(len(source))
        if source.startswith("from public.tournament_lifecycle_receipts as receipt", index)
    ]
    assert any(index > operation_lock for index in receipt_reads)


def test_fixture_repairs_are_generic_and_never_discard_entrants() -> None:
    source = _source()

    assert "women(''s|’s)?" in source
    assert "set gender_restriction = 'WOMEN'" in source
    assert "set capacity_teams = 4" in source
    assert "set capacity_teams = 16" in source
    assert "tournament_registration_selections" in source
    assert "tournament_teams" in source
    assert "sijpxjxvdtrehmqvirfi" not in source
    assert "dnoockbwfenunhcibwfn" not in source
    assert "delete from public.tournament_teams" not in source.lower()


def test_reconcile_preserves_existing_games_and_cancel_retains_configuration() -> None:
    source = _source().lower()
    reconcile = source.split(
        "create or replace function public.admin_reconcile_tournament_round_robin_games_cas",
        1,
    )[1].split(
        "create or replace function public.admin_transition_tournament_terminal_status_cas",
        1,
    )[0]

    assert "delete from public.tournament_games" not in reconcile
    assert "insert into public.tournament_games" in reconcile
    assert "score_a is not null and game.score_b is not null" in reconcile
    assert "set status = 'cancelled'" in source
    assert "set enabled = false, status = 'cancelled'" in source
    assert "delete from public.tournament_event_options" not in source
    assert "delete from public.tournament_event_draws" not in source


def test_recovery_serializes_writers_and_uses_draw_owned_scope() -> None:
    source = _source().lower()

    assert source.count(
        "lock table public.tournament_games in share row exclusive mode"
    ) == 2
    assert source.count(
        "v_draw.registration_day_id, v_draw.event_option_id"
    ) == 2
    assert "coalesce(pg_catalog.nullif(payload.registration_day_id" not in source
    assert "coalesce(pg_catalog.nullif(payload.event_option_id" not in source


def test_recovery_and_cancellation_cannot_mutate_terminal_tournaments() -> None:
    source = _source().lower()

    assert source.count(
        "not in ('completed', 'archived')"
    ) >= 4
