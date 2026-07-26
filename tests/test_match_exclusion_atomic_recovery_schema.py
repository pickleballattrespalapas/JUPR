from pathlib import Path
import re


MIGRATION = Path(
    "supabase/migrations/20260726143742_match_exclusion_atomic_recovery.sql"
)
NONRETRYABLE_CONFLICT_MIGRATION = Path(
    "supabase/migrations/"
    "20260726204954_match_exclusion_nonretryable_conflicts.sql"
)
INVENTORY = Path("docs/migrations.md")


def _source() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def _nonretryable_conflict_source() -> str:
    return NONRETRYABLE_CONFLICT_MIGRATION.read_text(
        encoding="utf-8"
    ).lower()


def _function_block(source: str, function_name: str) -> str:
    start = source.index(
        f"create or replace function public.{function_name}("
    )
    end = source.index("$function$;", start) + len("$function$;")
    return source[start:end]


def test_match_exclusion_migration_is_in_canonical_inventory() -> None:
    assert MIGRATION.is_file()
    assert NONRETRYABLE_CONFLICT_MIGRATION.is_file()
    inventory = INVENTORY.read_text(encoding="utf-8")
    assert MIGRATION.name in inventory
    assert NONRETRYABLE_CONFLICT_MIGRATION.name in inventory
    assert "43 SQL files: 42 deployable migrations" in inventory


def test_expected_match_conflicts_use_nonretryable_sqlstate() -> None:
    original = _source()
    correction = _nonretryable_conflict_source()

    original_bump = _function_block(original, "bump_match_row_version")
    corrected_bump = _function_block(correction, "bump_match_row_version")
    assert original_bump.count("errcode = '40001'") == 1
    assert corrected_bump == original_bump.replace(
        "errcode = '40001'",
        "errcode = 'p0001'",
    )

    original_apply = _function_block(
        original,
        "apply_match_exclusions_atomic",
    )
    corrected_apply = _function_block(
        correction,
        "apply_match_exclusions_atomic",
    )
    assert original_apply.count("errcode = '40001'") == 3
    assert corrected_apply == original_apply.replace(
        "errcode = '40001'",
        "errcode = 'p0001'",
    )

    assert correction.count("errcode = 'p0001'") == 4
    assert "errcode = '40001'" not in correction
    for marker in (
        "jupr_match_row_version_immutable",
        "jupr_match_exclusion_stale",
        "jupr_match_exclusion_duplicate_keeper_stale",
    ):
        assert marker in correction

    assert (
        "revoke all on function public.bump_match_row_version()"
        in correction
    )
    assert (
        "grant execute on function public.bump_match_row_version()"
        in correction
    )
    assert (
        "revoke all on function public.apply_match_exclusions_atomic("
        in correction
    )
    assert (
        "grant execute on function public.apply_match_exclusions_atomic("
        in correction
    )
    assert "from public, anon, authenticated" in correction
    assert "to service_role" in correction
    assert "notify pgrst, 'reload schema'" in correction


def test_matches_have_immutable_row_version_and_exact_cas_targets() -> None:
    source = _source()

    assert "add column if not exists row_version bigint not null default 1" in source
    assert "create or replace function public.bump_match_row_version()" in source
    assert "new.row_version := old.row_version + 1" in source
    assert "trg_matches_bump_row_version" in source
    assert "new.row_version is distinct from old.row_version" in source

    assert "create or replace function public.apply_match_exclusions_atomic(" in source
    assert "p_mode text" in source
    assert "p_targets jsonb" in source
    assert "p_badge_ids text[]" in source
    assert "p_badge_contract_version text" in source
    assert "'match_id'" in source
    assert "'expected_row_version'" in source
    assert "match_row.row_version = v_target.expected_row_version" in source
    assert "for update" in source
    assert "order by (target.item ->> 'match_id')::bigint" in source


def test_duplicate_cleanup_locks_and_records_a_surviving_canonical_keeper() -> None:
    source = _source()

    assert (
        "create or replace function "
        "public.match_exclusion_canonical_duplicate_key(" in source
    )
    for canonical_field in (
        "p_league text",
        "p_week_tag text",
        "p_match_type text",
        "p_t1_p1 bigint",
        "p_t1_p2 bigint",
        "p_t2_p1 bigint",
        "p_t2_p2 bigint",
        "p_score_t1 integer",
        "p_score_t2 integer",
    ):
        assert canonical_field in source

    assert "duplicate_keeper_json jsonb not null" in source
    assert "keeper_row.id < v_target.match_id" in source
    assert "not (keeper_row.id = any(v_target_ids))" in source
    assert "keeper_row.deleted_at is null" in source
    assert "order by keeper_row.id" in source
    assert "limit 1\n      for update" in source
    assert "jupr_match_exclusion_duplicate_keeper_stale" in source
    assert "'keeper_match_id', v_keeper.id" in source
    assert "'keeper_row_version', v_keeper.row_version" in source


def test_atomic_exclusion_freezes_full_recovery_evidence_before_replay() -> None:
    source = _source()

    assert "create table if not exists public.match_exclusion_operations" in source
    for field in (
        "targets_json jsonb not null",
        "duplicate_keeper_json jsonb not null",
        "before_json jsonb not null",
        "after_json jsonb not null",
        "excluded_match_ids bigint[] not null",
        "affected_player_ids bigint[] not null",
        "badge_ids text[] not null",
        "badge_contract_version text not null",
        "replay_job_id uuid not null",
    ):
        assert field in source

    assert "mode in ('exclude', 'duplicate_cleanup')" in source
    assert "'pending_replay'" in source
    assert "'pending_badge_reconcile'" in source
    assert "'recovery_required'" in source
    assert "'succeeded'" in source
    assert "v_replay_key := 'match-exclusion:' || p_operation_id::text" in source
    assert "v_replay_target <> 'all (full system reset)'" in source
    assert "insert into public.replay_jobs" in source
    assert "insert into public.match_exclusion_operations" in source
    assert "insert into public.admin_activity_log" in source

    assert "deleted_at = v_now" in source
    assert "deleted_by = v_actor_email" in source
    assert "deleted_source = v_source" in source
    assert "delete_note = v_delete_note" in source
    assert "delete from public.matches" not in source


def test_apply_idempotency_reuses_exact_body_and_rejects_changed_body() -> None:
    source = _source()

    assert "match_exclusion_operations_club_key_unique" in source
    assert "where operation.club_id = v_club_id" in source
    assert "and operation.idempotency_key = v_idempotency_key" in source
    assert "v_existing.targets_json is distinct from v_targets" in source
    assert "v_existing.badge_ids is distinct from v_badge_ids" in source
    assert "v_existing.badge_contract_version <> v_badge_contract_version" in source
    assert "different request body" in source
    assert "'operation_id', v_existing.id" in source
    assert "'idempotent', true" in source


def test_replay_jobs_use_one_club_lease_and_exact_token_cas() -> None:
    source = _source()

    for field in (
        "lease_token uuid",
        "leased_by text",
        "lease_expires_at timestamptz",
        "heartbeat_at timestamptz",
        "last_claimed_at timestamptz",
    ):
        assert field in source

    assert "replay_jobs_lease_shape_check" in source
    assert "replay_jobs_one_running_per_club_uidx" in source
    assert "where status = 'running'" in source
    assert "jupr:replay-club:" in source

    assert "create or replace function public.claim_replay_job_atomic(" in source
    assert "create or replace function public.heartbeat_replay_job_atomic(" in source
    assert "create or replace function public.finish_replay_job_atomic(" in source
    assert "p_retry_failed boolean default false" in source
    assert "v_job.lease_token is distinct from p_lease_token" in source
    assert "v_job.leased_by is distinct from v_worker_id" in source
    assert "v_job.lease_expires_at <= v_now" in source
    assert "full replay must attest singles_replay_supported=true" in source


def test_replay_projection_writes_are_transactionally_fenced_and_idempotent() -> None:
    source = _source()

    assert (
        "create or replace function "
        "public.assert_replay_write_fence_atomic(" in source
    )
    assert (
        "create or replace function "
        "public.apply_replay_write_batch_atomic(" in source
    )
    fence_start = source.index(
        "create or replace function public.assert_replay_write_fence_atomic("
    )
    fence_end = source.index("$function$;", fence_start)
    fence = source[fence_start:fence_end]
    assert "jupr:replay-club:" in fence
    assert "pg_advisory_xact_lock" in fence
    assert "for update" in fence
    assert "v_now := pg_catalog.clock_timestamp()" in fence
    assert fence.index("for update") < fence.index(
        "v_now := pg_catalog.clock_timestamp()"
    )
    for predicate in (
        "v_job.status <> 'running'",
        "v_job.target_reset is distinct from v_target_reset",
        "v_job.lease_token is distinct from p_lease_token",
        "v_job.leased_by is distinct from v_worker_id",
        "v_job.lease_expires_at <= v_now",
    ):
        assert predicate in fence

    batch_start = source.index(
        "create or replace function public.apply_replay_write_batch_atomic("
    )
    batch_end = source.index("$function$;", batch_start)
    batch = source[batch_start:batch_end]
    assert "perform public.assert_replay_write_fence_atomic(" in batch
    for write_kind in (
        "'players_stats'",
        "'player_singles_stats'",
        "'delete_league_ratings'",
        "'insert_league_ratings'",
        "'match_snapshots'",
    ):
        assert write_kind in batch
    assert (
        "on conflict (club_id, player_id, league_name) do update" in batch
    )
    assert "where not exists (" in batch
    assert "is distinct from row(" in batch
    assert "is not distinct from row(" in batch
    assert "last_game_at = data.last_game_at" in batch
    assert "player.last_game_at" in batch
    assert "data.last_game_at" in batch
    assert "where not (row_item.item ? 'last_game_at')" in batch
    assert "jupr_replay_write_batch_incomplete" in batch


def test_replay_lease_time_is_sampled_only_after_waiting_for_locks() -> None:
    source = _source()

    for function_name in (
        "claim_replay_job_atomic",
        "heartbeat_replay_job_atomic",
        "finish_replay_job_atomic",
    ):
        start = source.index(
            f"create or replace function public.{function_name}("
        )
        end = source.index("$function$;", start)
        block = source[start:end]
        assert "pg_advisory_xact_lock" in block
        assert "for update" in block
        assert "v_now timestamptz;" in block
        assert "v_now := pg_catalog.clock_timestamp()" in block
        assert block.index("for update") < block.index(
            "v_now := pg_catalog.clock_timestamp()"
        )


def test_only_one_open_exclusion_and_only_its_replay_can_be_claimed() -> None:
    source = _source()

    assert "match_exclusion_operations_one_open_per_club_uidx" in source
    assert (
        "on public.match_exclusion_operations (club_id)\n"
        "  where status <> 'succeeded'" in source
    )
    assert "'match_exclusion_operation_in_progress'" in source
    assert "and operation.status <> 'succeeded'" in source
    assert "v_open_operation.replay_job_id <> p_job_id" in source
    assert (
        "'another replay is blocked while match exclusion recovery is open.'"
        in source
    )


def test_full_reset_freezes_all_club_players_for_badge_reconciliation() -> None:
    source = _source()

    assert (
        "pg_catalog.cardinality(affected_player_ids) between 1 and 10000"
        in source
    )
    assert (
        "select pg_catalog.array_agg(player_row.id order by player_row.id)"
        in source
    )
    assert "from public.players as player_row" in source
    assert "where player_row.club_id::text = v_club_id" in source
    assert "jupr_match_exclusion_badge_scope_invalid" in source


def test_recovery_transition_accepts_exact_replay_and_stage_retry() -> None:
    source = _source()

    assert (
        "create or replace function public.transition_match_exclusion_after_replay("
        in source
    )
    assert (
        "create or replace function public.mark_match_exclusion_recovery_required("
        in source
    )
    assert "v_operation.recovery_stage <> 'replay'" in source
    assert "where replay_job.id = p_replay_job_id" in source
    assert "v_job.status <> 'succeeded'" in source
    assert "v_job.target_reset <> 'all (full system reset)'" in source
    assert "insert into public.match_exclusion_badge_progress" in source
    assert "status = 'pending_badge_reconcile'" in source
    assert "recovery_stage in ('replay', 'badge_reconcile')" in source


def test_badge_reconciliation_is_per_player_leased_and_narrow() -> None:
    source = _source()

    assert (
        "create table if not exists public.match_exclusion_badge_progress"
        in source
    )
    assert "unique (operation_id, player_id)" in source
    assert (
        "create or replace function public.claim_match_exclusion_badge_progress("
        in source
    )
    assert (
        "create or replace function public.apply_match_exclusion_badge_reconciliation("
        in source
    )
    assert (
        "create or replace function public.fail_match_exclusion_badge_progress("
        in source
    )
    assert (
        "create or replace function public.finalize_match_exclusion_operation("
        in source
    )

    assert "player_badge.player_id = v_progress.player_id" in source
    assert "player_badge.badge_id = any(v_operation.badge_ids)" in source
    assert (
        "player_badge.context_type = desired.item ->> 'context_type'"
        in source
    )
    assert (
        "player_badge.context_type =\n"
        "        v_desired_badge ->> 'context_type'" in source
    )
    assert "player_badge.awarded_by = 'engine'" in source
    assert "player_badge.awarded_by is distinct from 'engine'" in source
    assert (
        "is distinct from v_operation.badge_contract_version" in source
    )
    assert "jupR_badge_reconcile_protected_key".lower() in source
    assert "revoked_at = v_now" in source
    assert "revoked_at = null" in source
    assert "'inserted_count', v_inserted_count" in source
    assert "'updated_count', v_updated_count" in source
    assert "'revoked_count', v_revoked_count" in source
    assert "delete from public.player_badges" not in source


def test_badge_identity_and_uuid_safe_revocation_match_live_schema() -> None:
    source = _source()

    assert "v_revoked_by_type" in source
    assert "v_revoked_by_type not in ('uuid', 'text')" in source
    assert (
        "unique (club_id, player_id, badge_id, context_type, context_id)"
        in source
    )
    assert (
        "group by\n"
        "      desired_key.badge_id,\n"
        "      desired_key.context_type,\n"
        "      desired_key.context_id" in source
    )
    assert (
        "badge_id/context_type/context_id keys must be unique" in source
    )
    assert "revoked_by = v_actor_email" not in source
    assert "revoked_by = null" in source
    assert "|| ' by '\n      || v_actor_email" in source


def test_all_new_rpc_surfaces_are_invoker_safe_and_service_only() -> None:
    source = _source()
    rpc_names = (
        "apply_match_exclusions_atomic",
        "claim_replay_job_atomic",
        "heartbeat_replay_job_atomic",
        "finish_replay_job_atomic",
        "assert_replay_write_fence_atomic",
        "apply_replay_write_batch_atomic",
        "transition_match_exclusion_after_replay",
        "mark_match_exclusion_recovery_required",
        "claim_match_exclusion_badge_progress",
        "apply_match_exclusion_badge_reconciliation",
        "fail_match_exclusion_badge_progress",
        "finalize_match_exclusion_operation",
    )

    assert "security definer" not in source
    function_blocks = re.findall(
        r"create or replace function public\.(\w+)\(.*?\n\$function\$;",
        source,
        flags=re.DOTALL,
    )
    assert set(rpc_names).issubset(set(function_blocks))

    for name in rpc_names:
        declaration = source.index(f"create or replace function public.{name}(")
        body = source.index("$function$;", declaration)
        block = source[declaration:body]
        assert "security invoker" in block
        assert "set search_path = ''" in block
        assert f"revoke all on function public.{name}(" in source
        assert f"grant execute on function public.{name}(" in source

    assert "alter table public.match_exclusion_operations enable row level security" in source
    assert "alter table public.match_exclusion_operations force row level security" in source
    assert "alter table public.match_exclusion_badge_progress enable row level security" in source
    assert "alter table public.match_exclusion_badge_progress force row level security" in source
    assert "from public, anon, authenticated" in source
    assert "to service_role" in source
    assert "notify pgrst, 'reload schema'" in source


def test_sql_does_not_schema_qualify_special_coalesce_expression() -> None:
    source = _source()
    assert "pg_catalog.coalesce" not in source
    assert "pg_catalog.greatest" not in source
    assert "pg_catalog.least" not in source
