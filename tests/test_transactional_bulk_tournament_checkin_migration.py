from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261108019000_transactional_bulk_tournament_check_in.sql"
)


def migration_sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_bulk_check_in_rpc_is_one_private_service_role_transaction() -> None:
    sql = migration_sql()
    signature = (
        "public.admin_bulk_upsert_tournament_registration_check_ins(\n"
        "  text, text, text, uuid, jsonb, text, text\n"
        ")"
    )

    assert "create or replace function public.admin_bulk_upsert_tournament_registration_check_ins" in sql
    assert "language plpgsql" in sql
    assert "security invoker" in sql
    assert "set search_path = ''" in sql
    assert f"revoke all on function {signature}" in sql
    assert ") from public, anon, authenticated, service_role;" in sql
    assert f"grant execute on function {signature}" in sql
    assert ") to service_role;" in sql
    assert "security definer" not in sql


def test_bulk_check_in_canonicalizes_one_bounded_unique_request() -> None:
    sql = migration_sql()

    assert "jsonb_array_length(p_updates) > 100" in sql
    assert "count(distinct pg_catalog.btrim" in sql
    assert "each registration may appear only once" in sql
    assert "order by pg_catalog.btrim(requested.element ->> 'registration_id') collate \"c\"" in sql
    assert "substitutions and unrecognized row fields are not supported" in sql
    assert "not (v_patch ? 'attendance_status')" in sql
    assert "and not (v_patch ? 'waiver_verified')" in sql
    assert "and not (v_patch ? 'notes')" in sql


def test_bulk_check_in_uses_one_durable_batch_idempotency_ledger() -> None:
    sql = migration_sql()

    assert "insert into public.tournament_admin_operations" in sql
    assert "'tournament_check_in_bulk_update'" in sql
    assert "'tournament_registration_day'" in sql
    assert "client_idempotency_key" in sql
    assert "extensions.digest(v_canonical_request::text, 'sha256')" in sql
    assert "pg_catalog.pg_advisory_xact_lock" in sql
    assert "'jupr:tournament-admin-operation:' || pg_catalog.btrim(p_club_id)" in sql
    assert "':tournament_live:' || p_operation_key::text" in sql
    advisory_section = sql.split("pg_catalog.pg_advisory_xact_lock", 1)[1].split(
        "select operation.*", 1
    )[0]
    assert "v_internal_operation_key" not in advisory_section
    assert "v_operation.request_json is distinct from v_canonical_request" in sql
    assert "jupr_check_in_bulk_idempotency_conflict" in sql
    assert "return v_operation.result_json || pg_catalog.jsonb_build_object" in sql
    assert "'idempotent_replay', true" in sql
    assert "last_operation_key = pg_catalog.md5(" in sql
    assert "v_internal_operation_key || ':' || v_registration.id" in sql
    assert "last_operation_key = p_operation_key" not in sql


def test_bulk_check_in_locks_and_preflights_every_row_before_any_write() -> None:
    sql = migration_sql()
    preflight = sql.index("-- preflight every selected row")
    operation_insert = sql.index("insert into public.tournament_admin_operations")
    first_check_in_write = min(
        sql.index("update public.tournament_registration_check_ins"),
        sql.index("insert into public.tournament_registration_check_ins"),
    )

    assert "order by registration.id collate \"c\"\n   for update" in sql
    assert "order by selection.id collate \"c\"\n   for share" in sql
    assert "order by check_in.registration_id collate \"c\"\n   for update" in sql
    draw_lock = sql.index("perform draw.id")
    assert "perform team.id" not in sql
    assert sql.index("order by draw.id\n   for share", draw_lock) > draw_lock
    assert "v_existing.updated_at is distinct from v_expected_updated_at" in sql
    assert "jupr_check_in_bulk_stale" in sql
    assert preflight < operation_insert < first_check_in_write


def test_bulk_check_in_preflights_the_entire_roster_in_one_statement_snapshot() -> None:
    sql = migration_sql()
    roster_preflight = sql.index(
        "-- resolve roster authority for the complete canonical registration set"
    )
    row_preflight = sql.index("-- preflight every selected row")
    roster_section = sql[roster_preflight:row_preflight]
    row_section = sql[
        row_preflight : sql.index("insert into public.tournament_admin_operations")
    ]

    assert (
        "one sql statement and therefore one read committed statement snapshot"
        in roster_section
    )
    assert "into v_rostered_registration_count" in roster_section
    assert "requested_registration.id = any(v_registration_ids)" in roster_section
    assert "selection.registration_id = requested_registration.id" in roster_section
    assert "v_rostered_registration_count <> v_requested_count" in roster_section
    assert "v_rostered_selection_count" not in sql
    assert "public.tournament_event_draws" not in row_section
    assert "public.tournament_teams" not in row_section


def test_bulk_check_in_enforces_selected_day_draw_roster_authority() -> None:
    sql = migration_sql()

    assert "registration_day.enabled is true" in sql
    assert "event.scheduled_day_ids ? pg_catalog.btrim(p_registration_day_id)" in sql
    assert "coalesce(draw.hidden_from_primary_ops, false) is false" in sql
    assert "pg_catalog.upper(coalesce(draw.draw_kind, 'standard')) = 'standard'" in sql
    assert "primary_scope.draw_count = 0" in sql
    assert "primary_scope.draw_count = 1" in sql
    assert "requested_registration.player_id in (" in sql
    assert "team.player1_id" in sql
    assert "team.player2_id" in sql
    assert (
        "exact_registration.player_id = requested_registration.player_id" in sql
    )
    assert "player.club_id::text = pg_catalog.btrim(p_club_id)" in sql
    assert "player.active is true" in sql
    assert "jupr_check_in_bulk_roster" in sql


def test_bulk_check_in_sparse_patches_preserve_untouched_fields_and_audit_actor() -> None:
    sql = migration_sql()

    assert "when v_patch ? 'attendance_status'" in sql
    assert "then v_existing.attendance_status" in sql
    assert "when v_patch ? 'waiver_verified'" in sql
    assert "then v_existing.waiver_verified" in sql
    assert "when v_patch ? 'notes'" in sql
    assert "then v_existing.notes" in sql
    assert "approved_substitute_player_id" in sql
    assert "bulk check-in cannot change or restore attendee identity" in sql
    assert "insert into public.admin_activity_log" in sql
    assert "'bulk_update_tournament_registration_check_in_admin'" in sql
    assert "v_actor_email" in sql
    assert "v_actor_role" in sql
    assert "'next_tournament_check_in'" in sql
    assert "notify pgrst, 'reload schema';" in sql


def test_bulk_check_in_uses_identical_unlinked_attendee_identity_in_both_passes() -> None:
    sql = migration_sql()
    identity_fragment = """pg_catalog.concat_ws(
          ':',
          'registration',
          v_registration.id,
          pg_catalog.lower(pg_catalog.btrim(coalesce(v_registration.display_name, ''))),
          pg_catalog.lower(pg_catalog.btrim(coalesce(v_registration.first_name, ''))),
          pg_catalog.lower(pg_catalog.btrim(coalesce(v_registration.last_name, ''))),
          pg_catalog.lower(pg_catalog.btrim(coalesce(v_registration.email, '')))
        )"""

    assert sql.count(identity_fragment) == 2
