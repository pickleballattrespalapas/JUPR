from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261026000000_tournament_registration_check_ins.sql"
)
PRIVILEGE_MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261027000000_harden_tournament_registration_check_in_privileges.sql"
)
DAY_ATTENDANCE_MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261028000000_tournament_check_in_day_attendance.sql"
)
DAY_FK_INDEX_MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261029000000_tournament_check_in_day_fk_index.sql"
)


def test_check_in_migration_is_private_service_role_only_and_indexed() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "create table if not exists public.tournament_registration_check_ins" in sql
    assert "alter table public.tournament_registration_check_ins enable row level security" in sql
    assert "alter table public.tournament_registration_check_ins force row level security" in sql
    assert "revoke all on table public.tournament_registration_check_ins from public, anon, authenticated, service_role" in sql
    assert "grant select, insert, update on table public.tournament_registration_check_ins to service_role" in sql
    assert "registration_id text not null references public.tournament_registrations(id)" in sql
    assert "attendee_identity_key text not null" in sql
    assert "approved_substitute_player_id integer null references public.players(id)" in sql
    assert "tournament_registration_check_ins_substitute_atomicity_guard" in sql
    assert "approved_substitute_player_id is null\n        and approved_substitute_name is null" in sql
    assert "approved_substitute_player_id is not null\n        and nullif(pg_catalog.btrim(approved_substitute_name), '') is not null" in sql
    assert "on public.tournament_registration_check_ins (tournament_id)" in sql
    assert "on public.tournament_registration_check_ins (registration_id)" in sql
    assert "on public.tournament_registration_check_ins (approved_substitute_player_id)" in sql


def test_check_in_rpc_is_compare_and_swap_and_resets_changed_attendee() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "create or replace function public.admin_upsert_tournament_registration_check_in" in sql
    assert "p_club_id text" in sql
    assert "jupr_check_in_substitute_atomicity" in sql
    assert "for share" in sql
    assert "for update" in sql
    assert "pg_catalog.upper(coalesce(v_registration.status, '')) not in" in sql
    assert "('active', 'approved', 'confirmed', 'registered')" in sql
    assert "v_existing.updated_at is distinct from p_expected_updated_at" in sql
    assert "exception when unique_violation" in sql
    assert "jupr_check_in_stale" in sql
    assert "v_attendee_identity_changed" in sql
    assert "v_existing.attendee_identity_key is distinct from v_attendee_identity_key" in sql
    assert "p_approved_substitute_player_id::text" in sql
    assert "'player', v_registration.player_id::text" in sql
    assert "v_registration.display_name" in sql
    assert "v_registration.email" in sql
    assert "when v_attendee_identity_changed then false" in sql
    assert "registration.tournament_id::text = p_tournament_id" in sql
    assert "check_in.tournament_id = v_tournament.id" in sql
    assert "revoke all on function public.admin_upsert_tournament_registration_check_in" in sql
    assert "grant execute on function public.admin_upsert_tournament_registration_check_in" in sql


def test_check_in_privilege_rebind_removes_platform_default_delete_access() -> None:
    sql = PRIVILEGE_MIGRATION.read_text(encoding="utf-8").lower()

    assert "revoke all on table public.tournament_registration_check_ins" in sql
    assert "from public, anon, authenticated, service_role" in sql
    assert "grant select, insert, update on table public.tournament_registration_check_ins" in sql
    assert "revoke all on function public.admin_upsert_tournament_registration_check_in" in sql
    assert "grant execute on function public.admin_upsert_tournament_registration_check_in" in sql


def test_day_attendance_migration_is_fail_closed_private_and_day_scoped() -> None:
    sql = DAY_ATTENDANCE_MIGRATION.read_text(encoding="utf-8").lower()

    assert "jupr_check_in_legacy_day_ambiguous" in sql
    assert "count(distinct scheduled.day_id)" in sql
    assert "scheduled_day_ids" in sql
    assert "registration_day.enabled is true" in sql
    assert "registration_day_id text null" in sql
    assert "attendance_status text null" in sql
    assert "last_operation_key uuid null" in sql
    assert "('expected', 'checked_in', 'absent')" in sql
    assert "unique (tournament_id, registration_day_id, registration_id)" in sql
    assert "create unique index if not exists idx_tournament_registration_check_ins_operation_key" in sql
    assert "checked_in = (attendance_status = 'checked_in')" in sql
    assert "force row level security" in sql
    assert "from public, anon, authenticated, service_role" in sql
    assert "grant select, insert, update" in sql
    assert "grant delete" not in sql


def test_day_attendance_rpc_is_cas_idempotent_and_retires_dayless_execute() -> None:
    sql = DAY_ATTENDANCE_MIGRATION.read_text(encoding="utf-8").lower()

    assert "p_registration_day_id text" in sql
    assert "p_attendance_status text" in sql
    assert "p_operation_key uuid" in sql
    assert "event.scheduled_day_ids ? p_registration_day_id" in sql
    assert "event.registration_day_id = p_registration_day_id" in sql
    assert "v_existing.last_operation_key = p_operation_key" in sql
    assert "v_operation_existing.registration_id is distinct from p_registration_id" in sql
    assert "v_existing.last_request_fingerprint is distinct from v_request_fingerprint" in sql
    assert "jupr_check_in_idempotency_conflict" in sql
    assert sql.index("v_existing.last_operation_key = p_operation_key") < sql.index(
        "v_existing.updated_at is distinct from p_expected_updated_at"
    )
    assert sql.count("when v_attendee_identity_changed then 'expected'") == 1
    assert "and check_in.registration_day_id = p_registration_day_id" in sql
    assert "security invoker" in sql
    assert "set search_path = ''" in sql
    assert "legacy dayless signature" in sql
    assert "text, text, text, timestamptz, boolean, boolean" in sql
    insert_race_handler = sql.rsplit("exception when unique_violation then", 1)[1]
    assert "into v_operation_existing" in insert_race_handler
    assert "v_operation_existing.registration_day_id = p_registration_day_id" in insert_race_handler
    assert "v_operation_existing.registration_id = p_registration_id" in insert_race_handler
    assert "v_operation_existing.last_request_fingerprint = v_request_fingerprint" in insert_race_handler
    assert "'idempotent_replay', true" in insert_race_handler
    assert sql.count("'idempotent_replay', true") == 2


def test_day_fk_follow_up_adds_the_advisor_required_leading_index() -> None:
    sql = DAY_FK_INDEX_MIGRATION.read_text(encoding="utf-8").lower()

    assert "create index if not exists idx_tournament_registration_check_ins_registration_day" in sql
    assert "on public.tournament_registration_check_ins (registration_day_id)" in sql
    assert "drop" not in sql
    assert "grant" not in sql
