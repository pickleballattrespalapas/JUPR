from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (
    ROOT
    / "supabase"
    / "migrations"
    / "20261026000000_tournament_registration_check_ins.sql"
)


def test_check_in_migration_is_private_service_role_only_and_indexed() -> None:
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "create table if not exists public.tournament_registration_check_ins" in sql
    assert "alter table public.tournament_registration_check_ins enable row level security" in sql
    assert "alter table public.tournament_registration_check_ins force row level security" in sql
    assert "revoke all on table public.tournament_registration_check_ins from public, anon, authenticated" in sql
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
