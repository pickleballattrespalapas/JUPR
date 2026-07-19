from pathlib import Path


MIGRATION = Path(
    "supabase/migrations/20260719160821_public_registration_edit_transaction.sql"
)


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_public_registration_edit_rpc_is_atomic_and_versioned() -> None:
    sql = _sql()

    assert "create or replace function public.server_update_public_tournament_registration_edit" in sql
    assert "security invoker" in sql
    assert "pg_advisory_xact_lock" in sql
    assert "private.lock_tournament_registration_selection_scope(v_existing_ids)" in sql
    assert "v_registration.updated_at is distinct from p_expected_updated_at" in sql
    assert "stale_selection_version" in sql
    assert "registration.updated_at = p_expected_updated_at" in sql
    assert "set_config('jupr.selection_edit_rpc', 'on', true)" in sql


def test_public_registration_edit_rpc_blocks_imported_draws_and_relationship_loss() -> None:
    sql = _sql()

    assert "join public.tournament_teams as team" in sql
    assert "registration_imported_to_draw" in sql
    assert "public.tournament_registration_partner_requests" in sql
    assert "public.tournament_registration_team_links" in sql
    assert "public.tournament_registration_team_members" in sql
    assert "registration_relationship_locked" in sql


def test_public_registration_edit_rpc_is_service_role_only() -> None:
    sql = _sql()

    signature = "public.server_update_public_tournament_registration_edit(\n  text,\n  text,\n  timestamptz,\n  jsonb,\n  jsonb,\n  jsonb\n)"
    assert f"revoke all on function {signature} from public, anon, authenticated" in sql
    assert f"grant execute on function {signature} to service_role" in sql
    assert "grant execute" not in sql.replace(
        f"grant execute on function {signature} to service_role", ""
    )


def test_public_registration_edit_migration_has_dependency_preflight() -> None:
    sql = _sql()

    assert "do $migration_preflight$" in sql
    assert "to_regclass('public.tournament_registrations') is null" in sql
    assert "to_regclass('public.tournament_teams') is null" in sql
    assert "to_regprocedure('private.lock_tournament_registration_selection_scope(text[])') is null" in sql
    assert "column_name = 'updated_at'" in sql
