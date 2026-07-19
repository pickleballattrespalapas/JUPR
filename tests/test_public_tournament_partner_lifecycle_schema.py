from pathlib import Path


MIGRATION = Path("supabase/migrations/20260719194500_public_partner_pairing_lifecycle.sql")


def test_partner_lifecycle_migration_is_transactional_and_server_only():
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    assert "create or replace function public.create_tournament_partner_request" in sql
    assert "create or replace function public.transition_tournament_partner_request" in sql
    assert sql.count("security invoker") == 2
    assert "security definer" not in sql
    assert "for update" in sql
    assert "pg_advisory_xact_lock" in sql
    assert sql.count("private.lock_tournament_registration_selection_scope") >= 2
    assert "uq_tournament_partner_pending_pair" in sql
    assert "jupr_partner_duplicate_pending" in sql
    assert "where status = 'pending'" in sql
    assert "grant execute on function public.create_tournament_partner_request" in sql
    assert "grant execute on function public.transition_tournament_partner_request" in sql
    assert sql.count("to service_role") == 2
    assert sql.count("from public, anon, authenticated") == 2


def test_accept_transaction_pairs_and_cancels_competing_requests_atomically():
    sql = MIGRATION.read_text(encoding="utf-8").lower()

    accept_section = sql.split("create or replace function public.transition_tournament_partner_request", 1)[1]
    assert "insert into public.tournament_registration_team_links" in accept_section
    assert "insert into public.tournament_registration_team_members" in accept_section
    assert "set partner_mode = 'has_partner', show_on_partner_board = false" in accept_section
    assert "with cancelled as" in accept_section
    assert "competing.status = 'pending'" in accept_section
    assert "wants_partner_board_contact" in accept_section
    assert "v_lock_selection_ids" in accept_section
    assert "'outcome', 'idempotent'" in accept_section
    assert "'outcome', 'stale'" in accept_section
