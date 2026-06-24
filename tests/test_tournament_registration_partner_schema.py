from pathlib import Path

from jupr_app.domain import tournament_registration_repo as repo


MIGRATION = Path("migrations/20261019_tournament_registration_partner_links.sql")


def _sql() -> str:
    return MIGRATION.read_text(encoding="utf-8").lower()


def test_partner_link_migration_preserves_existing_selection_entry_model():
    sql = _sql()
    assert "create table if not exists public.tournament_registration_partner_requests" in sql
    assert "create table if not exists public.tournament_registration_team_links" in sql
    assert "create table if not exists public.tournament_registration_team_members" in sql
    assert "event_entries" not in sql
    assert sql.count("id text primary key") == 3
    assert "requester_selection_id text not null references public.tournament_registration_selections(id)" in sql
    assert "target_selection_id text null references public.tournament_registration_selections(id)" in sql
    assert "selection1_id text not null references public.tournament_registration_selections(id)" in sql
    assert "selection2_id text not null references public.tournament_registration_selections(id)" in sql
    assert "accepted_request_id text null references public.tournament_registration_partner_requests(id)" in sql
    assert "team_link_id text not null references public.tournament_registration_team_links(id)" in sql
    assert "created_by_user_id text null" in sql


def test_partner_link_migration_adds_profile_link_and_active_membership_uniqueness():
    sql = _sql()
    assert "add column if not exists player_id integer null references public.players(id)" in sql
    assert "idx_tournament_registrations_player_id" in sql
    assert "uq_tournament_registrations_tournament_player" in sql
    assert "where player_id is not null" in sql
    assert "uq_tournament_team_members_active_event_selection" in sql
    assert "where status = 'active'" in sql


def test_registration_schema_contract_includes_partner_link_foundation():
    assert "migrations/20261019_tournament_registration_partner_links.sql" in repo.REGISTRATION_SCHEMA_CONTRACT_MIGRATIONS
    assert "player_id" in repo.REGISTRATION_SCHEMA_REQUIRED_COLUMNS["tournament_registrations"]
    assert "requester_selection_id" in repo.REGISTRATION_SCHEMA_REQUIRED_COLUMNS["tournament_registration_partner_requests"]
    assert "accepted_request_id" in repo.REGISTRATION_SCHEMA_REQUIRED_COLUMNS["tournament_registration_team_links"]
    assert "player_order" in repo.REGISTRATION_SCHEMA_REQUIRED_COLUMNS["tournament_registration_team_members"]
