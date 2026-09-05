from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_normalized_roster_migration_extends_fixed_pair_compatibly() -> None:
    base = _read(
        "supabase/migrations/20261030010000_team_league_composition_settings.sql"
    )
    migration = _read(
        "supabase/migrations/20261103000000_team_league_normalized_rosters_substitute_pool.sql"
    )

    assert "check (team_size = 2)" in base
    assert "drop constraint if exists team_league_settings_team_size_check" in migration
    assert "check (team_size between 2 and 4)" in migration
    assert "create table if not exists public.team_league_team_members" in migration
    assert "create table if not exists public.team_league_substitute_pool" in migration
    assert "team_league_assert_roster_policy_v1" in migration
    assert "validate_existing_normalized_team_rosters" in migration
    assert "TEAM_LEAGUE_EXISTING_ROSTER_CATEGORY_INVALID" in migration
    assert "team_league_project_legacy_pair_v1" in migration
    assert "active_primary_count = v_team_size" in migration
    assert ") = settings.team_size" in migration
    assert "p_settings ->> 'team_size'" in migration
    assert "p_settings ->> 'team_category'" in migration
    assert "p_settings ->> 'allow_substitutes'" in migration
    assert "team_league_save_settings_v2" in migration
    assert "team_league_apply_roster_action_v1" in migration
    assert "team_league_create_team_v1" in migration
    assert "alter column partner_player_id drop not null" in migration
    assert "team_league_team_members_operation_idx" in migration
    assert "team_league_substitute_pool_operation_idx" in migration
    assert "force row level security" in migration
    assert "revoke all" in migration
    assert "grant select, insert, update on table public.team_league_team_members" in migration
    assert "grant select, insert, update, delete on table public.team_league_team_members" not in migration
    assert "service_role" in migration
    assert "allow_alternates" not in migration

    # Status-only completeness reconciliation must never re-project and
    # resurrect a removed legacy partner.
    assert "if not v_identity_or_invite_changed and not v_lifecycle_sync then" in migration
    assert "Never use that status-only transition to resurrect" in migration
    assert "member.status in ('active', 'removed') then member.status" in migration
    assert "existing.player_id = new.captain_player_id" in migration
    assert "existing.player_id = new.partner_player_id" in migration
    projection = migration[
        migration.index("create or replace function public.team_league_project_legacy_pair_v1"):
        migration.index("drop trigger if exists team_league_project_legacy_pair")
    ]
    assert "if not found then" not in projection
    assert "TEAM_LEAGUE_CAPTAIN_ROLE_INVALID" in migration
    assert "old.role in ('captain', 'primary')" in migration
    assert "TEAM_LEAGUE_MEMBER_SCOPE_IMMUTABLE" in migration
    assert "TEAM_LEAGUE_ACTIVE_CAPTAIN_REQUIRED" in migration
    assert "old.team_id is distinct from new.team_id" in migration
    assert "v_normalized_completion" in migration
    assert "and not v_normalized_completion" in migration
    assert "pg_catalog.to_jsonb(old) - 'status' - 'updated_at'" in migration

    # Receipt replay precedes CAS, and settings locks teams before settings.
    roster_rpc = migration.index("create or replace function public.team_league_apply_roster_action_v1")
    roster_body = migration[roster_rpc:]
    assert roster_body.index("select operation.*") < roster_body.index(
        "v_settings.roster_version <> p_expected_roster_version"
    )
    settings_rpc = migration.index("create or replace function public.team_league_save_settings_v2")
    settings_body = migration[settings_rpc:roster_rpc]
    assert "'pcs:team-league-roster:' || v_club_id || ':' || v_league_name" in settings_body
    assert migration.count("lock table public.team_league_teams in exclusive mode") == 1
    assert settings_body.index(
        "lock table public.team_league_teams in exclusive mode"
    ) < settings_body.index("order by team.id")
    assert settings_body.index("order by team.id") < settings_body.index(
        "select settings.*"
    )
    member_trigger = migration[
        migration.index("create or replace function public.team_league_validate_member_v1"):
        migration.index("drop trigger if exists team_league_validate_member")
    ]
    assert member_trigger.index("select team.*") < member_trigger.index(
        "select settings.*"
    )


def test_admin_surfaces_advertise_normalized_rosters_but_public_form_stays_pair_only() -> None:
    admin = _read("apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx")
    setup = _read(
        "apps/web/app/admin/league-manager/settings/TeamLeagueSetupPanel.tsx"
    )
    public_list = _read("apps/web/app/clubs/[clubSlug]/team-leagues/page.tsx")
    public_detail = _read(
        "apps/web/app/clubs/[clubSlug]/team-leagues/[leagueName]/page.tsx"
    )
    registration = _read(
        "apps/web/app/clubs/[clubSlug]/team-leagues/[leagueName]/"
        "TeamLeagueRegistrationForm.tsx"
    )
    api_types = _read("apps/web/lib/teamLeagueApi.ts")
    admin_surfaces = "\n".join((admin, setup))
    surfaces = "\n".join(
        (admin, setup, public_list, public_detail, registration, api_types)
    )

    for phrase in (
        "Primary roster",
        "2 players",
        "3 players",
        "4 players",
        "Team eligibility",
        "Men&apos;s",
        "Women&apos;s",
        "Mixed",
        "Allow substitutes",
        "Shared substitute pool",
        "Assigned alternate",
        "Create a forming team",
    ):
        assert phrase in admin_surfaces
    assert "Need a substitute for one match?" in registration
    assert "team_size} players per team" in public_detail
    assert "league.team_size" in public_list
    assert "team_size: 2 | 3 | 4" in api_types
    assert "max_alternates" in surfaces
    assert "substitute_pool_enabled" in surfaces
    assert "partner_player_id" in registration
    assert "alternate_management_supported" not in registration
    assert "substitute_pool_registration_supported" not in registration
