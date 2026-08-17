from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_composition_migration_is_fixed_pair_and_rpc_compatible() -> None:
    migration = _read(
        "supabase/migrations/20261030010000_team_league_composition_settings.sql"
    )

    assert "add column if not exists team_size" in migration
    assert "add column if not exists team_category" in migration
    assert "check (team_size = 2)" in migration
    assert "team_category in ('open', 'mens', 'womens', 'mixed')" in migration
    assert "p_settings ->> 'team_size'" in migration
    assert "<> 2" in migration
    assert "p_settings ->> 'team_category'" in migration
    assert "p_settings ->> 'allow_substitutes'" in migration
    assert "team_league_save_settings_v1" in migration
    assert "revoke all" in migration
    assert "service_role" in migration
    for unsupported in (
        "allow_alternates",
        "max_alternates",
        "substitute_pool_enabled",
    ):
        assert unsupported not in migration


def test_admin_and_public_surfaces_only_advertise_supported_pair_policy() -> None:
    admin = _read("apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx")
    public_list = _read("apps/web/app/clubs/[clubSlug]/team-leagues/page.tsx")
    public_detail = _read(
        "apps/web/app/clubs/[clubSlug]/team-leagues/[leagueName]/page.tsx"
    )
    registration = _read(
        "apps/web/app/clubs/[clubSlug]/team-leagues/[leagueName]/"
        "TeamLeagueRegistrationForm.tsx"
    )
    api_types = _read("apps/web/lib/teamLeagueApi.ts")
    surfaces = "\n".join((admin, public_list, public_detail, registration, api_types))

    for phrase in (
        "Primary roster",
        "2 players",
        "Team eligibility",
        "Men&apos;s",
        "Women&apos;s",
        "Mixed",
        "Allow substitutes",
    ):
        assert phrase in admin
    assert "substitutes are allowed" in registration
    assert "2-player roster" in public_detail
    for unsupported in (
        "Allow team alternates",
        "Shared substitute pool",
        "allow_alternates",
        "max_alternates",
        "substitute_pool_enabled",
        "alternate_management_supported",
        "substitute_pool_registration_supported",
    ):
        assert unsupported not in surfaces
