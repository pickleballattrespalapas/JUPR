from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_PROJECT_ID = "dnoockbwfenunhcibwfn"


def _read(path: str) -> str:
    return (ROOT / path).read_text()


def test_team_league_migrations_are_private_versioned_and_recoverable() -> None:
    schema = _read(
        "supabase/migrations/20260728030000_team_leagues_and_awards.sql"
    )
    hardening = _read(
        "supabase/migrations/20260728040000_team_league_awards_hardening.sql"
    )
    registration_recovery = _read(
        "supabase/migrations/"
        "20260728041000_team_league_registration_identity_recovery.sql"
    )

    for table in (
        "team_league_settings",
        "team_league_teams",
        "team_league_solo_waitlist",
        "team_league_fixtures",
        "team_league_operations",
        "league_award_result_sets",
        "league_award_result_records",
    ):
        assert f"public.{table}" in schema + hardening
        assert f"alter table public.{table} enable row level security" in (
            schema + hardening
        )
        assert f"alter table public.{table} force row level security" in (
            schema + hardening
        )
    assert "team_league_replace_schedule_v2" in hardening
    assert "p_expected_roster_version" in hardening
    assert "p_confirmed_roster_fingerprint" in hardening
    assert "TEAM_LEAGUE_REGULAR_RESULT_LOCKED_AFTER_PLAYOFF_SEEDING" in hardening
    assert "teamfx:" in hardening
    assert "league_awards_save_config_v1" in hardening
    assert "league_awards_apply_workflow_v2" in hardening
    assert (
        "team_league_recover_public_registration_v1"
        in registration_recovery
    )
    assert "recovered_by_business_identity" in registration_recovery
    assert "TEAM_LEAGUE_REGISTRATION_IDENTITY_CONFLICT" in registration_recovery
    assert "partner_invite_token_hash = p_invite_token_hash" in registration_recovery
    assert "service_role" in registration_recovery
    assert "admin_activity_log" in schema
    assert "revoke all" in schema + hardening
    assert "service_role" in schema + hardening
    assert PRODUCTION_PROJECT_ID not in schema + hardening + registration_recovery


def test_all_team_league_mutation_decorators_are_static_literals_and_gated() -> None:
    for path, gate in (
        (
            "services/api/public_team_league_routes.py",
            "require_public_team_league_write_or_403",
        ),
        (
            "services/api/admin_team_league_routes.py",
            "require_admin_team_league_write_or_403",
        ),
    ):
        source = _read(path)
        tree = ast.parse(source)
        unsafe_count = 0
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                attribute = decorator.func
                if not isinstance(attribute, ast.Attribute):
                    continue
                if attribute.attr not in {"post", "put", "patch", "delete"}:
                    continue
                unsafe_count += 1
                assert decorator.args
                route_path = ast.literal_eval(decorator.args[0])
                assert isinstance(route_path, str)
                body = ast.get_source_segment(source, node) or ""
                if not route_path.endswith("/schedule-preview/{phase}"):
                    assert gate in body
        assert unsafe_count >= 2


def test_every_team_league_route_checks_feature_gate_before_any_data_access() -> None:
    route_count = 0
    for path in (
        "services/api/public_team_league_routes.py",
        "services/api/admin_team_league_routes.py",
    ):
        source = _read(path)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_route = any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr in {"get", "post", "put", "patch", "delete"}
                for decorator in node.decorator_list
            )
            if not is_route:
                continue
            route_count += 1
            assert node.body
            first = node.body[0]
            assert isinstance(first, ast.Expr)
            assert isinstance(first.value, ast.Call)
            assert isinstance(first.value.func, ast.Name)
            assert first.value.func.id == "require_team_leagues_enabled_or_403"
    assert route_count == 14

    feature_gate = _read("services/api/team_league_feature.py")
    assert "JUPR_ENABLE_TEAM_LEAGUES" in feature_gate
    assert "LOCAL_TEST_ENVIRONMENTS" in feature_gate
    assert PRODUCTION_PROJECT_ID not in feature_gate


def test_team_league_routes_are_installed_in_the_api_aggregator() -> None:
    main = _read("services/api/main.py")

    assert "install_public_team_league_routes" in main
    assert "install_admin_team_league_routes" in main
    assert "install_public_team_league_routes(" in main
    assert "install_admin_team_league_routes(" in main


def test_partner_secret_never_appears_in_query_or_browser_history() -> None:
    confirmation = _read(
        "apps/web/app/clubs/[clubSlug]/team-league-partner-confirmation/"
        "PartnerConfirmationPanel.tsx"
    )
    service = _read("jupr_app/services/team_league_service.py")
    route = _read("services/api/public_team_league_routes.py")

    assert "window.location.hash" in confirmation
    assert "history.replaceState" in confirmation
    assert "token:" in confirmation
    assert "?token=" not in confirmation
    assert "#token=" in service
    assert "team-league-partner:v1:" in service
    assert "get_next_web_base_url" in route
    assert 'club.get("public_base_url")' not in route
    assert "https://juprleagues.com" not in route


def test_team_league_and_award_admin_pages_are_complete_and_responsive() -> None:
    nav = _read("apps/web/app/admin/league-manager/LeagueManagerNav.tsx")
    team_panel = _read(
        "apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx"
    )
    roster_panel = _read(
        "apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx"
    )
    awards_panel = _read(
        "apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx"
    )
    registration = _read(
        "apps/web/app/clubs/[clubSlug]/team-leagues/[leagueName]/"
        "TeamLeagueRegistrationForm.tsx"
    )
    public_detail = _read(
        "apps/web/app/clubs/[clubSlug]/team-leagues/[leagueName]/page.tsx"
    )

    for target in ("settings", "roster", "teams", "live", "awards"):
        assert f"/admin/league-manager/{target}" in nav
    for phrase in (
        "Allow substitutes",
        "No playoffs",
        "weekly schedule",
        "SAVE TEAM LEAGUE RESULT",
        "RECONCILE TEAM LEAGUE RESULT",
        "FINALIZE TEAM LEAGUE RECOVERY",
    ):
        assert phrase in team_panel
    assert "SAVE LEAGUE ROSTER BATCH" in roster_panel
    assert "Select visible" in roster_panel
    assert "award_catalog" in awards_panel
    assert "Measurable league results" in awards_panel
    assert "Team measures" in awards_panel
    assert 'useState<"success" | "error" | null>' in registration
    assert 'messageTone === "error" ? "alert" : "status"' in registration
    assert 'boxSizing: "border-box"' in registration
    assert "minmax(min(100%, 220px)" in registration
    assert "minmax(min(100%, 200px)" in public_detail
    assert "minmax(min(100%, 120px)" in public_detail
    assert 'overflowX: "auto", maxWidth: "100%"' in public_detail
    for source in (team_panel, roster_panel, awards_panel):
        assert "repeat(auto-fit" in source
        assert "overflowX: \"auto\"" in source
        assert "SUPABASE_SERVICE_ROLE_KEY" not in source
        assert "supabase.table" not in source


def test_acceptance_criteria_cover_approved_team_and_award_decisions() -> None:
    criteria = _read("docs/team_leagues_and_awards_acceptance.md").lower()

    for phrase in (
        "payment remains offline",
        "same two partners",
        "substitutes are disabled",
        "every team meets every opponent exactly once",
        "no more than one match per week",
        "playoffs are optional",
        "head-to-head",
        "wins above expected",
        "missing ratings never become an invented 50/50",
        "club owner can enable any supported player or team award",
        "exact configuration",
        "lost registration response is recoverable",
        "distinct durable award identities",
        "production supabase",
    ):
        assert phrase in criteria
