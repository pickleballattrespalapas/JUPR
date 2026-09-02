import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def test_league_manager_landing_is_manager_only() -> None:
    page = read("app/admin/league-manager/page.tsx")
    panel = read("app/admin/league-manager/LeagueManagerPanel.tsx")

    assert "Create a new league or open an existing league." in page
    assert "Guarded Next/FastAPI league operations" not in page
    assert "League Manager admin session" not in page
    assert "Previous-month Top 50" not in page
    assert "League night printout" not in page
    assert 'href="/admin/league-manager/create"' in panel
    assert "Create league draft" in panel
    assert "router.push(leagueHomeHref(league))" in panel
    assert "Available leagues" not in panel


def test_create_and_selected_league_homes_are_separate() -> None:
    create_page = read("app/admin/league-manager/create/page.tsx")
    league_page = read("app/admin/league-manager/league/page.tsx")
    league_panel = read("app/admin/league-manager/league/LeagueHomePanel.tsx")
    nav = read("app/admin/league-manager/LeagueManagerNav.tsx")

    assert "Start league setup" in create_page
    assert "league setup wizard before activation" in create_page
    assert "redirect(\"/admin/league-manager\")" in league_page
    assert "League tools" in league_panel
    assert "League night printout" in league_panel
    assert 'label: "League Manager Home"' in nav
    assert 'label: "League Home"' in nav
    assert "if (hasLeague && leagueName)" in nav
    assert "if (isTeamLeagueType(leagueType))" in nav


def test_selected_league_modules_preserve_context() -> None:
    for path in (
        "app/admin/league-manager/results/page.tsx",
        "app/admin/league-manager/settings/page.tsx",
        "app/admin/league-manager/roster/page.tsx",
        "app/admin/league-manager/live/page.tsx",
        "app/admin/league-manager/awards/page.tsx",
        "app/admin/league-manager/teams/page.tsx",
        "app/admin/league-manager/print/page.tsx",
    ):
        source = read(path)
        assert "searchParams" in source
        assert "leagueName" in source
        assert "LeagueManagerNav" in source
        assert "readLeagueRouteContext(searchParams)" in source
        assert "leagueId={context.leagueId}" in source

    scope = read("app/admin/league-manager/SelectedLeaguePanelScope.tsx")
    assert "MutationObserver" in scope
    assert 'select.dispatchEvent(new Event("change", { bubbles: true }))' in scope


def test_selected_league_routes_use_explicit_id_and_results_fail_closed() -> None:
    context = read("lib/leagueRouteContext.ts")
    landing = read("app/admin/league-manager/LeagueManagerPanel.tsx")
    create = read("app/admin/league-manager/create/LeagueCreatePanel.tsx")
    home = read("app/admin/league-manager/league/LeagueHomePanel.tsx")
    nav = read("app/admin/league-manager/LeagueManagerNav.tsx")
    results = read("app/admin/league-manager/results/page.tsx")
    results_panel = read(
        "app/admin/league-manager/results/LeagueResultsPanel.tsx"
    )

    assert 'params.set("league_id", context.leagueId)' in context
    assert 'readValue(searchParams, "league_id") || legacyLeague' in context
    assert "leagueId: leagueIdentifier(league)" in landing
    assert "leagueIdentifier(item) === leagueId" in landing
    assert "payload.league?.league_id || createdName" in create
    assert "leagueHomeHref(createdId, createdName, createdType)" in create
    assert "leagueId={leagueId}" in home
    assert "leagueRouteHref(path, context)" in nav
    assert 'leagueHref("/admin/league-manager/results", context)' in nav
    assert "readLeagueRouteContext(searchParams)" in results
    assert "getClubLeagueResults" not in results
    assert "useAdminSession" in results_panel
    assert "Authorization" in results_panel
    assert "/league-manager/leagues/${encodeURIComponent(initialLeague)}/results" in results_panel
    assert "The selected league did not match the returned results." in results_panel


def test_team_results_redirect_to_the_team_league_surface() -> None:
    results = read("app/admin/league-manager/results/page.tsx")

    assert "isTeamLeagueType(context.leagueType)" in results
    assert 'redirect(leagueRouteHref("/admin/league-manager/teams", context))' in results


def test_node_20_component_suite_covers_league_route_and_status_behavior() -> None:
    package = json.loads(read("package.json"))
    command = package["scripts"]["test:component"]

    assert "node tests/tournament-draw-operational-status.mjs" in command
    assert "node tests/league-manager-route-status.cjs" in command
    assert "--experimental-strip-types" not in command


def test_reports_are_in_admin_tools_not_league_manager() -> None:
    tools = read("app/admin/tools/page.tsx")
    manager = read("app/admin/league-manager/page.tsx")
    assert "Previous-month Top 50" in tools
    assert 'href="/admin/top-players-printable"' in tools
    assert "Previous-month Top 50" not in manager


def test_no_temporary_publisher_workflows_remain() -> None:
    assert not (ROOT / ".github" / "workflows" / "tmp-selected-league-panel-patch.yml").exists()
