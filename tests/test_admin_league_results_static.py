from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_admin_results_panel_uses_authenticated_admin_endpoint() -> None:
    page = _read("apps/web/app/admin/league-manager/results/page.tsx")
    panel = _read(
        "apps/web/app/admin/league-manager/results/LeagueResultsPanel.tsx"
    )

    assert "getClubLeagueResults" not in page
    assert "useAdminSession" in panel
    assert "Authorization" in panel
    assert "Bearer ${accessToken}" in panel
    assert "/league-manager/leagues/${encodeURIComponent(initialLeague)}/results" in panel
    assert "useAuthenticatedAutoLoad" in panel
    assert "results.publicly_visible" in panel
    assert "Historical results are admin-only." in panel


def test_results_navigation_preserves_identity_and_redirects_team_mode() -> None:
    page = _read("apps/web/app/admin/league-manager/results/page.tsx")
    panel = _read(
        "apps/web/app/admin/league-manager/results/LeagueResultsPanel.tsx"
    )

    assert "readLeagueRouteContext(searchParams)" in page
    assert "leagueId={context.leagueId}" in page
    assert "initialLeagueId={context.leagueId}" in page
    assert "isTeamLeagueType(context.leagueType)" in page
    assert 'leagueRouteHref("/admin/league-manager/teams", context)' in page
    assert 'router.replace(' in panel
    assert 'leagueRouteHref("/admin/league-manager/teams"' in panel


def test_admin_results_route_authorizes_before_building_historical_results() -> None:
    routes = _read("services/api/admin_league_manager_routes.py")
    route = routes.split("def get_admin_league_manager_results", 1)[1].split(
        '@app.get("/admin/clubs/{club_id}/league-manager/top-players-printable")',
        1,
    )[0]

    assert "_resolve_league_manager_role_or_403(" in route
    assert "build_admin_league_results(" in route
    assert route.index("_resolve_league_manager_role_or_403(") < route.index(
        "build_admin_league_results("
    )
