from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def test_public_header_uses_leagues_and_tournaments_hubs() -> None:
    header = read("components/PublicSiteHeader.tsx")
    layout = read("app/layout.tsx")

    assert 'label: "Leagues"' in header
    assert 'href: `${clubBase}/leagues`' in header
    assert 'label: "Tournaments"' in header
    assert 'href: `${clubBase}/tournaments`' in header
    assert 'label: "Register"' not in header
    assert 'label: "Roster"' not in header
    assert 'label: "Partner Board"' not in header
    assert 'pathname.startsWith(`${clubBase}/tournament-`)' in header
    assert 'pathname.startsWith("/admin/")' in header
    assert "PublicSiteHeader" in layout
    assert 'href="/clubs/tres-palapas/leagues"' in layout
    assert 'href="/clubs/tres-palapas/tournaments"' in layout


def test_admin_sidebar_is_authorized_session_only_and_collapsible() -> None:
    shell = read("components/AdminShell.tsx")
    styles = read("components/AdminShell.module.css")
    layout = read("app/admin/layout.tsx")

    assert "useAdminSession" in shell
    assert "if (authPage || !accessToken)" in shell
    assert 'pathname === "/admin/login"' in shell
    assert 'pathname === "/admin/reset-password"' in shell
    assert 'aria-label="Admin workspace navigation"' in shell
    assert 'label: "Admin Home"' in shell
    assert 'label: "Match Uploader"' in shell
    assert 'label: "Match Log"' in shell
    assert 'label: "League Manager"' in shell
    assert 'label: "Tournament Manager"' in shell
    assert 'label: "Player Updates"' in shell
    assert 'label: "Admin Tools"' in shell
    assert 'label: "Public site"' in shell
    assert 'label: "Public Home ↗"' in shell
    assert 'label: "Leagues ↗"' in shell
    assert 'label: "Tournaments ↗"' in shell
    assert 'target={item.newTab ? "_blank" : undefined}' in shell
    assert "sidebarCollapsed" in shell
    assert "collapsedGroups" in shell
    assert 'aria-label={sidebarCollapsed ? "Expand admin sidebar"' in shell
    assert "toggleGroup" in shell
    assert "shellCollapsed" in styles
    assert "sidebarCollapsed" in styles
    assert "groupToggle" in styles
    assert "signOutAdminSession" in shell
    assert "AdminShell" in layout


def test_public_tournament_selection_is_separate_from_selected_workspace() -> None:
    hub = read("app/clubs/[clubSlug]/tournaments/page.tsx")
    nav = read("components/PublicTournamentNav.tsx")
    registration = read("app/clubs/[clubSlug]/tournament-registration/page.tsx")
    roster = read("app/clubs/[clubSlug]/tournament-roster/page.tsx")
    partner = read("app/clubs/[clubSlug]/tournament-partner-board/page.tsx")

    assert "getClubTournamentRegistration" in hub
    assert "const explicitSelection = Boolean(registrationSlug || tournamentId)" in hub
    assert "if (!explicitSelection)" in hub
    assert "Choose a tournament" in hub
    assert "Select a tournament to open its Tournament Home" in hub
    assert "if (!tournament)" in hub
    assert "← Choose another tournament" in hub
    assert "Tournament Home" in hub
    assert "Tournament pages" in hub
    assert "PublicTournamentNav" in hub
    assert 'active="overview"' in hub
    assert '["registration", "Register"]' in nav
    assert '["roster", "Roster"]' in nav
    assert '["partner-board", "Partner Board"]' in nav
    assert "registrationSlug" in nav
    assert "tournamentId" in nav
    assert "PublicTournamentNav" in registration
    assert 'active="registration"' in registration
    assert "PublicTournamentModuleHeader" in roster
    assert 'active="roster"' in roster
    assert "PublicTournamentModuleHeader" in partner
    assert 'active="partner-board"' in partner


def test_leagues_hub_lists_active_leagues_and_opens_league_home() -> None:
    leagues = read("app/clubs/[clubSlug]/leagues/page.tsx")
    league_home = read("app/clubs/[clubSlug]/leagues/[leagueName]/page.tsx")
    club = read("app/clubs/[clubSlug]/page.tsx")
    site_map = read("app/site-map/page.tsx")
    sitemap = read("app/sitemap.ts")

    assert "getClubLeagueResults" in leagues
    assert "Choose a league" in leagues
    assert "data?.leagues || []" in leagues
    assert "publicLeagueHomeHref" in leagues
    assert "Open League Home" in leagues
    assert "PublicLeagueNav" in league_home
    assert "League pages" in league_home
    assert "Team Leagues" not in leagues
    assert "Club Leaderboards" not in leagues
    assert 'title: "Leagues"' in club
    assert 'href: `${base}/leagues`' in club
    assert 'title: "Tournaments"' in club
    assert 'href: `${base}/tournaments`' in club
    assert '"Leagues", "/clubs/tres-palapas/leagues"' in site_map
    assert '"Tournaments", "/clubs/tres-palapas/tournaments"' in site_map
    assert '"/clubs/tres-palapas/leagues"' in sitemap
    assert '"/clubs/tres-palapas/tournaments"' in sitemap
