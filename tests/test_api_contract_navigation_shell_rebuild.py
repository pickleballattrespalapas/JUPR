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


def test_admin_sidebar_is_authorized_session_only() -> None:
    shell = read("components/AdminShell.tsx")
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
    assert "signOutAdminSession" in shell
    assert "AdminShell" in layout


def test_public_tournament_hub_owns_registration_roster_and_partner_board() -> None:
    hub = read("app/clubs/[clubSlug]/tournaments/page.tsx")
    nav = read("components/PublicTournamentNav.tsx")
    registration = read("app/clubs/[clubSlug]/tournament-registration/page.tsx")
    roster_layout = read("app/clubs/[clubSlug]/tournament-roster/layout.tsx")
    partner_layout = read("app/clubs/[clubSlug]/tournament-partner-board/layout.tsx")

    assert "getClubTournamentRegistration" in hub
    assert "Choose a tournament" in hub
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
    assert "PublicTournamentRouteNav" in roster_layout
    assert 'active="roster"' in roster_layout
    assert "PublicTournamentRouteNav" in partner_layout
    assert 'active="partner-board"' in partner_layout


def test_leagues_hub_and_club_home_use_grouped_public_navigation() -> None:
    leagues = read("app/clubs/[clubSlug]/leagues/page.tsx")
    club = read("app/clubs/[clubSlug]/page.tsx")
    site_map = read("app/site-map/page.tsx")
    sitemap = read("app/sitemap.ts")

    for label in ("League Results", "Team Leagues", "Challenge Ladder"):
        assert label in leagues
    assert 'title: "Leagues"' in club
    assert 'href: `${base}/leagues`' in club
    assert 'title: "Tournaments"' in club
    assert 'href: `${base}/tournaments`' in club
    assert '"Leagues", "/clubs/tres-palapas/leagues"' in site_map
    assert '"Tournaments", "/clubs/tres-palapas/tournaments"' in site_map
    assert '"/clubs/tres-palapas/leagues"' in sitemap
    assert '"/clubs/tres-palapas/tournaments"' in sitemap
