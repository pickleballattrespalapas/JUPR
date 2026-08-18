from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def test_public_leagues_open_dedicated_league_home() -> None:
    hub = read("app/clubs/[clubSlug]/leagues/page.tsx")
    home = read("app/clubs/[clubSlug]/leagues/[leagueName]/page.tsx")
    nav = read("components/PublicLeagueNav.tsx")
    route_nav = read("components/PublicLeagueResultsRouteNav.tsx")
    results_layout = read("app/clubs/[clubSlug]/league-results/layout.tsx")

    assert "publicLeagueHomeHref" in hub
    assert "Open League Home" in hub
    assert "PublicLeagueNav" in home
    assert 'active="home"' in home
    assert "League pages" in home
    assert "Awards race" in home
    assert "Rating standings preview" in home
    assert "All leagues" in nav
    assert "League Home" in nav
    assert "Standings" in nav
    assert "Weekly History" in nav
    assert "Player Summaries" in nav
    assert "useSearchParams" in route_nav
    assert "PublicLeagueResultsRouteNav" in results_layout


def test_roster_is_a_selected_tournament_workspace() -> None:
    page = read("app/clubs/[clubSlug]/tournament-roster/page.tsx")
    layout = read("app/clubs/[clubSlug]/tournament-roster/layout.tsx")

    assert "PublicTournamentModuleHeader" in page
    assert 'active="roster"' in page
    assert "Roster overview" in page
    assert "Players looking for partners" in page
    assert "groupedEntries" in page
    assert "Apply filters" in page
    assert "data.tournaments.map" not in page
    assert "PublicTournamentRouteNav" not in layout
    assert 'redirect(`/clubs/${params.clubSlug}/tournaments`)' in page


def test_partner_board_is_a_selected_tournament_workspace() -> None:
    page = read("app/clubs/[clubSlug]/tournament-partner-board/page.tsx")
    layout = read("app/clubs/[clubSlug]/tournament-partner-board/layout.tsx")

    assert "PublicTournamentModuleHeader" in page
    assert 'active="partner-board"' in page
    assert "Open partner requests" in page
    assert "Want to contact or accept a player?" in page
    assert "PartnerRequestReviewPanel" in page
    assert "PairingInterestPanel" in page
    assert "data.tournaments.map" not in page
    assert "PublicTournamentRouteNav" not in layout
    assert 'redirect(`/clubs/${params.clubSlug}/tournaments`)' in page


def test_shared_tournament_module_header_preserves_selected_context() -> None:
    header = read("components/PublicTournamentModuleHeader.tsx")

    assert "PublicTournamentNav" in header
    assert "publicTournamentHref" in header
    assert "← Tournament Home" in header
    assert "tournamentId" in header
    assert "registrationSlug" in header
