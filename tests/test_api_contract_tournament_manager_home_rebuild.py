from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def test_tournament_manager_landing_is_create_or_open_only() -> None:
    page = read("app/admin/tournaments/page.tsx")
    panel = read("app/admin/tournaments/TournamentAdminPanel.tsx")

    assert "Create a new tournament or open an existing tournament." in page
    assert "Create tournament" in panel
    assert 'href="/admin/tournaments/create"' in panel
    assert "Open tournament" in panel
    assert "router.push(tournamentHomeHref(tournament))" in panel
    assert "Admin session" not in panel
    assert "Tournament home" not in panel
    assert "Status</strong>" not in page


def test_selected_tournament_home_exposes_scoped_modules() -> None:
    page = read("app/admin/tournaments/tournament/page.tsx")
    panel = read("app/admin/tournaments/tournament/TournamentHomePanel.tsx")

    assert 'first(searchParams?.tournament)' in page
    assert 'redirect("/admin/tournaments")' in page
    assert "Tournament tools" in panel
    for label in (
        "Setup",
        "Registrations",
        "Reports",
        "Extras & fulfillment",
        "Ratings & team play",
        "Operations",
        "Results",
        "Live runner",
        "Official publish",
        "Status & recovery",
    ):
        assert label in panel
    assert "Delete draft" in panel


def test_tournament_navigation_separates_manager_and_selected_context() -> None:
    nav = read("components/TournamentAdminNav.tsx")
    assert 'label: "Tournament Manager Home"' in nav
    assert 'const hasTournament = Boolean(tournamentId)' in nav
    assert 'const tournamentItems = hasTournament ? selectedItems' in nav
    assert 'label: "Tournament Home"' in nav
    assert 'label: "Registrations"' in nav
    assert 'label: "Reports"' in nav
    assert 'label: "Status & recovery"' in nav
    assert "Tournament workspace" not in nav
    assert "Setup, registrations, event operations, and live play" not in nav


def test_selected_tournament_scope_removes_repeated_selection_and_preserves_links() -> None:
    scope = read("app/admin/tournaments/SelectedTournamentPanelScope.tsx")
    assert "MutationObserver" in scope
    assert 'candidate.dispatchEvent(new Event("change", { bubbles: true }))' in scope
    assert "preserveTournamentContext" in scope
    assert 'url.searchParams.set("tournament", tournamentId)' in scope
    assert '"registration reporting session"' in scope
    assert '"1. create tournament shell"' in scope
    assert '"2. select tournament"' in scope


def test_selected_tournament_module_pages_require_context() -> None:
    pages = (
        "app/admin/tournaments/editor/page.tsx",
        "app/admin/tournaments/registrations/page.tsx",
        "app/admin/tournaments/bulk/page.tsx",
        "app/admin/tournaments/commerce/page.tsx",
        "app/admin/tournaments/team-competition/page.tsx",
        "app/admin/tournaments/status/page.tsx",
        "app/admin/tournaments/delete-draft/page.tsx",
        "app/admin/tournaments/ops/page.tsx",
        "app/admin/tournaments/ops/draws/page.tsx",
        "app/admin/tournaments/ops/import/page.tsx",
        "app/admin/tournaments/ops/results/page.tsx",
        "app/admin/tournaments/ops/publish/page.tsx",
        "app/admin/tournament-setup/page.tsx",
        "app/admin/tournament-live/page.tsx",
    )
    for path in pages:
        source = read(path)
        assert "searchParams" in source, path
        assert "tournamentId" in source, path
        assert 'redirect("/admin/tournaments")' in source or 'redirect("/admin/tournaments/create")' in source, path


def test_league_printout_is_available_in_selected_league_navigation() -> None:
    nav = read("app/admin/league-manager/LeagueManagerNav.tsx")
    assert 'label: "League night printout"' in nav
    assert 'leagueHref("/admin/league-manager/print"' in nav
