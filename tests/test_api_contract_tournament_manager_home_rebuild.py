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


def test_selected_tournament_home_exposes_lifecycle_phases() -> None:
    page = read("app/admin/tournaments/tournament/page.tsx")
    panel = read("app/admin/tournaments/tournament/TournamentHomePanel.tsx")

    assert 'first(searchParams?.tournament)' in page
    assert 'redirect("/admin/tournaments")' in page
    assert "Tournament workflow" in panel
    assert "Next action" in panel
    for label in (
        'title: "Setup"',
        'title: "Registration"',
        'title: "Live Operations"',
        'title: "Publish"',
    ):
        assert label in panel
    assert "Results become official only through Publish" in panel
    assert "Delete draft" in panel


def test_tournament_navigation_separates_manager_and_lifecycle_context() -> None:
    nav = read("components/TournamentAdminNav.tsx")
    assert 'label: "Tournament Manager Home"' in nav
    assert 'const hasTournament = Boolean(tournamentId)' in nav
    assert "const tournamentItems = hasTournament" in nav
    assert 'label: "Tournament Home"' in nav
    assert 'label: "Tournament Builder"' in nav
    assert 'label: "Registration"' in nav
    assert 'label: "Live Operations"' in nav
    assert 'label: "Publish"' in nav
    for old_label in (
        'label: "Bulk actions"',
        'label: "Extras & fulfillment"',
        'label: "Ratings & team play"',
        'label: "Operations"',
        'label: "Results"',
        'label: "Live runner"',
        'label: "Official publish"',
        'label: "Status & recovery"',
    ):
        assert old_label not in nav
    assert "Tournament workspace" not in nav
    assert "Setup, registrations, event operations, and live play" not in nav


def test_selected_tournament_scope_removes_repeated_selection_and_preserves_links() -> None:
    scope = read("app/admin/tournaments/SelectedTournamentPanelScope.tsx")
    assert "MutationObserver" in scope
    assert 'candidate.dispatchEvent(new Event("change", { bubbles: true }))' in scope
    assert "preserveTournamentContext" in scope
    assert 'url.searchParams.set("tournament", tournamentId)' in scope
    assert 'querySelectorAll<HTMLAnchorElement>("a[href]")' in scope
    assert '"registration reporting session"' in scope
    assert '"bulk registration actions"' in scope
    assert '"1. create tournament shell"' in scope
    assert '"2. select tournament"' in scope
    assert "hasVisibleTransientText" in scope
    assert "Loading {tournamentName" in scope


def test_selected_tournament_module_pages_require_context() -> None:
    pages = (
        "app/admin/tournaments/setup/page.tsx",
        "app/admin/tournaments/registration/page.tsx",
        "app/admin/tournaments/registration/registrants/page.tsx",
        "app/admin/tournaments/registration/partners/page.tsx",
        "app/admin/tournaments/live-operations/page.tsx",
        "app/admin/tournaments/live-operations/check-in/page.tsx",
        "app/admin/tournaments/publish/page.tsx",
        "app/admin/tournaments/publish/closeout/page.tsx",
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
