from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import jupr_app.services.admin_tournament_service as tournament_service


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

    assert "readTournamentRouteContext(searchParams)" in page
    assert "context.tournamentId" in page
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
    assert "const context = readTournamentRouteContext(searchParams)" in nav
    assert 'const hasTournament = Boolean(context.tournamentId)' in nav
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


def test_selected_tournament_context_is_constructed_without_dom_mutation() -> None:
    helper = read("lib/tournamentRouteContext.ts")
    assert "readTournamentRouteContext" in helper
    assert "tournamentRouteHref" in helper
    assert 'params.set("draw", context.drawId)' in helper
    assert not (WEB / "app/admin/tournaments/SelectedTournamentPanelScope.tsx").exists()
    tournament_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (WEB / "app/admin/tournaments").rglob("*.tsx")
    )
    assert "MutationObserver" not in tournament_sources


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
        if "TournamentLiveRoute" in source:
            route = read("app/admin/tournaments/live-operations/TournamentLiveRoute.tsx")
            assert "readTournamentRouteContext(searchParams)" in route, path
            assert "context.tournamentId" in route, path
            assert 'redirect("/admin/tournaments")' in route, path
        else:
            assert "tournamentId" in source, path
            assert 'redirect("/admin/tournaments")' in source or 'redirect("/admin/tournaments/create")' in source, path


def test_registration_handoff_targets_the_guarded_import_page(monkeypatch) -> None:
    detail = {
        "tournament": {
            "id": "tournament-1",
            "name": "Staging Summer Classic / 2026",
        },
        "registrations": [
            {"id": "registration-1", "registration_status": "confirmed"}
        ],
        "selections": [],
        "state_fingerprint": "reviewed-state",
    }
    monkeypatch.setattr(
        tournament_service,
        "get_admin_tournament_detail",
        lambda *args, **kwargs: detail,
    )

    handoff = tournament_service.build_admin_tournament_registration_import_handoff(
        object(),
        club_id="club-1",
        tournament_id="tournament-1",
    )
    destination = urlsplit(handoff["ops_path"])

    assert destination.path == "/admin/tournaments/ops/import"
    assert parse_qs(destination.query) == {
        "tournament": ["tournament-1"],
        "name": ["Staging Summer Classic / 2026"],
    }
    assert "tournament_id" not in parse_qs(destination.query)


def test_legacy_import_handoff_links_redirect_to_the_guarded_import_page() -> None:
    page = read("app/admin/tournaments/ops/page.tsx")

    assert "searchParams?.tournament_id" in page
    assert 'redirect(tournamentRouteHref("/admin/tournaments/ops/import", context))' in page


def test_league_printout_is_available_in_selected_league_navigation() -> None:
    nav = read("app/admin/league-manager/LeagueManagerNav.tsx")
    assert 'label: "League night printout"' in nav
    assert 'leagueHref("/admin/league-manager/print"' in nav
