from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def test_selected_tournament_and_draw_have_one_shared_route_contract() -> None:
    helper = read("lib/tournamentRouteContext.ts")
    nav = read("components/TournamentAdminNav.tsx")
    phases = read("components/TournamentPhaseNav.tsx")

    assert "readTournamentRouteContext" in helper
    assert 'searchParams.get("tournament_id")' in helper
    assert 'searchParams.get("draw_id")' in helper
    assert 'params.set("draw", context.drawId)' in helper
    assert "readTournamentRouteContext" in nav
    assert "tournamentRouteHref" in nav
    assert "readTournamentRouteContext" in phases
    assert "tournamentRouteHref" in phases


def test_setup_registration_and_home_preserve_the_selected_draw_without_dom_rewrites() -> None:
    home_page = read("app/admin/tournaments/tournament/page.tsx")
    home_panel = read("app/admin/tournaments/tournament/TournamentHomePanel.tsx")
    registration_page = read("app/admin/tournaments/registration/page.tsx")
    registration_overview = read("app/admin/tournaments/TournamentLifecycleOverviewPanel.tsx")
    setup_page = read("app/admin/tournaments/setup/TournamentSetupWizardPage.tsx")
    setup_nav = read("components/TournamentSetupWizardNav.tsx")
    partner_page = read("app/admin/tournaments/registration/partners/page.tsx")

    assert "initialDrawId={context.drawId" in home_page
    assert "?draw_id=${encodeURIComponent(initialDrawId)}" in home_panel
    assert "drawId={context.drawId}" in registration_page
    assert "tournamentRouteHref" in registration_overview
    assert "drawId={context.drawId}" in setup_page
    assert "drawId" in setup_nav
    assert 'tournament_id: context.tournamentId' in partner_page
    assert 'partnerBoardParams.set("draw", context.drawId)' in partner_page
    assert 'tournamentRouteHref("/clubs/tres-palapas/tournament-partner-board"' not in partner_page
    assert not (WEB / "app/admin/tournaments/SelectedTournamentPanelScope.tsx").exists()


def test_live_workflow_uses_focused_routes_and_one_draw_scoped_shell() -> None:
    phases = read("components/TournamentPhaseNav.tsx")
    shell = read("app/admin/tournaments/live-operations/TournamentLiveRoute.tsx")

    for path in (
        "/admin/tournaments/live-operations",
        "/admin/tournaments/live-operations/check-in",
        "/admin/tournaments/live-operations/draws",
        "/admin/tournament-live",
        "/admin/tournaments/live-operations/corrections",
        "/admin/tournaments/live-operations/podium",
    ):
        assert path in phases
    assert "readTournamentRouteContext" in shell
    assert "initialTournamentId={context.tournamentId}" in shell
    assert "initialDrawId={context.drawId}" in shell


def test_live_runner_locks_tournament_scope_and_exposes_a_url_synced_draw_selector() -> None:
    panel = read("app/admin/tournament-live/TournamentLivePanel.tsx")
    css = read("app/admin/tournament-live/TournamentLivePanel.module.css")

    assert "const lockedTournamentId = initialTournamentId;" in panel
    assert 'aria-label="Tournament operating scope"' in panel
    assert 'id="working-draw"' in panel
    assert "Locked to this tournament workspace" in panel
    assert "Refresh available draws" in panel
    assert "Reload selected draw" in panel
    assert "router.replace(tournamentRouteHref(pathname, nextContext), { scroll: false })" in panel
    assert "Change or refresh selection" not in panel
    assert "Include archived tournaments" not in panel
    assert "Refresh tournaments" not in panel
    assert "selectTournament" not in panel
    assert "/tournaments/admin/ops/tournaments" not in panel
    assert "const preferredDrawStillAvailable" in panel
    assert ": !preferredDrawId && operableDraws.length === 1 ? operableDraws[0].id : \"\";" in panel
    assert "if (!nextSelectedDrawId) {" in panel
    assert "setSnapshot(null);" in panel
    select_draw = panel.split("function selectDraw(drawId: string)", 1)[1].split("function selectScoreGame", 1)[0]
    assert select_draw.index("boardRequest.invalidate") < select_draw.index("setSnapshot(null)") < select_draw.index("loadLiveBoard(drawId)")
    assert "const generation = boardRequest.begin();" in panel
    assert "if (!boardRequest.isCurrent(generation)) return;" in panel
    assert "assertSnapshotIdentity(payload, tournamentId, drawId);" in panel
    assert "assertSnapshotIdentity(payload, lockedTournamentId, drawId);" in panel
    assert "@media (max-width: 1200px)" in css


def test_live_draw_selector_uses_authoritative_progress_instead_of_stale_draft_status() -> None:
    panel = read("app/admin/tournament-live/TournamentLivePanel.tsx")
    status_helper = read("lib/tournamentDrawOperationalStatus.mjs")

    draw_label = status_helper.split("export function drawOperationalStatus(", 1)[1]
    assert 'from "@/lib/tournamentDrawOperationalStatus.mjs"' in panel
    assert "lifecycleDraw" in draw_label
    assert 'liveOperations === "in_progress"' in draw_label
    assert 'liveOperations === "complete"' in draw_label
    assert 'officialPublish === "complete"' in draw_label
    assert 'return `Not started · ${games} ${gameWord}`' in draw_label
    assert 'return `In progress · ${finalizedGames} of ${games} scored`' in draw_label
    assert 'return `Scores complete · ${finalizedGames} of ${games} scored`' in draw_label
    assert 'return `Published · ${publishedGames} official ${matchWord}`' in draw_label
    assert 'return `Publish recovery needed · ${publishedGames} of ${games} official`' in draw_label
    assert 'return "No games scheduled"' in draw_label
    assert 'return "Status unavailable"' in draw_label
    assert "INACTIVE_DRAW_STATUSES.has" in draw_label
    assert "draw.status || \"draft\"" not in draw_label
    assert "setDrawLifecycle(payload.lifecycle?.draws || [])" in panel
    assert "drawLifecycleById.get(draw.id)" in panel
    assert "selectedLifecycleDraw?.counts" in panel
    assert "disabled={inactive || !lifecycleDraw}" in panel
    assert "`${selectedFinalizedGames} of ${selectedTotalGames} games scored; ${selectedOpenGames} open.`" in panel
    assert "{selectedFinalizedGames} of {selectedTotalGames} games finalized" in panel
    assert "{selectedDrawCounts?.published_games || 0} of {selectedRatingPublishEligibleGames} played games published" in panel


def test_live_runner_is_human_readable_and_validates_before_confirmation() -> None:
    panel = read("app/admin/tournament-live/TournamentLivePanel.tsx")
    css = read("app/admin/tournament-live/TournamentLivePanel.module.css")

    for phrase in (
        "Confirm & save",
        "Edit score",
        "Proposed winner",
        "Before correction",
        "After correction",
        "Recent operations and reconciliation",
        "Technical operation evidence",
        "Tournament draw scores can be corrected here only before official publication",
    ):
        assert phrase in panel
    assert "validateScoreDraft" in panel
    assert "setScoreConfirmation" in panel
    assert "onClick={validateScoreDraft}" in panel
    assert "!scoreConfirmation" in panel
    assert 'const actionScope = `${accessToken}\\u0000${selectedTournamentId}\\u0000${selectedDrawId}`' in panel
    assert panel.count("actionRequest.invalidate();") >= 3
    assert '`#${playerId}`' not in panel
    assert "overflow-x: auto" in css
    assert "min-width: 0" in css
    assert "@media (max-width: 1200px)" in css


def test_authoritative_lifecycle_drives_home_results_publish_closeout_and_recovery() -> None:
    home = read("app/admin/tournaments/tournament/TournamentHomePanel.tsx")
    panel = read("app/admin/tournament-live/TournamentLivePanel.tsx")
    status = read("app/admin/tournaments/status/page.tsx")

    assert "lifecycle" in home
    assert "finalized_games" in home
    assert "open_games" in home
    assert "Continue scoring" in home
    assert "domain_readiness.official_publish" in home
    assert "Authoritative tournament state unavailable" in home
    assert "readiness is not inferred" in home
    assert "?draw_id=${encodeURIComponent(initialDrawId)}" in home
    assert 'label: "Prepare Live Operations"' not in home
    for phrase in (
        "Review results",
        "Import results",
        "Runtime capability",
        "Tournament readiness",
        "Completion readiness",
        "Communications",
        "Payments, extras, and fulfillment",
    ):
        assert phrase in panel
    assert "status.official_publish_writes_enabled" in panel
    assert "status.official_publish_write_flag?.enabled" in panel
    assert "dedicated official-publish permission" in panel
    assert "Complete tournament" in panel
    assert "Move to hidden archive" in panel
    assert "Restore completed tournament" in panel
    assert "archiveTournament" not in panel
    assert "Archive tournament" not in status


def test_local_browser_fixture_covers_operator_safety_story_at_desktop_widths() -> None:
    spec = read("e2e/tournament-operator.local.spec.ts")
    config = read("playwright.tournament-operator.config.ts")

    assert "page.route" in spec
    assert "1 of 21 games scored" in spec
    assert "9–9" in spec or "9-9" in spec
    assert "Confirm & save" in spec
    assert "Corrections & recovery" in spec
    assert "Best-of-three score entry records every rating game" in spec
    assert "BEST_2_OF_3" in spec
    assert "game_scores" in spec
    assert 'name: "Game 3"' in spec
    assert "Publish official matches" in spec
    assert "Complete tournament" in spec
    assert "1280" in spec
    assert "1440" in spec
    assert "forbidOnly: true" in config
    assert "fullyParallel: false" in config
