from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_tournament_ops_write_readiness_requires_every_runtime_gate() -> None:
    panel = _read("apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx")
    status_type = _read("apps/web/lib/adminTournamentApi.ts")

    readiness = panel.split("const operationsWriteReady = Boolean(", 1)[1].split(");", 1)[0]
    assert "status.mutation_runtime?.service_role_ready" in readiness
    assert "status.mutation_runtime?.surface_flags?.operations?.enabled" in readiness
    assert "status.operations_runtime?.operations_mutations_enabled" in readiness
    assert "operations_runtime?:" in status_type
    assert "operations_mutations_enabled: boolean" in status_type


def test_tournament_ops_status_is_never_served_from_a_stale_next_cache() -> None:
    api = _read("apps/web/lib/adminTournamentApi.ts")

    assert 'cache: "no-store"' in api
    assert "revalidate" not in api


def test_selected_tournament_ops_keeps_read_only_snapshots_and_hides_post_controls() -> None:
    panel = _read("apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx")

    assert 'data-testid="tournament-ops-read-only-banner"' in panel
    assert "Tournament Ops is read-only" in panel
    assert "Tournament and draw snapshots remain available." in panel
    assert "POST-backed previews and mutation controls are hidden" in panel
    assert 'operationsWriteReady && snapshot && shows("draws")' in panel
    assert 'operationsWriteReady && shows("import")' in panel
    assert 'operationsWriteReady && shows("draws")' in panel
    assert 'operationsWriteReady && shows("results")' in panel
    assert 'operationsWriteReady && shows("publish")' in panel

    results_section = panel.split('{operationsWriteReady && shows("results") ? (', 1)[1].split(
        '{operationsWriteReady && shows("publish")', 1
    )[0]
    assert "Preview without writing" in results_section
    assert "{operationsWriteReady ? <div" in results_section
    assert "Commit reviewed results" in results_section

    assert "initialTournamentId: string" in panel
    assert "useAuthenticatedAutoLoad(" in panel
    assert '() => loadOps(initialTournamentId, initialDrawId || "")' in panel
    assert "Refresh tournaments" not in panel
    assert "selectTournament(event.target.value)" not in panel
    assert "loadTournaments" not in panel
    assert "Reload selected draw" in panel
    assert ">Load tournaments<" not in panel
    assert "Load ops snapshot" not in panel
    assert "This legacy operations editor cannot publish official matches" in panel
    assert "This legacy editor cannot mint awards" in panel
    assert "awardPodium" not in panel
    assert "publishOfficialMatches" not in panel
    assert "Publish rating game" not in panel
    assert 'data-testid="legacy-ops-human-summary"' in panel
    assert "Raw draw, player, team, game, and podium identifiers are intentionally hidden" in panel
    assert "GenericRowsTable" not in panel
    assert 'placeholder="player id"' not in panel
    assert 'placeholder="optional player id"' not in panel
    assert "Player names must match the club roster" in panel
    assert "Player names or IDs" not in panel
    assert "draw.name || draw.id" not in panel
    assert "gameLabel(game, teamsById, players)" in panel
    assert "importedTeamLabel(match.team_a_ref)" in panel
    assert "importedTeamLabel(match.team_b_ref)" in panel
    assert "{importedTeamLabel(teamRef)}</option>" in panel


def test_legacy_streamlit_tournament_pages_have_no_direct_unarchive_action() -> None:
    for relative_path in (
        "jupr_app/ui/pages/tournaments.py",
        "jupr_app/ui/pages/tournament_ops.py",
    ):
        source = _read(relative_path)
        assert "unarchive_tournament" not in source
        assert 'button("Unarchive Tournament"' not in source
        assert "Direct unarchive is unavailable" in source


def test_setup_operator_labels_never_fall_back_to_registration_ids() -> None:
    panel = _read("apps/web/app/admin/tournaments/setup/TournamentSetupWizardPanel.tsx")

    assert "registration.display_name || registration.email || registration.registration_id" not in panel
    assert 'registration.display_name || registration.email || "Registration needs details"' in panel


def test_tournament_ops_workflow_header_preserves_phase_and_tournament_context() -> None:
    page = _read("apps/web/app/admin/tournaments/ops/TournamentOpsWorkflowPage.tsx")

    assert "tournamentId: string;" in page
    assert "TournamentPhaseNav" in page
    assert 'const phase = workflow === "results" || workflow === "publish" ? "publish" : "live";' in page
    assert "getAdminTournamentStatus" in page
    assert "Tournament Operations are unavailable" in page
    assert "TournamentOpsPanel" in page
    assert "initialTournamentId={tournamentId}" in page
    assert "operationsWriteReady" not in page


def test_registration_import_surfaces_excluded_needs_partner_warning_in_success_result() -> None:
    panel = _read("apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx")

    assert "function warningSuffix(payload: AdminTournamentWriteResponse)" in panel
    assert "const importWarning = warningSuffix(payload);" in panel
    assert 'actionSuccess("Registration teams imported"' in panel
    assert 'imported.${importWarning}`' in panel


def test_legacy_admin_tools_cannot_bypass_canonical_tournament_publish() -> None:
    next_tools = _read("apps/web/app/admin/tools/AdminToolsPanel.tsx")
    streamlit_tools = _read("jupr_app/ui/pages/admin_tools.py")

    assert "Preview missing tournament matches" in next_tools
    assert "Tournament match backfill is diagnostic and read-only" in next_tools
    assert "No write is available here" in next_tools
    assert "applyTournamentMatchBackfill" not in next_tools
    assert "Apply selected tournament matches" not in next_tools

    assert "Tournament match backfill writes are retired" in streamlit_tools
    assert "Backfill Missing Tournament Matches" not in streamlit_tools
    assert "_run_tournament_match_backfill" not in streamlit_tools
    assert "submit_match_batch" not in streamlit_tools


def test_legacy_tournament_score_surfaces_reject_blank_and_lock_non_played_results() -> None:
    live = _read("apps/web/app/admin/tournament-live/TournamentLivePanel.tsx")
    ops = _read("apps/web/app/admin/tournaments/ops/TournamentOpsPanel.tsx")

    for source in (live, ops):
        assert "NON_PLAYED_RESULT_TYPES" in source
        assert "isNonPlayedGame" in source
        assert "scoreA.trim()" in source
        assert "scoreB.trim()" in source
        assert "cannot be changed through ordinary score entry" in source
        assert "not played" in source

    assert "scoreableGames" in ops
    assert 'aria-label="Non-played tournament outcomes"' in ops
    assert "editable && !nonPlayed" in live
    assert "guarded Day Workspace" in live


def test_tournament_home_and_closeout_handle_terminal_publication_without_a_draw() -> None:
    home = _read("apps/web/app/admin/tournaments/tournament/TournamentHomePanel.tsx")
    live = _read("apps/web/app/admin/tournament-live/TournamentLivePanel.tsx")

    assert "officialPublishComplete" in home
    assert 'official_publish.state || "").toLowerCase() === "complete"' in home
    assert "!officialPublishComplete" in home
    assert "Review completed tournament" in home
    assert '["overview", "results", "publish-overview", "closeout", "status"].includes(view) ? payload : null' in live
    assert "setSnapshot(board)" in live
    assert "lifecycle: current.lifecycle" in live
