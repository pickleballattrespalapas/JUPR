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
    assert "() => loadOps(initialTournamentId, \"\")" in panel
    assert "Refresh tournaments" not in panel
    assert "selectTournament(event.target.value)" not in panel
    assert "loadTournaments" not in panel
    assert "Reload selected draw" in panel
    assert ">Load tournaments<" not in panel
    assert "Load ops snapshot" not in panel
    for read_only_table in ("Draws", "Teams", "Games", "Podium"):
        assert f">{read_only_table}</h2><GenericRowsTable" in panel


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
