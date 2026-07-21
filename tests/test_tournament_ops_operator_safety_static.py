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


def test_tournament_ops_read_only_mode_hides_post_backed_controls_but_keeps_snapshots() -> None:
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

    assert "Load tournaments" in panel
    assert "Load ops snapshot" in panel
    for read_only_table in ("Draws", "Teams", "Games", "Podium"):
        assert f">{read_only_table}</h2><GenericRowsTable" in panel


def test_tournament_ops_workflow_header_reports_ops_specific_mode() -> None:
    page = _read("apps/web/app/admin/tournaments/ops/TournamentOpsWorkflowPage.tsx")

    assert "Tournament Ops status is temporarily unavailable" in page
    assert "Tournament Ops mode" in page
    assert "Operations runtime" in page
    assert "Guarded writes ready" in page
    assert "Read-only" in page
    assert "data.operations_runtime?.operations_mutations_enabled" in page
