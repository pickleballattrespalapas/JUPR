from pathlib import Path


PANEL = Path("apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx")


def test_tournament_setup_selection_auto_loads_and_retains_manual_reload() -> None:
    source = PANEL.read_text()

    assert "function selectTournament(id: string)" in source
    assert "setSelectedId(id);" in source
    assert "if (id) void loadDetail(id);" in source
    assert "if (nextId) await loadDetail(nextId);" in source
    assert "onChange={(event) => selectTournament(event.target.value)}" in source
    assert '"Reload setup"' in source
    assert ">Load setup</button>" not in source


def test_tournament_setup_replacement_load_preserves_visible_detail_and_blocks_stale_writes() -> None:
    source = PANEL.read_text()

    assert 'const [loadedDetailId, setLoadedDetailId] = useState("");' in source
    assert 'const [detailLoadingId, setDetailLoadingId] = useState("");' in source
    assert "The current setup remains visible until its replacement is ready." in source
    assert "Showing the previously loaded setup until the selected tournament is ready." in source
    assert "aria-busy={Boolean(detailLoadingId)}" in source
    assert "const detailIsCurrent = Boolean(detail && loadedDetailId === selectedId);" in source
    assert "disabled={!detailIsCurrent}" in source
    assert "disabled={!impactReview || !detailIsCurrent}" in source


def test_tournament_setup_selector_layout_wraps_for_narrow_viewports() -> None:
    source = PANEL.read_text()

    assert 'gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))"' in source


def test_post_write_success_does_not_hide_a_failed_authoritative_reload() -> None:
    source = PANEL.read_text()

    assert "async function loadDetail(id = selectedId): Promise<boolean>" in source
    assert 'if (!id) { setMessage("Choose a tournament first."); return false; }' in source
    assert source.count("const reloaded = await loadDetail(selectedId);") == 2
    assert source.count("if (reloaded) setMessage(payload.idempotent_replay") == 2
