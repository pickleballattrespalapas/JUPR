from pathlib import Path


PANEL = Path("apps/web/app/admin/tournament-setup/TournamentSetupPanel.tsx")


def test_tournament_setup_selection_auto_loads_and_retains_manual_reload() -> None:
    source = PANEL.read_text()

    assert "function selectTournament(id: string)" in source
    assert "setSelectedId(id);" in source
    assert "if (id) void loadDetail(id);" in source
    assert "await loadDetail(nextId);" in source
    assert "onChange={(event) => selectTournament(event.target.value)}" in source
    assert "useAuthenticatedAutoLoad(status?.enabled ? accessToken : \"\", loadTournaments)" in source
    assert ">Refresh list</button>" in source
    assert ">Load list</button>" not in source
    assert '"Reload setup"' in source
    assert ">Load setup</button>" not in source


def test_tournament_setup_replacement_load_clears_visible_detail_and_blocks_stale_writes() -> None:
    source = PANEL.read_text()

    assert 'const [loadedDetailId, setLoadedDetailId] = useState("");' in source
    assert 'const [detailLoadingId, setDetailLoadingId] = useState("");' in source
    assert "function clearDetailState()" in source
    selection = source.split("function selectTournament", 1)[1].split(
        "useAuthenticatedAutoLoad", 1
    )[0]
    assert selection.index("clearDetailState();") < selection.index("setSelectedId(id);")
    assert "current setup remains visible" not in source
    assert "previously loaded setup" not in source
    assert "aria-busy={Boolean(detailLoadingId)}" in source
    assert "const detailIsCurrent = Boolean(detail && loadedDetailId === selectedId);" in source
    assert "{detail && detailIsCurrent ? <>" in source
    assert "disabled={!detailIsCurrent}" in source
    assert "const builderReady = detailIsCurrent && builderIssues.length === 0;" in source
    assert "disabled={!impactReview || !builderReady}" in source
    assert "const detailRequest = useLatestRequestGuard(accessToken);" in source
    assert "if (!detailRequest.isCurrent(generation)) return false;" in source


def test_tournament_setup_selector_layout_wraps_for_narrow_viewports() -> None:
    source = PANEL.read_text()

    assert 'gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))"' in source


def test_post_write_success_does_not_hide_a_failed_authoritative_reload() -> None:
    source = PANEL.read_text()

    assert "async function loadDetail(id = selectedId): Promise<boolean>" in source
    detail_loader = source.split(
        "async function loadDetail", 1
    )[1].split("function selectTournament", 1)[0]
    assert detail_loader.index("clearDetailState();") < detail_loader.index(
        "await requestJson<DetailResponse>"
    )
    assert "if (id !== loadedDetailId) clearDetailState();" not in detail_loader
    assert 'setMessage("Choose a tournament first.");' in source
    assert "return false;" in source
    assert source.count("const reloaded = await loadDetail(selectedId);") == 3
    assert source.count("if (reloaded) setMessage(payload.idempotent_replay") == 3
