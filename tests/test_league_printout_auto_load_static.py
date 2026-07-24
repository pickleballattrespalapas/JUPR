from pathlib import Path


PANEL = Path(
    "apps/web/app/admin/league-manager/print/LeaguePrintoutPanel.tsx"
).read_text(encoding="utf-8")


def test_printout_auto_load_is_token_scoped_and_logout_clears_workspace() -> None:
    assert "useAuthenticatedAutoLoad(status.enabled ? accessToken : \"\", loadLeagues)" in PANEL
    assert "const listRequest = useLatestRequestGuard(accessToken, resetWorkspace);" in PANEL
    assert "const detailRequest = useLatestRequestGuard(accessToken);" in PANEL
    reset = PANEL.split("function resetWorkspace()", 1)[1].split(
        "const listRequest", 1
    )[0]
    for statement in (
        "setLeagues([]);",
        'setLeagueName("");',
        "setPrintout(null);",
        'setWeekNum("");',
        "setLoadingLeagues(false);",
        "setLoadingPrintout(false);",
    ):
        assert statement in reset


def test_printout_selector_changes_auto_load_and_clear_stale_results() -> None:
    assert "function selectLeague(selectedLeague: string)" in PANEL
    assert 'void loadDetail(selectedLeague, "");' in PANEL
    assert "function selectWeek(selectedWeek: string)" in PANEL
    assert "void loadDetail(leagueName, selectedWeek);" in PANEL
    load_detail = PANEL.split("async function loadDetail", 1)[1].split(
        "function selectLeague", 1
    )[0]
    assert "setPrintout(null);" in load_detail
    assert "if (!detailRequest.isCurrent(generation)) return;" in load_detail


def test_printout_refresh_preserves_valid_selection_and_week() -> None:
    load_leagues = PANEL.split("async function loadLeagues", 1)[1].split(
        "async function loadDetail", 1
    )[0]
    assert "const selectedLeagueBeforeRefresh = leagueName;" in load_leagues
    assert "const selectedWeekBeforeRefresh = weekNum;" in load_leagues
    assert "names.includes(selectedLeagueBeforeRefresh)" in load_leagues
    assert "await loadDetail(selectedLeague, selectedWeek);" in load_leagues
    assert '"Refresh leagues"' in PANEL
    assert ">Load leagues</button>" not in PANEL
