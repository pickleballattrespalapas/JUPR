from pathlib import Path


PANEL = Path(
    "apps/web/app/admin/league-manager/print/LeaguePrintoutPanel.tsx"
).read_text(encoding="utf-8")
PAGE = Path(
    "apps/web/app/admin/league-manager/print/page.tsx"
).read_text(encoding="utf-8")


def test_printout_auto_load_is_scoped_to_the_selected_league_and_token() -> None:
    assert 'initialLeague: string;' in PANEL
    assert 'useLatestRequestGuard(`${accessToken}\\u0000${initialLeague}`, clearProtectedState)' in PANEL
    assert 'useAuthenticatedAutoLoad(status.enabled ? `${accessToken}\\u0000${initialLeague}` : "", () => loadDetail(""))' in PANEL
    clear = PANEL.split("function clearProtectedState()", 1)[1].split(
        "async function requestJson", 1
    )[0]
    for statement in (
        "setPrintout(null);",
        'setWeekNum("");',
        "setBusy(false);",
        "setMessage(null);",
    ):
        assert statement in clear


def test_printout_week_changes_reload_without_a_second_league_selector() -> None:
    assert "function selectWeek(selectedWeek: string)" in PANEL
    assert "void loadDetail(selectedWeek);" in PANEL
    assert "encodeURIComponent(initialLeague)" in PANEL
    assert "setPrintout(null);" not in PANEL.split("async function loadDetail", 1)[1].split(
        "function selectWeek", 1
    )[0]
    assert "Select league" not in PANEL
    assert "Refresh leagues" not in PANEL
    assert "Reload printout" in PANEL


def test_printout_page_requires_selected_league_context() -> None:
    assert "searchParams" in PAGE
    assert "readLeagueRouteContext(searchParams)" in PAGE
    assert 'if (!context.leagueId) redirect("/admin/league-manager")' in PAGE
    assert "const leagueName = context.leagueName || context.leagueId" in PAGE
    assert "LeagueManagerNav" in PAGE
    assert "leagueId={context.leagueId}" in PAGE
    assert "leagueName={leagueName}" in PAGE
    assert "initialLeague={leagueName}" in PAGE
