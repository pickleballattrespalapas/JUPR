from pathlib import Path


PANEL = Path("apps/web/app/admin/league-manager/LeagueManagerPanel.tsx")


def test_league_manager_open_dropdown_has_counted_lifecycle_filters() -> None:
    panel = PANEL.read_text(encoding="utf-8")

    assert 'useState<LeagueStatusFilter>("active")' in panel
    assert 'aria-label="Filter leagues by status"' in panel
    assert "LEAGUE_STATUS_FILTERS.map" in panel
    assert "statusCounts[filter.key]" in panel
    assert "filterLeaguesByStatus(leagues, leagueFilter)" in panel
    assert "filteredLeagues.map" in panel
    assert "Choose another status filter." in panel


def test_status_filter_defines_all_requested_lifecycle_groups() -> None:
    helper = Path("apps/web/lib/leagueStatusFilter.ts").read_text(encoding="utf-8")

    for key in ("active", "draft", "paused", "ended", "archived", "all"):
        assert f'key: "{key}"' in helper
    assert 'normalized === "inactive"' in helper
    assert 'return "draft"' in helper
