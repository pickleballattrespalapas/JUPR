from pathlib import Path


PAGE = Path("apps/web/app/clubs/[clubSlug]/league-results/page.tsx")


def test_league_results_page_hydrates_all_deep_link_selectors() -> None:
    source = PAGE.read_text(encoding="utf-8")

    for selector in (
        '"league"',
        '"section"',
        '"week"',
        '"player"',
        '"weekly_min_games"',
        '"print"',
    ):
        assert selector in source
    assert "getClubLeagueResults(" in source
    assert "weeklyMinGames" in source


def test_league_results_page_has_distinct_scopes_and_complete_player_detail() -> None:
    source = PAGE.read_text(encoding="utf-8")

    assert "data.season_highlights" in source
    assert "data.weekly_highlights" in source
    assert "rank_delta" in source
    assert "RecentMatchesTable" in source
    assert "data.recent_matches" in source
    assert "playerCandidates.map" in source
    assert "playerCandidates.slice" not in source
