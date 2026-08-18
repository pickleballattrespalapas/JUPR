from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_award_configuration_only_lives_in_draft_settings() -> None:
    guided = _read(
        "apps/web/app/admin/league-manager/GuidedLeagueSettingsEditor.tsx"
    )
    settings = _read(
        "apps/web/app/admin/league-manager/settings/LeagueSettingsPanel.tsx"
    )
    setup = _read(
        "apps/web/app/admin/league-manager/settings/LeagueAwardsSetupPanel.tsx"
    )
    awards = _read(
        "apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx"
    )

    assert "LeagueAwardsSetupPanel" in settings
    assert 'leagueStatus === "draft"' in setup
    assert 'awardPath("/config")' in setup
    assert "Save award setup" in setup
    assert "awards_config" not in guided
    assert 'leagueAwardsPath("/config")' not in awards
    assert "Save award category choices" not in awards
    assert "Live award progress" in awards
    assert "Award setup is managed before the league starts in the Settings tab" in awards


def test_team_configuration_lives_in_settings_and_operations_stay_in_team_tab() -> None:
    settings = _read(
        "apps/web/app/admin/league-manager/settings/LeagueSettingsPanel.tsx"
    )
    setup = _read(
        "apps/web/app/admin/league-manager/settings/TeamLeagueSetupPanel.tsx"
    )
    operations = _read(
        "apps/web/app/admin/league-manager/teams/TeamLeaguesPanel.tsx"
    )

    assert "TeamLeagueSetupPanel" in settings
    for phrase in (
        "Team eligibility",
        "Primary roster size",
        "Maximum alternates",
        "Allow substitutes",
        "Registration open",
        "No playoffs",
    ):
        assert phrase in setup
    assert 'team-leagues/${encodeURIComponent(leagueName)}/settings' in setup
    assert 'teamLeaguePath("/settings")' not in operations
    assert "Team setup is managed in League Settings." in operations
    for phrase in (
        "Teams and standings",
        "Assigned rosters and substitute pool",
        "Schedule and playoffs",
        "Fixtures and results",
        "Safe recovery",
    ):
        assert phrase in operations


def test_public_awards_precede_standings_and_wait_for_api_progress() -> None:
    standings = _read(
        "apps/web/app/clubs/[clubSlug]/leagues/[leagueName]/standings/page.tsx"
    )
    team = _read(
        "apps/web/app/clubs/[clubSlug]/team-leagues/[leagueName]/page.tsx"
    )

    assert "data.award_progress.awards.length" in standings
    assert standings.index("<h2>Awards</h2>") < standings.index("<h2>Current standings</h2>")
    assert "data.award_progress.awards.length" in team
    assert team.index(">Awards<") < team.index(">Standings<")


def test_player_summary_selection_is_immediate_and_singles_hides_partner() -> None:
    page = _read(
        "apps/web/app/clubs/[clubSlug]/leagues/[leagueName]/players/page.tsx"
    )
    selector = _read(
        "apps/web/app/clubs/[clubSlug]/leagues/[leagueName]/players/LeaguePlayerSelect.tsx"
    )

    assert "Open summary" not in page
    assert "router.push" in selector
    assert "onChange" in selector
    assert 'data.league?.match_format === "singles"' in page
    assert "Partner:" in page


def test_print_css_hides_admin_sidebar_and_uses_full_width() -> None:
    print_panel = _read(
        "apps/web/app/admin/league-manager/print/LeaguePrintoutPanel.tsx"
    )
    shell_css = _read("apps/web/components/AdminShell.module.css")

    assert "nav, header, footer, aside, .no-print" in print_panel
    assert "width: 100% !important" in print_panel
    assert "@media print" in shell_css
    assert ".sidebar" in shell_css
    assert "display: none !important" in shell_css
    assert ".shellCollapsed" in shell_css
