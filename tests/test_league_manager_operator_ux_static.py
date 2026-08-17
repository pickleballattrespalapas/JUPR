from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_roster_rating_input_is_an_add_player_value_not_a_filter() -> None:
    source = _read("apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx")

    assert "Starting JUPR or Elo for newly added players" in source
    assert "this is not a roster filter" in source
    assert 'rosterMutable && action === "activate"' in source
    assert "This roster is available for review only while the league is" in source


def test_awards_and_live_panels_do_not_expose_deployment_instructions() -> None:
    awards = _read("apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx")
    live = _read("apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx")

    assert "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE" not in awards
    assert "server-only service-role key" not in awards
    assert "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN" not in live
    assert "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT" not in live
    assert "Award changes are unavailable in this build." in awards
    assert "League Live is not available yet" in live


def test_empty_award_evidence_cannot_advance_the_workflow() -> None:
    source = _read("apps/web/app/admin/league-manager/awards/LeagueAwardsPanel.tsx")

    assert "const hasMeasurableResults" in source
    assert "disabled={!writeReady || !hasMeasurableResults}" in source
    assert "disabled={busy || !writeReady || !hasMeasurableResults}" in source


def test_guided_settings_keep_saved_configuration_visible_after_draft() -> None:
    source = _read(
        "apps/web/app/admin/league-manager/GuidedLeagueSettingsEditor.tsx"
    )

    assert 'fieldset disabled={!isDraft} aria-label="League configuration"' in source
    assert "its complete saved configuration is shown below" in source
    assert "Saved configuration · read-only" in source
    assert "structured Streamlit overview" not in source
    assert "disabled={!canWrite || !hasChanges}" in source
    assert "disabled={saving || !hasChanges}" in source
    assert "Choose a week count or an end date" in source
    assert "Weeks (or use an end date)" in source
    assert 'color: localMessageIsError ? "#b91c1c" : "#166534"' in source


def test_roster_filters_and_actions_open_in_the_same_mode() -> None:
    source = _read(
        "apps/web/app/admin/league-manager/roster/LeagueRosterPanel.tsx"
    )

    assert 'useState<RosterFilter>("not_in_league")' in source
    assert 'nextFilter === "not_in_league"' in source
    assert 'setAction("activate")' in source
    assert 'nextFilter === "in_league"' in source
    assert 'setAction("deactivate")' in source
    assert "Eligible to add" in source
    assert "Current members" in source
    assert 'action === "activate" ? !row.in_league : row.in_league' in source
    assert "disabled={!rowSelectable(row)}" in source
    service = _read("jupr_app/services/admin_league_manager_service.py")
    roster_service = _read(
        "jupr_app/services/admin_league_manager_roster_service.py"
    )
    batch_service = _read(
        "jupr_app/services/admin_league_manager_roster_batch_service.py"
    )
    assert '"roster_mutable": status in {"draft", "active"}' in service
    assert "update_admin_league_manager_roster_batch(" in roster_service
    assert "_rollback_roster_membership" not in roster_service
    assert '"admin_apply_league_roster_batch_atomic_v2"' in batch_service
    roster_guard = _read(
        "supabase/migrations/20261104000000_historic_singles_league_rating_backfill.sql"
    )
    assert "v_league_status not in ('draft', 'active')" in roster_guard
    assert "if v_has_operation then" in roster_guard


def test_selected_awards_scope_keeps_one_clear_recovery_control() -> None:
    source = _read(
        "apps/web/app/admin/league-manager/SelectedLeaguePanelScope.tsx"
    )

    assert 'headingText === "Select and recover league"' in source
    assert 'buttonText === "Refresh leagues"' in source
    assert 'buttonText === "Retry saved state"' in source
    assert 'button.textContent = "Reload saved awards"' in source


def test_selected_league_navigation_scrolls_in_one_row() -> None:
    source = _read(
        "apps/web/app/admin/league-manager/LeagueManagerNav.tsx"
    )

    assert 'flexWrap: "nowrap"' in source
    assert 'overflowX: "auto"' in source
    assert 'flex: "0 0 auto"' in source


def test_league_home_guidance_follows_the_league_lifecycle() -> None:
    source = _read(
        "apps/web/app/admin/league-manager/league/LeagueHomePanel.tsx"
    )

    assert 'if (status === "paused")' in source
    assert 'if (status === "ended")' in source
    assert 'if (status === "archived")' in source
    assert "new rounds stay paused" in source
    assert "Review the final league roster" in source
    assert "Review the archived league configuration" in source
    assert "League setup needs attention before play resumes" in source
    assert 'startsWith("league lifecycle state is inconsistent")' in source
    assert "&& !hasLifecycleIntegrityError" in source
    assert 'role={setupNotesAreHistorical ? undefined : "alert"}' in source
    assert "no lifecycle action is required" in source


def test_league_live_uses_operator_language_instead_of_runtime_jargon() -> None:
    source = _read(
        "apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx"
    )

    for old_copy in (
        "Python suggested",
        "Python roster and bench suggestion",
        "Preview Python movement",
        "Python-authoritative plan",
        "FastAPI/Python publishes",
        "Python / FastAPI",
    ):
        assert old_copy not in source

    assert "Roster and bench suggestion" in source
    assert "Movement planning:" in source
    assert "Preview movement" in source
    assert "League Live publishes the whole round" in source
    assert "function leagueLiveOperatorMessage" in source
    assert ".replace(/\\bPython\\b/gi, \"League Live\")" in source
    assert ".replace(/\\bFastAPI\\b/gi, \"the League Live service\")" in source
    assert "leagueLiveOperatorMessage(operation.error_text" in source


def test_league_print_styles_are_compact_without_forced_blank_pages() -> None:
    admin = _read(
        "apps/web/app/admin/league-manager/print/LeaguePrintoutPanel.tsx"
    )
    public_results = _read(
        "apps/web/app/clubs/[clubSlug]/league-results/page.tsx"
    )
    public_standings = _read(
        "apps/web/app/clubs/[clubSlug]/leagues/[leagueName]/standings/page.tsx"
    )

    assert "print-break-before" not in admin
    assert "page-break-before: always" not in admin
    assert ".print-section { padding: 3mm !important" in admin
    assert "@page { size: auto; margin: 8mm; }" in admin
    assert 'pageBreakInside: "avoid"' not in public_results
    assert "article, table, h1, h2, h3 { break-inside: avoid; }" not in public_results
    assert "article { padding: 3mm !important; }" in public_results
    assert "article { padding: 3mm !important; break-inside: avoid" in public_standings
