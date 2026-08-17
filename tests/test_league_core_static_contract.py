from pathlib import Path


def test_top_players_page_uses_authenticated_python_export() -> None:
    page = Path("apps/web/app/admin/top-players-printable/page.tsx").read_text(encoding="utf-8")
    panel = Path("apps/web/app/admin/top-players-printable/TopPlayersPrintablePanel.tsx").read_text(encoding="utf-8")
    tools = Path("apps/web/app/admin/tools/page.tsx").read_text(encoding="utf-8")

    assert "getClubPlayers" not in page
    assert "league-manager/top-players-printable" in panel
    assert "Authorization" in panel
    assert "previous UTC calendar month" in panel
    assert "@media print" in panel
    assert "Previous-month Top 50" in tools
    assert 'href="/admin/top-players-printable"' in tools


def test_league_printout_renders_true_leaders_and_print_contract() -> None:
    panel = Path("apps/web/app/admin/league-manager/print/LeaguePrintoutPanel.tsx").read_text(encoding="utf-8")
    page = Path("apps/web/app/admin/league-manager/print/page.tsx").read_text(encoding="utf-8")
    home = Path("apps/web/app/admin/league-manager/league/LeagueHomePanel.tsx").read_text(encoding="utf-8")
    service = Path("jupr_app/services/admin_league_print_service.py").read_text(encoding="utf-8")

    assert "/printout" in panel
    assert "initialLeague" in panel
    assert "encodeURIComponent(initialLeague)" in panel
    assert 'useAuthenticatedAutoLoad(status.enabled ? `${accessToken}\\u0000${initialLeague}` : "", () => loadDetail(""))' in panel
    assert "Select league" not in panel
    assert "Refresh leagues" not in panel
    assert "Reload printout" in panel
    assert "Load selected" not in panel
    assert 'disabled={busy || !hasPrintableData}' in panel
    assert "Nothing to print yet" in panel
    assert "Team standings" in panel
    assert "Shared substitute pool" in panel
    assert "printout.printable_sections.team_rosters" in panel
    assert "Weekly leaders" in panel
    assert "Season leaders" in panel
    assert "data-print-surface" in panel
    assert "@media print" in panel
    assert 'redirect("/admin/league-manager")' in page
    assert "League night printout" in home
    assert "calculate_hybrid_elo" in service
    assert "preview_admin_league_awards" in service


def test_league_core_mutations_require_server_only_supabase_key() -> None:
    routes = Path("services/api/admin_league_manager_routes.py").read_text(encoding="utf-8")
    roster = Path("jupr_app/services/admin_league_manager_roster_service.py").read_text(encoding="utf-8")
    settings = Path("jupr_app/services/admin_league_manager_update_service.py").read_text(encoding="utf-8")
    lifecycle = Path("jupr_app/services/admin_league_manager_lifecycle_service.py").read_text(encoding="utf-8")

    assert "SUPABASE_SERVICE_ROLE_KEY" in routes
    assert routes.count("_require_league_manager_service_role()") >= 6
    assert "update_admin_league_manager_roster_batch(" in roster
    assert "CONFIRM_ROSTER_BATCH" in roster
    assert "player_ids=[pid]" in roster
    assert "_rollback_roster_membership" not in roster
    assert "write_admin_activity_log" not in roster
    assert "validate_admin_league_manager_lifecycle_state" in settings
    assert "validate_admin_league_manager_lifecycle_state" in lifecycle


def test_league_browser_evidence_is_registered() -> None:
    smoke = Path("apps/web/e2e/staging.smoke.spec.ts").read_text(encoding="utf-8")
    evidence = Path("apps/web/e2e/league-core.staging.spec.ts").read_text(encoding="utf-8")

    assert 'path: "/admin/league-manager/print"' in smoke
    assert 'path: "/admin/top-players-printable"' in smoke
    assert "JUPR_STAGING_ADMIN_ACCESS_TOKEN" in evidence
    assert "Python-authoritative leaders" in evidence
