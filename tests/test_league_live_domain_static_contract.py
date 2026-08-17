from pathlib import Path


PANEL = Path("apps/web/app/admin/league-manager/live/LeagueLiveRoundPanel.tsx")
PAGE = Path("apps/web/app/admin/league-manager/live/page.tsx")
ROUTES = Path("services/api/admin_league_manager_routes.py")
SERVICE = Path("jupr_app/services/admin_league_live_service.py")


def test_browser_does_not_own_league_live_movement_math() -> None:
    panel = PANEL.read_text(encoding="utf-8")
    assert "buildMovementPlan" not in panel
    assert "Round Wins" not in panel
    assert "Client-supplied movement" not in panel
    assert "/plan`" in panel
    assert "expected_operation_key" in panel
    assert "movement_overrides" in panel
    assert "The browser displays plans but never ranks players" in panel


def test_league_selection_auto_loads_roster_without_showing_stale_detail() -> None:
    panel = PANEL.read_text(encoding="utf-8")
    assert "selectLeague(event.target.value)" in panel
    assert "void loadLeagueDetail(selectedLeague)" in panel
    assert "useAuthenticatedAutoLoad(" in panel
    assert "Refresh leagues" in panel
    assert ">Load leagues<" not in panel
    assert "Reload roster" in panel
    assert ">Load roster<" not in panel
    assert "const suggestion = await fetchRosterSuggestion(payload);" in panel
    assert "clearPersistedSessionBinding();" in panel
    assert "Session writes remain unavailable until the replacement roster is ready." in panel


def test_league_live_binds_session_to_selected_league_and_pauses_edits_while_busy() -> None:
    panel = PANEL.read_text(encoding="utf-8")
    assert "loadedSessionId === sessionId" in panel
    assert "sessionLeagueName === leagueName" in panel
    assert "sessionLeagueName === loadedLeagueName" in panel
    assert "clearPersistedSessionBinding()" in panel
    assert "requireCurrentSession(" in panel
    assert panel.count("const requestedSessionId = loadedSessionId;") >= 7
    assert panel.count("encodeURIComponent(requestedSessionId)") >= 7
    assert "setDetail(null);" in panel
    assert "setRosterSuggestion(null);" in panel
    assert "detail.league.league_name !== leagueName" in panel
    assert 'disabled={busy || !sessionIsCurrentLeague}' in panel
    assert "disabled={busy}" in panel


def test_league_live_court_and_bench_controls_are_phone_responsive() -> None:
    panel = PANEL.read_text(encoding="utf-8")
    assert "data-responsive-court-grid" in panel
    assert "data-responsive-bench-controls" in panel
    assert 'minmax(min(100%, 180px), 1fr)' in panel
    assert 'minmax(min(100%, 240px), 1fr)' in panel
    assert 'gridTemplateColumns: "120px 180px 1fr auto"' not in panel


def test_page_fetches_python_domain_readiness_and_fails_closed() -> None:
    page = PAGE.read_text(encoding="utf-8")
    panel = PANEL.read_text(encoding="utf-8")
    assert "getAdminLeagueLiveStatus" in page
    assert "liveDomainStatus={liveDomainStatus}" in page
    assert "!liveDomainStatus.enabled" in panel
    assert "Live round scoring remains unavailable in this build." in panel


def test_fastapi_contract_requires_service_role_stale_guard_and_operation_key() -> None:
    routes = ROUTES.read_text(encoding="utf-8")
    service = SERVICE.read_text(encoding="utf-8")
    assert "SUPABASE_SERVICE_ROLE_KEY" in routes
    assert "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN" in service
    assert "expected_updated_at" in routes
    assert "expected_operation_key" in routes
    assert "Client-supplied movement is not accepted" in service
    assert "idempotent_replay" in service
