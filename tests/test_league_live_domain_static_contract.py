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


def test_page_fetches_python_domain_readiness_and_fails_closed() -> None:
    page = PAGE.read_text(encoding="utf-8")
    panel = PANEL.read_text(encoding="utf-8")
    assert "getAdminLeagueLiveStatus" in page
    assert "liveDomainStatus={liveDomainStatus}" in page
    assert "!liveDomainStatus.enabled" in panel
    assert "Streamlit League Manager" in panel


def test_fastapi_contract_requires_service_role_stale_guard_and_operation_key() -> None:
    routes = ROUTES.read_text(encoding="utf-8")
    service = SERVICE.read_text(encoding="utf-8")
    assert "SUPABASE_SERVICE_ROLE_KEY" in routes
    assert "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN" in service
    assert "expected_updated_at" in routes
    assert "expected_operation_key" in routes
    assert "Client-supplied movement is not accepted" in service
    assert "idempotent_replay" in service
