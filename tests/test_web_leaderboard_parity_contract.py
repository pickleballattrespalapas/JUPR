from pathlib import Path


PAGE = Path("apps/web/app/clubs/[clubSlug]/leaderboards/page.tsx")
API = Path("apps/web/lib/api.ts")
E2E = Path("apps/web/e2e/leaderboards.staging.spec.ts")


def test_leaderboard_page_has_parity_controls_and_states():
    source = PAGE.read_text(encoding="utf-8")

    assert 'return "active";' in source
    assert 'name="q"' in source
    assert 'data-testid="leaderboard-player-snapshot"' in source
    assert 'data-testid="leaderboard-pagination"' in source
    assert 'data-testid="leaderboard-empty-state"' in source
    assert 'data-testid="leaderboard-filter-empty-state"' in source
    assert 'data-testid="leaderboard-error-state"' in source
    for label in ("Gain", "Gap", "Qualification", "Badges"):
        assert f">{label}<" in source


def test_leaderboard_api_client_forwards_only_explicit_filters():
    source = API.read_text(encoding="utf-8")

    for query_key in ("league_name", "status", "q", "sort", "player_id", "limit", "offset"):
        assert f'params.set("{query_key}"' in source
    assert "subscription_token" not in source


def test_leaderboard_route_has_loading_error_and_route_specific_browser_evidence():
    loading = PAGE.with_name("loading.tsx").read_text(encoding="utf-8")
    error = PAGE.with_name("error.tsx").read_text(encoding="utf-8")
    e2e = E2E.read_text(encoding="utf-8")

    assert 'data-testid="leaderboard-loading-state"' in loading
    assert 'data-testid="leaderboard-route-error-state"' in error
    assert "leaderboard-status-active" in e2e
    assert "leaderboard-player-snapshot" in e2e
    assert "leaderboard-filter-empty-state" in e2e
    assert "leaderboard-error-state" in e2e
