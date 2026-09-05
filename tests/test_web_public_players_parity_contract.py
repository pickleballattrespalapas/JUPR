from pathlib import Path


DIRECTORY = Path("apps/web/app/clubs/[clubSlug]/players/page.tsx")
PROFILE = Path("apps/web/app/clubs/[clubSlug]/players/[playerId]/page.tsx")
API = Path("apps/web/lib/api.ts")
E2E = Path("apps/web/e2e/players.staging.spec.ts")


def test_directory_has_visible_search_active_default_privacy_states_and_stable_links():
    source = DIRECTORY.read_text(encoding="utf-8")

    assert 'return "active";' in source
    assert 'name="q"' in source
    assert 'data-testid="players-search-form"' in source
    assert "players-status-${item}" in source
    assert 'data-testid="players-row"' in source
    assert 'data-testid="players-filter-empty-state"' in source
    assert 'data-testid="players-error-state"' in source
    assert "Share ${player.name}" in source
    assert "Player profiles are unavailable right now" in source
    assert "stable row link" not in source
    assert "No private player data was exposed" not in source
    assert "{error}" not in source


def test_profile_renders_every_parity_projection_and_format_label():
    source = PROFILE.read_text(encoding="utf-8")

    for test_id in (
        "player-public-identity",
        "player-rating-trend",
        "player-format-row",
        "player-overview",
        "player-league-positions",
        "player-trophies",
        "player-badges",
        "player-trophy",
        "player-badge",
        "player-best-partner",
        "player-rival",
        "player-social",
        "player-match-history",
        "player-history-all",
    ):
        assert test_id in source
    assert "Request player updates" in source
    assert "Request an alias or privacy review" in source
    assert "Singles" in source
    assert "Doubles" in source
    assert "Contact information and other private details are not shown" in source
    assert 'data-rating-series={item.format}' in source
    assert 'data-rating-point={item.format}' in source
    assert "Each format has its own line" in source
    assert "publicSocialEventTypeLabel(event.event_type)" in source
    assert "publicBadgeRarityLabel(badge.rarity)" in source
    assert "coordinates.filter(({ point }) => point.match_format === definition.format)" in source
    assert "Leaderboard snapshot" not in source
    assert "Badge codex" not in source
    assert "Major honors only" in source
    assert "{error}" not in source


def test_player_api_client_forwards_explicit_directory_and_history_filters():
    source = API.read_text(encoding="utf-8")
    public_player_type = source.split("export type PublicPlayer =", 1)[1].split("export type PublicLeagueRating =", 1)[0]

    for query_key in ("q", "status", "sort", "limit", "offset", "recent_limit", "history_limit"):
        assert f'params.set("{query_key}"' in source
    for private_field in ("email", "phone", "legal_name", "normalized_name", "unsubscribe_token"):
        assert private_field not in public_player_type


def test_players_routes_have_loading_errors_and_route_specific_browser_evidence():
    directory_loading = DIRECTORY.with_name("loading.tsx").read_text(encoding="utf-8")
    directory_error = DIRECTORY.with_name("error.tsx").read_text(encoding="utf-8")
    profile_loading = PROFILE.with_name("loading.tsx").read_text(encoding="utf-8")
    profile_error = PROFILE.with_name("error.tsx").read_text(encoding="utf-8")
    e2e = E2E.read_text(encoding="utf-8")

    assert "players-loading-state" in directory_loading
    assert "players-route-error-state" in directory_error
    assert "player-profile-loading-state" in profile_loading
    assert "player-profile-route-error-state" in profile_error
    assert "players-status-active" in e2e
    assert "players-search-form" in e2e
    assert 'getByRole("searchbox", { name: "Find player" })' in e2e
    assert 'getByRole("textbox", { name: "Find player" })' not in e2e
    assert "player-public-identity" in e2e
    assert "player-history-all" in e2e
    assert "player-profile-error-state" in e2e
    assert "route.fulfill" not in e2e
