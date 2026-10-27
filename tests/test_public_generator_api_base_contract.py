from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (WEB / path).read_text(encoding="utf-8")


def test_public_generator_setup_pages_use_the_deployed_api_base() -> None:
    for path in (
        "app/clubs/[clubSlug]/round-robin-generator/page.tsx",
        "app/clubs/[clubSlug]/ladder-generator/page.tsx",
    ):
        source = read(path)
        assert "process.env.JUPR_API_BASE_URL" in source
        assert "process.env.NEXT_PUBLIC_JUPR_API_BASE_URL" in source
        assert "apiBase={serverApiBase()}" in source
        assert 'apiBase="/api"' not in source


def test_public_generator_round_pages_use_the_deployed_api_base() -> None:
    for path in (
        "app/clubs/[clubSlug]/round-robin-generator/sessions/[sessionKey]/rounds/[roundNumber]/page.tsx",
        "app/clubs/[clubSlug]/ladder-generator/sessions/[sessionKey]/rounds/[roundNumber]/page.tsx",
    ):
        source = read(path)
        assert "process.env.JUPR_API_BASE_URL" in source
        assert "process.env.NEXT_PUBLIC_JUPR_API_BASE_URL" in source
        assert "apiBase={apiBase()}" in source
        assert 'apiBase="/api"' not in source


def test_public_generator_client_requests_match_fastapi_routes() -> None:
    workspace = read(
        "app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx"
    )
    roster = read("components/GeneratorRosterSetup.tsx")
    player_search = read("lib/playerSearch.ts")
    runner = read(
        "app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx"
    )

    assert "/play-generators/preview" in workspace
    assert "/play-generators/sessions" in workspace
    assert "fetchAllPublicPlayers<DirectoryPlayer>" in roster
    assert '/clubs/${encodeURIComponent(publicClubSlug)}/players?status=all&sort=name' in roster
    # The shared helper preserves that club/status URL while fetching every
    # page; player-search.cjs exercises a player beyond the first 1,000 rows.
    assert "const pageUrl = new URL(url)" in player_search
    assert 'pageUrl.searchParams.set("limit", "1000")' in player_search
    assert 'pageUrl.searchParams.set("offset", String(rows.length))' in player_search
    assert "rows.push(...players)" in player_search
    assert "if (!page.pagination?.has_more || !players.length) return rows" in player_search
    assert "/play-generators/sessions/" in runner
