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
    runner = read(
        "app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx"
    )

    assert "/play-generators/preview" in workspace
    assert "/play-generators/sessions" in workspace
    assert "/players?status=active&sort=name&limit=1000" in roster
    assert "/play-generators/sessions/" in runner
