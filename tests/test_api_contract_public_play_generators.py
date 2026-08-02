from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_public_play_generator_routes_and_safety_are_installed() -> None:
    main = read("services/api/main.py")
    routes = read("services/api/public_play_generator_routes.py")
    waves = read("scripts/staging_write_waves.py")
    assert "install_public_play_generator_routes" in main
    for route in (
        "/play-generators/preview",
        "/play-generators/sessions",
        "/rounds/{round_number}/scores",
        "/rounds/{round_number}/skip",
        "/sessions/{session_key}/advance",
        "/sessions/{session_key}/roster",
        "/sessions/{session_key}/complete",
    ):
        assert route in routes
    for route in (
        '("POST", "/clubs/{club_slug}/play-generators/preview")',
        '("POST", "/clubs/{club_slug}/play-generators/sessions")',
        '("PATCH", "/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/scores")',
        '("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/rounds/{round_number}/skip")',
        '("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/advance")',
        '("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/roster")',
        '("POST", "/clubs/{club_slug}/play-generators/sessions/{session_key}/complete")',
    ):
        assert route in waves


def test_public_modules_mirror_generator_functionality_without_official_publish() -> None:
    workspace = read("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx")
    runner = read("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx")
    header = read("apps/web/components/PublicSiteHeader.tsx")
    live = read("apps/web/app/clubs/[clubSlug]/live/page.tsx")
    assert 'label: "Play"' in header
    assert "Round-Robin Generator" in read("apps/web/app/clubs/[clubSlug]/play/page.tsx")
    assert "Ladder Generator" in read("apps/web/app/clubs/[clubSlug]/play/page.tsx")
    assert "Download CSV" in workspace
    assert "Download one-sheet PDF" in workspace
    assert "Preview matchups" in workspace
    assert "Start session" in workspace
    assert "Save round scores" in runner
    assert "Skip round" in runner
    assert "Adaptive roster" in runner
    assert "Substitute player" in runner
    assert "View-only link" in runner
    assert "Public generator sessions are unrated" in runner
    assert "Publish official matches" not in runner
    assert "redirect(`/clubs/${params.clubSlug}/play`)" in live


def test_public_service_uses_edit_tokens_and_durable_operation_ledger() -> None:
    service = read("jupr_app/services/public_play_generator_service.py")
    assert "begin_public_live_operation" in service
    assert "edit_token_matches" in service
    assert "hash_edit_token" in service
    assert "version" in service
    assert "create_generator_preview" in service
    assert "save_generator_round" in service
    assert "skip_generator_round" in service
    assert "mutate_generator_roster" in service
    assert "advance_generator_event" in service
    assert "public_play_generator" in service
