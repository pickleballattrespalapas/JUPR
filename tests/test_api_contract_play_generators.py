from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "apps" / "web"


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_round_robin_and_ladder_generators_replace_jupr_live_admin_navigation() -> None:
    shell = read("apps/web/components/AdminShell.tsx")
    redirect = read("apps/web/app/admin/jupr-live/page.tsx")

    assert 'label: "Round-Robin Generator"' in shell
    assert 'href: "/admin/round-robin-generator"' in shell
    assert 'label: "Ladder Generator"' in shell
    assert 'href: "/admin/ladder-generator"' in shell
    assert 'label: "JUPR Live"' not in shell
    assert 'redirect("/admin/round-robin-generator")' in redirect


def test_generator_preview_and_round_runner_contracts() -> None:
    workspace = read("apps/web/app/admin/play-generators/GeneratorWorkspace.tsx")
    runner = read("apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx")
    package = read("apps/web/package.json")

    assert "Preview matchups" in workspace
    assert "Download CSV" in workspace
    assert "Download one-sheet PDF" in workspace
    assert 'import("jspdf")' in workspace
    assert "Start session" in workspace
    assert "Only Round 1 is shown" in workspace
    assert '"jspdf"' in package

    assert "Save round scores" in runner
    assert "Skip round" in runner
    assert "Round {roundNumber} results" in runner
    assert "Adaptive roster" in runner
    assert "Add player" in runner
    assert "Remove player" in runner
    assert "Substitute player" in runner
    assert "One round only" in runner
    assert "Rest of session" in runner
    assert "Generate Round" in runner


def test_skip_round_preserves_acknowledgement_and_partial_advance_recovery() -> None:
    runners = (
        read("apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx"),
        read("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx"),
    )

    for runner in runners:
        # The root-owned confirmation snapshots onAcknowledge when it opens.
        # A stable ref must therefore carry the destination produced later by
        # the mutation instead of a stale render closure.
        assert "const skipDestinationRef = useRef" in runner
        assert "skipDestinationRef.current = nextRound" in runner
        assert "const destination = skipDestinationRef.current" in runner

        # Skip and automatic advance are two durable writes. Once the first is
        # confirmed, any error in the second stage is a partial/uncertain result
        # and recovery must retain both exact identities.
        assert "let skipCommitted = false" in runner
        assert "skipCommitted = true" in runner
        assert "skipCommitted || isUncertainRequestError(error)" in runner
        assert "Both exact operation keys are retained" in runner
        assert "request.advanceIdempotencyKey" in runner
        assert "() => executeSkipRound(request)" in runner


def test_generator_backend_routes_and_adaptive_engine_are_installed() -> None:
    main = read("services/api/main.py")
    routes = read("services/api/admin_play_generator_routes.py")
    engine = read("jupr_app/domain/adaptive_play_engine.py")
    service = read("jupr_app/services/admin_play_generator_service.py")

    assert "install_admin_play_generator_routes" in main
    assert "/play-generators/preview" in routes
    assert "/play-generators/sessions" in routes
    assert "/rounds/{round_number}/scores" in routes
    assert "/rounds/{round_number}/skip" in routes
    assert "/sessions/{session_key}/roster" in routes
    assert "/sessions/{session_key}/advance" in routes
    assert "/sessions/{session_key}/publish" in routes

    assert "create_generator_preview" in engine
    assert "generator_kind" in engine
    assert "play_format" in engine
    assert "skip_generator_round" in engine
    assert "mutate_generator_roster" in engine
    assert "substitute_scope" in engine
    assert "schedule_export_rows" in engine

    assert "generator_match_play_format" in service
    assert "payloads_by_format" in service
    assert "match_format=match_format" in service
    assert "Official publication requires every participant" in service
    assert '!= "COMPLETE SESSION"' in routes
    assert '!= "PUBLISH MATCHES"' in routes
    assert "request_payload=_model_payload(payload)" in routes


def test_round_robin_previews_all_rounds_but_ladder_generates_adaptively() -> None:
    engine = read("jupr_app/domain/adaptive_play_engine.py")

    assert 'if kind == "round_robin":' in engine
    assert 'event["rounds"] = [_create_ladder_round(event, 1, ordered_ids)]' in engine
    assert 'if kind == "ladder":' in engine
    assert "_ladder_next_order" in engine
    assert "Save or skip the current round before continuing." in engine


def test_every_play_generator_unsafe_route_is_classified_for_staging() -> None:
    waves = read("scripts/staging_write_waves.py")
    required = (
        '("POST", "/admin/clubs/{club_id}/play-generators/preview")',
        '("POST", "/admin/clubs/{club_id}/play-generators/sessions")',
        '("PATCH", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/scores")',
        '("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/skip")',
        '("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/advance")',
        '("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/roster")',
        '("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/complete")',
        '("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/publish")',
    )
    for route in required:
        assert route in waves
