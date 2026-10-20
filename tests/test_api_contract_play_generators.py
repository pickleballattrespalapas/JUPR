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
    assert "Start ${ratingMode} session" in workspace
    assert "Only Round 1 is shown" in workspace
    assert '"jspdf"' in package

    assert "Save round scores" in runner
    assert "Skip round" in runner
    assert "Round {roundNumber} results" in runner
    assert "Adaptive roster" in runner
    assert "Add player" in runner
    assert "Remove player" in runner
    assert "Substitute player" in runner
    assert "Swap players" in runner
    assert "swapRosterPositions" in runner
    assert "Next round only" in runner
    assert "Rest of session" in runner
    assert "Generate Round" in runner


def test_skip_round_preserves_acknowledgement_and_partial_advance_recovery() -> None:
    admin_runner = read("apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx")
    public_runner = read("apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx")
    runners = (admin_runner, public_runner)

    for runner in runners:
        # The root-owned confirmation snapshots onAcknowledge when it opens.
        # A stable ref must therefore carry the destination produced later by
        # the mutation instead of a stale render closure.
        assert "const skipDestinationRef = useRef" in runner
        assert "skipDestinationRef.current = nextRound" in runner
        assert "const destination = skipDestinationRef.current" in runner

        # Skip and automatic advance are separate writes. Once the first is
        # confirmed, any error in the second stage is a partial/uncertain result
        # and recovery must retain the same request details internally.
        assert "let skipCommitted = false" in runner
        assert "skipCommitted = true" in runner
        assert "skipCommitted || isUncertainRequestError(error)" in runner
        assert "request.skipBody.idempotency_key" in runner
        assert "request.advanceIdempotencyKey" in runner
        assert "() => executeSkipRound(request)" in runner

    assert "Both exact operation keys are retained" in admin_runner
    assert "Both exact operation keys are retained" not in public_runner
    assert '"Update not confirmed"' in public_runner
    assert '"Try again"' in public_runner
    assert "false\n        );" in public_runner


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
    assert "/rounds/{round_number}/reopen" in routes
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
        '("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/rounds/{round_number}/reopen")',
        '("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/advance")',
        '("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/roster")',
        '("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/complete")',
        '("POST", "/admin/clubs/{club_id}/play-generators/sessions/{session_key}/publish")',
    )
    for route in required:
        assert route in waves


def test_sparse_admin_profiles_reach_preview_start_and_durable_request(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from services.api import admin_play_generator_routes as routes
    from test_admin_play_generator_service import FakeSupabase

    db = FakeSupabase()
    db.storage["players"] = [{"id": 21, "club_id": "club", "name": "Linked player"}]
    monkeypatch.setattr(routes, "_resolve_role_or_403", lambda **kwargs: ("staff@example.invalid", "administrator"))
    monkeypatch.setattr(routes, "_require_write_gate", lambda: None)
    recorded = []

    def record_intent(_db, **kwargs):
        recorded.append(kwargs["request_payload"])
        return kwargs["mutate"]()

    monkeypatch.setattr(routes, "run_durable_admin_operation", record_intent)
    app = FastAPI()
    routes.install_admin_play_generator_routes(app, get_supabase_client=lambda: db)
    client = TestClient(app)
    body = {"generator_kind": "round_robin", "play_format": "doubles", "title": "Mixed roster",
            "participant_names": ["Guest A", "Linked player", "Guest B", "Guest C"],
            "participant_player_ids": {"Linked player": 21}, "total_rounds": 3, "court_count": 1}
    preview = client.post("/admin/clubs/club/play-generators/preview", json=body)
    assert preview.status_code == 200, preview.text
    assert [p.get("player_id") for p in preview.json()["preview"]["participants"]] == [None, 21, None, None]
    started = client.post("/admin/clubs/club/play-generators/sessions", json={**body,
        "preview_fingerprint": preview.json()["preview"]["previewFingerprint"],
        "expected_version": "new", "idempotency_key": "sparse-links-start-1"})
    assert started.status_code == 200, started.text
    assert [p.get("player_id") for p in started.json()["session"]["event"]["participants"]] == [None, 21, None, None]
    assert recorded[0]["participant_player_ids"] == {"Linked player": 21}


def test_new_optional_profile_map_keeps_legacy_start_intent_fingerprint():
    from services.api.admin_play_generator_routes import GeneratorStartRequest, _model_payload
    body = {"generator_kind": "round_robin", "play_format": "doubles", "participant_names": ["A", "B", "C", "D"],
            "player_ids": [1, 2, 3, 4], "expected_version": "new", "idempotency_key": "retained-old-start"}
    model = GeneratorStartRequest(**body)
    legacy_payload = model.model_dump()
    legacy_payload.pop("participant_player_ids")
    assert _model_payload(model) == legacy_payload
    assert _model_payload(GeneratorStartRequest(**body, participant_player_ids={})) == legacy_payload
    linked = GeneratorStartRequest(**{**body, "player_ids": [], "participant_player_ids": {"A": 1}})
    assert _model_payload(linked)["participant_player_ids"] == {"A": 1}
