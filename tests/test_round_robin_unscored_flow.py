from pathlib import Path

import pytest

from jupr_app.domain.adaptive_play_engine import (
    advance_generator_event,
    create_generator_preview,
    generator_event_standings,
    history_before_round,
    mark_generator_round_played,
    save_generator_round,
    start_generator_event,
)

ROOT = Path(__file__).resolve().parents[1]


def _matches(round_row: dict) -> list[dict]:
    if round_row.get("matches"):
        return list(round_row.get("matches") or [])
    return [
        match
        for court in round_row.get("courts") or []
        for match in court.get("matches") or []
    ]


def _score_round(event: dict, round_number: int) -> dict:
    round_row = next(row for row in event["rounds"] if row["number"] == round_number)
    return save_generator_round(
        event,
        round_number=round_number,
        scores=[
            {"match_id": match["id"], "score_a": 11, "score_b": 7}
            for match in _matches(round_row)
        ],
    )


def _function_block(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def test_scoring_mode_changes_preview_fingerprint_and_defaults_to_scored() -> None:
    common = dict(
        generator_kind="round_robin",
        play_format="singles",
        title="Fingerprint",
        participant_names=["A", "B", "C", "D"],
        total_rounds=2,
        court_count=2,
    )
    scored = create_generator_preview(**common)
    unscored = create_generator_preview(**common, scoring_mode="unscored")
    assert scored["scoringMode"] == "scored"
    assert unscored["scoringMode"] == "unscored"
    assert scored["previewFingerprint"] != unscored["previewFingerprint"]


def test_scored_round_robin_lifecycle_saves_standings_and_completes() -> None:
    event = start_generator_event(
        create_generator_preview(
            generator_kind="round_robin",
            play_format="singles",
            title="Scored lifecycle",
            participant_names=["A", "B", "C", "D"],
            total_rounds=2,
            court_count=2,
        )
    )
    event = _score_round(event, 1)
    assert event["rounds"][0]["status"] == "saved"
    assert any(row["matches"] > 0 for row in generator_event_standings(event))
    event = advance_generator_event(event)
    assert event["currentRoundNumber"] == 2
    assert event["rounds"][1]["status"] == "active"
    event = _score_round(event, 2)
    event = advance_generator_event(event)
    assert event["status"] == "completed"


def test_unscored_round_played_is_idempotent_and_preserves_history() -> None:
    event = start_generator_event(
        create_generator_preview(
            generator_kind="round_robin",
            play_format="singles",
            title="Unscored lifecycle",
            participant_names=["A", "B", "C"],
            total_rounds=2,
            court_count=1,
            scoring_mode="unscored",
        )
    )
    with pytest.raises(ValueError, match="Round Played"):
        save_generator_round(event, round_number=1, scores=[])
    played = mark_generator_round_played(event, round_number=1)
    replay = mark_generator_round_played(played, round_number=1)
    assert replay == played
    assert played["rounds"][0]["status"] == "played"
    assert sum(history_before_round(played, 2)["games"].values()) == 2
    event = advance_generator_event(played)
    assert event["currentRoundNumber"] == 2
    event = mark_generator_round_played(event, round_number=2)
    event = advance_generator_event(event)
    assert event["status"] == "completed"
    assert generator_event_standings(event) == []


def test_ladder_rejects_unscored_mode() -> None:
    with pytest.raises(ValueError, match="requires scored rounds"):
        create_generator_preview(
            generator_kind="ladder",
            play_format="doubles",
            title="Bad ladder",
            participant_names=["A", "B", "C", "D"],
            total_rounds=3,
            court_count=1,
            scoring_mode="unscored",
        )


def test_round_played_api_is_one_durable_navigation_action() -> None:
    admin_routes = (ROOT / "services/api/admin_play_generator_routes.py").read_text()
    public_routes = (ROOT / "services/api/public_play_generator_routes.py").read_text()
    admin_service = (ROOT / "jupr_app/services/admin_play_generator_service.py").read_text()
    public_service = (ROOT / "jupr_app/services/public_play_generator_service.py").read_text()

    admin_block = _function_block(
        admin_service,
        "def mark_play_generator_round_played(",
        "def skip_play_generator_round(",
    )
    public_block = _function_block(
        public_service,
        "def mark_public_play_generator_round_played(",
        "def skip_public_play_generator_round(",
    )
    for block in (admin_block, public_block):
        assert "mark_generator_round_played" in block
        assert "advance_generator_event" in block

    assert "run_durable_admin_operation" in admin_routes
    assert 'operation_type="mark_round_played"' in admin_routes
    assert "idempotency_key=payload.idempotency_key" in admin_routes
    assert 'action="played"' in public_service
    assert "idempotency_key=idempotency_key" in public_service
    assert "/played" in admin_routes
    assert "/played" in public_routes
    assert "scoring_mode" in admin_routes
    assert "scoring_mode" in public_routes
    assert (
        "Official publishing is unavailable for unscored Round-Robin sessions."
        in admin_service
    )


def test_public_and_staff_component_navigation_contracts() -> None:
    admin_setup = (
        ROOT / "apps/web/app/admin/play-generators/GeneratorWorkspace.tsx"
    ).read_text()
    public_setup = (
        ROOT
        / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx"
    ).read_text()
    admin_runner = (
        ROOT / "apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx"
    ).read_text()
    public_runner = (
        ROOT
        / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx"
    ).read_text()

    for setup in (admin_setup, public_setup):
        assert "Unscored — mark each round played" in setup
        assert "standingsSort" in setup
        assert "scoringMode" in setup

    admin_ui = _function_block(
        admin_runner,
        "  async function markRoundPlayed",
        "  async function executeSkipRound",
    )
    public_ui = _function_block(
        public_runner,
        "  async function markRoundPlayed",
        "  async function executeSkipRound",
    )
    for block in (admin_ui, public_ui):
        assert "/played" in block
        assert "/advance" not in block
        assert "current_round_number" in block

    for runner in (admin_runner, public_runner):
        assert "Round Played" in runner
        assert "View standings and continue" in runner
        assert "Session complete" in runner

    assert "Boolean(editToken)" in public_runner
    assert "{scoredSession ? (" in admin_runner
    assert "Official results" in admin_runner


def test_standings_pages_own_scored_progression_and_completion() -> None:
    admin = (
        ROOT / "apps/web/app/admin/play-generators/GeneratorStandings.tsx"
    ).read_text()
    public = (
        ROOT
        / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings.tsx"
    ).read_text()
    for text in (admin, public):
        assert "Continue to Round" in text
        assert "/advance" in text
        assert "This unscored Round-Robin does not use standings." in text
        assert "Session complete" in text
