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


def test_unscored_round_robin_marks_played_and_advances() -> None:
    preview = create_generator_preview(
        generator_kind="round_robin",
        play_format="singles",
        title="Unscored",
        participant_names=["A", "B", "C"],
        total_rounds=3,
        court_count=1,
        scoring_mode="unscored",
    )
    event = start_generator_event(preview)
    assert event["scoringMode"] == "unscored"
    assert generator_event_standings(event) == []
    with pytest.raises(ValueError, match="Round Played"):
        save_generator_round(event, round_number=1, scores=[])
    played = mark_generator_round_played(event, round_number=1)
    assert played["rounds"][0]["status"] == "played"
    history = history_before_round(played, 2)
    assert sum(history["games"].values()) == 2
    advanced = advance_generator_event(played)
    assert advanced["currentRoundNumber"] == 2
    assert advanced["rounds"][1]["status"] == "active"


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


def test_unscored_routes_setup_and_round_played_controls_exist() -> None:
    admin_routes = (ROOT / "services/api/admin_play_generator_routes.py").read_text()
    public_routes = (ROOT / "services/api/public_play_generator_routes.py").read_text()
    admin_setup = (ROOT / "apps/web/app/admin/play-generators/GeneratorWorkspace.tsx").read_text()
    public_setup = (ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorWorkspace.tsx").read_text()
    admin_runner = (ROOT / "apps/web/app/admin/play-generators/GeneratorRoundRunner.tsx").read_text()
    public_runner = (ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorRoundRunner.tsx").read_text()
    assert "/played" in admin_routes
    assert "/played" in public_routes
    assert "Unscored — mark each round played" in admin_setup
    assert "Unscored — mark each round played" in public_setup
    assert "Round Played" in admin_runner
    assert "Round Played" in public_runner
    assert "View standings and continue" in admin_runner
    assert "View standings and continue" in public_runner


def test_standings_pages_own_scored_progression() -> None:
    admin = (ROOT / "apps/web/app/admin/play-generators/GeneratorStandings.tsx").read_text()
    public = (ROOT / "apps/web/app/clubs/[clubSlug]/play-generators/PublicGeneratorStandings.tsx").read_text()
    for text in (admin, public):
        assert "Continue to Round" in text
        assert "/advance" in text
        assert "This unscored Round-Robin does not use standings." in text
