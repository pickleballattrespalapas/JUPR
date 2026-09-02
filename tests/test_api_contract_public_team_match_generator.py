from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURE = ROOT / "apps" / "web" / "app" / "clubs" / "[clubSlug]" / "team-match-generator"


def _source(relative: str) -> str:
    return (FEATURE / relative).read_text(encoding="utf-8")


def test_team_schedule_moves_from_setup_to_a_dedicated_session_page() -> None:
    setup = _source("TeamMatchGenerator.tsx")
    route = _source("sessions/[sessionKey]/page.tsx")

    assert "createSession(title, cleanedTeams)" in setup
    assert "router.push(" in setup
    assert "/team-match-generator/sessions/" in setup
    assert "TeamMatchSessionRunner" in route
    assert "matchups.map" not in setup


def test_mixed_lineups_are_selected_after_gender_doubles_by_player_name() -> None:
    state = _source("teamMatchState.ts")
    runner = _source("sessions/[sessionKey]/TeamMatchSessionRunner.tsx")

    assert 'gameByKey(matchup, "women").submitted' in state
    assert 'gameByKey(matchup, "men").submitted' in state
    assert 'return { kind: "mixed_lineups" }' in state
    assert "Mixed Doubles 1 woman" in runner
    assert "Mixed Doubles 1 man" in runner
    assert "teamA.women.map" in runner
    assert "teamA.men.map" in runner
    assert "remaining woman and man automatically play Mixed Doubles 2" in runner


def test_results_are_explicitly_submitted_one_game_at_a_time() -> None:
    runner = _source("sessions/[sessionKey]/TeamMatchSessionRunner.tsx")

    assert "Game {gameNumber(game.key)} of 4" in runner
    assert "Submit ${currentGame.label} result" in runner
    assert "submitted: true" in runner
    assert "Women’s Doubles submitted. Men’s Doubles is next." in runner
    assert "Mixed Doubles 1 submitted. Mixed Doubles 2 is next." in runner
    assert "Continue to next team matchup" in runner


def test_two_two_regulation_result_advances_to_dreambreaker() -> None:
    state = _source("teamMatchState.ts")
    runner = _source("sessions/[sessionKey]/TeamMatchSessionRunner.tsx")

    assert 'return { kind: "dreambreaker_lineups" }' in state
    assert 'return { kind: "dreambreaker_score" }' in state
    assert "Regulation is tied 2–2" in runner
    assert "Submit DreamBreaker rotations" in runner
    assert "Submit DreamBreaker result" in runner
    assert "DreamBreaker submitted. The team matchup is complete." in runner


def test_team_match_generator_uses_shared_confirmation_instead_of_native_prompt() -> None:
    setup = _source("TeamMatchGenerator.tsx")
    runner = _source("sessions/[sessionKey]/TeamMatchSessionRunner.tsx")

    assert "<ConfirmAction" in setup
    assert "window.confirm(" not in setup
    assert "window.confirm(" not in runner
