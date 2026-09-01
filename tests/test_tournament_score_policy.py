from __future__ import annotations

import pytest

from jupr_app.domain.tournaments.score_policy import (
    require_best_of_three_game_scores,
    require_tournament_score,
    resolve_tournament_scoring_format,
    review_tournament_score,
)


@pytest.mark.parametrize(
    ("format_code", "score", "expected_status"),
    [
        ("GAME_TO_11", (11, 7), "ordinary"),
        ("GAME_TO_11", (12, 10), "ordinary"),
        ("GAME_TO_15", (15, 13), "ordinary"),
        ("GAME_TO_21", (23, 21), "ordinary"),
        ("BEST_2_OF_3", (2, 0), "ordinary"),
        ("BEST_2_OF_3", (1, 2), "ordinary"),
        ("GAME_TO_11", (76, 11), "unusual"),
        ("GAME_TO_11", (76, 74), "unusual"),
        ("GAME_TO_15", (14, 7), "impossible"),
        ("GAME_TO_21", (21, 20), "impossible"),
        ("BEST_2_OF_3", (3, 1), "impossible"),
    ],
)
def test_score_policy_for_supported_formats(format_code, score, expected_status):
    review = review_tournament_score(*score, scoring_format=format_code)
    assert review["status"] == expected_status


def test_unusual_score_requires_explicit_acknowledgement():
    with pytest.raises(ValueError, match="explicit acknowledgement"):
        require_tournament_score(76, 11, scoring_format="GAME_TO_11")
    accepted = require_tournament_score(
        76,
        11,
        scoring_format="GAME_TO_11",
        unusual_score_acknowledged=True,
    )
    assert accepted["accepted"] is True
    assert accepted["acknowledged"] is True


def test_scoring_override_wins_over_event_default():
    assert resolve_tournament_scoring_format(
        {"scoring_default": "GAME_TO_15", "scoring_override": "GAME_TO_21"}
    ) == "GAME_TO_21"


def test_best_of_three_semantics_are_games_won_not_points():
    with pytest.raises(ValueError, match="games won"):
        require_tournament_score(11, 7, scoring_format="BEST_2_OF_3")


@pytest.mark.parametrize(
    ("game_scores", "expected_series"),
    [
        (
            [
                {"game_number": 1, "score_a": 11, "score_b": 7},
                {"game_number": 2, "score_a": 15, "score_b": 13},
            ],
            (2, 0),
        ),
        (
            [
                {"game_number": 1, "score_a": 11, "score_b": 8},
                {"game_number": 2, "score_a": 7, "score_b": 11},
                {"game_number": 3, "score_a": 12, "score_b": 10},
            ],
            (2, 1),
        ),
    ],
)
def test_best_of_three_preserves_and_derives_individual_games(
    game_scores, expected_series
):
    review = require_best_of_three_game_scores(game_scores)

    assert (review["score_a"], review["score_b"]) == expected_series
    assert review["individual_game_format"] == "GAME_TO_11"
    assert [row["game_number"] for row in review["game_scores"]] == list(
        range(1, len(game_scores) + 1)
    )


@pytest.mark.parametrize(
    ("game_scores", "message"),
    [
        (
            [
                {"game_number": 1, "score_a": 11, "score_b": 8},
                {"game_number": 2, "score_a": 7, "score_b": 11},
            ],
            "deciding third game",
        ),
        (
            [
                {"game_number": 1, "score_a": 11, "score_b": 8},
                {"game_number": 2, "score_a": 11, "score_b": 7},
                {"game_number": 3, "score_a": 11, "score_b": 9},
            ],
            "cannot be recorded",
        ),
        (
            [
                {"game_number": 1, "score_a": 11, "score_b": 10},
                {"game_number": 2, "score_a": 11, "score_b": 7},
            ],
            "two-point winning margin",
        ),
    ],
)
def test_best_of_three_rejects_incomplete_or_impossible_game_rows(
    game_scores, message
):
    with pytest.raises(ValueError, match=message):
        require_best_of_three_game_scores(game_scores)
