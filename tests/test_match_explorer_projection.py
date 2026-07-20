from __future__ import annotations

import pytest

from jupr_app.domain.match_explorer import (
    build_match_explorer_projection,
    equivalent_score_to_goal,
)
from jupr_app.domain.ratings import calculate_hybrid_elo


def test_equivalent_score_to_goal_golden_vectors() -> None:
    assert equivalent_score_to_goal(0.0) == {"you": 0, "opponents": 11, "label": "0\u201311"}
    assert equivalent_score_to_goal(0.5) == {"you": 11, "opponents": 11, "label": "11\u201311"}
    assert equivalent_score_to_goal(1.0) == {"you": 11, "opponents": 0, "label": "11\u20130"}
    assert equivalent_score_to_goal(11 / 18) == {"you": 11, "opponents": 7, "label": "11\u20137"}


def test_projection_is_canonical_python_rating_output() -> None:
    projection = build_match_explorer_projection(
        team_you_avg=1475.0,
        team_opponents_avg=1275.0,
        score_you=11,
        score_opponents=7,
        k_factor=24,
    )
    expected_delta = calculate_hybrid_elo(
        1475.0,
        1275.0,
        11,
        7,
        k_factor=24,
        min_win_delta=1.0,
        cap_loser_gain=16.0,
    )

    assert projection["expected"]["you"] == pytest.approx(0.7597469266)
    assert projection["expected"]["label"] == "Heavy Favorite"
    assert projection["expected"]["score_to_11"]["label"] == "11\u20133"
    assert projection["score"]["you_share"] == pytest.approx(11 / 18)
    assert projection["score"]["beat_expectation_pp"] == pytest.approx((11 / 18 - 0.7597469266) * 100)
    assert projection["rating_delta"]["you_team_elo"] == pytest.approx(expected_delta[0])
    assert projection["rating_delta"]["opponent_team_elo"] == pytest.approx(expected_delta[1])


def test_impact_chart_uses_rating_engine_for_every_score_share() -> None:
    projection = build_match_explorer_projection(
        team_you_avg=1200.0,
        team_opponents_avg=1200.0,
        score_you=0,
        score_opponents=0,
        k_factor=32,
    )

    points = projection["impact_chart"]["points"]
    assert len(points) == 101
    assert points[0]["score_to_11"]["label"] == "0\u201311"
    assert points[50]["you_team_elo"] == 0.0
    assert points[100]["score_to_11"]["label"] == "11\u20130"
    assert projection["impact_chart"]["selected_marker"] is None

    for index in (0, 25, 50, 75, 100):
        expected = calculate_hybrid_elo(
            1200.0,
            1200.0,
            index,
            100 - index,
            k_factor=32,
            min_win_delta=1.0,
            cap_loser_gain=16.0,
        )
        assert points[index]["you_team_elo"] == pytest.approx(expected[0])
        assert points[index]["opponent_team_elo"] == pytest.approx(expected[1])
