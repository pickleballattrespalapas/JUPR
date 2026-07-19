from __future__ import annotations

import math
from typing import Any

from jupr_app.domain.constants import CAP_LOSER_GAIN_ELO, MIN_WIN_DELTA_ELO
from jupr_app.domain.ratings import calculate_hybrid_elo


def _jupr(elo: float) -> float:
    return float(elo) / 400.0


def _win_label(probability: float) -> str:
    if probability >= 0.70:
        return "Heavy Favorite"
    if probability >= 0.55:
        return "Favored"
    if probability >= 0.45:
        return "Toss-up"
    if probability >= 0.30:
        return "Underdog"
    return "Heavy Underdog"


def _round_half_up(value: float) -> int:
    return int(math.floor(float(value) + 0.5))


def equivalent_score_to_goal(share: float, *, goal_points: int = 11) -> dict[str, Any]:
    """Translate a score share into the same to-11 label used by Match Explorer."""

    normalized = min(1.0, max(0.0, float(share)))
    goal = max(1, int(goal_points))
    if abs(normalized - 0.5) < 1e-12:
        you = goal
        opponents = goal
    elif normalized < 0.5:
        you = _round_half_up(goal * normalized / max(1e-12, 1.0 - normalized))
        opponents = goal
    else:
        you = goal
        opponents = _round_half_up(goal * (1.0 - normalized) / max(1e-12, normalized))
    return {"you": int(you), "opponents": int(opponents), "label": f"{you}\u2013{opponents}"}


def _rating_delta(
    *,
    team_you_avg: float,
    team_opponents_avg: float,
    score_you: int,
    score_opponents: int,
    k_factor: int,
) -> tuple[float, float]:
    return calculate_hybrid_elo(
        float(team_you_avg),
        float(team_opponents_avg),
        int(score_you),
        int(score_opponents),
        k_factor=int(k_factor),
        min_win_delta=float(MIN_WIN_DELTA_ELO),
        cap_loser_gain=float(CAP_LOSER_GAIN_ELO),
    )


def build_match_explorer_projection(
    *,
    team_you_avg: float,
    team_opponents_avg: float,
    score_you: int,
    score_opponents: int,
    k_factor: int,
) -> dict[str, Any]:
    """Return the complete, read-only projection used by every Match Explorer UI.

    Expected score, selected-score movement, and every impact-chart point are
    produced here through the canonical Python rating engine. Browser clients
    should render these values rather than reimplementing rating policy.
    """

    you_avg = float(team_you_avg)
    opponents_avg = float(team_opponents_avg)
    normalized_you = max(0, int(score_you))
    normalized_opponents = max(0, int(score_opponents))
    normalized_k = int(k_factor)

    expected_you = 1.0 / (1.0 + 10.0 ** ((opponents_avg - you_avg) / 400.0))
    expected_score = equivalent_score_to_goal(expected_you)
    delta_you_elo, delta_opponents_elo = _rating_delta(
        team_you_avg=you_avg,
        team_opponents_avg=opponents_avg,
        score_you=normalized_you,
        score_opponents=normalized_opponents,
        k_factor=normalized_k,
    )

    total_points = normalized_you + normalized_opponents
    selected_share = (float(normalized_you) / float(total_points)) if total_points > 0 else None
    beat_expectation_pp = None
    if selected_share is not None and normalized_you != normalized_opponents:
        beat_expectation_pp = (selected_share - expected_you) * 100.0

    chart_points: list[dict[str, Any]] = []
    for index in range(101):
        share = float(index) / 100.0
        point_you_elo, point_opponents_elo = _rating_delta(
            team_you_avg=you_avg,
            team_opponents_avg=opponents_avg,
            score_you=index,
            score_opponents=100 - index,
            k_factor=normalized_k,
        )
        chart_points.append(
            {
                "score_share": share,
                "score_to_11": equivalent_score_to_goal(share),
                "you_team_elo": float(point_you_elo),
                "opponent_team_elo": float(point_opponents_elo),
                "you_team_jupr": _jupr(float(point_you_elo)),
                "opponent_team_jupr": _jupr(float(point_opponents_elo)),
            }
        )

    selected_score_to_11 = equivalent_score_to_goal(selected_share) if selected_share is not None else None
    chart_tick_shares = (0.0, 3.0 / 14.0, 6.0 / 17.0, 9.0 / 20.0, 0.5, 11.0 / 20.0, 11.0 / 17.0, 11.0 / 14.0, 1.0)
    return {
        "expected": {
            "you": float(expected_you),
            "opponents": float(1.0 - expected_you),
            "label": _win_label(float(expected_you)),
            "score_to_11": expected_score,
        },
        "score": {
            "you": int(normalized_you),
            "opponents": int(normalized_opponents),
            "you_share": selected_share,
            "opponents_share": (1.0 - selected_share) if selected_share is not None else None,
            "beat_expectation_pp": beat_expectation_pp,
            "score_to_11": selected_score_to_11,
        },
        "rating_delta": {
            "you_team_elo": float(delta_you_elo),
            "opponent_team_elo": float(delta_opponents_elo),
            "you_team_jupr": _jupr(float(delta_you_elo)),
            "opponent_team_jupr": _jupr(float(delta_opponents_elo)),
        },
        "impact_chart": {
            "points": chart_points,
            "score_ticks": [
                {"score_share": share, "score_to_11": equivalent_score_to_goal(share)}
                for share in chart_tick_shares
            ],
            "expected_marker": {
                "score_share": float(expected_you),
                "score_to_11": expected_score,
            },
            "selected_marker": (
                {
                    "score_share": float(selected_share),
                    "score_to_11": selected_score_to_11,
                    "you_team_elo": float(delta_you_elo),
                    "you_team_jupr": _jupr(float(delta_you_elo)),
                }
                if selected_share is not None
                else None
            ),
        },
    }
