from typing import Tuple
import math


def calculate_hybrid_elo(
    t1_avg,
    t2_avg,
    score_t1,
    score_t2,
    k_factor=32,
    min_win_delta=1.0,
    cap_loser_gain=16,
):
    """
    Returns (delta_for_team1_players, delta_for_team2_players) in ELO points (not JUPR).

    Policy:
      - Winner hard rule: winner delta must be > 0. If computed <= 0, set to +min_win_delta.
      - Loser may gain if they beat expectations (non-zero-sum behavior).
      - ONLY cap: if the loser delta is positive, cap it to cap_loser_gain.
    """
    # Normalize inputs
    s1 = int(score_t1 or 0)
    s2 = int(score_t2 or 0)

    # No movement on ties or empty scores
    total_points = s1 + s2
    if total_points <= 0 or s1 == s2:
        return 0.0, 0.0

    # Expected outcomes from ratings
    expected_t1 = 1 / (1 + 10 ** ((t2_avg - t1_avg) / 400))
    expected_t2 = 1 - expected_t1

    # Observed performance proxy from score share
    share_t1 = s1 / total_points
    share_t2 = 1.0 - share_t1

    # Base deltas (symmetric)
    d1 = float(k_factor) * 2.0 * (share_t1 - expected_t1)
    d2 = float(k_factor) * 2.0 * (share_t2 - expected_t2)  # == -d1

    # Apply winner floor + loser-positive cap only
    if s1 > s2:
        # Team 1 wins
        if d1 <= 0:
            d1 = float(min_win_delta)

        if cap_loser_gain is not None and d2 > 0:
            d2 = min(d2, float(cap_loser_gain))

        return float(d1), float(d2)

    else:
        # Team 2 wins
        if d2 <= 0:
            d2 = float(min_win_delta)

        if cap_loser_gain is not None and d1 > 0:
            d1 = min(d1, float(cap_loser_gain))

        return float(d1), float(d2)

def elo_to_jupr(elo_score):
    try:
        return float(elo_score) / 400.0
    except Exception:
        return 0.0