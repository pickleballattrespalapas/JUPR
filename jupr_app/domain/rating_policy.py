from __future__ import annotations

from typing import Any

from jupr_app.domain.constants import (
    CAP_LOSER_GAIN_ELO,
    DEFAULT_K_FACTOR,
    MIN_WIN_DELTA_ELO,
)


RATING_ALGORITHM_VERSION = "jupr-hybrid-score-share-v1"
RATING_PARAMETER_VERSION = "flat-k32-floor1-loser-cap16-v1"
ELO_PER_JUPR = 400.0


def rating_parameter_snapshot(
    *,
    overall_k_factor: float = DEFAULT_K_FACTOR,
    min_win_delta_elo: float = MIN_WIN_DELTA_ELO,
    cap_loser_gain_elo: float | None = CAP_LOSER_GAIN_ELO,
    league_k_factor: float | None = None,
) -> dict[str, Any]:
    """Return the JSON-safe settings needed to reproduce a rating update."""

    snapshot: dict[str, Any] = {
        "overall_k_factor": float(overall_k_factor),
        "min_win_delta_elo": float(min_win_delta_elo),
        "cap_loser_gain_elo": (
            None
            if cap_loser_gain_elo is None
            else float(cap_loser_gain_elo)
        ),
        "elo_per_jupr": float(ELO_PER_JUPR),
        "winner_must_gain": True,
        "loser_may_gain_for_outperformance": True,
    }
    if league_k_factor is not None:
        snapshot["league_k_factor"] = float(league_k_factor)
    return snapshot


def rating_calculation_metadata(
    *,
    overall_k_factor: float = DEFAULT_K_FACTOR,
    min_win_delta_elo: float = MIN_WIN_DELTA_ELO,
    cap_loser_gain_elo: float | None = CAP_LOSER_GAIN_ELO,
    league_k_factor: float | None = None,
) -> dict[str, Any]:
    return {
        "rating_algorithm_version": RATING_ALGORITHM_VERSION,
        "rating_parameter_version": RATING_PARAMETER_VERSION,
        "rating_parameters": rating_parameter_snapshot(
            overall_k_factor=overall_k_factor,
            min_win_delta_elo=min_win_delta_elo,
            cap_loser_gain_elo=cap_loser_gain_elo,
            league_k_factor=league_k_factor,
        ),
    }
