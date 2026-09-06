from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Iterable

from jupr_app.domain.rating_backtest import (
    _calibration,
    _match_order,
    _metric_summary,
    _safe_float,
    _safe_player_id,
    _timestamp,
)


BAYESIAN_DOUBLES_ALGORITHM_VERSION = "bayesian-doubles-score-gaussian-v1"


@dataclass(frozen=True)
class BayesianDoublesConfig:
    """Private-only parameters for the Bayesian doubles benchmark."""

    prior_sigma_elo: float = 200.0
    performance_sigma_elo: float = 200.0
    score_smoothing_points: float = 1.0
    fallback_initial_rating_elo: float = 1200.0
    minimum_variance_elo2: float = 25.0

    def validate(self) -> None:
        if self.prior_sigma_elo <= 0:
            raise ValueError("prior_sigma_elo must be positive")
        if self.performance_sigma_elo <= 0:
            raise ValueError("performance_sigma_elo must be positive")
        if self.score_smoothing_points <= 0:
            raise ValueError("score_smoothing_points must be positive")
        if self.minimum_variance_elo2 <= 0:
            raise ValueError("minimum_variance_elo2 must be positive")


@dataclass
class _PlayerBelief:
    mean: float
    variance: float


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _expected_score_share(mean_gap: float, predictive_variance: float) -> float:
    """Approximate E[logistic(Y)] for a Gaussian Elo-performance gap."""

    scale = math.log(10.0) / 400.0
    natural_mean = mean_gap * scale
    natural_variance = max(0.0, predictive_variance) * scale * scale
    adjusted = natural_mean / math.sqrt(1.0 + math.pi * natural_variance / 8.0)
    if adjusted >= 0:
        return 1.0 / (1.0 + math.exp(-adjusted))
    exp_value = math.exp(adjusted)
    return exp_value / (1.0 + exp_value)


def _parse_boundary(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    parsed = value if isinstance(value, datetime) else _timestamp(value)
    if parsed is None:
        raise ValueError(f"Invalid evaluation boundary: {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _score_gap_elo(score_t1: float, score_t2: float, smoothing: float) -> float:
    return 400.0 * math.log10(
        (score_t1 + smoothing) / (score_t2 + smoothing)
    )


def _shadow_metrics(
    outcome_predictions: list[tuple[float, float, float]],
    score_predictions: list[tuple[float, float, float]],
) -> dict[str, float | int | None]:
    metrics = _metric_summary(outcome_predictions)
    metrics["score_share_mae"] = _metric_summary(score_predictions)[
        "score_share_mae"
    ]
    return metrics


def run_bayesian_doubles_backtest(
    rows: Iterable[dict[str, Any]],
    *,
    config: BayesianDoublesConfig = BayesianDoublesConfig(),
    evaluation_start: str | datetime | None = None,
    evaluation_end: str | datetime | None = None,
) -> dict[str, Any]:
    """Evaluate a score-aware Bayesian doubles model in strict chronology.

    Player beliefs are private shadow state. The factorised Gaussian update is
    an online approximation: each score supplies evidence about the average
    skill gap between the two teams, and changing partnerships gradually
    separate individual player estimates.
    """

    config.validate()
    start_at = _parse_boundary(evaluation_start)
    end_at = _parse_boundary(evaluation_end)
    if start_at is not None and end_at is not None and start_at >= end_at:
        raise ValueError("evaluation_start must be earlier than evaluation_end")

    prepared: list[tuple[datetime, tuple[int, int | str], int, dict[str, Any]]] = []
    invalid_dates = 0
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        played_at = _timestamp(row.get("date"))
        if played_at is None:
            invalid_dates += 1
            continue
        prepared.append((played_at, _match_order(row.get("id")), index, row))
    prepared.sort(key=lambda item: (item[0], item[1], item[2]))

    beliefs: dict[tuple[str, str], _PlayerBelief] = {}
    predictions: list[tuple[float, float, float]] = []
    score_predictions: list[tuple[float, float, float]] = []
    by_month: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_month_score: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_club: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_club_score: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    if start_at is None and invalid_dates:
        skipped["invalid_date"] = invalid_dates
    first_match_at: datetime | None = None
    last_match_at: datetime | None = None
    training_matches = 0

    for played_at, _match_id, _index, row in prepared:
        if end_at is not None and played_at >= end_at:
            break
        in_evaluation = start_at is None or played_at >= start_at

        if row.get("deleted_at") not in (None, ""):
            if in_evaluation:
                skipped["deleted"] += 1
            continue
        if str(row.get("rating_scope") or "").strip().casefold() == "unrated":
            if in_evaluation:
                skipped["unrated"] += 1
            continue
        explicit_format = str(row.get("match_format") or "").strip().casefold()
        inferred_format = (
            "singles"
            if str(row.get("match_type") or "").strip().casefold() == "singles"
            else "doubles"
        )
        if (explicit_format or inferred_format) != "doubles":
            if in_evaluation:
                skipped["non_doubles"] += 1
            continue

        score_t1 = _safe_float(row.get("score_t1"))
        score_t2 = _safe_float(row.get("score_t2"))
        if (
            score_t1 is None
            or score_t2 is None
            or score_t1 < 0
            or score_t2 < 0
            or score_t1 == score_t2
            or score_t1 + score_t2 <= 0
        ):
            if in_evaluation:
                skipped["invalid_score"] += 1
            continue

        slots = (
            (("t1_p1", "t1_p1_r"), ("t1_p2", "t1_p2_r")),
            (("t2_p1", "t2_p1_r"), ("t2_p2", "t2_p2_r")),
        )
        teams: list[list[str]] = []
        seed_fields: list[list[str]] = []
        valid = True
        for team_slots in slots:
            team: list[str] = []
            fields: list[str] = []
            for player_field, rating_field in team_slots:
                player_id = _safe_player_id(row.get(player_field))
                if player_id is None:
                    valid = False
                    break
                team.append(player_id)
                fields.append(rating_field)
            teams.append(team)
            seed_fields.append(fields)
        flat_players = [player for team in teams for player in team]
        if not valid or len(set(flat_players)) != 4:
            if in_evaluation:
                skipped["invalid_players"] += 1
            continue

        club_id = str(row.get("club_id") or "__single_club_input__")
        for team, fields in zip(teams, seed_fields):
            for player_id, rating_field in zip(team, fields):
                key = (club_id, player_id)
                if key not in beliefs:
                    seed = _safe_float(row.get(rating_field))
                    beliefs[key] = _PlayerBelief(
                        mean=(
                            seed
                            if seed is not None
                            else float(config.fallback_initial_rating_elo)
                        ),
                        variance=float(config.prior_sigma_elo) ** 2,
                    )

        weighted_players = [
            ((club_id, teams[0][0]), 0.5),
            ((club_id, teams[0][1]), 0.5),
            ((club_id, teams[1][0]), -0.5),
            ((club_id, teams[1][1]), -0.5),
        ]
        mean_gap = sum(beliefs[key].mean * weight for key, weight in weighted_players)
        latent_variance = sum(
            beliefs[key].variance * weight * weight
            for key, weight in weighted_players
        )
        predictive_variance = (
            latent_variance + float(config.performance_sigma_elo) ** 2
        )
        probability = _normal_cdf(mean_gap / math.sqrt(predictive_variance))
        predicted_score_share = _expected_score_share(mean_gap, predictive_variance)
        outcome = 1.0 if score_t1 > score_t2 else 0.0
        actual_score_share = score_t1 / (score_t1 + score_t2)
        outcome_observation = (probability, outcome, actual_score_share)
        score_observation = (predicted_score_share, outcome, actual_score_share)

        if in_evaluation:
            predictions.append(outcome_observation)
            score_predictions.append(score_observation)
            by_month[played_at.strftime("%Y-%m")].append(outcome_observation)
            by_month_score[played_at.strftime("%Y-%m")].append(score_observation)
            by_club[club_id].append(outcome_observation)
            by_club_score[club_id].append(score_observation)
            first_match_at = first_match_at or played_at
            last_match_at = played_at
        else:
            training_matches += 1

        observed_gap = _score_gap_elo(
            score_t1,
            score_t2,
            float(config.score_smoothing_points),
        )
        residual = observed_gap - mean_gap
        prior_variances = {
            key: beliefs[key].variance for key, _weight in weighted_players
        }
        for key, weight in weighted_players:
            belief = beliefs[key]
            variance = prior_variances[key]
            gain = variance * weight / predictive_variance
            belief.mean += gain * residual
            belief.variance = max(
                float(config.minimum_variance_elo2),
                variance - (variance * weight) ** 2 / predictive_variance,
            )

    metrics = _shadow_metrics(predictions, score_predictions)
    naive = _metric_summary([(0.5, outcome, share) for _, outcome, share in predictions])
    brier = metrics["brier_score"]
    naive_brier = naive["brier_score"]
    return {
        "model": {
            "algorithm_version": BAYESIAN_DOUBLES_ALGORITHM_VERSION,
            "player_facing": False,
            "uses_private_uncertainty": True,
            "uses_recency_drift": False,
            "parameters": {
                "prior_sigma_elo": float(config.prior_sigma_elo),
                "performance_sigma_elo": float(config.performance_sigma_elo),
                "score_smoothing_points": float(config.score_smoothing_points),
                "fallback_initial_rating_elo": float(
                    config.fallback_initial_rating_elo
                ),
                "minimum_variance_elo2": float(config.minimum_variance_elo2),
            },
        },
        "method": "chronological_walk_forward",
        "evaluation_window": {
            "start_inclusive": start_at.isoformat() if start_at else None,
            "end_exclusive": end_at.isoformat() if end_at else None,
        },
        "training_matches_before_window": training_matches,
        "first_match_at": first_match_at.isoformat() if first_match_at else None,
        "last_match_at": last_match_at.isoformat() if last_match_at else None,
        "players_simulated": len(beliefs),
        "metrics": metrics,
        "score_share_prediction_note": (
            "score_share_mae uses the private Gaussian-logistic score forecast; "
            "win metrics use the Gaussian probability of a positive score gap"
        ),
        "naive_50_percent_baseline": naive,
        "brier_improvement_vs_50_percent": (
            float(naive_brier) - float(brier)
            if brier is not None and naive_brier is not None
            else None
        ),
        "by_format": {"doubles": dict(metrics)},
        "by_club": {
            key: _shadow_metrics(value, by_club_score[key])
            for key, value in sorted(by_club.items())
        },
        "by_month": {
            key: _shadow_metrics(value, by_month_score[key])
            for key, value in sorted(by_month.items())
        },
        "calibration": _calibration(predictions),
        "skipped": dict(sorted(skipped.items())),
    }


def default_bayesian_candidate_grid() -> tuple[BayesianDoublesConfig, ...]:
    """Predeclared grid selected without looking at the final holdout."""

    return tuple(
        BayesianDoublesConfig(
            prior_sigma_elo=prior_sigma,
            performance_sigma_elo=performance_sigma,
            score_smoothing_points=smoothing,
        )
        for prior_sigma in (100.0, 200.0, 300.0)
        for performance_sigma in (100.0, 150.0, 200.0, 300.0)
        for smoothing in (0.5, 1.0)
    )
