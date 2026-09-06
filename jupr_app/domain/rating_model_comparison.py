from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable, Sequence

from jupr_app.domain.bayesian_doubles_shadow import (
    BayesianDoublesConfig,
    default_bayesian_candidate_grid,
    run_bayesian_doubles_backtest,
)
from jupr_app.domain.rating_backtest import run_chronological_backtest
from jupr_app.domain.rating_backtest import _timestamp


def _metrics(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report.get("metrics") or {})


def _comparison(jupr: dict[str, Any], bayesian: dict[str, Any]) -> dict[str, Any]:
    jupr_metrics = _metrics(jupr)
    bayesian_metrics = _metrics(bayesian)

    def difference(metric: str) -> float | None:
        left = bayesian_metrics.get(metric)
        right = jupr_metrics.get(metric)
        if left is None or right is None:
            return None
        return float(left) - float(right)

    return {
        "current_jupr": jupr_metrics,
        "bayesian_shadow": bayesian_metrics,
        "bayesian_minus_jupr": {
            "brier_score": difference("brier_score"),
            "log_loss": difference("log_loss"),
            "outcome_accuracy": difference("outcome_accuracy"),
            "score_share_mae": difference("score_share_mae"),
        },
        "interpretation": {
            "lower_is_better": ["brier_score", "log_loss", "score_share_mae"],
            "higher_is_better": ["outcome_accuracy"],
        },
    }


def compare_jupr_with_bayesian_shadow(
    rows: Iterable[dict[str, Any]],
    *,
    validation_start: str | datetime,
    validation_end: str | datetime,
    holdout_start: str | datetime,
    holdout_end: str | datetime | None = None,
    candidates: Sequence[BayesianDoublesConfig] | None = None,
) -> dict[str, Any]:
    """Select Bayesian parameters on validation, then inspect one holdout once."""

    materialized = []
    for raw_row in rows:
        row = dict(raw_row)
        explicit = str(row.get("match_format") or "").strip().casefold()
        inferred = (
            "singles"
            if str(row.get("match_type") or "").strip().casefold() == "singles"
            else "doubles"
        )
        if (explicit or inferred) == "doubles":
            materialized.append(row)

    validation_start_at = _timestamp(validation_start)
    validation_end_at = _timestamp(validation_end)
    holdout_start_at = _timestamp(holdout_start)
    holdout_end_at = _timestamp(holdout_end) if holdout_end is not None else None
    if not all((validation_start_at, validation_end_at, holdout_start_at)):
        raise ValueError("Validation and holdout boundaries must be valid timestamps")
    assert validation_start_at is not None
    assert validation_end_at is not None
    assert holdout_start_at is not None
    if not validation_start_at < validation_end_at <= holdout_start_at:
        raise ValueError(
            "Validation must end on or before the untouched holdout starts"
        )
    if holdout_end_at is not None and holdout_start_at >= holdout_end_at:
        raise ValueError("holdout_start must be earlier than holdout_end")
    candidate_grid = tuple(candidates or default_bayesian_candidate_grid())
    if not candidate_grid:
        raise ValueError("At least one Bayesian candidate is required")

    leaderboard: list[dict[str, Any]] = []
    selected_config: BayesianDoublesConfig | None = None
    selected_key: tuple[float, float, float, float] | None = None
    for config in candidate_grid:
        report = run_bayesian_doubles_backtest(
            materialized,
            config=config,
            evaluation_start=validation_start,
            evaluation_end=validation_end,
        )
        metrics = _metrics(report)
        if not metrics.get("matches"):
            raise ValueError("Validation window contains no eligible doubles matches")
        key = (
            float(metrics["brier_score"]),
            float(metrics["log_loss"]),
            float(config.performance_sigma_elo),
            float(config.prior_sigma_elo),
        )
        leaderboard.append(
            {
                "parameters": report["model"]["parameters"],
                "validation_metrics": metrics,
            }
        )
        if selected_key is None or key < selected_key:
            selected_key = key
            selected_config = config

    assert selected_config is not None
    leaderboard.sort(
        key=lambda item: (
            float(item["validation_metrics"]["brier_score"]),
            float(item["validation_metrics"]["log_loss"]),
        )
    )

    jupr_validation = run_chronological_backtest(
        materialized,
        evaluation_start=validation_start,
        evaluation_end=validation_end,
    )
    bayesian_validation = run_bayesian_doubles_backtest(
        materialized,
        config=selected_config,
        evaluation_start=validation_start,
        evaluation_end=validation_end,
    )
    jupr_holdout = run_chronological_backtest(
        materialized,
        evaluation_start=holdout_start,
        evaluation_end=holdout_end,
    )
    bayesian_holdout = run_bayesian_doubles_backtest(
        materialized,
        config=selected_config,
        evaluation_start=holdout_start,
        evaluation_end=holdout_end,
    )
    jupr_full = run_chronological_backtest(materialized)
    bayesian_full = run_bayesian_doubles_backtest(
        materialized,
        config=selected_config,
    )

    return {
        "schema_version": 1,
        "purpose": "private_shadow_benchmark_only",
        "official_rating_model": jupr_full["model"],
        "shadow_model": bayesian_full["model"],
        "selection": {
            "rule": "lowest validation Brier score; log loss breaks ties",
            "validation_window": bayesian_validation["evaluation_window"],
            "candidate_count": len(candidate_grid),
            "selected_parameters": bayesian_full["model"]["parameters"],
            "leaderboard": leaderboard,
        },
        "validation": _comparison(jupr_validation, bayesian_validation),
        "holdout": {
            "window": bayesian_holdout["evaluation_window"],
            **_comparison(jupr_holdout, bayesian_holdout),
        },
        "full_history": _comparison(jupr_full, bayesian_full),
        "official_policy_checks": jupr_full["policy_checks"],
        "guardrails": {
            "changes_official_ratings": False,
            "visible_to_players": False,
            "uses_production_writes": False,
            "eligible_for_automatic_promotion": False,
        },
    }
