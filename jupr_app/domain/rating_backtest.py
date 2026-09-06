from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any, Iterable

from jupr_app.domain.constants import (
    CAP_LOSER_GAIN_ELO,
    DEFAULT_K_FACTOR,
    MIN_WIN_DELTA_ELO,
)
from jupr_app.domain.rating_policy import (
    RATING_ALGORITHM_VERSION,
    RATING_PARAMETER_VERSION,
    rating_parameter_snapshot,
)
from jupr_app.domain.ratings import (
    calculate_hybrid_elo,
    expected_team1_performance,
)


@dataclass(frozen=True)
class BacktestConfig:
    algorithm_version: str = RATING_ALGORITHM_VERSION
    parameter_version: str = RATING_PARAMETER_VERSION
    k_factor: float = float(DEFAULT_K_FACTOR)
    min_win_delta_elo: float = float(MIN_WIN_DELTA_ELO)
    cap_loser_gain_elo: float | None = float(CAP_LOSER_GAIN_ELO)
    fallback_initial_rating_elo: float = 1200.0


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _safe_player_id(value: Any) -> str | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    clean = str(value).strip()
    return clean or None


def _timestamp(value: Any) -> datetime | None:
    clean = str(value or "").strip()
    if not clean:
        return None
    try:
        parsed = datetime.fromisoformat(clean.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _match_format(row: dict[str, Any]) -> str:
    explicit = str(row.get("match_format") or "").strip().casefold()
    if explicit in {"singles", "doubles"}:
        return explicit
    return (
        "singles"
        if str(row.get("match_type") or "").strip().casefold() == "singles"
        else "doubles"
    )


def _match_order(value: Any) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value or ""))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _metric_summary(predictions: list[tuple[float, float, float]]) -> dict[str, float | int | None]:
    if not predictions:
        return {
            "matches": 0,
            "brier_score": None,
            "log_loss": None,
            "outcome_accuracy": None,
            "score_share_mae": None,
        }
    brier = []
    log_losses = []
    accuracy = []
    share_errors = []
    for probability, outcome, score_share in predictions:
        clipped = min(max(probability, 1e-12), 1.0 - 1e-12)
        brier.append((probability - outcome) ** 2)
        log_losses.append(
            -(
                outcome * math.log(clipped)
                + (1.0 - outcome) * math.log(1.0 - clipped)
            )
        )
        accuracy.append(float((probability >= 0.5) == (outcome == 1.0)))
        share_errors.append(abs(probability - score_share))
    return {
        "matches": len(predictions),
        "brier_score": _mean(brier),
        "log_loss": _mean(log_losses),
        "outcome_accuracy": _mean(accuracy),
        "score_share_mae": _mean(share_errors),
    }


def _calibration(predictions: list[tuple[float, float, float]]) -> list[dict[str, Any]]:
    buckets: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for probability, outcome, _score_share in predictions:
        bucket = min(9, int(probability * 10.0))
        buckets[bucket].append((probability, outcome))
    return [
        {
            "range": f"{bucket / 10:.1f}–{(bucket + 1) / 10:.1f}",
            "matches": len(values),
            "mean_prediction": _mean([item[0] for item in values]),
            "team1_win_rate": _mean([item[1] for item in values]),
        }
        for bucket, values in sorted(buckets.items())
    ]


def run_chronological_backtest(
    rows: Iterable[dict[str, Any]],
    *,
    config: BacktestConfig = BacktestConfig(),
    evaluation_start: str | datetime | None = None,
    evaluation_end: str | datetime | None = None,
) -> dict[str, Any]:
    """Walk forward through matches without using any future result.

    A player is seeded from their first stored pre-match snapshot when one is
    available, otherwise from ``fallback_initial_rating_elo``. Every later
    prediction uses only the simulated state produced by earlier matches.
    """

    start_at = (
        evaluation_start
        if isinstance(evaluation_start, datetime)
        else _timestamp(evaluation_start)
    )
    end_at = (
        evaluation_end
        if isinstance(evaluation_end, datetime)
        else _timestamp(evaluation_end)
    )
    if evaluation_start is not None and start_at is None:
        raise ValueError(f"Invalid evaluation_start: {evaluation_start!r}")
    if evaluation_end is not None and end_at is None:
        raise ValueError(f"Invalid evaluation_end: {evaluation_end!r}")
    if isinstance(start_at, datetime) and start_at.tzinfo is None:
        start_at = start_at.replace(tzinfo=timezone.utc)
    if isinstance(end_at, datetime) and end_at.tzinfo is None:
        end_at = end_at.replace(tzinfo=timezone.utc)
    if start_at is not None and end_at is not None and start_at >= end_at:
        raise ValueError("evaluation_start must be earlier than evaluation_end")

    prepared: list[
        tuple[datetime, tuple[int, int | str], int, dict[str, Any]]
    ] = []
    skipped: dict[str, int] = defaultdict(int)
    for index, raw_row in enumerate(rows):
        row = dict(raw_row)
        played_at = _timestamp(row.get("date"))
        if played_at is None:
            skipped["invalid_date"] += 1
            continue
        prepared.append((played_at, _match_order(row.get("id")), index, row))
    prepared.sort(key=lambda item: (item[0], item[1], item[2]))

    ratings: dict[tuple[str, str, str], float] = {}
    predictions: list[tuple[float, float, float]] = []
    by_format: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_month: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    by_club: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    player_ids_seen: set[tuple[str, str]] = set()
    winner_gain_violations = 0
    loser_outperformance_gains = 0
    first_match_at: datetime | None = None
    last_match_at: datetime | None = None

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

        match_format = _match_format(row)
        club_id = str(row.get("club_id") or "__single_club_input__")
        if match_format == "singles":
            slots = (
                (("t1_p1", "t1_p1_r"),),
                (("t2_p1", "t2_p1_r"),),
            )
        else:
            slots = (
                (("t1_p1", "t1_p1_r"), ("t1_p2", "t1_p2_r")),
                (("t2_p1", "t2_p1_r"), ("t2_p2", "t2_p2_r")),
            )

        teams: list[list[str]] = []
        team_seed_fields: list[list[str]] = []
        valid = True
        for team_slots in slots:
            team: list[str] = []
            seed_fields: list[str] = []
            for player_field, rating_field in team_slots:
                player_id = _safe_player_id(row.get(player_field))
                if player_id is None:
                    valid = False
                    break
                team.append(player_id)
                seed_fields.append(rating_field)
            teams.append(team)
            team_seed_fields.append(seed_fields)
        if not valid or len(
            {player for team in teams for player in team}
        ) != sum(map(len, teams)):
            if in_evaluation:
                skipped["invalid_players"] += 1
            continue

        for team, seed_fields in zip(teams, team_seed_fields):
            for player_id, rating_field in zip(team, seed_fields):
                key = (club_id, match_format, player_id)
                if key not in ratings:
                    stored_seed = _safe_float(row.get(rating_field))
                    ratings[key] = (
                        stored_seed
                        if stored_seed is not None
                        else float(config.fallback_initial_rating_elo)
                    )
                player_ids_seen.add((club_id, player_id))

        team1_rating = _mean(
            [ratings[(club_id, match_format, player)] for player in teams[0]]
        )
        team2_rating = _mean(
            [ratings[(club_id, match_format, player)] for player in teams[1]]
        )
        probability = expected_team1_performance(team1_rating, team2_rating)
        outcome = 1.0 if score_t1 > score_t2 else 0.0
        score_share = score_t1 / (score_t1 + score_t2)
        observation = (probability, outcome, score_share)
        if in_evaluation:
            predictions.append(observation)
            by_format[match_format].append(observation)
            by_month[played_at.strftime("%Y-%m")].append(observation)
            by_club[club_id].append(observation)

        delta1, delta2 = calculate_hybrid_elo(
            team1_rating,
            team2_rating,
            int(score_t1),
            int(score_t2),
            k_factor=float(config.k_factor),
            min_win_delta=float(config.min_win_delta_elo),
            cap_loser_gain=config.cap_loser_gain_elo,
        )
        winner_bonus = max(0.0, _safe_float(row.get("rating_bonus_elo")) or 0.0)
        winning_delta = delta1 if outcome == 1.0 else delta2
        losing_delta = delta2 if outcome == 1.0 else delta1
        if in_evaluation:
            if winning_delta + winner_bonus <= 0:
                winner_gain_violations += 1
            if losing_delta > 0:
                loser_outperformance_gains += 1
        for player in teams[0]:
            ratings[(club_id, match_format, player)] += delta1 + (
                winner_bonus if outcome == 1.0 else 0.0
            )
        for player in teams[1]:
            ratings[(club_id, match_format, player)] += delta2 + (
                winner_bonus if outcome == 0.0 else 0.0
            )

        if in_evaluation:
            first_match_at = first_match_at or played_at
            last_match_at = played_at

    metrics = _metric_summary(predictions)
    naive = _metric_summary([(0.5, outcome, share) for _, outcome, share in predictions])
    brier = metrics["brier_score"]
    naive_brier = naive["brier_score"]
    return {
        "model": {
            "algorithm_version": config.algorithm_version,
            "parameter_version": config.parameter_version,
            "parameters": rating_parameter_snapshot(
                overall_k_factor=config.k_factor,
                min_win_delta_elo=config.min_win_delta_elo,
                cap_loser_gain_elo=config.cap_loser_gain_elo,
            ),
        },
        "method": "chronological_walk_forward",
        "evaluation_window": {
            "start_inclusive": start_at.isoformat() if start_at else None,
            "end_exclusive": end_at.isoformat() if end_at else None,
        },
        "first_match_at": first_match_at.isoformat() if first_match_at else None,
        "last_match_at": last_match_at.isoformat() if last_match_at else None,
        "players_simulated": len(player_ids_seen),
        "rating_states_simulated": len(ratings),
        "metrics": metrics,
        "naive_50_percent_baseline": naive,
        "brier_improvement_vs_50_percent": (
            float(naive_brier) - float(brier)
            if brier is not None and naive_brier is not None
            else None
        ),
        "by_format": {
            key: _metric_summary(value) for key, value in sorted(by_format.items())
        },
        "by_club": {
            key: _metric_summary(value) for key, value in sorted(by_club.items())
        },
        "by_month": {
            key: _metric_summary(value) for key, value in sorted(by_month.items())
        },
        "calibration": _calibration(predictions),
        "policy_checks": {
            "winner_gain_violations": winner_gain_violations,
            "loser_outperformance_gains": loser_outperformance_gains,
        },
        "skipped": dict(sorted(skipped.items())),
    }
