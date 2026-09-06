from __future__ import annotations

from jupr_app.domain.bayesian_doubles_shadow import (
    BayesianDoublesConfig,
    run_bayesian_doubles_backtest,
)
from jupr_app.domain.rating_model_comparison import (
    compare_jupr_with_bayesian_shadow,
)


def _match(
    match_id: int,
    date: str,
    team1: tuple[int, int],
    team2: tuple[int, int],
    score: tuple[int, int],
) -> dict:
    return {
        "id": match_id,
        "club_id": "club-a",
        "date": date,
        "league": "Ladder",
        "match_format": "doubles",
        "rating_scope": "overall_only",
        "t1_p1": team1[0],
        "t1_p2": team1[1],
        "t2_p1": team2[0],
        "t2_p2": team2[1],
        "score_t1": score[0],
        "score_t2": score[1],
        "t1_p1_r": 1200,
        "t1_p2_r": 1200,
        "t2_p1_r": 1200,
        "t2_p2_r": 1200,
    }


def test_equal_priors_begin_with_even_prediction() -> None:
    report = run_bayesian_doubles_backtest(
        [_match(1, "2026-01-01", (1, 2), (3, 4), (11, 7))]
    )

    assert report["metrics"]["matches"] == 1
    assert report["metrics"]["brier_score"] == 0.25
    assert report["model"]["player_facing"] is False
    assert report["model"]["uses_recency_drift"] is False


def test_score_margin_changes_future_bayesian_prediction() -> None:
    close_rows = [
        _match(1, "2026-01-01", (1, 2), (3, 4), (11, 10)),
        _match(2, "2026-01-02", (1, 2), (3, 4), (11, 9)),
    ]
    blowout_rows = [
        _match(1, "2026-01-01", (1, 2), (3, 4), (11, 0)),
        _match(2, "2026-01-02", (1, 2), (3, 4), (11, 9)),
    ]

    close = run_bayesian_doubles_backtest(
        close_rows, evaluation_start="2026-01-02"
    )
    blowout = run_bayesian_doubles_backtest(
        blowout_rows, evaluation_start="2026-01-02"
    )

    assert blowout["metrics"]["brier_score"] < close["metrics"]["brier_score"]


def test_rotating_partners_supply_individual_evidence() -> None:
    rows = [
        _match(1, "2026-01-01", (1, 2), (3, 4), (11, 4)),
        _match(2, "2026-01-02", (1, 3), (2, 4), (11, 5)),
        _match(3, "2026-01-03", (1, 4), (2, 3), (11, 6)),
        _match(4, "2026-01-04", (1, 2), (3, 4), (11, 7)),
    ]

    report = run_bayesian_doubles_backtest(
        rows, evaluation_start="2026-01-04"
    )

    assert report["metrics"]["outcome_accuracy"] == 1.0
    assert report["training_matches_before_window"] == 3


def test_input_order_does_not_change_chronological_result() -> None:
    rows = [
        _match(2, "2026-01-02", (1, 2), (3, 4), (7, 11)),
        _match(1, "2026-01-01", (1, 2), (3, 4), (11, 4)),
    ]

    forward = run_bayesian_doubles_backtest(rows)
    reverse = run_bayesian_doubles_backtest(list(reversed(rows)))

    assert forward["metrics"] == reverse["metrics"]


def test_parameter_selection_cannot_see_holdout() -> None:
    rows = [
        _match(1, "2026-01-01", (1, 2), (3, 4), (11, 5)),
        _match(2, "2026-03-10", (1, 2), (3, 4), (11, 7)),
        _match(3, "2026-04-10", (1, 2), (3, 4), (11, 9)),
    ]
    candidates = (
        BayesianDoublesConfig(prior_sigma_elo=100, performance_sigma_elo=100),
        BayesianDoublesConfig(prior_sigma_elo=300, performance_sigma_elo=300),
    )

    first = compare_jupr_with_bayesian_shadow(
        rows,
        validation_start="2026-03-01",
        validation_end="2026-04-01",
        holdout_start="2026-04-01",
        candidates=candidates,
    )
    rows[-1]["score_t1"], rows[-1]["score_t2"] = 1, 11
    changed_holdout = compare_jupr_with_bayesian_shadow(
        rows,
        validation_start="2026-03-01",
        validation_end="2026-04-01",
        holdout_start="2026-04-01",
        candidates=candidates,
    )

    assert (
        first["selection"]["selected_parameters"]
        == changed_holdout["selection"]["selected_parameters"]
    )
    assert first["holdout"] != changed_holdout["holdout"]
