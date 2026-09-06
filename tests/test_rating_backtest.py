from __future__ import annotations

from jupr_app.domain.rating_backtest import run_chronological_backtest
from jupr_app.domain.rating_policy import (
    RATING_ALGORITHM_VERSION,
    RATING_PARAMETER_VERSION,
)


def _match(match_id: int, date: str, score_t1: int, score_t2: int) -> dict:
    return {
        "id": match_id,
        "date": date,
        "match_format": "doubles",
        "rating_scope": "overall_only",
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "score_t1": score_t1,
        "score_t2": score_t2,
        "t1_p1_r": 1200,
        "t1_p2_r": 1200,
        "t2_p1_r": 1200,
        "t2_p2_r": 1200,
    }


def test_backtest_is_chronological_and_uses_only_prior_updates() -> None:
    rows = [
        _match(2, "2026-01-02T00:00:00Z", 0, 11),
        _match(1, "2026-01-01T00:00:00Z", 11, 0),
    ]

    report = run_chronological_backtest(rows)

    assert report["metrics"]["matches"] == 2
    assert report["metrics"]["outcome_accuracy"] == 0.5
    assert report["first_match_at"] == "2026-01-01T00:00:00+00:00"
    assert report["last_match_at"] == "2026-01-02T00:00:00+00:00"
    assert report["policy_checks"]["winner_gain_violations"] == 0


def test_backtest_separates_singles_state_and_reports_skips() -> None:
    singles = {
        "id": 3,
        "date": "2026-01-03",
        "match_format": "singles",
        "rating_scope": "overall_only",
        "t1_p1": 1,
        "t2_p1": 3,
        "score_t1": 11,
        "score_t2": 9,
        "t1_p1_r": 1300,
        "t2_p1_r": 1200,
    }
    unrated = {**_match(4, "2026-01-04", 11, 5), "rating_scope": "unrated"}

    report = run_chronological_backtest([singles, unrated])

    assert report["by_format"]["singles"]["matches"] == 1
    assert report["skipped"]["unrated"] == 1
    assert report["model"]["algorithm_version"] == RATING_ALGORITHM_VERSION
    assert report["model"]["parameter_version"] == RATING_PARAMETER_VERSION


def test_backtest_records_a_loser_gain_for_score_outperformance() -> None:
    row = {
        **_match(5, "2026-01-05", 11, 10),
        "t1_p1_r": 1600,
        "t1_p2_r": 1600,
        "t2_p1_r": 1000,
        "t2_p2_r": 1000,
    }

    report = run_chronological_backtest([row])

    assert report["policy_checks"]["winner_gain_violations"] == 0
    assert report["policy_checks"]["loser_outperformance_gains"] == 1


def test_backtest_keeps_club_rating_pools_isolated() -> None:
    club_a = {
        **_match(6, "2026-01-06", 11, 7),
        "club_id": "club-a",
    }
    club_b = {
        **_match(7, "2026-01-07", 7, 11),
        "club_id": "club-b",
    }

    report = run_chronological_backtest([club_a, club_b])

    assert report["by_club"]["club-a"]["matches"] == 1
    assert report["by_club"]["club-b"]["matches"] == 1
    assert report["players_simulated"] == 8
    assert report["rating_states_simulated"] == 8


def test_backtest_evaluation_window_trains_on_earlier_matches() -> None:
    rows = [
        _match(8, "2026-01-01", 11, 0),
        _match(9, "2026-02-01", 11, 9),
    ]

    report = run_chronological_backtest(
        rows,
        evaluation_start="2026-02-01",
        evaluation_end="2026-03-01",
    )

    assert report["metrics"]["matches"] == 1
    assert report["metrics"]["outcome_accuracy"] == 1.0
    assert report["first_match_at"] == "2026-02-01T00:00:00+00:00"
    assert report["evaluation_window"] == {
        "start_inclusive": "2026-02-01T00:00:00+00:00",
        "end_exclusive": "2026-03-01T00:00:00+00:00",
    }
