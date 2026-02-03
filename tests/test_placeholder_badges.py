from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_types import BadgeEvaluationContext
from jupr_app.domain.gamification.evaluators import (
    evaluate_above_expectations,
    evaluate_breakthrough,
    evaluate_dominant_run,
    evaluate_high_output,
)


def _context_with_facts(facts: pd.DataFrame) -> BadgeEvaluationContext:
    return BadgeEvaluationContext(
        club_id="club",
        league_id=None,
        as_of=None,
        ctx=SimpleNamespace(),
        facts=facts,
        matches=facts,
    )


def test_dominant_run_awards_streak_milestones():
    rows = []
    for idx in range(1, 21):
        rows.append(
            {
                "player_id": 7,
                "league": "League A",
                "win": True,
                "match_id": f"m{idx}",
                "date_dt": pd.Timestamp(f"2024-01-{idx:02d}T10:00:00Z"),
            }
        )
    facts = pd.DataFrame(rows)
    ctx = _context_with_facts(facts)

    candidates = list(evaluate_dominant_run(ctx))

    assert [c.value_num for c in candidates] == [5.0, 10.0, 20.0]


def test_high_output_awards_once_per_week():
    rows = [
        {
            "player_id": 1,
            "win": True,
            "points_for": 11,
            "points_against": 5,
            "week_key": "2024-W01",
            "match_id": "m1",
            "date_dt": pd.Timestamp("2024-01-02T10:00:00Z"),
        },
        {
            "player_id": 1,
            "win": True,
            "points_for": 11,
            "points_against": 7,
            "week_key": "2024-W01",
            "match_id": "m2",
            "date_dt": pd.Timestamp("2024-01-03T10:00:00Z"),
        },
        {
            "player_id": 1,
            "win": True,
            "points_for": 11,
            "points_against": 6,
            "week_key": "2024-W02",
            "match_id": "m3",
            "date_dt": pd.Timestamp("2024-01-09T10:00:00Z"),
        },
    ]
    facts = pd.DataFrame(rows)
    ctx = _context_with_facts(facts)

    candidates = list(evaluate_high_output(ctx))

    assert len(candidates) == 2
    assert {c.context_id for c in candidates} == {"week:2024-W01", "week:2024-W02"}


def test_above_expectations_awards_underdog_win():
    facts = pd.DataFrame(
        [
            {
                "player_id": 3,
                "win": True,
                "expected_win_prob": 0.4,
                "match_id": "m1",
                "date_dt": pd.Timestamp("2024-01-02T10:00:00Z"),
            },
            {
                "player_id": 3,
                "win": True,
                "expected_win_prob": 0.41,
                "match_id": "m2",
                "date_dt": pd.Timestamp("2024-01-03T10:00:00Z"),
            },
        ]
    )
    ctx = _context_with_facts(facts)

    candidates = list(evaluate_above_expectations(ctx))

    assert len(candidates) == 1
    assert candidates[0].match_id == "m1"


def test_above_expectations_inactive_without_probability():
    facts = pd.DataFrame(
        [
            {
                "player_id": 3,
                "win": True,
                "match_id": "m1",
                "date_dt": pd.Timestamp("2024-01-02T10:00:00Z"),
            }
        ]
    )
    ctx = _context_with_facts(facts)

    assert list(evaluate_above_expectations(ctx)) == []


def test_breakthrough_awards_milestones_once():
    facts = pd.DataFrame(
        [
            {
                "player_id": 9,
                "match_id": "m1",
                "rating_pre": 3.1,
                "rating_post": 3.3,
                "date_dt": pd.Timestamp("2024-01-02T10:00:00Z"),
            },
            {
                "player_id": 9,
                "match_id": "m2",
                "rating_pre": 3.6,
                "rating_post": 3.8,
                "date_dt": pd.Timestamp("2024-01-05T10:00:00Z"),
            },
        ]
    )
    ctx = _context_with_facts(facts)

    candidates = list(evaluate_breakthrough(ctx))

    assert [c.value_num for c in candidates] == [3.25, 3.75]
