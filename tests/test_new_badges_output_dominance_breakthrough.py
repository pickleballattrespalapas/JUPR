from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_types import BadgeEvaluationContext
from jupr_app.domain.gamification.evaluators import (
    evaluate_above_expectations,
    evaluate_breakthrough,
    evaluate_dominant_run,
    evaluate_high_output,
)


def _context_with_facts(facts: pd.DataFrame, df_leagues: pd.DataFrame | None = None) -> BadgeEvaluationContext:
    ctx = SimpleNamespace(df_leagues=df_leagues)
    return BadgeEvaluationContext(
        club_id="club",
        league_id=None,
        as_of=None,
        ctx=ctx,
        facts=facts,
        matches=facts,
    )


def test_dominant_run_awards_at_10th_win():
    rows = []
    for idx in range(10):
        rows.append(
            {
                "player_id": 1,
                "match_id": idx + 1,
                "league": "A" if idx % 2 == 0 else "B",
                "date_dt": pd.Timestamp(2024, 1, idx + 1, tz="UTC"),
                "win": True,
                "margin": 5,
            }
        )
    facts = pd.DataFrame(rows)
    ctx = _context_with_facts(facts)

    candidates = list(evaluate_dominant_run(ctx))

    assert any(
        candidate.badge_id == "dominant_run" and candidate.match_id == "10"
        for candidate in candidates
    )


def test_high_output_awards_on_window_end():
    rows = []
    for idx in range(25):
        rows.append(
            {
                "player_id": 2,
                "match_id": f"m{idx + 1}",
                "league": "L1",
                "date_dt": pd.Timestamp(2024, 2, idx + 1, tz="UTC"),
                "win": idx < 20,
                "margin": 4,
            }
        )
    facts = pd.DataFrame(rows)
    ctx = _context_with_facts(facts)

    candidates = list(evaluate_high_output(ctx))

    assert any(
        candidate.badge_id == "high_output" and candidate.match_id == "m25"
        for candidate in candidates
    )


def test_breakthrough_awards_top25_and_top10():
    players = []
    for pid in range(1, 31):
        players.append(
            {
                "player_id": pid,
                "league_name": "Premier",
                "starting_rating": 2000 - pid,
                "rating": 2000 - pid,
                "matches_played": 12,
            }
        )
    players[-1]["player_id"] = 99
    players[-1]["starting_rating"] = 1700
    players[-1]["rating"] = 2600
    df_leagues = pd.DataFrame(players)
    ctx = _context_with_facts(pd.DataFrame(), df_leagues=df_leagues)

    candidates = [c for c in evaluate_breakthrough(ctx) if c.player_id == 99]
    tiers = sorted(c.value_json["tier"] for c in candidates)

    assert tiers == [10, 25]


def test_above_expectations_awards_with_quality_filter():
    rows = [
        {
            "player_id": 5,
            "match_id": "m1",
            "league": "L1",
            "date_dt": pd.Timestamp(2024, 3, 1, tz="UTC"),
            "win": True,
            "margin": 4,
            "expected_win_prob": 0.35,
            "abs_elo_delta": 10.0,
        },
        {
            "player_id": 6,
            "match_id": "m2",
            "league": "L1",
            "date_dt": pd.Timestamp(2024, 3, 2, tz="UTC"),
            "win": True,
            "margin": 1,
            "expected_win_prob": 0.6,
            "abs_elo_delta": 1.0,
        },
        {
            "player_id": 7,
            "match_id": "m3",
            "league": "L1",
            "date_dt": pd.Timestamp(2024, 3, 3, tz="UTC"),
            "win": False,
            "margin": -2,
            "expected_win_prob": 0.2,
            "abs_elo_delta": 2.0,
        },
        {
            "player_id": 8,
            "match_id": "m4",
            "league": "L1",
            "date_dt": pd.Timestamp(2024, 3, 4, tz="UTC"),
            "win": True,
            "margin": 4,
            "expected_win_prob": 0.45,
            "abs_elo_delta": 3.0,
        },
    ]
    facts = pd.DataFrame(rows)
    ctx = _context_with_facts(facts)

    candidates = list(evaluate_above_expectations(ctx))

    assert any(candidate.match_id == "m1" for candidate in candidates)
