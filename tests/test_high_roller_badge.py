from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_types import BadgeEvaluationContext
from jupr_app.domain.gamification.evaluators import evaluate_high_roller


def _context_with_facts(facts: pd.DataFrame) -> BadgeEvaluationContext:
    return BadgeEvaluationContext(
        club_id="club",
        league_id=None,
        as_of=None,
        ctx=SimpleNamespace(),
        facts=facts,
        matches=facts,
    )


def test_high_roller_counts_distinct_match_wins():
    rows = []
    for match_index in range(1, 25):
        match_id = f"m{match_index}"
        for _ in range(5):
            rows.append({"player_id": 1, "match_id": match_id, "win": True})
    facts = pd.DataFrame(rows)
    ctx = _context_with_facts(facts)

    candidates = list(evaluate_high_roller(ctx))

    assert candidates == []


def test_high_roller_awards_at_100_wins():
    rows = [{"player_id": 7, "match_id": f"win-{i}", "win": True} for i in range(100)]
    rows.extend({"player_id": 7, "match_id": f"loss-{i}", "win": False} for i in range(20))
    facts = pd.DataFrame(rows)
    ctx = _context_with_facts(facts)

    candidates = list(evaluate_high_roller(ctx))

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.player_id == 7
    assert candidate.value_json["wins"] == 100
