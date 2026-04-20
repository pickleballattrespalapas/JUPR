from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_audit import build_high_roller_diagnostic_report
from jupr_app.domain.gamification.badge_types import BadgeEvaluationContext
from jupr_app.domain.gamification.evaluators import compute_high_roller_win_counts, evaluate_high_roller


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


def test_compute_high_roller_win_counts_112_qualifies_99_not():
    rows = [{"player_id": 11, "match_id": f"j-{i}", "win": True} for i in range(112)]
    rows.extend({"player_id": 22, "match_id": f"t-{i}", "win": True} for i in range(99))
    rows.extend({"player_id": 11, "match_id": f"j-loss-{i}", "win": False} for i in range(10))
    counts = compute_high_roller_win_counts(pd.DataFrame(rows))

    assert int(counts[11]) == 112
    assert int(counts[22]) == 99
    assert int(counts[11]) >= 100
    assert int(counts[22]) < 100


def test_high_roller_diagnostic_report_exclusions_and_stable_counts():
    rows = []
    for i in range(112):
        rows.append(
            {
                "id": f"j-{i}",
                "club_id": "club",
                "league": "A",
                "date": f"2024-01-{(i % 28) + 1:02d}",
                "score_t1": 11,
                "score_t2": 7,
                "t1_p1": 11,
                "t2_p1": 33,
                "match_type": "League",
                "is_valid": True,
                "context_type": "LEAGUE",
                "tournament_id": None,
            }
        )
    for i in range(99):
        rows.append(
            {
                "id": f"t-{i}",
                "club_id": "club",
                "league": "A",
                "date": f"2024-02-{(i % 28) + 1:02d}",
                "score_t1": 11,
                "score_t2": 8,
                "t1_p1": 22,
                "t2_p1": 44,
                "match_type": "League",
                "is_valid": True,
                "context_type": "LEAGUE",
                "tournament_id": None,
            }
        )
    rows.extend(
        [
            {
                "id": "popup-1",
                "club_id": "club",
                "date": "2024-03-01",
                "score_t1": 11,
                "score_t2": 5,
                "t1_p1": 11,
                "t2_p1": 33,
                "match_type": "PopUp",
                "is_valid": True,
                "context_type": "LEAGUE",
                "tournament_id": None,
            },
            {
                "id": "tournament-1",
                "club_id": "club",
                "date": "2024-03-02",
                "score_t1": 11,
                "score_t2": 5,
                "t1_p1": 11,
                "t2_p1": 33,
                "match_type": "League",
                "is_valid": True,
                "context_type": "TOURNAMENT",
                "tournament_id": None,
            },
            {
                "id": "invalid-1",
                "club_id": "club",
                "date": "2024-03-03",
                "score_t1": 11,
                "score_t2": 5,
                "t1_p1": 11,
                "t2_p1": 33,
                "match_type": "League",
                "is_valid": False,
                "context_type": "LEAGUE",
                "tournament_id": None,
            },
            {
                "id": None,
                "match_id": "j-0",
                "club_id": "club",
                "date": "2024-03-04",
                "score1": 11,
                "score2": 7,
                "player1_id": 11,
                "player3_id": 33,
                "match_type": "League",
                "is_valid": True,
                "context_type": "LEAGUE",
                "tournament_id": None,
            },
        ]
    )
    ctx = SimpleNamespace(club_id="club", df_matches=pd.DataFrame(rows))

    report = build_high_roller_diagnostic_report(
        supabase=SimpleNamespace(),
        club_id="club",
        player_id=11,
        ctx=ctx,
    )

    selected = report["selected_player"]
    assert selected["hybrid_unique_win_match_ids"] == 112
    assert selected["canonical_unique_win_match_ids"] == 112
    assert selected["qualifies_high_roller_hybrid"] is True
    assert report["hybrid_unique_win_match_ids_by_player"]["22"] == 99
    assert len(report["top_20_players_by_hybrid_unique_win_count"]) >= 2

    removed_by_step = {step["step_name"]: step["removed_count"] for step in report["filter_steps"]["steps"]}
    assert removed_by_step["exclude_popups"] == 1
    assert removed_by_step["is_valid"] == 1
    assert removed_by_step["exclude_context_type_tournament"] == 1
