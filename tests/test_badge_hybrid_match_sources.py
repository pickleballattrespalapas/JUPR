from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club
from jupr_app.domain.gamification.match_facts import (
    build_canonical_player_match_facts,
    build_hybrid_player_match_facts,
    build_legacy_safe_player_match_facts,
)
from jupr_app.domain.gamification.participation import compute_lifetime_games


def _ctx(df_matches: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        club_id="club",
        df_matches=df_matches,
        df_players_all=pd.DataFrame([{"id": 1, "rating": 1200.0}, {"id": 2, "rating": 1200.0}]),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        df_badges=pd.DataFrame(
            [
                {"badge_id": "participant", "state": "live"},
                {"badge_id": "first_win", "state": "live"},
                {"badge_id": "high_roller", "state": "live"},
                {"badge_id": "david_vs_goliath", "state": "live"},
                {"badge_id": "above_expectations", "state": "live"},
            ]
        ),
    )


def test_legacy_safe_rows_award_hybrid_badges_but_not_canonical_only():
    rows = []
    for i in range(100):
        rows.append(
            {
                "id": f"legacy-{i}",
                "club_id": "club",
                "date": f"2024-01-{(i % 28) + 1:02d}",
                "league_name": "Open",
                "team1_player1": 1,
                "team2_player1": 2,
                "team1_score": 11,
                "team2_score": 7,
            }
        )
    ctx = _ctx(pd.DataFrame(rows))
    candidates = list(compute_candidates_for_club("club", ctx=ctx))
    badge_ids = {(c.player_id, c.badge_id) for c in candidates}
    assert (1, "participant") in badge_ids
    assert (1, "first_win") in badge_ids
    assert (1, "high_roller") in badge_ids
    assert (1, "david_vs_goliath") not in badge_ids
    assert (1, "above_expectations") not in badge_ids


def test_hybrid_facts_prefer_canonical_when_same_player_match_pair_exists():
    df_matches = pd.DataFrame(
        [
            {
                "id": "m1",
                "club_id": "club",
                "date": "2024-01-01",
                "league": "Open",
                "t1_p1": 1,
                "t2_p1": 2,
                "score_t1": 11,
                "score_t2": 9,
                "t1_p1_r": 1200.0,
                "t2_p1_r": 1200.0,
                "elo_delta": 8.0,
            },
            {
                "id": "m1",
                "club_id": "club",
                "date": "2024-01-01",
                "league_name": "Open",
                "team1_player1": 1,
                "team2_player1": 2,
                "team1_score": 11,
                "team2_score": 9,
            },
        ]
    )
    ctx = _ctx(df_matches)
    canonical = build_canonical_player_match_facts(ctx)
    legacy = build_legacy_safe_player_match_facts(ctx)
    hybrid = build_hybrid_player_match_facts(ctx)

    assert len(canonical) == 2
    assert len(legacy) == 2
    assert len(hybrid) == 2
    p1_row = hybrid[(hybrid["player_id"] == 1) & (hybrid["match_id"] == "m1")].iloc[0]
    assert p1_row["fact_source"] == "canonical"


def test_compute_lifetime_games_counts_legacy_safe_rows_with_dedupe():
    df_matches = pd.DataFrame(
        [
            {
                "id": "m1",
                "club_id": "club",
                "date": "2024-01-01",
                "team1_player1": 1,
                "team2_player1": 2,
                "team1_score": 11,
                "team2_score": 8,
            },
            {
                "id": "m1",
                "club_id": "club",
                "date": "2024-01-01",
                "team1_player1": 1,
                "team2_player1": 2,
                "team1_score": 11,
                "team2_score": 8,
            },
        ]
    )
    counts = compute_lifetime_games(_ctx(df_matches))
    assert counts == {1: 1, 2: 1}
