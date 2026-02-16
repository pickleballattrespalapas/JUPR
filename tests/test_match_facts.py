from types import SimpleNamespace
from pathlib import Path

import pandas as pd
import pytest

from jupr_app.domain.gamification.match_facts import build_player_match_facts


def test_build_player_match_facts_filters_and_outputs():
    df_matches = pd.DataFrame(
        [
            {
                "id": "m1",
                "club_id": "club",
                "league": "A",
                "date": "2024-01-01",
                "score_t1": 11,
                "score_t2": 5,
                "t1_p1": 1,
                "t1_p2": None,
                "t2_p1": 2,
                "t2_p2": None,
                "t1_p1_r": 1500,
                "t2_p1_r": 1400,
                "match_type": "League",
                "is_valid": True,
                "context_type": "LEAGUE",
                "tournament_id": None,
                "elo_delta": 10,
            },
            {
                "id": "m2",
                "club_id": "club",
                "league": "A",
                "date": "2024-01-02",
                "score_t1": 11,
                "score_t2": 9,
                "t1_p1": 1,
                "t1_p2": 3,
                "t2_p1": 2,
                "t2_p2": 4,
                "t1_p1_r": 1500,
                "t1_p2_r": 1550,
                "t2_p1_r": 1400,
                "t2_p2_r": 1450,
                "match_type": "League",
                "is_valid": True,
                "context_type": "LEAGUE",
                "tournament_id": None,
                "elo_delta": 5,
            },
            {
                "id": "m3",
                "club_id": "club",
                "league": "A",
                "date": "2024-01-03",
                "score_t1": 11,
                "score_t2": 7,
                "t1_p1": 1,
                "t2_p1": 2,
                "match_type": "PopUp",
                "is_valid": True,
                "context_type": "LEAGUE",
                "tournament_id": None,
            },
            {
                "id": "m4",
                "club_id": "club",
                "league": "A",
                "date": "2024-01-04",
                "score_t1": 11,
                "score_t2": 6,
                "t1_p1": 1,
                "t2_p1": 2,
                "match_type": "League",
                "is_valid": True,
                "context_type": "TOURNAMENT",
                "tournament_id": None,
            },
            {
                "id": "m5",
                "club_id": "club",
                "league": "A",
                "date": "2024-01-05",
                "score_t1": 11,
                "score_t2": 8,
                "t1_p1": 1,
                "t2_p1": 2,
                "match_type": "League",
                "is_valid": False,
                "context_type": "LEAGUE",
                "tournament_id": None,
            },
            {
                "id": "m6",
                "club_id": "club",
                "league": "A",
                "date": "2024-01-06",
                "score_t1": 11,
                "score_t2": 3,
                "t1_p1": 1,
                "t2_p1": 2,
                "match_type": "League",
                "is_void": True,
                "context_type": "LEAGUE",
                "tournament_id": None,
            },
            {
                "id": "m7",
                "club_id": "club",
                "league": "A",
                "date": "2024-01-07",
                "score_t1": 11,
                "score_t2": 3,
                "t1_p1": 1,
                "t2_p1": 2,
                "match_type": "League",
                "deleted": True,
                "context_type": "LEAGUE",
                "tournament_id": None,
            },
            {
                "id": "m8",
                "club_id": "club",
                "league": "A",
                "date": "2024-01-08",
                "score_t1": 0,
                "score_t2": 0,
                "t1_p1": 1,
                "t2_p1": 2,
                "match_type": "League",
                "is_valid": True,
                "context_type": "LEAGUE",
                "tournament_id": None,
            },
            {
                "id": "m9",
                "club_id": "other",
                "league": "A",
                "date": "2024-01-09",
                "score_t1": 11,
                "score_t2": 9,
                "t1_p1": 1,
                "t2_p1": 2,
                "match_type": "League",
                "is_valid": True,
                "context_type": "LEAGUE",
                "tournament_id": None,
            },
        ]
    )

    ctx = SimpleNamespace(df_matches=df_matches, club_id="club")
    facts = build_player_match_facts(ctx)

    expected_columns = [
        "club_id",
        "player_id",
        "match_id",
        "league",
        "date_dt",
        "week_key",
        "month_key",
        "season_key",
        "win",
        "points_for",
        "points_against",
        "margin",
        "partner_id",
        "opponent_ids",
        "expected_win_prob",
        "elo_delta_signed",
        "abs_elo_delta",
        "opp_max_rating",
        "lobby_avg_rating",
    ]
    assert list(facts.columns) == expected_columns
    assert len(facts) == 6

    m1_p1 = facts[(facts["match_id"] == "m1") & (facts["player_id"] == 1)].iloc[0]
    assert bool(m1_p1.win) is True
    assert m1_p1.margin == 6
    assert m1_p1.points_for == 11
    assert m1_p1.points_against == 5
    assert pd.isna(m1_p1.partner_id)
    assert m1_p1.opponent_ids == [2]
    expected_win = 1.0 / (1.0 + 10 ** ((1400 - 1500) / 400.0))
    assert m1_p1.expected_win_prob == pytest.approx(expected_win)

    m2_p1 = facts[(facts["match_id"] == "m2") & (facts["player_id"] == 1)].iloc[0]
    assert bool(m2_p1.win) is True
    assert m2_p1.partner_id == 3
    assert sorted(m2_p1.opponent_ids) == [2, 4]
    assert m2_p1.margin == 2

    week_key = pd.Timestamp("2024-01-01", tz="UTC").isocalendar()
    assert m1_p1.week_key == f"{week_key.year}-W{int(week_key.week):02d}"
    assert m1_p1.month_key == "2024-01"
    assert m1_p1.season_key == "2024"


def test_no_gamification_badge_rules_imports_in_domain_modules():
    repo_root = Path(__file__).resolve().parents[1]
    roots = [repo_root / "jupr_app" / "domain", repo_root / "jupr_app" / "domain" / "gamification"]
    disallowed = "badge_rules"

    for root in roots:
        for path in root.rglob("*.py"):
            contents = path.read_text(encoding="utf-8")
            assert disallowed not in contents, f"Found '{disallowed}' in {path}"
