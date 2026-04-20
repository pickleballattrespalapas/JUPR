from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club


def _match_row(match_id, date, t1_p1, t2_p1, score_t1, score_t2, league="A"):
    return {
        "id": match_id,
        "club_id": "club",
        "league": league,
        "date": date,
        "t1_p1": t1_p1,
        "t1_p2": None,
        "t2_p1": t2_p1,
        "t2_p2": None,
        "score_t1": score_t1,
        "score_t2": score_t2,
        "t1_p1_r": 1200.0,
        "t2_p1_r": 1200.0,
    }


def test_first_win_idempotent():
    df_matches = pd.DataFrame(
        [
            _match_row("m1", "2024-01-02", 1, 2, 11, 5),
        ]
    )
    df_players = pd.DataFrame([{"id": 1, "rating": 1200}, {"id": 2, "rating": 1200}])
    ctx = SimpleNamespace(df_matches=df_matches, df_players_all=df_players, club_id="club")

    candidates = list(compute_candidates_for_club("club", ctx=ctx))
    first_win = [a for a in candidates if a.badge_id == "first_win"]
    assert len(first_win) == 1
    assert first_win[0].context_id == "first_win"


def test_hot_streak_tiers_context():
    rows = []
    for i in range(20):
        rows.append(_match_row(f"m{i+1}", f"2024-02-{i+1:02d}", 1, 2, 11, 4))
    df_matches = pd.DataFrame(rows)
    df_players = pd.DataFrame([{"id": 1, "rating": 1400}, {"id": 2, "rating": 1400}])
    ctx = SimpleNamespace(df_matches=df_matches, df_players_all=df_players, club_id="club")

    candidates = list(compute_candidates_for_club("club", ctx=ctx))
    streaks = [a for a in candidates if a.badge_id == "hot_streak"]
    context_ids = {a.context_id for a in streaks}
    assert {"A:streak:5:m5", "A:streak:10:m10", "A:streak:20:m20"} == context_ids


def test_weekly_regular_consecutive_weeks():
    rows = [
        _match_row("m1", "2024-01-03", 1, 2, 11, 3),
        _match_row("m2", "2024-01-10", 1, 2, 11, 4),
        _match_row("m3", "2024-01-17", 1, 2, 11, 6),
        _match_row("m4", "2024-01-24", 1, 2, 11, 7),
    ]
    df_matches = pd.DataFrame(rows)
    df_players = pd.DataFrame([{"id": 1, "rating": 1400}, {"id": 2, "rating": 1400}])
    ctx = SimpleNamespace(df_matches=df_matches, df_players_all=df_players, club_id="club")

    candidates = list(compute_candidates_for_club("club", ctx=ctx))
    weekly = [a for a in candidates if a.badge_id == "weekly_regular" and a.player_id == 1]
    assert len(weekly) == 1
    assert weekly[0].context_id == "A:2024"


def test_upset_champion_picks_lowest_expected():
    rows = [
        _match_row("m1", "2024-03-05", 1, 2, 11, 9),
        _match_row("m2", "2024-03-10", 1, 2, 11, 9),
    ]
    df_matches = pd.DataFrame(rows)
    df_matches.loc[0, "t1_p1_r"] = 1200.0
    df_matches.loc[0, "t2_p1_r"] = 1500.0
    df_matches.loc[1, "t1_p1_r"] = 1300.0
    df_matches.loc[1, "t2_p1_r"] = 1400.0
    df_players = pd.DataFrame([{"id": 1, "rating": 1400}, {"id": 2, "rating": 1400}])
    ctx = SimpleNamespace(df_matches=df_matches, df_players_all=df_players, club_id="club")

    candidates = list(compute_candidates_for_club("club", ctx=ctx))
    upset = [a for a in candidates if a.badge_id == "upset_champion"]
    assert upset
    assert any(a.context_id.endswith("match:m1") for a in upset)


def test_canonical_only_badges_do_not_use_legacy_safe_only_rows():
    df_matches = pd.DataFrame(
        [
            {
                "id": "legacy-only",
                "club_id": "club",
                "date": "2024-01-02",
                "league_name": "A",
                "team1_player1": 1,
                "team2_player1": 2,
                "team1_score": 11,
                "team2_score": 5,
            }
        ]
    )
    df_players = pd.DataFrame([{"id": 1, "rating": 1400}, {"id": 2, "rating": 1400}])
    ctx = SimpleNamespace(df_matches=df_matches, df_players_all=df_players, club_id="club")
    candidates = list(compute_candidates_for_club("club", ctx=ctx))
    awarded = {(c.player_id, c.badge_id) for c in candidates}
    assert (1, "david_vs_goliath") not in awarded
    assert (1, "above_expectations") not in awarded
