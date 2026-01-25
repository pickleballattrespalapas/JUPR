import pandas as pd

from jupr_app.domain.story_stats import (
    StoryStatsConfig,
    compute_best_partner,
    compute_rival,
)


def _match(
    date,
    t1_p1,
    t1_p2,
    t2_p1,
    t2_p2,
    s1,
    s2,
    league="Alpha",
    match_type="League",
):
    return {
        "club_id": "1",
        "date": date,
        "league": league,
        "match_type": match_type,
        "t1_p1": t1_p1,
        "t1_p2": t1_p2,
        "t2_p1": t2_p1,
        "t2_p2": t2_p2,
        "score_t1": s1,
        "score_t2": s2,
    }


def test_compute_rival_prefers_games_then_balance_then_date():
    matches = []
    for i in range(8):
        score = (11, 9) if i % 2 == 0 else (9, 11)
        matches.append(_match(f"2024-01-{i+1:02d}", 1, None, 2, None, *score))
    for i in range(8):
        score = (11, 3) if i < 6 else (3, 11)
        matches.append(_match(f"2024-02-{i+1:02d}", 1, None, 3, None, *score))

    df = pd.DataFrame(matches)
    rival = compute_rival(1, {"club_id": "1"}, df)

    assert rival is not None
    assert rival["opponent_id"] == 2
    assert rival["games"] == 8
    assert rival["wins"] == 4


def test_compute_rival_fallback_when_no_balanced_opponent():
    matches = []
    for i in range(10):
        score = (11, 2) if i < 8 else (2, 11)
        matches.append(_match(f"2024-03-{i+1:02d}", 1, None, 4, None, *score))
    for i in range(8):
        score = (11, 6) if i < 7 else (6, 11)
        matches.append(_match(f"2024-04-{i+1:02d}", 1, None, 5, None, *score))

    df = pd.DataFrame(matches)
    rival = compute_rival(1, {"club_id": "1"}, df)

    assert rival is not None
    assert rival["opponent_id"] == 4
    assert rival["games"] == 10


def test_compute_rival_breaks_ties_with_recent_date():
    matches = []
    for i in range(6):
        score = (11, 9) if i % 2 == 0 else (9, 11)
        matches.append(_match(f"2024-08-{i+1:02d}", 1, None, 7, None, *score))
    for i in range(6):
        score = (11, 9) if i % 2 == 0 else (9, 11)
        matches.append(_match(f"2024-09-{i+1:02d}", 1, None, 8, None, *score))

    df = pd.DataFrame(matches)
    rival = compute_rival(1, {"club_id": "1"}, df)

    assert rival is not None
    assert rival["opponent_id"] == 8


def test_compute_best_partner_uses_win_pct_then_games_then_date():
    matches = []
    for i in range(6):
        score = (11, 7) if i < 5 else (7, 11)
        matches.append(_match(f"2024-05-{i+1:02d}", 1, 2, 3, 4, *score))
    for i in range(7):
        score = (11, 7) if i < 6 else (7, 11)
        matches.append(_match(f"2024-06-{i+1:02d}", 1, 5, 3, 4, *score))

    df = pd.DataFrame(matches)
    partner = compute_best_partner(1, {"club_id": "1"}, df)

    assert partner is not None
    assert partner["partner_id"] == 5
    assert partner["games"] == 7


def test_compute_best_partner_requires_min_games():
    matches = [
        _match("2024-07-01", 1, 6, 3, 4, 11, 8),
        _match("2024-07-02", 1, 6, 3, 4, 11, 8),
        _match("2024-07-03", 1, 6, 3, 4, 11, 8),
    ]

    df = pd.DataFrame(matches)
    config = StoryStatsConfig(min_games_partner=4)
    partner = compute_best_partner(1, {"club_id": "1"}, df, config=config)
    assert partner is None
