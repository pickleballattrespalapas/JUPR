from datetime import date
from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.recaps.weekly_recap import compute_weekly_recap


def _ctx(df_matches: pd.DataFrame) -> SimpleNamespace:
    return SimpleNamespace(
        club_id="club-1",
        supabase=None,
        df_matches=df_matches,
        id_to_name={1: "A", 2: "B", 3: "C", 4: "D"},
        df_players_all=pd.DataFrame(
            [
                {"id": 1, "rating": 1200},
                {"id": 2, "rating": 1200},
                {"id": 3, "rating": 1200},
                {"id": 4, "rating": 1200},
            ]
        ),
    )


def test_recap_is_date_range_driven_not_week_driven():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "date": "2025-02-01T10:00:00Z",
                "league": "Alpha",
                "match_type": "League",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 5,
            },
            {
                "id": 2,
                "date": "2025-02-08T10:00:00Z",
                "league": "Alpha",
                "match_type": "League",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
            },
        ]
    )

    recap = compute_weekly_recap(_ctx(df), start_date=date(2025, 2, 1), end_date=date(2025, 2, 1))

    assert recap["numbers"]["matches"] == 1
    assert recap["start_date"] == "2025-02-01"
    assert recap["end_date"] == "2025-02-01"


def test_tournament_matches_are_conditionally_included():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "date": "2025-02-01T10:00:00Z",
                "league": "Alpha",
                "match_type": "League",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 8,
            },
            {
                "id": 2,
                "date": "2025-02-01T11:00:00Z",
                "league": "Alpha",
                "match_type": "Tournament",
                "context_type": "TOURNAMENT",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 6,
            },
        ]
    )
    ctx = _ctx(df)

    included = compute_weekly_recap(ctx, start_date=date(2025, 2, 1), end_date=date(2025, 2, 1), include_tournaments=True)
    excluded = compute_weekly_recap(ctx, start_date=date(2025, 2, 1), end_date=date(2025, 2, 1), include_tournaments=False)

    assert included["numbers"]["matches"] == 2
    assert excluded["numbers"]["matches"] == 1
