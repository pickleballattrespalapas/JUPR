import pandas as pd

from jupr_app.domain.recaps.weekly_recap import _compute_stats


def test_rating_delta_from_snapshots():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "date": "2025-01-06T10:00:00Z",
                "league": "Alpha",
                "match_type": "League",
                "week_tag": "Week 1",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
                "t1_p1_r": 1200,
                "t1_p2_r": 1200,
                "t2_p1_r": 1200,
                "t2_p2_r": 1200,
                "t1_p1_r_end": 1240,
                "t1_p2_r_end": 1210,
                "t2_p1_r_end": 1190,
                "t2_p2_r_end": 1190,
            }
        ]
    )
    df["date_dt"] = pd.to_datetime(df["date"], utc=True)
    stats, _, _, _ = _compute_stats(df, rating_map={})
    assert round(stats[1]["delta_jupr"], 2) == 0.10
