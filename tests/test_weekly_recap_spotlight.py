import pandas as pd

from jupr_app.domain.recaps.weekly_recap import _build_spotlight_candidates, _compute_stats


def test_giant_slayer_selects_max_gap():
    df = pd.DataFrame(
        [
            {
                "id": 10,
                "date": "2025-01-07T10:00:00Z",
                "league": "Alpha",
                "match_type": "League",
                "week_tag": "Week 1",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 9,
                "t1_p1_r": 1100,
                "t1_p2_r": 1100,
                "t2_p1_r": 1300,
                "t2_p2_r": 1300,
                "t1_p1_r_end": 1110,
                "t1_p2_r_end": 1110,
                "t2_p1_r_end": 1290,
                "t2_p2_r_end": 1290,
            },
            {
                "id": 11,
                "date": "2025-01-08T10:00:00Z",
                "league": "Beta",
                "match_type": "League",
                "week_tag": "Week 1",
                "t1_p1": 5,
                "t1_p2": 6,
                "t2_p1": 7,
                "t2_p2": 8,
                "score_t1": 11,
                "score_t2": 8,
                "t1_p1_r": 1000,
                "t1_p2_r": 1000,
                "t2_p1_r": 1400,
                "t2_p2_r": 1400,
                "t1_p1_r_end": 1020,
                "t1_p2_r_end": 1020,
                "t2_p1_r_end": 1380,
                "t2_p2_r_end": 1380,
            },
        ]
    )
    df["date_dt"] = pd.to_datetime(df["date"], utc=True)
    stats, _, _, giant_slayer = _compute_stats(df, rating_map={})
    candidates = _build_spotlight_candidates(stats, giant_slayer, id_to_name={})
    best = candidates["GIANT_SLAYER_WEEK"][0]
    assert best.value_json["match_id"] == 11
