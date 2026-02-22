from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from jupr_app.domain.recaps.weekly_recap import _load_week_matches, compute_weekly_recap, get_week_bounds
from jupr_app.ui.components.weekly_recap_layout import build_weekly_recap_html


class _Ctx:
    def __init__(self, *, club_id: str, df_matches: pd.DataFrame, id_to_name: dict[int, str] | None = None):
        self.club_id = club_id
        self.df_matches = df_matches
        self.id_to_name = id_to_name or {}
        self.df_players_all = pd.DataFrame()
        self.supabase = None


def test_load_week_matches_filters_local_df_by_club_id():
    week_start = date(2025, 1, 6)
    start_dt, end_dt = get_week_bounds(week_start, "UTC")
    df = pd.DataFrame(
        [
            {"id": 1, "club_id": "club-a", "date": "2025-01-06T10:00:00Z", "score_t1": 11, "score_t2": 9},
            {"id": 2, "club_id": "club-b", "date": "2025-01-06T11:00:00Z", "score_t1": 11, "score_t2": 9},
        ]
    )

    loaded = _load_week_matches(df, None, "club-a", start_dt, end_dt)

    assert set(loaded["id"].tolist()) == {1}


def test_compute_weekly_recap_and_layout_non_empty_with_valid_data():
    df = pd.DataFrame(
        [
            {
                "id": 1,
                "club_id": "club-a",
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
                "t1_p1_r_end": 1215,
                "t1_p2_r_end": 1210,
                "t2_p1_r_end": 1190,
                "t2_p2_r_end": 1185,
                "context_type": "LEAGUE",
                "context_id": "alpha",
                "tournament_id": None,
            }
        ]
    )
    ctx = _Ctx(club_id="club-a", df_matches=df, id_to_name={1: "A", 2: "B", 3: "C", 4: "D"})

    recap = compute_weekly_recap(ctx, date(2025, 1, 6), tz_name="UTC")
    html = build_weekly_recap_html(recap, print_view=False)

    assert recap["numbers"]["matches"] == 1
    assert "Spotlight Reel" in html
    assert "Around the Club" in html


def test_compute_weekly_recap_and_layout_handle_empty_dataset():
    ctx = _Ctx(club_id="club-a", df_matches=pd.DataFrame())

    recap = compute_weekly_recap(ctx, date(2025, 1, 6), tz_name="UTC")
    html = build_weekly_recap_html(recap, print_view=False)

    assert recap["numbers"]["matches"] == 0
    assert "Tres Palapas Weekly Recap" in html
    assert len(html.strip()) > 0
