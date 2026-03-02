from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from jupr_app.ui.pages.top_players_printable import (
    _build_ranked_rows,
    _build_top_players_pdf,
    _previous_month_subtitle,
    _previous_month_window,
    build_top50_previous_month_df,
)


def test_previous_month_window_filters_only_previous_calendar_month_matches():
    now_utc = datetime(2026, 3, 15, 9, 30, tzinfo=timezone.utc)
    start_dt, end_dt = _previous_month_window(now_utc)

    df_matches = pd.DataFrame(
        [
            {
                "date_dt": datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc),
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 8,
            },
            {
                "date_dt": datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc),
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 9,
            },
            {
                "date_dt": datetime(2026, 1, 20, 12, 0, tzinfo=timezone.utc),
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
            },
        ]
    )

    filtered = df_matches[(df_matches["date_dt"] >= start_dt) & (df_matches["date_dt"] < end_dt)]

    assert len(filtered) == 1
    assert filtered.iloc[0]["date_dt"] == datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc)


def test_single_page_pdf_contains_previous_month_subtitle_jupr_and_no_generated_stamp():
    id_to_name = {1: "Alice", 2: "Bob", 3: "Carla", 4: "Diego"}
    rating_map = {1: 2000.0, 2: 1600.0, 3: 1400.0, 4: 1200.0}

    matches = []
    for idx in range(10):
        matches.append(
            {
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 8 if idx % 2 == 0 else 9,
            }
        )
    df_matches = pd.DataFrame(matches)

    rows = _build_ranked_rows(df_matches, id_to_name, rating_map)
    subtitle = _previous_month_subtitle(datetime(2026, 3, 15, tzinfo=timezone.utc))
    pdf_bytes = _build_top_players_pdf(rows, "Tres Palapas -- Top 50 Players", subtitle)
    text = pdf_bytes.decode("latin-1")

    assert "Tres Palapas -- Top 50 Players" in text
    assert "February 2026" in text
    assert "Generated:" not in text
    assert "5.000" in text
    assert "2000" not in text
    assert "/Count 1" in text
    assert text.count("/Type /Page") == 2


def test_previous_month_subtitle_uses_calendar_month_and_year():
    assert _previous_month_subtitle(datetime(2026, 3, 15, tzinfo=timezone.utc)) == "February 2026"
    assert _previous_month_subtitle(datetime(2026, 1, 2, tzinfo=timezone.utc)) == "December 2025"


def test_build_top50_previous_month_df_applies_window_min_games_and_csv_shape():
    now_utc = datetime(2026, 3, 15, 9, 30, tzinfo=timezone.utc)
    prior_month_match = {
        "date_dt": datetime(2026, 2, 14, 12, 0, tzinfo=timezone.utc),
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "score_t1": 11,
        "score_t2": 8,
    }
    current_month_match = {
        "date_dt": datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc),
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "score_t1": 11,
        "score_t2": 9,
    }

    matches = [prior_month_match for _ in range(10)] + [current_month_match for _ in range(20)]
    ctx = SimpleNamespace(
        df_matches=pd.DataFrame(matches),
        df_players_all=pd.DataFrame(
            [
                {"id": 1, "elo": 2000},
                {"id": 2, "elo": 1800},
                {"id": 3, "elo": 1600},
                {"id": 4, "elo": 1200},
            ]
        ),
        id_to_name={1: "Alice", 2: "Bob", 3: "Carla", 4: "Diego"},
    )

    df_out = build_top50_previous_month_df(ctx, now_utc=now_utc)

    assert list(df_out.columns) == ["rank", "player_id", "player_name", "jupr", "wins", "losses", "games", "wl"]
    assert set(df_out["player_id"].tolist()) == {1, 2, 3, 4}
    assert (df_out["games"] >= 10).all()
    assert "5.000" in df_out["jupr"].tolist()

    csv_text = df_out.to_csv(index=False)
    assert csv_text.splitlines()[0] == "rank,player_id,player_name,jupr,wins,losses,games,wl"
    assert "5.000" in csv_text
