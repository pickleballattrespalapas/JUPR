from datetime import datetime, timezone

import pandas as pd

from jupr_app.ui.pages.top_players_printable import (
    _build_top_players_pdf,
    _previous_month_subtitle,
    list_leagues,
)


class _Ctx:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_build_top_players_pdf_contains_multi_league_pages_and_jupr_values():
    league_pages = [
        {
            "league_label": "Verified Men 4.0 (Week 9)",
            "rows": [
                {"rank": 1, "name": "Alice", "jupr": 5.000, "jupr_str": "5.000", "wins": 10, "losses": 2, "wl_str": "10-2"}
            ],
        },
        {
            "league_label": "Verified Women 3.5 (Week 9)",
            "rows": [
                {"rank": 1, "name": "Bob", "jupr": 4.700, "jupr_str": "4.700", "wins": 8, "losses": 4, "wl_str": "8-4"}
            ],
        },
    ]

    pdf_bytes = _build_top_players_pdf(league_pages, "Tres Palapas -- Top 50 Players", "February 2026")
    text = pdf_bytes.decode("latin-1")

    assert "Tres Palapas -- Top 50 Players" in text
    assert "February 2026" in text
    assert r"Verified Men 4.0 \(Week 9\)" in text
    assert r"Verified Women 3.5 \(Week 9\)" in text
    assert "Generated:" not in text
    assert "5.000" in text
    assert "/MediaBox [0 0 792 612]" in text
    assert "/Count 2" in text
    assert text.count("/Type /Page") == 3


def test_list_leagues_prefers_metadata_labels_when_available():
    df_recent = pd.DataFrame({"league_id": ["L2", "L1"], "score_t1": [11, 11], "score_t2": [5, 6]})
    ctx = _Ctx(df_leagues=pd.DataFrame({"league_id": ["L1", "L2"], "league_name": ["Alpha League", "Beta League"]}))

    leagues, league_col = list_leagues(ctx, df_recent)

    assert league_col == "league_id"
    assert leagues == [
        {"league_id": "L1", "league_label": "Alpha League"},
        {"league_id": "L2", "league_label": "Beta League"},
    ]


def test_previous_month_subtitle_uses_calendar_month_and_year():
    assert _previous_month_subtitle(datetime(2026, 3, 15, tzinfo=timezone.utc)) == "February 2026"
    assert _previous_month_subtitle(datetime(2026, 1, 2, tzinfo=timezone.utc)) == "December 2025"
