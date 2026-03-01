from datetime import datetime, timezone

from jupr_app.ui.pages.top_players_printable import _build_top_players_pdf, _previous_month_subtitle


def test_build_top_players_pdf_contains_new_header_and_jupr_values():
    rows = [
        {"rank": 1, "name": "Alice", "rating_elo": 2000.0, "jupr": 5.000, "wins": 10, "losses": 2},
        {"rank": 2, "name": "Bob", "rating_elo": 1880.0, "jupr": 4.700, "wins": 8, "losses": 4},
    ]

    pdf_bytes = _build_top_players_pdf(rows, "Tres Palapas -- Top 50 Players", "February 2026")
    text = pdf_bytes.decode("latin-1")

    assert "Tres Palapas -- Top 50 Players" in text
    assert "February 2026" in text
    assert "Generated:" not in text
    assert "Rank  Player" in text
    assert "5.000" in text
    assert "2000.00" not in text
    assert "Alice" in text


def test_previous_month_subtitle_uses_calendar_month_and_year():
    assert _previous_month_subtitle(datetime(2026, 3, 15, tzinfo=timezone.utc)) == "February 2026"
    assert _previous_month_subtitle(datetime(2026, 1, 2, tzinfo=timezone.utc)) == "December 2025"
