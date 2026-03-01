from jupr_app.ui.pages.top_players_printable import _build_top_players_pdf


def test_build_top_players_pdf_contains_expected_strings():
    rows = [
        {"rank": 1, "name": "Alice", "rating": 1420.25, "wins": 10, "losses": 2},
        {"rank": 2, "name": "Bob", "rating": 1388.5, "wins": 8, "losses": 4},
    ]

    pdf_bytes = _build_top_players_pdf(rows, "Top 50 Active Players", "2026-02-08T12:00:00+00:00")
    text = pdf_bytes.decode("latin-1")

    assert "Top 50 Active Players" in text
    assert "Rank  Player" in text
    assert "Alice" in text
