from jupr_app.ui.pages.weekly_recap import _build_recap_pdf


def test_build_recap_pdf_uses_spotlight_when_highlights_missing():
    recap = {
        "spotlight": [
            {"label": "Top Performer", "players": ["Alice 3-0", "Bob 2-1"]},
            {"label": "Biggest Jump", "players": ["Cara +0.12"]},
        ],
        "looking_ahead": ["Finals this Friday"],
    }

    pdf_bytes = _build_recap_pdf(recap, "2026-02-01", "2026-02-07")
    text = pdf_bytes.decode("latin-1")

    assert "Top Performer: Alice 3-0, Bob 2-1" in text
    assert "Biggest Jump: Cara +0.12" in text
