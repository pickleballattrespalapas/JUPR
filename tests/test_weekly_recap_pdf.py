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


def test_builder_legacy_pdf_and_page_honor_selected_edition(monkeypatch):
    from tests.test_recap_builder import fixture
    from jupr_app.domain.recaps.recap_builder import apply_builder
    from jupr_app.ui.components import weekly_recap_layout as layout
    recap = apply_builder(fixture(), {'sections': {'events': {'enabled': False}, 'numbers': {'enabled': False}}, 'custom_blocks': [{'id': 'coach', 'title': 'Coach tip', 'text': 'Stay patient'}]})
    pdf = _build_recap_pdf(recap, '2026-10-05', '2026-10-11').decode('latin-1')
    assert 'Tuesday Moneyball' not in pdf and 'Stay patient' in pdf
    displayed = []
    for method in ('header', 'caption', 'subheader', 'text', 'image', 'link_button'):
        monkeypatch.setattr(layout.st, method, lambda *args, **kw: displayed.extend(map(str, args)))
    layout.render_weekly_recap(recap, print_view=False)
    assert 'Stay patient' in displayed
    assert not any('Tuesday Moneyball' in item for item in displayed)
