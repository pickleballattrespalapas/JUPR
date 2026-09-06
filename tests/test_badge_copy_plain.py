import pytest

from jupr_app.domain.gamification.badge_copy import build_badge_copy_plain


def test_build_badge_copy_plain_keeps_text_plain():
    badge = {
        "requirements": "Win 10 games",
        "description_md": "Earn *10* wins.",
    }
    copy = build_badge_copy_plain(badge, earners_count=7)

    assert copy.req_text == "Win 10 games"
    assert copy.desc_text == "Win 10 games"
    assert copy.meta_text == "7 earners"


def test_build_badge_copy_plain_rejects_html():
    badge = {
        "requirements": "<div class='badge-card__req'>Win 10 games</div>",
        "description_md": "Plain desc",
    }
    with pytest.raises(AssertionError):
        build_badge_copy_plain(badge, earners_count=1)
