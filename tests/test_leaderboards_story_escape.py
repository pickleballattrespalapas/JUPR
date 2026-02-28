import math

from jupr_app.ui.pages import leaderboards


def test_safe_text_escapes_html_story():
    raw_story = "<b>Hi</b> & stuff"
    assert leaderboards._safe_text(raw_story) == "&lt;b&gt;Hi&lt;/b&gt; &amp; stuff"


def test_story_sanitize_neutralizes_script_tags():
    cleaned = leaderboards.sanitize_story_text("<script>alert(1)</script>")
    assert "<script>" not in cleaned
    assert "<" not in cleaned
    assert ">" not in cleaned


def test_story_sanitize_neutralizes_html_and_javascript_links():
    raw = '<a href="x">click</a> and [x](javascript:alert(1))'
    cleaned = leaderboards.sanitize_story_text(raw)
    assert "<a" not in cleaned
    assert "javascript:" not in cleaned.lower()
    assert "click" in cleaned


def test_story_sanitize_preserves_paragraphs_and_bullets():
    raw = "Line1\n\n- bullet1\n- bullet2"
    cleaned = leaderboards.sanitize_story_text(raw)
    assert "Line1\n\n- bullet1\n- bullet2" in cleaned


def test_extract_story_handles_none_and_nan():
    story, source = leaderboards._extract_story_from_row({"story": None, "story_text": math.nan})
    assert story is None
    assert source is None


def test_story_sanitize_fallback_for_none():
    assert leaderboards.sanitize_story_text(None) == "No story yet for this window."


def test_story_sanitize_caps_length():
    cleaned = leaderboards.sanitize_story_text("A" * 5000)
    assert len(cleaned) <= leaderboards.MAX_STORY_TEXT_LEN
    assert cleaned.endswith("…")


def test_story_sanitize_removes_legacy_html_wrappers():
    raw_story = (
        '<div class="lb-story-text">Active this season with 10 games logged.</div>'
        '<div class="lb-row" style="gap:6px;"></div>'
    )
    cleaned = leaderboards.sanitize_story_text(raw_story)
    assert "<div" not in cleaned
    assert "lb-row" in cleaned
    assert "Active this season with 10 games logged." in cleaned
