from jupr_app.ui.pages import leaderboards


def test_safe_text_escapes_html_story():
    raw_story = "<b>Hi</b> & stuff"
    assert leaderboards._safe_text(raw_story) == "&lt;b&gt;Hi&lt;/b&gt; &amp; stuff"


def test_safe_text_escapes_div_tags():
    raw_story = "<div>Already escaped</div>"
    assert leaderboards._safe_text(raw_story) == "&lt;div&gt;Already escaped&lt;/div&gt;"


def test_story_sanitize_removes_legacy_html_wrappers():
    raw_story = (
        '<div class="lb-story-text">Active this season with 10 games logged.</div>'
        '<div class="lb-row" style="gap:6px;"></div>'
    )
    cleaned = leaderboards.sanitize_story_text(raw_story)
    assert "<div" not in cleaned
    assert "lb-row" not in cleaned
    assert "Active this season with 10 games logged." in cleaned
