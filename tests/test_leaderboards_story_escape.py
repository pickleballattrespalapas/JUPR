from jupr_app.ui.pages import leaderboards


def test_safe_text_escapes_html_story():
    raw_story = "<b>Hi</b> & stuff"
    assert leaderboards._safe_text(raw_story) == "&lt;b&gt;Hi&lt;/b&gt; &amp; stuff"


def test_safe_text_escapes_div_tags():
    raw_story = "<div>Already escaped</div>"
    assert leaderboards._safe_text(raw_story) == "&lt;div&gt;Already escaped&lt;/div&gt;"
