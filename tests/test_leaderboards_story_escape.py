import pandas as pd

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


def test_story_html_in_row_is_regenerated_without_html():
    row = pd.Series(
        {
            "_pid": 101,
            "name": "Sample Player",
            "matches_played": 10,
            "wins": 6,
            "losses": 4,
            "rating_gain": 0.0,
            "JUPR": 3.1,
            "Win %": 0.6,
            "story_html": (
                '<div class="lb-story-text">Active this season with 10 games logged.</div>'
                '<div class="lb-row" style="gap:6px;"></div>'
            ),
        }
    )

    story_text, _ = leaderboards._build_story_text_for_row(
        row,
        story_badges=[],
        story_rivals_by_player={},
        story_partners_by_player={},
        id_to_name={},
        admin_logged_in=False,
    )
    story_text = leaderboards.sanitize_story_text(story_text)
    escaped_story = leaderboards.html.escape(story_text)

    assert "<div" not in escaped_story
    assert "lb-row" not in escaped_story
