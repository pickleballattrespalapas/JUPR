import pandas as pd

from jupr_app.ui.helpers import build_badge_story, sanitize_story_text


def test_build_badge_story_with_badges_and_stats():
    row = {
        "name": "Ava",
        "matches_played": 25,
        "Win %": 80,
        "Gain": 0.234,
    }
    badges = [
        {
            "badge_id": "upset_champion",
            "name": "Upset Champion",
            "prestige": 90,
            "category": "Prestige / Rarity",
            "earned_at_dt": pd.Timestamp("2024-05-01", tz="UTC"),
        }
    ]
    story = build_badge_story(row, badges)
    assert "Ava has earned Upset Champion" in story
    assert "+0.234" in story
    assert "80%" in story
    assert "25 games" in story


def test_build_badge_story_no_badges_zero_games():
    row = {"name": "Kai", "matches_played": 0}
    story = build_badge_story(row, [])
    assert story == "New to the standings—play your first matches to begin earning badges."


def test_build_badge_story_no_badges_low_games():
    row = {"name": "Kai", "matches_played": 3}
    story = build_badge_story(row, [])
    assert story == "New to the leaderboard—log a few matches to start earning badges."


def test_build_badge_story_no_badges_active():
    row = {"name": "Kai", "matches_played": 12}
    story = build_badge_story(row, [])
    assert (
        story
        == "Active this season with 12 games logged—badges will start appearing as the reel fills."
    )
    assert "<" not in sanitize_story_text(story)
    assert ">" not in sanitize_story_text(story)


def test_sanitize_story_text_strips_html():
    text = sanitize_story_text("<div>Hello<br>World</div>")
    assert text == "Hello World"
