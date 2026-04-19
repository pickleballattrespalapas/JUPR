from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.gamification.profile import build_gamification_summary
from jupr_app.domain.gamification.copy_pack import (
    assert_no_banned_words,
    get_badge_copy,
    pick_variant,
    render_template,
)
from jupr_app.domain.gamification.story_engine import compute_story_cards
from jupr_app.domain.gamification.badge_rules import BadgeAward


def test_profile_summary_locked_and_prestige():
    df_badges = pd.DataFrame(
        [
            {
                "badge_id": "first_win",
                "name": "First Win",
                "prestige": 15,
                "category": "Participation & Habit Loop",
                "rarity": "common",
                "icon_key": "first_win",
                "tier": None,
                "scope": "overall",
            },
            {
                "badge_id": "hot_streak",
                "name": "Hot Streak",
                "prestige": 50,
                "category": "Skill Growth & Momentum",
                "rarity": "epic",
                "icon_key": "hot_streak",
                "tier": None,
                "scope": "league",
            },
        ]
    )
    df_player_badges = pd.DataFrame(
        [
            {
                "player_id": 7,
                "badge_id": "first_win",
                "earned_at": "2024-01-02T00:00:00Z",
                "value_json": {"tape_excerpt": "The first win hit the archive."},
            }
        ]
    )
    summary = build_gamification_summary(7, df_badges, df_player_badges)
    assert summary["prestige_total"] == 15
    assert summary["collected_unique_count"] == 1
    assert summary["total_active_badge_types"] == 2
    assert summary["unlocked_badges"][0]["latest_tape_excerpt"]
    locked = summary["locked_badges"]
    assert locked
    assert "requirements" in locked[0]
    assert locked[0]["requirements"] == "Requirements TBD"


def test_story_generation_highlight_and_foreshadow_dedupe():
    facts = pd.DataFrame(
        [
            {
                "club_id": "club",
                "player_id": 1,
                "match_id": "m1",
                "league": "A",
                "date_dt": pd.Timestamp("2024-03-01", tz="UTC"),
                "week_key": "2024-W09",
                "month_key": "2024-03",
                "season_key": "2024",
                "win": True,
                "points_for": 11,
                "points_against": 4,
                "margin": 7,
                "partner_id": None,
                "opponent_ids": [2],
                "expected_win_prob": 0.2,
                "elo_delta_signed": 15.0,
                "abs_elo_delta": 15.0,
                "opp_max_rating": 1800.0,
                "lobby_avg_rating": 1500.0,
            },
            {
                "club_id": "club",
                "player_id": 1,
                "match_id": "m2",
                "league": "A",
                "date_dt": pd.Timestamp("2024-03-08", tz="UTC"),
                "week_key": "2024-W10",
                "month_key": "2024-03",
                "season_key": "2024",
                "win": True,
                "points_for": 11,
                "points_against": 6,
                "margin": 5,
                "partner_id": None,
                "opponent_ids": [3],
                "expected_win_prob": 0.3,
                "elo_delta_signed": 10.0,
                "abs_elo_delta": 10.0,
                "opp_max_rating": 1600.0,
                "lobby_avg_rating": 1500.0,
            },
            {
                "club_id": "club",
                "player_id": 1,
                "match_id": "m3",
                "league": "A",
                "date_dt": pd.Timestamp("2024-03-15", tz="UTC"),
                "week_key": "2024-W11",
                "month_key": "2024-03",
                "season_key": "2024",
                "win": True,
                "points_for": 11,
                "points_against": 7,
                "margin": 4,
                "partner_id": None,
                "opponent_ids": [4],
                "expected_win_prob": 0.3,
                "elo_delta_signed": 12.0,
                "abs_elo_delta": 12.0,
                "opp_max_rating": 1600.0,
                "lobby_avg_rating": 1500.0,
            },
        ]
    )
    awards = [
        BadgeAward(
            player_id=1,
            badge_id="first_win",
            context_type="overall",
            context_id="first_win",
            match_id="m1",
            value_num=None,
            value_json={"tape_excerpt": "The first win hit the archive."},
        )
    ]
    ctx = SimpleNamespace(club_id="club")
    stories = compute_story_cards(ctx, facts, awards)
    story_types = {(s["story_type"], s["context_id"]) for s in stories}
    assert any(st[0].startswith("highlight.badge.") for st in story_types)
    assert any(st[0].startswith("foreshadow.") for st in story_types)

    highlight = next(story for story in stories if story["story_type"].startswith("highlight.badge."))
    badge_copy = get_badge_copy("first_win")
    seed = "1:first_win:first_win:highlight"
    expected_title = render_template(
        pick_variant(badge_copy["highlight"]["titles"], f"{seed}:title"),
        {"badge_name": "First Win", "tape_excerpt": "The first win hit the archive."},
    ) or "Highlight — First Win"
    assert highlight["title"] == expected_title


def test_player_facing_copy_has_no_banned_words():
    df_badges = pd.DataFrame(
        [
            {
                "badge_id": "first_win",
                "name": "First Win",
                "prestige": 15,
                "category": "Participation & Habit Loop",
                "rarity": "common",
                "icon_key": "first_win",
                "tier": None,
                "scope": "overall",
            }
        ]
    )
    df_player_badges = pd.DataFrame(
        [
            {
                "player_id": 9,
                "badge_id": "first_win",
                "earned_at": "2024-01-02T00:00:00Z",
                "value_json": {"tape_excerpt": "The first win hit the archive."},
            }
        ]
    )
    summary = build_gamification_summary(9, df_badges, df_player_badges)
    for badge in summary.get("unlocked_badges", []):
        assert_no_banned_words(
            " ".join([badge.get("name", ""), badge.get("latest_tape_excerpt", "")])
        )
    for badge in summary.get("locked_badges", []):
        assert_no_banned_words(" ".join([badge.get("name", "")]))


def test_story_cards_dedupe_by_type_and_context():
    facts = pd.DataFrame(
        [
            {
                "club_id": "club",
                "player_id": 1,
                "match_id": "m1",
                "league": "A",
                "date_dt": pd.Timestamp("2024-03-01", tz="UTC"),
                "week_key": "2024-W09",
                "month_key": "2024-03",
                "season_key": "2024",
                "win": True,
                "points_for": 11,
                "points_against": 4,
                "margin": 7,
                "partner_id": None,
                "opponent_ids": [2],
                "expected_win_prob": 0.2,
                "elo_delta_signed": 15.0,
                "abs_elo_delta": 15.0,
                "opp_max_rating": 1800.0,
                "lobby_avg_rating": 1500.0,
            },
            {
                "club_id": "club",
                "player_id": 1,
                "match_id": "m2",
                "league": "A",
                "date_dt": pd.Timestamp("2024-03-08", tz="UTC"),
                "week_key": "2024-W10",
                "month_key": "2024-03",
                "season_key": "2024",
                "win": True,
                "points_for": 11,
                "points_against": 6,
                "margin": 5,
                "partner_id": None,
                "opponent_ids": [3],
                "expected_win_prob": 0.3,
                "elo_delta_signed": 10.0,
                "abs_elo_delta": 10.0,
                "opp_max_rating": 1600.0,
                "lobby_avg_rating": 1500.0,
            },
            {
                "club_id": "club",
                "player_id": 1,
                "match_id": "m3",
                "league": "A",
                "date_dt": pd.Timestamp("2024-03-15", tz="UTC"),
                "week_key": "2024-W11",
                "month_key": "2024-03",
                "season_key": "2024",
                "win": True,
                "points_for": 11,
                "points_against": 7,
                "margin": 4,
                "partner_id": None,
                "opponent_ids": [4],
                "expected_win_prob": 0.3,
                "elo_delta_signed": 12.0,
                "abs_elo_delta": 12.0,
                "opp_max_rating": 1600.0,
                "lobby_avg_rating": 1500.0,
            },
        ]
    )
    awards = [
        BadgeAward(
            player_id=1,
            badge_id="first_win",
            context_type="overall",
            context_id="first_win",
            match_id="m1",
            value_num=None,
            value_json={"tape_excerpt": "The first win hit the archive."},
        )
    ]
    ctx = SimpleNamespace(club_id="club")
    stories = compute_story_cards(ctx, facts, awards)
    keys = [(s["story_type"], s["context_id"]) for s in stories]
    assert len(keys) == len(set(keys))


def test_profile_summary_supports_raw_player_badges():
    df_badges = pd.DataFrame(
        [
            {
                "badge_id": "participant_grace",
                "name": "Grace",
                "prestige": 10,
                "category": "Participation",
                "rarity": "common",
            }
        ]
    )
    df_player_badges = pd.DataFrame(
        [
            {
                "player_id": 42,
                "badge_id": "participant_grace",
                "earned_at": "2024-02-01T00:00:00Z",
            }
        ]
    )

    summary = build_gamification_summary(42, df_badges, df_player_badges)

    assert summary["collected_unique_count"] == 1
    assert summary["prestige_total"] == 10
    assert summary["unlocked_badges"][0]["name"] == "Grace"


def test_profile_summary_supports_enriched_player_badges():
    df_badges = pd.DataFrame(
        [
            {
                "badge_id": "participant_grace",
                "name": "Grace",
                "prestige": 10,
                "category": "Participation",
                "rarity": "common",
            }
        ]
    )
    df_player_badges = pd.DataFrame(
        [
            {
                "player_id": "42",
                "badge_id": "participant_grace",
                "earned_at": "2024-02-01T00:00:00Z",
                "name": "Grace (enriched)",
                "prestige": 15,
                "category": "Participation",
                "rarity": "rare",
                "requirements": "Play one match",
                "is_stackable": False,
            }
        ]
    )

    summary = build_gamification_summary(42, df_badges, df_player_badges)

    assert summary["collected_unique_count"] == 1
    assert summary["prestige_total"] == 15
    assert summary["unlocked_badges"][0]["name"] == "Grace (enriched)"
    assert summary["unlocked_badges"][0]["rarity"] == "rare"


def test_profile_summary_handles_def_suffix_without_crash():
    df_badges = pd.DataFrame(
        [
            {
                "badge_id": "participant_grace",
                "name": "Grace",
                "prestige": 10,
                "category": "Participation",
                "rarity": "common",
            }
        ]
    )
    df_player_badges = pd.DataFrame(
        [
            {
                "player_id": "42",
                "badge_id": "participant_grace",
                "earned_at": "2024-02-01T00:00:00Z",
                "prestige_def": 99,
                "name_def": "Grace def",
            },
            {
                "player_id": "not-a-number",
                "badge_id": "participant_grace",
                "earned_at": "2024-02-02T00:00:00Z",
            },
        ]
    )

    summary = build_gamification_summary(42, df_badges, df_player_badges)

    assert summary["collected_unique_count"] == 1
    assert summary["prestige_total"] == 10
    assert summary["unlocked_badges"][0]["name"] == "Grace"


def test_profile_summary_single_participant_badge_is_unlocked():
    df_badges = pd.DataFrame(
        [
            {
                "badge_id": "participant_grace",
                "name": "Grace",
                "prestige": 10,
                "category": "Participation",
                "rarity": "common",
            }
        ]
    )
    df_player_badges = pd.DataFrame(
        [
            {
                "player_id": 7.0,
                "badge_id": "participant_grace",
                "earned_at": "2024-02-01T00:00:00Z",
            }
        ]
    )

    summary = build_gamification_summary(7, df_badges, df_player_badges)

    assert len(summary["unlocked_badges"]) == 1
    assert summary["collected_unique_count"] == 1
    assert summary["prestige_total"] > 0
