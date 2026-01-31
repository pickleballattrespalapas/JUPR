import pandas as pd

from jupr_app.domain.gamification.profile import build_gamification_summary, select_featured_badges


def test_aggregation_counts_and_prestige():
    df_badges = pd.DataFrame(
        [
            {
                "badge_id": "giant_slayer",
                "name": "Giant Slayer",
                "prestige": 75,
                "category": "Rivalries",
                "rarity": "legendary",
                "is_active": True,
            },
            {
                "badge_id": "first_win",
                "name": "First Win",
                "prestige": 15,
                "category": "Participation",
                "rarity": "common",
                "is_active": True,
            },
            {
                "badge_id": "hot_streak",
                "name": "Hot Streak",
                "prestige": 50,
                "category": "Momentum",
                "rarity": "epic",
                "is_active": True,
            },
        ]
    )
    df_player_badges = pd.DataFrame(
        [
            {
                "player_id": 5,
                "badge_id": "giant_slayer",
                "earned_at": "2024-01-02T00:00:00Z",
                "value_json": {},
            },
            {
                "player_id": 5,
                "badge_id": "giant_slayer",
                "earned_at": "2024-02-02T00:00:00Z",
                "value_json": {},
            },
            {
                "player_id": 5,
                "badge_id": "first_win",
                "earned_at": "2024-03-02T00:00:00Z",
                "value_json": {"tape_excerpt": "A first win hit the logbook."},
            },
        ]
    )
    summary = build_gamification_summary(5, df_badges, df_player_badges)
    assert summary["prestige_total"] == 165
    assert summary["collected_unique_count"] == 2
    assert summary["total_active_badge_types"] == 3

    giant = next(b for b in summary["unlocked_badges"] if b["badge_id"] == "giant_slayer")
    assert giant["stack_count"] == 2
    assert giant["latest_tape_excerpt"]

    locked = summary["locked_badges"]
    assert any(b["badge_id"] == "hot_streak" for b in locked)


def test_featured_badges_unique():
    unlocked = [
        {"badge_id": "giant_slayer", "prestige": 75, "last_earned_at": "2024-02-02T00:00:00Z"},
        {"badge_id": "giant_slayer", "prestige": 75, "last_earned_at": "2024-01-02T00:00:00Z"},
        {"badge_id": "first_win", "prestige": 15, "last_earned_at": "2024-03-02T00:00:00Z"},
    ]
    featured = select_featured_badges(unlocked, max_count=5, sort_mode="recent")
    badge_ids = [b["badge_id"] for b in featured]
    assert len(badge_ids) == len(set(badge_ids))
