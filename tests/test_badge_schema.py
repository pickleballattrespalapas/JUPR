from jupr_app.domain.gamification.badge_registry import awardable_badge_ids
from jupr_app.domain.gamification.badge_schema import load_badge_definitions


def test_badge_schema_loads():
    badges = load_badge_definitions()
    assert badges


def test_badge_schema_required_fields():
    badges = load_badge_definitions()
    for badge in badges:
        assert badge.id
        assert badge.title
        assert isinstance(badge.prestige, int)
        assert badge.status
        assert badge.scope
        assert badge.award_timing
        assert badge.display.requirements


def test_badge_schema_no_elo_in_display():
    badges = load_badge_definitions()
    for badge in badges:
        display_text = " ".join(
            [
                badge.display.requirements or "",
                badge.display.flavor or "",
            ]
        ).lower()
        assert "elo" not in display_text


def test_badge_ids_snapshot():
    expected = [
        "participant",
        "dedicated_participant_50",
        "lifetime_participant_200",
        "first_win",
        "weekly_regular",
        "iron_week",
        "marathon_month",
        "level_up",
        "rocket_start",
        "most_improved_monthly",
        "mountain_climber",
        "hot_streak",
        "bounce_back",
        "breakthrough",
        "above_expectations",
        "ice_in_veins",
        "clutch_performer",
        "pickle_perfection",
        "blowout_artist",
        "untouchable",
        "clean_sweep_week",
        "high_roller",
        "dominant_run",
        "high_output",
        "social_butterfly",
        "network_builder",
        "draft_master",
        "swiss_army_knife",
        "giant_slayer",
        "david_vs_goliath",
        "upset_champion",
        "legendary_upset",
        "nemesis_found",
        "rivalry_win",
        "rivalry_streak",
        "settled_the_score",
        "battle_tested",
        "consistency",
        "steady_hand",
        "mr_reliable",
        "league_champion",
        "league_runner_up",
        "league_third_place",
        "tournament_champion",
        "tournament_runner_up",
        "tournament_third_place",
        "top_performer_highest_rating",
        "top_performer_most_improved",
        "top_performer_best_win_pct",
        "top_performer_most_wins",
        "podium",
        "hall_of_fame_night",
        "good_sport",
        "community_builder",
        "mentor",
    ]
    badge_ids = [badge.id for badge in load_badge_definitions()]
    assert badge_ids == expected


def test_awardable_badges_are_live_only():
    awardable = awardable_badge_ids()
    assert "participant" in awardable
    assert "top_performer_highest_rating" not in awardable
    assert "tournament_champion" not in awardable
