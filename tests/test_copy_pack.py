from jupr_app.domain.gamification.copy_pack import (
    assert_no_banned_words,
    get_badge_copy,
    load_copy_pack,
    pick_variant,
    render_template,
)


def test_copy_pack_has_required_badges():
    required = [
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
        "clutch_performer",
        "ice_in_veins",
        "pickle_perfection",
        "blowout_artist",
        "untouchable",
        "clean_sweep_week",
        "high_roller",
        "social_butterfly",
        "network_builder",
        "draft_master",
        "swiss_army_knife",
        "giant_slayer",
        "david_vs_goliath",
        "upset_champion",
        "hall_of_fame_night",
        "legendary_upset",
        "nemesis_found",
        "rivalry_win",
        "rivalry_streak",
        "settled_the_score",
        "steady_hand",
        "mr_reliable",
        "league_champion",
        "podium",
        "good_sport",
        "community_builder",
        "mentor",
        "breakthrough",
        "above_expectations",
        "dominant_run",
        "high_output",
        "battle_tested",
        "consistency",
    ]
    pack = load_copy_pack()
    assert "badges" in pack
    for badge_id in required:
        copy = get_badge_copy(badge_id)
        assert copy["lore"]
        assert copy["hint"]
        assert copy["tape_excerpts"]


def test_pick_variant_deterministic():
    options = ["alpha", "bravo", "charlie"]
    assert pick_variant(options, "seed-123") == pick_variant(options, "seed-123")


def test_render_template_removes_missing_placeholders():
    rendered = render_template("Hello {name} {missing}", {"name": "Riley"})
    assert rendered == "Hello Riley"


def test_copy_pack_player_facing_words_clean():
    pack = load_copy_pack()
    style = pack.get("style_guide", {}) if isinstance(pack, dict) else {}
    forbidden = style.get("banned_words", [])
    for badge_id, entry in pack.get("badges", {}).items():
        if not isinstance(entry, dict):
            continue
        texts = [
            entry.get("lore", ""),
            entry.get("hint", ""),
            *(entry.get("tape_excerpts", []) or []),
        ]
        highlight = entry.get("highlight", {}) or {}
        foreshadow = entry.get("foreshadow", {}) or {}
        texts.extend(highlight.get("titles", []) or [])
        texts.extend(highlight.get("bodies", []) or [])
        texts.extend(foreshadow.get("titles", []) or [])
        texts.extend(foreshadow.get("bodies", []) or [])
        joined = " ".join(str(t) for t in texts)
        assert_no_banned_words(joined)
        if forbidden:
            for word in forbidden:
                assert str(word).lower() not in joined.lower(), f"{badge_id} includes forbidden word: {word}"
