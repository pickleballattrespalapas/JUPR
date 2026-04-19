import pandas as pd

from jupr_app.ui.pages.badge_codex import (
    _availability_label,
    _normalize_category_label,
    _split_badges_for_catalog,
    get_all_badges,
    get_badge_earners_page,
)


def test_get_all_badges_includes_requirements_and_counts():
    df_badges = pd.DataFrame(
        [
            {"badge_id": "b1", "name": "Alpha", "requirements": "Win 1 match"},
            {"badge_id": "b2", "name": "Beta", "requirements": "Win 2 matches"},
        ]
    )
    df_player_badges = pd.DataFrame(
        [
            {"badge_id": "b1", "player_id": 1, "earned_at": "2023-01-01"},
            {"badge_id": "b1", "player_id": 1, "earned_at": "2023-02-01"},
            {"badge_id": "b1", "player_id": 2, "earned_at": "2023-03-01"},
        ]
    )

    badges = get_all_badges(df_badges, df_player_badges)

    assert [badge["badge_id"] for badge in badges] == ["b1", "b2"]
    assert badges[0]["requirements"] == "Win 1 match"
    assert badges[0]["earners_count"] == 2
    assert badges[1]["earners_count"] == 0


def test_get_badge_earners_page_paginates_and_maps_names():
    df_player_badges = pd.DataFrame(
        [
            {"badge_id": "b1", "player_id": 1, "earned_at": "2023-01-01"},
            {"badge_id": "b1", "player_id": 2, "earned_at": "2023-02-01"},
            {"badge_id": "b1", "player_id": 3, "earned_at": "2023-03-01"},
        ]
    )
    df_players = pd.DataFrame(
        [
            {"id": 1, "name": "Ada"},
            {"id": 2, "name": "Bela"},
            {"id": 3, "name": "Chen"},
        ]
    )

    page_one, total = get_badge_earners_page(df_player_badges, df_players, "b1", 0, 2)
    page_two, total_two = get_badge_earners_page(df_player_badges, df_players, "b1", 2, 2)

    assert total == 3
    assert total_two == 3
    assert [row["name"] for row in page_one] == ["Chen", "Bela"]
    assert [row["name"] for row in page_two] == ["Ada"]


def test_get_all_badges_carries_status_fields_with_defaults():
    df_badges = pd.DataFrame(
        [
            {"badge_id": "live", "name": "Live Badge"},
            {
                "badge_id": "manual",
                "name": "Manual Badge",
                "badge_status": "live",
                "badge_award_timing": "manual",
                "badge_scope": "league",
            },
        ]
    )

    badges = get_all_badges(df_badges, pd.DataFrame())

    assert badges[0]["badge_status"] == "live"
    assert badges[0]["badge_award_timing"] == "live"
    assert badges[0]["badge_scope"] is None
    assert badges[1]["badge_award_timing"] == "manual"
    assert badges[1]["badge_scope"] == "league"


def test_split_badges_for_catalog_separates_live_and_non_live():
    badges = [
        {"badge_id": "a", "badge_status": "live", "badge_award_timing": "live"},
        {"badge_id": "b", "badge_status": "tracked", "badge_award_timing": "live"},
        {"badge_id": "c", "badge_status": "live", "badge_award_timing": "manual"},
        {"badge_id": "d", "badge_status": "live", "badge_award_timing": "on_league_close"},
    ]

    grouped = _split_badges_for_catalog(badges)

    assert [b["badge_id"] for b in grouped["Live Now"]] == ["a"]
    assert [b["badge_id"] for b in grouped["Tracked / Disabled"]] == ["b"]
    assert [b["badge_id"] for b in grouped["Manual / Curated"]] == ["c"]
    assert [b["badge_id"] for b in grouped["Seasonal / League Close"]] == ["d"]


def test_deprecated_zero_earner_hidden_unless_included():
    df_badges = pd.DataFrame(
        [
            {"badge_id": "dep0", "name": "Old Badge", "badge_status": "deprecated"},
            {"badge_id": "dep1", "name": "Old Earned", "badge_status": "deprecated"},
        ]
    )
    df_player_badges = pd.DataFrame([{"badge_id": "dep1", "player_id": 1}])

    hidden = get_all_badges(df_badges, df_player_badges, include_deprecated=False)
    included = get_all_badges(df_badges, df_player_badges, include_deprecated=True)

    assert [badge["badge_id"] for badge in hidden] == ["dep1"]
    assert [badge["badge_id"] for badge in included] == ["dep0", "dep1"]


def test_category_normalization_collapses_capitalization_variants():
    assert _normalize_category_label("dominance") == "Dominance"
    assert _normalize_category_label("Dominance") == "Dominance"
    assert _normalize_category_label("  DOMINANCE  ") == "Dominance"


def test_availability_label_for_manual_and_tracked():
    assert _availability_label({"badge_status": "live", "badge_award_timing": "manual"}) == "Manual award"
    assert _availability_label({"badge_status": "tracked", "badge_award_timing": "live"}) == "Tracked only"
