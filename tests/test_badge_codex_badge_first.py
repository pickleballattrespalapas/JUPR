import pandas as pd

from jupr_app.ui.pages.badge_codex import get_all_badges, get_badge_earners_page


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
