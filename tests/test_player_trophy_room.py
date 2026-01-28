import pandas as pd

from jupr_app.ui.pages.players import build_completed_league_podium, build_inactive_league_options


def test_build_inactive_league_options_prefers_completed():
    df_meta = pd.DataFrame(
        [
            {
                "league_name": "Spring 2024 Ladder",
                "is_active": False,
                "status": "completed",
                "end_date": "2024-05-01",
                "season_label": "Spring 2024",
            },
            {
                "league_name": "Summer 2024 Ladder",
                "is_active": False,
                "status": "completed",
                "end_date": "2024-08-01",
                "season_label": "Summer 2024",
            },
            {
                "league_name": "Fall 2024 Ladder",
                "is_active": True,
                "status": "active",
                "end_date": "2099-11-01",
                "season_label": "Fall 2024",
            },
        ]
    )
    options = build_inactive_league_options(df_meta, pd.DataFrame(), pd.DataFrame())
    assert options["league_name"].tolist() == ["Summer 2024 Ladder", "Spring 2024 Ladder"]


def test_build_completed_league_podium_orders_by_wins_and_rating():
    df_matches = pd.DataFrame(
        [
            {
                "league": "Spring 2024 Ladder",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 5,
                "week_tag": "Week 1",
            },
            {
                "league": "Spring 2024 Ladder",
                "t1_p1": 1,
                "t1_p2": 3,
                "t2_p1": 2,
                "t2_p2": 4,
                "score_t1": 8,
                "score_t2": 11,
                "week_tag": "Week 2",
            },
            {
                "league": "Spring 2024 Ladder",
                "t1_p1": 1,
                "t1_p2": 4,
                "t2_p1": 2,
                "t2_p2": 3,
                "score_t1": 11,
                "score_t2": 9,
                "week_tag": "Week 3",
            },
        ]
    )
    df_leagues = pd.DataFrame(
        [
            {"league_name": "Spring 2024 Ladder", "player_id": 1, "rating": 1500},
            {"league_name": "Spring 2024 Ladder", "player_id": 2, "rating": 1400},
            {"league_name": "Spring 2024 Ladder", "player_id": 3, "rating": 1300},
            {"league_name": "Spring 2024 Ladder", "player_id": 4, "rating": 1200},
        ]
    )
    id_to_name = {1: "Alice", 2: "Ben", 3: "Cory", 4: "Dia"}

    podium = build_completed_league_podium(
        df_matches,
        df_leagues,
        id_to_name,
        "Spring 2024 Ladder",
    )

    assert podium["player_id"].tolist() == [1, 2, 4]
    assert podium["rank"].tolist() == [1, 2, 3]
