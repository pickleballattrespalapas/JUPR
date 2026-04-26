import math
from types import SimpleNamespace

import pandas as pd

from jupr_app.domain.player_ratings_source import build_seed_rating_maps, current_seed_rating
from jupr_app.ui.pages.league_results import (
    _build_current_standings_from_league_ratings,
    _build_scoped_league_standings,
)


class FailingSupabase:
    def table(self, _name):
        raise RuntimeError("db unavailable")


def _fixture_frames():
    league_name = "2026 Spring Ladder"
    df_players_all = pd.DataFrame(
        [
            {"id": 1, "name": "Joe Baumann", "rating": 2066.0},
            {"id": 2, "name": "Toby McElravey", "rating": 1900.0},
        ]
    )
    df_players_active = pd.DataFrame([{"id": 1}, {"id": 2}])
    df_leagues = pd.DataFrame(
        [
            {
                "league_name": league_name,
                "player_id": 1,
                "rating": 2056.8,
                "starting_rating": 2083.2,
                "wins": 9,
                "losses": 3,
                "matches_played": 12,
            },
            {
                "league_name": league_name,
                "player_id": 2,
                "rating": 1866.0,
                "starting_rating": 1878.0,
                "wins": 7,
                "losses": 5,
                "matches_played": 12,
            },
        ]
    )
    return SimpleNamespace(
        league_name=league_name,
        df_players_all=df_players_all,
        df_players_active=df_players_active,
        df_leagues=df_leagues,
    )


def test_current_standings_use_league_ratings_values():
    fx = _fixture_frames()
    standings = _build_current_standings_from_league_ratings(
        fx.df_leagues, fx.df_players_all, fx.df_players_active, fx.league_name
    )

    joe = standings[standings["player_name"] == "Joe Baumann"].iloc[0]
    toby = standings[standings["player_name"] == "Toby McElravey"].iloc[0]

    assert math.isclose(float(joe["JUPR"]), 5.142, abs_tol=1e-9)
    assert math.isclose(float(toby["JUPR"]), 4.665, abs_tol=1e-9)


def test_current_standings_prefer_league_ratings_over_replay_values():
    fx = _fixture_frames()
    overall_stats = pd.DataFrame(
        [
            {"player_id": 1, "player_name": "Joe Baumann", "games": 12, "wins": 9, "losses": 3},
            {"player_id": 2, "player_name": "Toby McElravey", "games": 12, "wins": 7, "losses": 5},
        ]
    )
    replay_df = pd.DataFrame(
        [
            {"player_id": 1, "start_rating": 2083.2, "end_rating": 2076.4},  # 5.191 replay JUPR
            {"player_id": 2, "start_rating": 1878.0, "end_rating": 1866.0},
        ]
    )

    replay_standings = _build_scoped_league_standings(overall_stats, replay_df, {1: "Joe Baumann", 2: "Toby McElravey"})
    league_standings = _build_current_standings_from_league_ratings(
        fx.df_leagues, fx.df_players_all, fx.df_players_active, fx.league_name
    )

    joe_replay = replay_standings[replay_standings["player_id"] == 1].iloc[0]
    joe_league = league_standings[league_standings["player_id"] == 1].iloc[0]
    assert math.isclose(float(joe_replay["JUPR"]), 5.191, abs_tol=1e-9)
    assert math.isclose(float(joe_league["JUPR"]), 5.142, abs_tol=1e-9)
    assert float(joe_league["JUPR"]) != float(joe_replay["JUPR"])


def test_league_results_and_league_manager_seed_rating_agree():
    fx = _fixture_frames()
    standings = _build_current_standings_from_league_ratings(
        fx.df_leagues, fx.df_players_all, fx.df_players_active, fx.league_name
    )
    overall_map, league_map, from_live = build_seed_rating_maps(
        supabase=FailingSupabase(),
        club_id="club-1",
        player_ids={1, 2},
        league_names={fx.league_name},
        df_players_all=fx.df_players_all,
        df_leagues=fx.df_leagues,
    )
    assert from_live is False

    joe_seed = current_seed_rating(
        player_id=1,
        league_name=fx.league_name,
        overall_map=overall_map,
        league_map=league_map,
    )
    toby_seed = current_seed_rating(
        player_id=2,
        league_name=fx.league_name,
        overall_map=overall_map,
        league_map=league_map,
    )
    joe_row = standings[standings["player_id"] == 1].iloc[0]
    toby_row = standings[standings["player_id"] == 2].iloc[0]

    assert math.isclose(joe_seed / 400.0, float(joe_row["JUPR"]), abs_tol=1e-9)
    assert math.isclose(toby_seed / 400.0, float(toby_row["JUPR"]), abs_tol=1e-9)
