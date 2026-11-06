from copy import deepcopy

import pytest

from jupr_app.domain.leaderboard_metrics import LEADERBOARD_CARD_KEYS
from jupr_app.services import leaderboard_service as service
from tests.test_leaderboard_seasons import fixture
from tests.test_overall_leaderboard_metrics import game, team_setup
from tests.test_public_leaderboard_projection import _fixture


@pytest.mark.parametrize("stat", LEADERBOARD_CARD_KEYS)
def test_stat_ranking_reuses_card_calculation_and_selected_period(stat):
    db, config = team_setup()
    config["cards"] = list(LEADERBOARD_CARD_KEYS)
    config["card_options"] = {stat: {"minimum": 0, "depth": 1}}
    db.store["matches"] = [
        game(1, score_t2=9, t1_p1_r=1200, t1_p2_r=1200),
        game(2, t1_p2=3, t2_p1=2, score_t2=9, t1_p1_r=1100, t1_p2_r=1100),
        game(3, score_t1=9, score_t2=11, t2_p1_r=1100, t2_p2_r=1100),
    ]
    before = deepcopy(db.store)
    cards = service.build_public_leaderboard(db, club_id="club-1", status="all")
    details = service.build_public_leaderboard(db, club_id="club-1", status="all", stat=stat)
    ranking = details["stat_ranking"]
    assert ranking["key"] == stat
    assert cards["period"] == details["period"]
    identity_and_stat = ("player_id", "team_key", "rank", "metric_value", "metric_display", "metric_sample")
    project = lambda rows: [{key: row.get(key) for key in identity_and_stat} for row in rows]
    assert project(ranking["entries"][:1]) == project(cards["highlights"][stat])
    assert ranking["total_qualified"] == len(ranking["entries"])
    assert [row["rank"] for row in ranking["entries"]] == list(range(1, len(ranking["entries"]) + 1))
    assert db.store == before


def test_pagination_search_keeps_metric_positions_and_does_not_load_badges(monkeypatch):
    db = fixture()
    config = db.store["pcs_club_sites"][0]["published"]["leaderboard"]
    config["default_season_id"] = None
    config["cards"] = ["highest_rating"]
    config["card_options"] = {"highest_rating": {"minimum": 0, "depth": 1}}
    template = db.store["players"][0]
    db.store["players"] = [{**template, "id": index, "name": f"Player {index}", "rating": 2000 - index} for index in range(1, 16)]
    monkeypatch.setattr(service, "_badge_map", lambda *_args, **_kwargs: pytest.fail("Stat pages must not load badges"))
    page = service.build_public_leaderboard(db, club_id="club-1", stat="highest_rating", limit=5, offset=5)["stat_ranking"]
    assert page["total_qualified"] == 15
    assert [row["rank"] for row in page["entries"]] == [6, 7, 8, 9, 10]
    assert page["pagination"] == {"total": 15, "offset": 5, "limit": 5, "has_more": True}
    found = service.build_public_leaderboard(db, club_id="club-1", stat="highest_rating", search="Player 12")["stat_ranking"]
    assert [row["rank"] for row in found["entries"]] == [12]
    assert found["total_qualified"] == 15
    assert found["pagination"]["total"] == 1
    assert db.match_queries == []


@pytest.mark.parametrize("stat", ["best_partnership", "biggest_upset"])
def test_pair_search_matches_either_member_after_ranking_without_duplicates(stat):
    db, config = team_setup()
    config["cards"] = [stat]
    config["card_options"] = {stat: {"minimum": 1, "depth": 1}}
    db.store["matches"] = [game(1, t1_p1_r=1200, t1_p2_r=1200),
                           game(2, t1_p2=3, t2_p1=2, t1_p1_r=1100, t1_p2_r=1100)]
    full = service.build_public_leaderboard(db, club_id="club-1", stat=stat)["stat_ranking"]
    assert len(full["entries"]) > 1
    found = service.build_public_leaderboard(db, club_id="club-1", stat=stat, search="Blake")["stat_ranking"]
    assert len({row["team_key"] for row in found["entries"]}) == len(found["entries"])
    by_pair = {row["team_key"]: row for row in full["entries"]}
    assert found["entries"] == [row for row in full["entries"] if any(member["player_name"] == "Blake Baseline" for member in row["team_members"])]
    assert all(row["rank"] == by_pair[row["team_key"]]["rank"] for row in found["entries"])
    if stat == "biggest_upset":
        best = full["entries"][0]
        assert best["upset_score_for"] == 11
        assert best["upset_score_against"] == 7
        assert len(best["upset_opponents"]) == 2


def test_qualification_status_and_empty_data_are_preserved():
    db, config = team_setup()
    config["cards"] = ["close_game_record"]
    config["min_games"] = 3
    config["card_options"] = {"close_game_record": {"minimum": 2, "depth": 1}}
    db.store["matches"] = [game(1, score_t2=9), game(2, score_t2=9)]
    empty = service.build_public_leaderboard(db, club_id="club-1", stat="close_game_record")["stat_ranking"]
    assert empty["entries"] == []
    assert (empty["min_games"], empty["minimum"], empty["minimum_metric"]) == (3, 2, "close_games")
    db.store["matches"].append(game(3))
    db.store["players"][0]["active"] = False
    active = service.build_public_leaderboard(db, club_id="club-1", stat="close_game_record")["stat_ranking"]
    all_players = service.build_public_leaderboard(db, club_id="club-1", stat="close_game_record", status="all")["stat_ranking"]
    assert 1 not in [row["player_id"] for row in active["entries"]]
    assert 1 in [row["player_id"] for row in all_players["entries"]]
    assert all(row["close_games"] == 2 for row in all_players["entries"])


@pytest.mark.parametrize("stat", LEADERBOARD_CARD_KEYS[:5])
def test_league_stat_top_matches_existing_league_card(stat):
    db = _fixture()
    cards = service.build_public_leaderboard(db, club_id="club-1", league_name="Pro", status="all")
    assert cards["selected_scope"] == "Pro"
    full = service.build_public_leaderboard(db, club_id="club-1", league_name="Pro", status="all", stat=stat)
    assert full["selected_scope"] == cards["selected_scope"]
    assert [row["player_id"] for row in full["stat_ranking"]["entries"][:5]] == [row["player_id"] for row in cards["highlights"][stat]]


@pytest.mark.parametrize("kwargs", [{"stat": "secret"}, {"stat": "most_wins", "league_name": "Missing league"},
                                    {"stat": "hot_hand", "league_name": "Pro"}])
def test_invalid_stats_or_scopes_do_not_silently_fall_back(kwargs):
    with pytest.raises(service.LeaderboardStatInvalid):
        service.build_public_leaderboard(_fixture(), club_id="club-1", **kwargs)


def test_public_stat_response_allowlists_support_and_nested_player_fields(monkeypatch):
    from fastapi.testclient import TestClient
    from services.api import main

    monkeypatch.setattr(main, "get_club", lambda slug: {"id": "club-1", "name": "Club", "slug": slug})
    monkeypatch.setattr(main, "get_supabase_client", object)
    def projection(*args, **kwargs):
        assert kwargs["stat"] == "biggest_upset"
        return {"stat_ranking": {"key": "biggest_upset", "entries": [{
            "rank": 11, "metric_value": .5, "metric_sample": 2, "email": "private",
            "team_members": [{"player_id": 1, "player_name": "A", "email": "private"}],
            "upset_opponents": [{"player_id": 2, "player_name": "B", "email": "private"}],
            "upset_score_for": 11, "upset_score_against": 7,
        }], "total_qualified": 20, "minimum": 1, "minimum_metric": "upset_wins",
        "pagination": {"total": 20, "offset": 10, "limit": 10, "has_more": False}}}
    monkeypatch.setattr(main, "build_public_leaderboard", projection)
    client = TestClient(main.app)
    response = client.get("/clubs/club/leaderboards?stat=biggest_upset&offset=10&limit=10")
    assert response.status_code == 200
    assert "private" not in response.text
    row = response.json()["stat_ranking"]["entries"][0]
    assert row["rank"] == 11
    assert row["upset_opponents"] == [{"player_id": 2, "player_name": "B"}]
    assert row["upset_score_for"] == 11


def test_public_stat_invalid_query_returns_422(monkeypatch):
    from fastapi.testclient import TestClient
    from services.api import main
    monkeypatch.setattr(main, "get_club", lambda slug: {"id": "club-1", "name": "Club", "slug": slug})
    monkeypatch.setattr(main, "get_supabase_client", lambda: _fixture())
    for query in ("stat=unknown", "stat=most_wins&league_name=Unknown"):
        assert TestClient(main.app).get(f"/clubs/club/leaderboards?{query}").status_code == 422


def test_partnership_supporting_record_does_not_invent_losses_for_legacy_ties():
    db, _ = team_setup()
    db.store["matches"] = [game(1), game(2, score_t1=9, score_t2=9)]
    ranking = service.build_public_leaderboard(db, club_id="club-1", stat="best_partnership")["stat_ranking"]
    pair = next(row for row in ranking["entries"] if row["team_key"] == "1:2")
    assert (pair["wins"], pair["losses"], pair["matches_played"]) == (1, 0, 2)
    assert pair["metric_value"] == 50
