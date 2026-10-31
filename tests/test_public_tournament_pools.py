from __future__ import annotations

from itertools import combinations
import json

import pytest

from jupr_app.services.public_tournament_results_service import build_public_tournament_results
from tests.test_public_tournament_registration_service import FakeSupabase
from tests.test_public_tournament_results_service import _results_storage


def _pool_storage() -> dict:
    storage = _results_storage()
    draw_id = "draw-private-id"
    teams = [
        {
            "id": f"private-team-{number}",
            "tournament_id": "t1",
            "draw_id": draw_id,
            "team_number": number,
            "player1_id": number,
        }
        for number in range(1, 8)
    ]
    storage["tournament_teams"] = teams
    storage["players"] = [
        {"id": number, "club_id": "club-1", "name": f"Player {number}"}
        for number in range(1, 8)
    ]
    storage["tournament_event_draws"][0]["pool_config"] = {
        "version": 1,
        "pools": [
            {"id": "A", "name": "Pool A", "advance_count": 2, "team_ids": [row["id"] for row in teams[:4]]},
            {"id": "B", "name": "Pool B", "advance_count": 2, "team_ids": [row["id"] for row in teams[4:]]},
        ],
    }
    games = []
    for group in (teams[:4], teams[4:]):
        for team_a, team_b in combinations(group, 2):
            games.append({
                "id": f"private-game-{len(games)}",
                "tournament_id": "t1",
                "draw_id": draw_id,
                "stage": "ROUND_ROBIN",
                "rr_round_number": len(games) + 1,
                "rr_slot_number": 1,
                "team_a_id": team_a["id"],
                "team_b_id": team_b["id"],
                "score_a": 11,
                "score_b": 5,
                "winner_team_id": team_a["id"],
                "finalized_at": "2026-10-31T20:00:00Z",
            })
    games.append({
        "id": "private-final",
        "tournament_id": "t1",
        "draw_id": draw_id,
        "stage": "PLAYOFF",
        "playoff_round": "Final",
        "playoff_game_code": "P3",
        "team_a_id": teams[0]["id"],
        "team_b_id": teams[4]["id"],
        "score_a": 11,
        "score_b": 8,
        "winner_team_id": teams[0]["id"],
        "finalized_at": "2026-10-31T22:00:00Z",
    })
    storage["tournament_games"] = games
    storage["tournament_podium"] = [
        {"id": "private-podium", "tournament_id": "t1", "draw_id": draw_id, "placement": 1, "team_id": teams[0]["id"]}
    ]
    return storage


def _public_draw(storage: dict) -> dict:
    return build_public_tournament_results(FakeSupabase(storage), club_id="club-1", tournament_id="t1")["draws"][0]


def test_public_unequal_pools_have_independent_ranks_and_one_division_bracket() -> None:
    draw = _public_draw(_pool_storage())

    assert draw["standings"] == []
    assert draw["tiebreak_explanations"] == []
    assert draw["round_robin_complete"] is True
    assert [pool["name"] for pool in draw["pool_standings"]] == ["Pool A", "Pool B"]
    assert [row["rank"] for row in draw["pool_standings"][0]["standings"]] == [1, 2, 3, 4]
    assert [row["rank"] for row in draw["pool_standings"][1]["standings"]] == [1, 2, 3]
    assert [pool["standings"][0]["wins"] for pool in draw["pool_standings"]] == [3, 2]
    assert all(pool["advance_count"] == 2 for pool in draw["pool_standings"])
    assert len(draw["scores"]) == 9
    assert len(draw["bracket"]) == 1
    assert draw["bracket"][0]["pool_name"] is None
    assert draw["bracket"][0]["public_pool_key"] is None
    assert len(draw["podium"]) == 1
    assert [row["pool_name"] for row in draw["scores"]].count("Pool A") == 6
    public_keys = {pool["public_pool_key"] for pool in draw["pool_standings"]}
    assert {game["public_pool_key"] for game in draw["scores"]} == public_keys
    serialized = json.dumps(draw)
    for private_value in ("private-team-", "private-game-", "draw-private-id", "pool_config", "team_ids"):
        assert private_value not in serialized


def test_pool_results_hide_draft_scores_and_mark_only_that_pool_incomplete() -> None:
    storage = _pool_storage()
    draft = storage["tournament_games"][-2]
    draft["score_a"] = 99
    draft["finalized_at"] = None
    draw = _public_draw(storage)

    assert draw["round_robin_complete"] is False
    assert [pool["round_robin_complete"] for pool in draw["pool_standings"]] == [True, False]
    assert len(draw["scores"]) == 8
    assert all(row["points_for"] < 99 for pool in draw["pool_standings"] for row in pool["standings"])


def test_public_pool_standings_refuse_cross_pool_round_robin_results() -> None:
    storage = _pool_storage()
    storage["tournament_games"][0]["team_b_id"] = "private-team-5"
    with pytest.raises(ValueError):
        _public_draw(storage)


def test_partial_saved_schedule_is_not_misrepresented_as_completed_pool_play() -> None:
    storage = _pool_storage()
    storage["tournament_games"].pop(0)
    draw = _public_draw(storage)

    assert draw["round_robin_complete"] is False
    assert draw["state"] == "LIVE"
    assert [pool["round_robin_complete"] for pool in draw["pool_standings"]] == [False, True]
