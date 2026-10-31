from copy import deepcopy

import pytest

from jupr_app.domain.tournaments.pools import build_draw_round_robin_games
from jupr_app.services.admin_tournament_game_service import (
    generate_admin_tournament_round_robin_games,
    reconcile_admin_tournament_round_robin_games,
    rebuild_admin_tournament_round_robin_games,
)
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_admin_tournament_schedule_recovery_service import _tables, _team_versions, UPDATED


def pool_tables(sizes):
    tables = _tables(team_count=sum(sizes), existing_count=0)
    pools = []
    offset = 0
    for index, size in enumerate(sizes):
        pools.append({
            "id": chr(65 + index),
            "name": f"Pool {chr(65 + index)}",
            "team_ids": [row["id"] for row in tables["tournament_teams"][offset:offset+size]],
            "advance_count": 2,
        })
        offset += size
    tables["tournament_event_draws"][0]["pool_config"] = {"version": 1, "pools": pools}
    return tables


def service_kwargs(tables):
    return dict(
        club_id="club", tournament_id="tour-1", draw_id="draw-1",
        actor_email="director@example.com", actor_role="club_owner",
        expected_draw_updated_at=UPDATED, expected_team_versions=_team_versions(tables),
    )


@pytest.fixture(autouse=True)
def enabled(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "test")


@pytest.mark.parametrize("sizes,expected", [([3, 3], 6), ([4, 4], 12), ([4, 3], 9)])
def test_generate_pool_draw_only_schedules_within_each_pool(sizes, expected):
    tables = pool_tables(sizes)
    result = generate_admin_tournament_round_robin_games(
        FakeSupabase(tables), **service_kwargs(tables), confirmation_text="GENERATE GAMES"
    )
    assert result["game_count"] == expected
    pools = tables["tournament_event_draws"][0]["pool_config"]["pools"]
    mapping = {team_id: pool["id"] for pool in pools for team_id in pool["team_ids"]}
    assert all(mapping[game["team_a_id"]] == mapping[game["team_b_id"]] for game in result["games"])
    assert len({(game["rr_round_number"], game["rr_slot_number"]) for game in result["games"]}) == expected


def partial_pool_tables():
    tables = pool_tables([3, 3])
    games = build_draw_round_robin_games(
        tournament_id="tour-1", teams=tables["tournament_teams"],
        pool_config=tables["tournament_event_draws"][0]["pool_config"],
    )
    tables["tournament_games"] = [
        {**row, "id": f"game-{index}", "draw_id": "draw-1",
         "updated_at": UPDATED, "score_a": None, "score_b": None}
        for index, row in enumerate(games[:2])
    ]
    first = tables["tournament_games"][0]
    first.update(score_a=11, score_b=7, winner_team_id=first["team_a_id"],
                 loser_team_id=first["team_b_id"], finalized_at=UPDATED)
    return tables


def test_reconcile_pool_draw_preserves_scores_and_does_not_add_crossovers():
    tables = partial_pool_tables()
    before = deepcopy(tables["tournament_games"][0])
    result = reconcile_admin_tournament_round_robin_games(
        FakeSupabase(tables), **service_kwargs(tables), confirmation_text="RECONCILE GAMES",
        allow_non_atomic_test_adapter=True,
    )
    assert result["game_count"] == 6
    assert result["inserted_game_count"] == 4
    assert tables["tournament_games"][0] == before
    assert len(tables["tournament_games"]) == 6


def test_reconcile_rejects_cross_pool_game_without_mutating():
    tables = partial_pool_tables()
    tables["tournament_games"][1]["team_a_id"] = "team-1"
    tables["tournament_games"][1]["team_b_id"] = "team-4"
    before = deepcopy(tables)
    with pytest.raises(ValueError, match="[Pp]ool"):
        reconcile_admin_tournament_round_robin_games(
            FakeSupabase(tables), **service_kwargs(tables), confirmation_text="RECONCILE GAMES",
            allow_non_atomic_test_adapter=True,
        )
    assert tables == before


def test_rebuild_unstarted_pool_schedule_keeps_the_split():
    tables = partial_pool_tables()
    for game in tables["tournament_games"]:
        for field in ("score_a", "score_b", "winner_team_id", "loser_team_id", "finalized_at"):
            game[field] = None
    result = rebuild_admin_tournament_round_robin_games(
        FakeSupabase(tables), **service_kwargs(tables), confirmation_text="REBUILD GAMES",
        allow_non_atomic_test_adapter=True,
    )
    assert result["game_count"] == 6
    assert len(tables["tournament_games"]) == 6
