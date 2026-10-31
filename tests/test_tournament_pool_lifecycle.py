from copy import deepcopy

import pytest

from jupr_app.services.admin_tournament_lifecycle_service import build_admin_tournament_lifecycle
from jupr_app.services.admin_tournament_podium_review_service import _validate_reviewable_podium
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_admin_tournament_lifecycle_service import _add_awards, _review_draw
from tests.test_admin_tournament_podium_review_service import podium_review_tables


def pool_lifecycle_tables(*, final=True):
    tables = podium_review_tables()
    updated = tables["tournament_event_draws"][0]["updated_at"]
    fourth = {**tables["tournament_teams"][2], "id": "team-4", "team_number": 4,
              "player1_id": 7, "player2_id": 8}
    tables["tournament_teams"].append(fourth)
    tables["players"].extend({"club_id": "club", "id": i, "name": f"Player {i}"} for i in (7, 8))
    tables["tournament_event_draws"][0]["pool_config"] = {
        "version": 1, "pools": [
            {"id": "A", "name": "Pool A", "team_ids": ["team-1", "team-2"], "advance_count": 1},
            {"id": "B", "name": "Pool B", "team_ids": ["team-3", "team-4"], "advance_count": 1},
        ]}
    pool_a = {**tables["tournament_games"][0], "rr_round_number": 1, "rr_slot_number": 1}
    pool_b = {**pool_a, "id": "game-2", "rr_slot_number": 2,
              "team_a_id": "team-3", "team_b_id": "team-4",
              "winner_team_id": "team-3", "loser_team_id": "team-4"}
    tables["tournament_games"] = [pool_a, pool_b]
    tables["tournament_podium"] = []
    if final:
        tables["tournament_games"].append({
            **pool_a, "id": "game-final", "stage": "PLAYOFF",
            "rr_round_number": None, "rr_slot_number": None,
            "playoff_game_code": "P1", "playoff_round": "Final",
            "team_a_id": "team-1", "team_b_id": "team-3",
            "winner_team_id": "team-1", "loser_team_id": "team-3",
        })
        tables["tournament_podium"] = [
            {"id": f"podium-{place}", "tournament_id": "tour-1", "draw_id": "draw-1",
             "placement": place, "team_id": team_id, "source": "PLAYOFF", "updated_at": updated}
            for place, team_id in ((1, "team-1"), (2, "team-3"))]
    return tables


def test_two_pool_winners_can_finish_one_final_and_review_two_medals(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = pool_lifecycle_tables()
    db = FakeSupabase(tables)
    _review_draw(db, tables, "draw-1")
    _add_awards(tables, "draw-1")
    lifecycle = build_admin_tournament_lifecycle(db, club_id="club", tournament_id="tour-1", selected_draw_id="draw-1")
    draw = lifecycle["draws"][0]
    assert len(draw["podium"]) == 2
    assert draw["standings"] == []
    assert [[row["team_id"] for row in pool["standings"]] for pool in draw["pool_standings"]] == [
        ["team-1", "team-2"], ["team-3", "team-4"]]
    assert "PODIUM_INCOMPLETE" not in {row["code"] for row in lifecycle["domain_readiness"]["official_publish"]["blockers"]}
    assert "PODIUM_REVIEW_REQUIRED" not in {row["code"] for row in lifecycle["domain_readiness"]["official_publish"]["blockers"]}


def test_pool_progress_does_not_publish_a_combined_round_robin_ranking(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = pool_lifecycle_tables(final=False)
    lifecycle = build_admin_tournament_lifecycle(FakeSupabase(tables), club_id="club", tournament_id="tour-1")
    draw = lifecycle["draws"][0]
    assert draw["standings"] == []
    assert [pool["pool_name"] for pool in draw["pool_standings"]] == ["Pool A", "Pool B"]
    assert [pool["standings"][0]["wins"] for pool in draw["pool_standings"]] == [1, 1]
    assert lifecycle["domain_readiness"]["official_publish"]["ready"] is False


def test_pool_podium_cannot_be_reviewed_from_aggregate_pool_scores():
    tables = pool_lifecycle_tables(final=False)
    podium = [{"placement": i, "team_id": f"team-{i}", "source": "ROUND_ROBIN"} for i in (1, 2, 3)]
    with pytest.raises(ValueError, match="[Pp]layoff|[Pp]ool"):
        _validate_reviewable_podium(
            draw=tables["tournament_event_draws"][0], teams=tables["tournament_teams"],
            games=tables["tournament_games"], podium=podium)


def test_pool_podium_must_match_actual_shared_final():
    tables = pool_lifecycle_tables()
    wrong = deepcopy(tables["tournament_podium"])
    wrong[0]["team_id"] = "team-2"
    with pytest.raises(ValueError, match="[Pp]layoff|[Pp]odium|[Pp]ool"):
        _validate_reviewable_podium(
            draw=tables["tournament_event_draws"][0], teams=tables["tournament_teams"],
            games=tables["tournament_games"], podium=wrong)
