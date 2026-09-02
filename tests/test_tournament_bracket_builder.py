from collections import Counter

import pytest

from jupr_app.domain.tournaments.bracket_builder import SUPPORTED_TEAM_COUNTS, build_round_robin_games


def test_supported_team_counts_include_six() -> None:
    assert SUPPORTED_TEAM_COUNTS == list(range(4, 17))


def test_round_robin_6_team_template_loads_from_csv() -> None:
    team_ids_by_number = {team_number: f"team-{team_number}" for team_number in range(1, 7)}

    games = build_round_robin_games(tournament_id="tour-1", team_ids_by_number=team_ids_by_number)

    assert len(games) == 15
    assert {game["rr_round_number"] for game in games} == {1, 2, 3, 4, 5}
    assert {game["rr_slot_number"] for game in games} == {1, 2, 3}

    pairings = {
        tuple(sorted((game["team_a_id"], game["team_b_id"])))
        for game in games
    }
    assert len(pairings) == 15


@pytest.mark.parametrize("team_count", SUPPORTED_TEAM_COUNTS)
def test_round_robins_schedule_every_pair_once_without_round_conflicts(
    team_count: int,
) -> None:
    team_ids = {
        team_number: f"team-{team_number}"
        for team_number in range(1, team_count + 1)
    }
    games = build_round_robin_games(
        tournament_id="tour-matrix",
        team_ids_by_number=team_ids,
    )

    expected_game_count = team_count * (team_count - 1) // 2
    expected_round_count = team_count if team_count % 2 else team_count - 1
    expected_games_per_round = team_count // 2
    expected_coordinates = {
        (round_number, slot_number)
        for round_number in range(1, expected_round_count + 1)
        for slot_number in range(1, expected_games_per_round + 1)
    }

    assert len(games) == expected_game_count
    assert {
        (game["rr_round_number"], game["rr_slot_number"])
        for game in games
    } == expected_coordinates
    assert len(
        {
            tuple(sorted((game["team_a_id"], game["team_b_id"])))
            for game in games
        }
    ) == expected_game_count

    appearances_by_team = Counter(
        team_id
        for game in games
        for team_id in (game["team_a_id"], game["team_b_id"])
    )
    assert appearances_by_team == Counter(
        {team_id: team_count - 1 for team_id in team_ids.values()}
    )

    for round_number in range(1, expected_round_count + 1):
        appearances = [
            team_id
            for game in games
            if game["rr_round_number"] == round_number
            for team_id in (game["team_a_id"], game["team_b_id"])
        ]
        assert len(appearances) == 2 * expected_games_per_round
        assert len(appearances) == len(set(appearances))
