from collections import Counter
from itertools import combinations

import pytest

from jupr_app.domain.tournament_four_player_team import (
    build_locked_team_games,
    build_team_playoff_matchups,
    build_team_round_robin_matchups,
    build_team_standings,
    build_team_tiebreak_game,
    calculate_team_podium,
    evaluate_team_match,
    validate_four_player_roster,
)


def _roster(offset: int = 0):
    return [
        {"slot": "MAN_1", "player_id": 1 + offset, "status": "ACCEPTED"},
        {"slot": "MAN_2", "player_id": 2 + offset, "status": "ACCEPTED"},
        {"slot": "WOMAN_1", "player_id": 3 + offset, "status": "ACCEPTED"},
        {"slot": "WOMAN_2", "player_id": 4 + offset, "status": "ACCEPTED"},
    ]


def test_roster_requires_two_men_two_women_and_distinct_players():
    assert [row["slot"] for row in validate_four_player_roster(_roster())] == [
        "MAN_1",
        "MAN_2",
        "WOMAN_1",
        "WOMAN_2",
    ]
    duplicate = _roster()
    duplicate[-1]["player_id"] = 1
    with pytest.raises(ValueError, match="one team roster slot"):
        validate_four_player_roster(duplicate)


@pytest.mark.parametrize("pairing", ["STRAIGHT", "CROSS"])
def test_locked_games_give_every_player_gender_and_one_mixed_game(pairing):
    games = build_locked_team_games(
        _roster(),
        mixed_pairing=pairing,
        singles_tiebreak_player_id=1,
    )
    appearances = Counter(
        player_id for game in games for player_id in game["player_ids"]
    )

    assert [game["game_code"] for game in games] == [
        "WOMENS",
        "MENS",
        "MIXED_1",
        "MIXED_2",
    ]
    assert appearances == Counter({1: 2, 2: 2, 3: 2, 4: 2})
    assert all(game["counts_for_rating"] for game in games)


def test_skinny_relay_is_not_rated_and_singles_tiebreak_is_rated():
    skinny = build_team_tiebreak_game(_roster(), tiebreak_mode="SKINNY_RELAY")
    singles = build_team_tiebreak_game(
        _roster(),
        tiebreak_mode="SINGLES",
        singles_tiebreak_player_id=2,
    )

    assert skinny["counts_for_rating"] is False
    assert singles["counts_for_rating"] is True
    assert singles["player_ids"] == [2]


def test_tiebreak_is_requested_only_after_two_two_regulation_result():
    tied = [
        {"game_code": code, "winner_side": "A" if index % 2 else "B"}
        for index, code in enumerate(("WOMENS", "MENS", "MIXED_1", "MIXED_2"))
    ]
    decided = [{**row, "winner_side": "A"} for row in tied]

    assert evaluate_team_match(
        tied,
        tiebreak_mode="SKINNY_RELAY",
    )["status"] == "TIEBREAK_REQUIRED"
    assert evaluate_team_match(
        decided,
        tiebreak_mode="SKINNY_RELAY",
    )["status"] == "FINAL"


@pytest.mark.parametrize("count", [4, 5])
def test_round_robin_pairs_each_team_once_and_handles_odd_byes(count):
    team_ids = [f"team-{index}" for index in range(1, count + 1)]
    matchups = build_team_round_robin_matchups(team_ids)
    observed = {
        frozenset((row["team_a_id"], row["team_b_id"])) for row in matchups
    }
    expected = {frozenset(pair) for pair in combinations(team_ids, 2)}

    assert observed == expected
    assert len(matchups) == count * (count - 1) // 2


def test_standings_use_head_to_head_then_game_differential():
    teams = [{"id": value, "name": value} for value in ("A", "B", "C")]
    matchups = [
        {
            "stage": "ROUND_ROBIN",
            "status": "FINAL",
            "team_a_id": "A",
            "team_b_id": "B",
            "winner_team_id": "A",
            "loser_team_id": "B",
            "team_a_game_wins": 3,
            "team_b_game_wins": 2,
        },
        {
            "stage": "ROUND_ROBIN",
            "status": "FINAL",
            "team_a_id": "B",
            "team_b_id": "C",
            "winner_team_id": "B",
            "loser_team_id": "C",
            "team_a_game_wins": 4,
            "team_b_game_wins": 0,
        },
        {
            "stage": "ROUND_ROBIN",
            "status": "FINAL",
            "team_a_id": "C",
            "team_b_id": "A",
            "winner_team_id": "C",
            "loser_team_id": "A",
            "team_a_game_wins": 3,
            "team_b_game_wins": 2,
        },
    ]

    standings = build_team_standings(teams, matchups)

    assert [row["team_id"] for row in standings] == ["B", "A", "C"]


def test_playoff_topology_and_podium_are_result_derived():
    standings = [
        {"rank": rank, "team_id": f"team-{rank}"} for rank in range(1, 5)
    ]
    bracket = build_team_playoff_matchups(
        standings,
        playoff_format="TOP_4_SEMIFINALS_WITH_BRONZE",
    )
    assert [row["playoff_game_code"] for row in bracket] == [
        "SF1",
        "SF2",
        "FINAL",
        "BRONZE",
    ]
    results = [
        {
            **row,
            "status": "FINAL",
            "winner_team_id": {
                "SF1": "team-1",
                "SF2": "team-2",
                "FINAL": "team-1",
                "BRONZE": "team-3",
            }[row["playoff_game_code"]],
            "loser_team_id": {
                "SF1": "team-4",
                "SF2": "team-3",
                "FINAL": "team-2",
                "BRONZE": "team-4",
            }[row["playoff_game_code"]],
        }
        for row in bracket
    ]

    assert calculate_team_podium(
        playoff_format="TOP_4_SEMIFINALS_WITH_BRONZE",
        standings=standings,
        playoff_matchups=results,
    ) == [
        {"placement": 1, "team_id": "team-1"},
        {"placement": 2, "team_id": "team-2"},
        {"placement": 3, "team_id": "team-3"},
    ]


def test_unfinished_final_refuses_podium():
    with pytest.raises(ValueError, match="final must be completed"):
        calculate_team_podium(
            playoff_format="TOP_2_FINAL",
            standings=[
                {"rank": 1, "team_id": "A"},
                {"rank": 2, "team_id": "B"},
                {"rank": 3, "team_id": "C"},
            ],
            playoff_matchups=[
                {"playoff_game_code": "FINAL", "status": "READY"}
            ],
        )
