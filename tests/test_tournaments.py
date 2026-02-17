import pytest

from jupr_app.domain.tournaments import (
    build_playoff_games,
    build_round_robin_games,
    compute_podium_from_rr,
    compute_round_robin_standings,
    resolve_series_results,
    resolve_playoff_dependencies,
    validate_podium_placements,
)


def test_round_robin_schedule_4():
    team_ids = {1: "t1", 2: "t2", 3: "t3", 4: "t4"}
    games = build_round_robin_games(tournament_id="tour1", team_ids_by_number=team_ids)

    matchups = [
        (g["rr_round_number"], g["rr_slot_number"], g["team_a_id"], g["team_b_id"]) for g in games
    ]

    assert matchups == [
        (1, 1, "t2", "t1"),
        (1, 2, "t3", "t4"),
        (2, 1, "t4", "t2"),
        (2, 2, "t1", "t3"),
        (3, 1, "t4", "t1"),
        (3, 2, "t2", "t3"),
    ]


def test_round_robin_schedule_6_rr_template():
    team_ids = {1: "t1", 2: "t2", 3: "t3", 4: "t4", 5: "t5", 6: "t6"}
    games = build_round_robin_games(tournament_id="tour1", team_ids_by_number=team_ids)

    by_round = {}
    pairings = set()
    for game in games:
        by_round.setdefault(game["rr_round_number"], []).append(game)
        pairings.add(tuple(sorted((game["team_a_id"], game["team_b_id"]))))

    assert len(by_round) == 5
    for round_games in by_round.values():
        assert len(round_games) == 3
        teams_in_round = [g["team_a_id"] for g in round_games] + [g["team_b_id"] for g in round_games]
        assert len(set(teams_in_round)) == 6

    assert len(games) == 15
    assert len(pairings) == 15

    matchups = [
        (g["rr_round_number"], g["rr_slot_number"], g["team_a_id"], g["team_b_id"])
        for g in sorted(games, key=lambda x: (x["rr_round_number"], x["rr_slot_number"]))
    ]

    assert matchups == [
        (1, 1, "t2", "t1"),
        (1, 2, "t3", "t6"),
        (1, 3, "t4", "t5"),
        (2, 1, "t3", "t4"),
        (2, 2, "t6", "t1"),
        (2, 3, "t2", "t5"),
        (3, 1, "t6", "t4"),
        (3, 2, "t2", "t3"),
        (3, 3, "t1", "t5"),
        (4, 1, "t4", "t1"),
        (4, 2, "t5", "t3"),
        (4, 3, "t2", "t6"),
        (5, 1, "t5", "t6"),
        (5, 2, "t1", "t3"),
        (5, 3, "t2", "t4"),
    ]


def test_round_robin_standings_head_to_head():
    teams = [
        {"id": "t1", "team_number": 1},
        {"id": "t2", "team_number": 2},
        {"id": "t3", "team_number": 3},
        {"id": "t4", "team_number": 4},
    ]
    games = [
        {"team_a_id": "t1", "team_b_id": "t2", "score_a": 11, "score_b": 9},
        {"team_a_id": "t1", "team_b_id": "t3", "score_a": 9, "score_b": 11},
        {"team_a_id": "t1", "team_b_id": "t4", "score_a": 11, "score_b": 9},
        {"team_a_id": "t2", "team_b_id": "t3", "score_a": 11, "score_b": 9},
        {"team_a_id": "t2", "team_b_id": "t4", "score_a": 11, "score_b": 9},
    ]

    standings = compute_round_robin_standings(teams, games)
    seeds = {row["team_id"]: row["seed"] for row in standings}

    assert seeds["t1"] == 1
    assert seeds["t2"] == 2


def test_round_robin_podium_from_standings():
    teams = [
        {"id": "t1", "team_number": 1},
        {"id": "t2", "team_number": 2},
        {"id": "t3", "team_number": 3},
    ]
    games = [
        {"team_a_id": "t1", "team_b_id": "t2", "score_a": 11, "score_b": 7},
        {"team_a_id": "t1", "team_b_id": "t3", "score_a": 11, "score_b": 6},
        {"team_a_id": "t2", "team_b_id": "t3", "score_a": 11, "score_b": 4},
    ]

    podium = compute_podium_from_rr(teams, games)

    assert [row["team_id"] for row in podium] == ["t1", "t2", "t3"]


def test_validate_podium_rejects_duplicate_teams():
    placements = [
        {"placement": 1, "team_id": "t1"},
        {"placement": 2, "team_id": "t1"},
    ]

    with pytest.raises(ValueError):
        validate_podium_placements(placements)


def test_build_playoff_games_best_of_series():
    standings = [
        {"seed": 1, "team_id": "t1"},
        {"seed": 2, "team_id": "t2"},
        {"seed": 3, "team_id": "t3"},
        {"seed": 4, "team_id": "t4"},
    ]

    games = build_playoff_games(
        tournament_id="tour1",
        advance_count=4,
        standings=standings,
        best_of=3,
    )

    p1_games = [g for g in games if g["playoff_game_code"] == "P1"]
    p2_games = [g for g in games if g["playoff_game_code"] == "P2"]
    p3_games = [g for g in games if g["playoff_game_code"] == "P3"]
    p4_games = [g for g in games if g["playoff_game_code"] == "P4"]

    assert len(games) == 12
    assert [g["series_game_number"] for g in p1_games] == [1, 2, 3]
    assert [g["series_game_number"] for g in p2_games] == [1, 2, 3]
    assert [g["series_game_number"] for g in p3_games] == [1, 2, 3]
    assert [g["series_game_number"] for g in p4_games] == [1, 2, 3]
    assert {g["team_a_id"] for g in p1_games} == {"t1"}
    assert {g["team_b_id"] for g in p1_games} == {"t4"}


def test_resolve_series_results_best_of_three():
    games = [
        {"id": "g1", "stage": "PLAYOFF", "playoff_game_code": "P1", "winner_team_id": "t1"},
        {"id": "g2", "stage": "PLAYOFF", "playoff_game_code": "P1", "winner_team_id": "t1"},
        {"id": "g3", "stage": "PLAYOFF", "playoff_game_code": "P1", "winner_team_id": "t2"},
        {"id": "g4", "stage": "PLAYOFF", "playoff_game_code": "P2", "winner_team_id": "t3"},
        {"id": "g5", "stage": "PLAYOFF", "playoff_game_code": "P2", "winner_team_id": None},
        {"id": "g6", "stage": "ROUND_ROBIN", "playoff_game_code": "P3", "winner_team_id": "t9"},
    ]

    updates = resolve_series_results(games)

    assert updates == [
        {"id": "g1", "series_winner_team_id": "t1"},
        {"id": "g2", "series_winner_team_id": "t1"},
        {"id": "g4", "series_winner_team_id": "t3"},
    ]


def test_best_of_three_series_winner():
    games = [
        {
            "id": "g1",
            "stage": "PLAYOFF",
            "playoff_game_code": "P1",
            "winner_team_id": "t1",
        },
        {
            "id": "g2",
            "stage": "PLAYOFF",
            "playoff_game_code": "P1",
            "winner_team_id": "t1",
        },
        {
            "id": "g3",
            "stage": "PLAYOFF",
            "playoff_game_code": "P1",
            "winner_team_id": None,
        },
    ]

    updates = resolve_series_results(games)

    assert len(updates) > 0


def test_resolve_playoff_dependencies():
    games = [
        {
            "id": "g1",
            "playoff_game_code": "P1",
            "stage": "PLAYOFF",
            "team_a_id": "t1",
            "team_b_id": "t4",
            "team_a_source": {"seed": 1},
            "team_b_source": {"seed": 4},
            "score_a": 11,
            "score_b": 5,
            "winner_team_id": "t1",
            "loser_team_id": "t4",
            "finalized_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": "g2",
            "playoff_game_code": "P2",
            "stage": "PLAYOFF",
            "team_a_id": "t2",
            "team_b_id": "t3",
            "team_a_source": {"seed": 2},
            "team_b_source": {"seed": 3},
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "t2",
            "loser_team_id": "t3",
            "finalized_at": "2024-01-01T00:00:00Z",
        },
        {
            "id": "g3",
            "playoff_game_code": "P3",
            "stage": "PLAYOFF",
            "team_a_id": None,
            "team_b_id": None,
            "team_a_source": {"winnerOf": "P1"},
            "team_b_source": {"winnerOf": "P2"},
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
        },
        {
            "id": "g4",
            "playoff_game_code": "P4",
            "stage": "PLAYOFF",
            "team_a_id": None,
            "team_b_id": None,
            "team_a_source": {"loserOf": "P1"},
            "team_b_source": {"loserOf": "P2"},
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
        },
    ]

    updates = resolve_playoff_dependencies(games)
    updates_by_id = {u["id"]: u for u in updates}

    assert updates_by_id["g3"]["team_a_id"] == "t1"
    assert updates_by_id["g3"]["team_b_id"] == "t2"
    assert updates_by_id["g4"]["team_a_id"] == "t4"
    assert updates_by_id["g4"]["team_b_id"] == "t3"

    games[2]["team_a_id"] = "t1"
    games[2]["team_b_id"] = "t2"
    games[2]["score_a"] = 11
    games[2]["score_b"] = 9
    games[2]["winner_team_id"] = "t1"
    games[2]["loser_team_id"] = "t2"
    games[2]["finalized_at"] = "2024-01-01T00:00:00Z"

    games[0]["winner_team_id"] = "t4"
    games[0]["loser_team_id"] = "t1"

    updates = resolve_playoff_dependencies(games)
    updates_by_id = {u["id"]: u for u in updates}

    assert updates_by_id["g3"]["team_a_id"] == "t4"
    assert updates_by_id["g3"]["score_a"] is None
    assert updates_by_id["g3"]["finalized_at"] is None
