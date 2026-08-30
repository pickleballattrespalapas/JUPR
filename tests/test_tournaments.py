import pytest

from jupr_app.domain.tournaments import (
    build_playoff_games,
    build_round_robin_games,
    compute_podium_from_rr,
    compute_round_robin_standings,
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


def test_retired_team_cannot_advance_to_playoffs():
    standings = [
        {"team_id": "t1", "seed": 1, "competition_status": "ACTIVE"},
        {"team_id": "t2", "seed": 2, "competition_status": "ACTIVE"},
        {"team_id": "t3", "seed": 3, "competition_status": "ACTIVE"},
        {
            "team_id": "t4",
            "seed": 4,
            "competition_status": "RETIRED",
            "retired": True,
        },
    ]

    with pytest.raises(ValueError, match="requires 4 active teams"):
        build_playoff_games(
            tournament_id="tour1",
            advance_count=4,
            standings=standings,
        )


def test_validate_podium_rejects_duplicate_teams():
    placements = [
        {"placement": 1, "team_id": "t1"},
        {"placement": 2, "team_id": "t1"},
    ]

    with pytest.raises(ValueError):
        validate_podium_placements(placements)


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


def test_round_robin_schedule_6_from_csv():
    team_ids = {idx: f"t{idx}" for idx in range(1, 7)}
    games = build_round_robin_games(tournament_id="tour6", team_ids_by_number=team_ids)

    assert len(games) == 15
    assert games[0]["rr_round_number"] == 1
    assert games[0]["team_a_id"] == "t1"
    assert games[0]["team_b_id"] == "t6"
    assert games[-1]["rr_round_number"] == 5
    assert games[-1]["team_a_id"] == "t4"
    assert games[-1]["team_b_id"] == "t5"
