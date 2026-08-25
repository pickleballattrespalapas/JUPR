from __future__ import annotations

from jupr_app.domain.tournaments import (
    compute_round_robin_standings,
    resolve_playoff_dependencies,
)
from jupr_app.services.admin_tournament_match_publish_service import (
    _build_official_match_payloads,
)


def test_synthetic_non_played_result_advances_standings_and_bracket() -> None:
    teams = [
        {"id": "t1", "team_number": 1},
        {"id": "t2", "team_number": 2},
    ]
    standings = compute_round_robin_standings(
        teams,
        [
            {
                "id": "rr-1",
                "team_a_id": "t1",
                "team_b_id": "t2",
                "score_a": 11,
                "score_b": 0,
                "winner_team_id": "t1",
                "result_type": "NO_SHOW",
            }
        ],
    )
    assert standings[0]["team_id"] == "t1"
    assert standings[0]["wins"] == 1

    games = [
        {
            "id": "p1",
            "stage": "PLAYOFF",
            "playoff_game_code": "P1",
            "team_a_id": "t1",
            "team_b_id": "t2",
            "team_a_source": {"seed": 1},
            "team_b_source": {"seed": 2},
            "score_a": 11,
            "score_b": 0,
            "winner_team_id": "t1",
            "loser_team_id": "t2",
            "finalized_at": "2026-08-25T12:00:00Z",
            "result_type": "FORFEIT",
        },
        {
            "id": "p2",
            "stage": "PLAYOFF",
            "playoff_game_code": "P2",
            "team_a_id": None,
            "team_b_id": "t3",
            "team_a_source": {"winnerOf": "P1"},
            "team_b_source": {"seed": 3},
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
        },
    ]
    updates = resolve_playoff_dependencies(games)
    assert len(updates) == 1
    assert updates[0]["id"] == "p2"
    assert updates[0]["team_a_id"] == "t1"


def test_non_played_results_are_excluded_from_official_rating_payloads() -> None:
    tournament = {
        "id": "tour-1",
        "name": "Summer Classic",
        "start_date": "2026-08-25",
    }
    draw = {"id": "draw-1", "name": "Women's 3.5"}
    event = {"id": "event-1", "division_name": "Women's 3.5"}
    teams = [
        {"id": "t1", "player1_id": 1, "player2_id": 2},
        {"id": "t2", "player1_id": 3, "player2_id": 4},
    ]
    games = [
        {
            "id": "played-1",
            "team_a_id": "t1",
            "team_b_id": "t2",
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "t1",
            "finalized_at": "2026-08-25T12:00:00Z",
            "result_type": "PLAYED",
            "stage": "ROUND_ROBIN",
        },
        {
            "id": "no-show-1",
            "team_a_id": "t1",
            "team_b_id": "t2",
            "score_a": 11,
            "score_b": 0,
            "winner_team_id": "t1",
            "finalized_at": "2026-08-25T12:30:00Z",
            "result_type": "NO_SHOW",
            "result_note": "Team 2 did not report.",
            "stage": "ROUND_ROBIN",
        },
    ]

    payloads = _build_official_match_payloads(
        tournament=tournament,
        draw=draw,
        event_option=event,
        teams=teams,
        games=games,
    )

    assert [row["tournament_game_id"] for row in payloads] == ["played-1"]
    assert all(row["tournament_game_id"] != "no-show-1" for row in payloads)
