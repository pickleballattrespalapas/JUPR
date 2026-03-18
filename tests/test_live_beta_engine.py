from jupr_app.domain.live_beta_engine import (
    build_league_movement,
    create_league_event,
    create_round_robin_event,
    create_tournament_event,
    league_aggregate_standings,
    mark_tournament_matches_saved,
    match_payloads_from_current_league_round,
    match_payloads_from_rr,
    resolve_payload_player_ids,
    round_robin_standings,
    set_pending_assignment,
    start_next_league_round,
    tournament_bracket_rows,
    tournament_champion,
    tournament_completed_match_payloads,
    update_tournament_score,
    update_league_score,
    update_round_robin_score,
)


def test_round_robin_standings_rank_scored_matches_only():
    event = create_round_robin_event(
        name="RR",
        participant_names=["Amy", "Brooke", "Chris", "Dana"],
    )

    update_round_robin_score(event, "rr-r1-m1", 11, 7)
    update_round_robin_score(event, "rr-r2-m1", 11, 8)
    standings = round_robin_standings(event)

    assert standings[0]["name"] == "Brooke"
    assert standings[0]["wins"] == 2
    assert standings[0]["rank"] == 1
    assert standings[-1]["matches"] == 2


def test_round_robin_payloads_require_resolved_player_ids():
    event = create_round_robin_event(
        name="Official RR",
        participant_names=["Amy", "Brooke", "Chris", "Dana"],
        resolved_ids={"Amy": 1, "Brooke": 2, "Chris": 3, "Dana": 4},
    )
    update_round_robin_score(event, "rr-r1-m1", 11, 7)

    payloads = match_payloads_from_rr(event)
    resolved = resolve_payload_player_ids(event, payloads)

    assert resolved == [
        {
            "round_number": 1,
            "match_id": "rr-r1-m1",
            "t1_p1": 2,
            "t1_p2": 1,
            "t2_p1": 3,
            "t2_p2": 4,
            "s1": 11,
            "s2": 7,
        }
    ]


def test_league_finalize_generates_next_round_using_pending_assignments():
    event = create_league_event(
        name="League",
        participant_names=["Amy", "Brooke", "Chris", "Dana", "Eli", "Finn", "Gia", "Hugo", "Ivy"],
        total_rounds=3,
    )
    round_one = event["rounds"][0]
    for court in round_one["courts"]:
        for mini_round in court["miniRounds"]:
            for match in mini_round["matches"]:
                update_league_score(event, match["id"], 11, 7)

    movement = build_league_movement(event)
    promoted_player = next(row["participantId"] for row in movement["rows"] if row["currentCourt"] == 2 and row["currentRank"] == 1)
    set_pending_assignment(event, promoted_player, 1)

    start_next_league_round(event)

    assert event["currentRoundNumber"] == 2
    assert len(event["rounds"]) == 2
    court_one_ids = event["rounds"][1]["courts"][0]["participantIds"]
    assert promoted_player in court_one_ids


def test_league_current_round_payloads_and_aggregate_standings():
    event = create_league_event(
        name="League",
        participant_names=["Amy", "Brooke", "Chris", "Dana", "Eli", "Finn", "Gia", "Hugo", "Ivy"],
        total_rounds=2,
        resolved_ids={
            "Amy": 1,
            "Brooke": 2,
            "Chris": 3,
            "Dana": 4,
            "Eli": 5,
            "Finn": 6,
            "Gia": 7,
            "Hugo": 8,
            "Ivy": 9,
        },
    )
    first_match = event["rounds"][0]["courts"][0]["miniRounds"][0]["matches"][0]
    update_league_score(event, first_match["id"], 11, 4)

    payloads = match_payloads_from_current_league_round(event)
    resolved = resolve_payload_player_ids(event, payloads)
    standings = league_aggregate_standings(event)

    assert len(payloads) == 1
    assert resolved[0]["s1"] == 11
    assert standings[0]["wins"] == 1


def test_tournament_byes_advance_and_champion_resolves():
    event = create_tournament_event(
        name="Club Championship",
        team_entries=[
            {"name": "Alpha", "player1_name": "Amy", "player2_name": "Brooke"},
            {"name": "Bravo", "player1_name": "Chris", "player2_name": "Dana"},
            {"name": "Charlie", "player1_name": "Eli", "player2_name": "Finn"},
            {"name": "Delta", "player1_name": "Gia", "player2_name": "Hugo"},
            {"name": "Echo", "player1_name": "Ivy", "player2_name": "Jules"},
        ],
    )

    bracket_rows = tournament_bracket_rows(event)
    assert bracket_rows[0]["team_a"] == "Alpha"
    assert bracket_rows[0]["team_b"] == "TBD"
    assert bracket_rows[-1]["winner"] == "Pending"

    update_tournament_score(event, 1, 2, 11, 7)
    update_tournament_score(event, 2, 1, 11, 5)
    update_tournament_score(event, 2, 2, 9, 11)
    update_tournament_score(event, 3, 1, 11, 4)

    assert tournament_champion(event) == "team-1"
    assert tournament_bracket_rows(event)[-1]["winner"] == "Alpha"


def test_tournament_payloads_only_return_unsaved_completed_matches():
    event = create_tournament_event(
        name="Official Tournament",
        official_context={"tournament_id": "t-123"},
        team_entries=[
            {"name": "Alpha", "player1_name": "Amy", "player2_name": "Brooke", "player1_id": 1, "player2_id": 2},
            {"name": "Bravo", "player1_name": "Chris", "player2_name": "Dana", "player1_id": 3, "player2_id": 4},
            {"name": "Charlie", "player1_name": "Eli", "player2_name": "Finn", "player1_id": 5, "player2_id": 6},
            {"name": "Delta", "player1_name": "Gia", "player2_name": "Hugo", "player1_id": 7, "player2_id": 8},
        ],
    )

    update_tournament_score(event, 1, 1, 11, 8)
    payloads = tournament_completed_match_payloads(event)

    assert payloads == [
        {
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 7,
            "t2_p2": 8,
            "s1": 11,
            "s2": 8,
            "score_t1": 11,
            "score_t2": 8,
            "date": payloads[0]["date"],
            "league": "Official Tournament",
            "match_type": "Tournament",
            "week_tag": "Tournament",
            "is_popup": False,
            "context_type": "TOURNAMENT",
            "context_id": "t-123",
            "tournament_id": "t-123",
            "tournament_game_id": "r1-s1",
        }
    ]

    mark_tournament_matches_saved(event, payloads)
    assert tournament_completed_match_payloads(event, unsaved_only=True) == []
