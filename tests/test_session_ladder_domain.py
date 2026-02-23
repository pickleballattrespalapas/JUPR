from jupr_app.domain.session_ladder import (
    CourtGameResult,
    apply_adjacent_court_movement,
    build_session_resume_pointer,
    compute_court_stats,
    rank_players_with_tiebreak,
    round_robin_template,
    transition_session_state,
)


def test_round_robin_template_for_four_players_has_three_games():
    template = round_robin_template(4)
    assert len(template) == 3
    assert all(item.sit_out_player_index is None for item in template)


def test_round_robin_template_for_five_players_has_one_sit_out_per_game():
    template = round_robin_template(5)
    assert len(template) == 5
    assert [item.sit_out_player_index for item in template] == [0, 1, 2, 3, 4]


def test_compute_court_stats_and_ranking_uses_expected_tie_break_order():
    players = [10, 11, 12, 13]
    results = [
        CourtGameResult(team_a=(10, 11), team_b=(12, 13), score_a=21, score_b=19),
        CourtGameResult(team_a=(10, 12), team_b=(11, 13), score_a=15, score_b=21),
        CourtGameResult(team_a=(10, 13), team_b=(11, 12), score_a=21, score_b=14),
    ]

    stats = {row.player_id: row for row in compute_court_stats(players, results)}
    assert stats[10].wins == 2
    assert stats[11].wins == 2
    assert stats[10].point_differential > stats[11].point_differential

    ranked = rank_players_with_tiebreak(players, results)
    assert ranked[0] == 13
    assert ranked.index(10) < ranked.index(11)


def test_apply_adjacent_court_movement_keeps_moves_to_one_boundary():
    courts = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
    ]

    moved = apply_adjacent_court_movement(courts, movers_per_boundary=1)
    assert moved == [
        [1, 2, 3, 5],
        [4, 6, 7, 9],
        [8, 10, 11, 12],
    ]


def test_session_state_machine_and_resume_pointer():
    assert transition_session_state("draft", "start") == "active"
    assert transition_session_state("active", "complete") == "completed"

    pointer = build_session_resume_pointer("abc123", 2, "court-2")
    assert pointer["route"] == "/sessions/abc123/rounds/2/courts/court-2"
