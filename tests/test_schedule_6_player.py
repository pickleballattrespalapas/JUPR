from jupr_app.domain.schedule import get_match_schedule


def test_6_player_schedule_has_9_matches_with_expected_key_rounds():
    players = [1, 2, 3, 4, 5, 6]

    matches = get_match_schedule("6-Player", players)

    assert len(matches) == 9
    assert matches[0] == {"t1": [1, 6], "t2": [2, 4], "desc": "Rnd 1"}
    assert matches[3] == {"t1": [3, 6], "t2": [1, 2], "desc": "Rnd 4"}
    assert matches[8] == {"t1": [2, 5], "t2": [1, 3], "desc": "Rnd 9"}
