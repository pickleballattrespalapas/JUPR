from jupr_app.ui.pages.tournament_registration import _resolve_existing_profile_for_next


def test_manual_search_selected_player_confirms_profile():
    active_players = [{"id": "p1", "display_name": "Alina Damian"}]

    resolved, selected_player_id, candidate_player_id, candidate_confirmed, rejected_likely = (
        _resolve_existing_profile_for_next(
            active_players,
            profile_mode="existing",
            selection_source="search",
            selected_player_id="",
            candidate_player_id="p1",
            candidate_confirmed=False,
            rejected_likely=True,
        )
    )

    assert resolved == active_players[0]
    assert selected_player_id == "p1"
    assert candidate_player_id == "p1"
    assert candidate_confirmed is True
    assert rejected_likely is False


def test_manual_search_missing_candidate_does_not_confirm():
    active_players = [{"id": "p1", "display_name": "Alina Damian"}]

    resolved, selected_player_id, candidate_player_id, candidate_confirmed, rejected_likely = (
        _resolve_existing_profile_for_next(
            active_players,
            profile_mode="existing",
            selection_source="search",
            selected_player_id="",
            candidate_player_id="missing",
            candidate_confirmed=False,
            rejected_likely=False,
        )
    )

    assert resolved is None
    assert selected_player_id == ""
    assert candidate_player_id == "missing"
    assert candidate_confirmed is False
    assert rejected_likely is False


def test_likely_match_flow_does_not_auto_confirm():
    active_players = [{"id": "p1", "display_name": "Alina Damian"}]

    resolved, selected_player_id, candidate_player_id, candidate_confirmed, rejected_likely = (
        _resolve_existing_profile_for_next(
            active_players,
            profile_mode="existing",
            selection_source="likely",
            selected_player_id="",
            candidate_player_id="p1",
            candidate_confirmed=False,
            rejected_likely=False,
        )
    )

    assert resolved is None
    assert selected_player_id == ""
    assert candidate_player_id == "p1"
    assert candidate_confirmed is False
    assert rejected_likely is False


def test_already_confirmed_existing_player_remains_confirmed():
    active_players = [{"id": "p1", "display_name": "Alina Damian"}]

    resolved, selected_player_id, candidate_player_id, candidate_confirmed, rejected_likely = (
        _resolve_existing_profile_for_next(
            active_players,
            profile_mode="existing",
            selection_source="likely",
            selected_player_id="p1",
            candidate_player_id="p1",
            candidate_confirmed=True,
            rejected_likely=False,
        )
    )

    assert resolved == active_players[0]
    assert selected_player_id == "p1"
    assert candidate_player_id == "p1"
    assert candidate_confirmed is True
    assert rejected_likely is False
