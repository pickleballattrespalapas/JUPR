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

from jupr_app.ui.pages.tournament_registration import _can_advance_profile_step, _confirm_suggested_profile_state


def test_confirm_suggested_profile_state_sets_confirmed_and_selected_player():
    state = _confirm_suggested_profile_state({}, player_id="p1", selection_source="likely", search_query="Ada")
    assert state["candidate_confirmed"] is True
    assert state["selected_player_id"] == "p1"
    assert state["candidate_player_id"] == "p1"


def test_confirmed_suggested_profile_can_advance_without_second_confirmation():
    assert _can_advance_profile_step(profile_mode="existing", selection_source="likely", candidate_player_id="p1", candidate_confirmed=True)


def test_manual_search_selected_profile_can_advance_for_auto_confirm_on_next():
    assert _can_advance_profile_step(profile_mode="existing", selection_source="search", candidate_player_id="p1", candidate_confirmed=False)


def test_unconfirmed_likely_profile_cannot_advance_to_generic_next_error():
    assert not _can_advance_profile_step(profile_mode="existing", selection_source="likely", candidate_player_id="p1", candidate_confirmed=False)
