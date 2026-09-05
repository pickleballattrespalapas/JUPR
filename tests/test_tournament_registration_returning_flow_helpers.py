import math

from jupr_app.ui.pages.tournament_registration import (
    _hydrate_registration_wizard_from_bundle,
    _mask_email,
    _partner_details_from_selections,
    _player_current_overall_jupr,
    _player_current_singles_jupr,
    _player_label,
    _selected_event_ids_from_selections,
)


def _bundle(selections):
    return {
        "registration": {
            "id": "reg_1",
            "first_name": "Ada",
            "last_name": "Lovelace",
            "display_name": "Ada Lovelace",
            "email": "ada@example.com",
            "phone": "555",
            "gender": "Female",
            "age": 35,
            "notes": "note",
            "dupr_id": "D123",
            "doubles_skill": 3.5,
            "singles_skill": 3.0,
        },
        "selections": selections,
    }


def test_mask_email():
    assert _mask_email("alina@example.com") == "a***a@example.com"


def test_missing_pandas_singles_rating_does_not_become_nan_skill():
    assert _player_current_singles_jupr(
        {"singles_rating": math.nan, "singles_matches_played": 1}
    ) is None


def test_default_singles_rating_without_singles_history_is_missing():
    assert _player_current_singles_jupr(
        {"singles_rating": 1200, "singles_matches_played": 0}
    ) is None


def test_overall_rating_does_not_fall_back_to_singles_skill():
    assert _player_current_overall_jupr(
        {"rating": None, "doubles_skill": None, "singles_skill": 4.0}
    ) is None


def test_official_singles_skill_requires_positive_match_history():
    assert _player_current_singles_jupr(
        {"singles_rating": 1400, "singles_matches_played": 1}
    ) == 3.5
    assert _player_current_singles_jupr({"singles_skill": 3.25}) is None


def test_player_label_shows_independent_doubles_and_singles_ratings():
    assert _player_label(
        {
            "name": "Jay Senior",
            "rating": 1600,
            "singles_rating": 1600,
            "singles_matches_played": 0,
        }
    ) == "Jay Senior · Doubles 4.000 · Singles not set"


def test_hydrate_wizard_from_one_event():
    wizard = _hydrate_registration_wizard_from_bundle({}, _bundle([{"event_option_id": "e1", "partner_mode": "NONE"}]))
    assert wizard["edit_mode"] is True
    assert wizard["email_locked"] is True
    assert wizard["edit_registration_id"] == "reg_1"
    assert wizard["step1"]["email"] == "ada@example.com"
    assert wizard["step3"]["selected_event_ids"] == ["e1"]


def test_hydrate_wizard_from_two_events():
    wizard = _hydrate_registration_wizard_from_bundle({}, _bundle([{"event_option_id": "e1"}, {"event_option_id": "e2"}]))
    assert wizard["step3"]["selected_event_ids"] == ["e1", "e2"]


def test_hydrate_edit_preserves_locked_player_for_missing_singles_entry():
    bundle = _bundle([])
    bundle["registration"]["player_id"] = 42

    wizard = _hydrate_registration_wizard_from_bundle({}, bundle)

    assert wizard["step2"]["profile_mode"] == "existing"
    assert wizard["step2"]["selected_player_id"] == "42"
    assert wizard["step2"]["candidate_player_id"] == "42"
    assert wizard["step2"]["candidate_confirmed"] is True
    assert wizard["step2"]["selection_source"] == "edit_link"


def test_partner_details_preserve_has_partner():
    details = _partner_details_from_selections([{"event_option_id": "e1", "partner_mode": "HAS_PARTNER", "partner_name": "Grace", "partner_email": "g@example.com", "partner_phone": "1", "partner_dupr_id": "D", "partner_skill": 4.0, "partner_age": 40, "show_on_partner_board": True, "partner_note": "ok"}])
    assert details["e1"]["partner_mode"] == "HAS_PARTNER"
    assert details["e1"]["partner_name"] == "Grace"
    assert details["e1"]["show_on_partner_board"] is True


def test_partner_details_preserve_needs_partner():
    details = _partner_details_from_selections([{"event_option_id": "e1", "partner_mode": "NEEDS_PARTNER", "show_on_partner_board": True}])
    assert details["e1"]["partner_mode"] == "NEEDS_PARTNER"


def test_selected_event_ids_filters_blank():
    assert _selected_event_ids_from_selections([{"event_option_id": "e1"}, {"event_option_id": ""}]) == ["e1"]


def test_hydrate_edit_registration_fresh_sets_current_step_1():
    wizard = _hydrate_registration_wizard_from_bundle({"current_step": 3}, _bundle([]))
    assert wizard["current_step"] == 1


def test_hydrate_same_edit_registration_preserves_progress():
    wizard = _hydrate_registration_wizard_from_bundle({}, _bundle([{"event_option_id": "e1"}]))
    wizard["current_step"] = 3
    wizard["step3"] = {"selected_event_ids": ["edited"]}
    wizard = _hydrate_registration_wizard_from_bundle(wizard, _bundle([{"event_option_id": "e1"}]), preserve_existing_progress=True)
    assert wizard["current_step"] == 3
    assert wizard["step3"]["selected_event_ids"] == ["edited"]


def test_hydrate_different_edit_registration_resets_progress():
    wizard = _hydrate_registration_wizard_from_bundle({}, _bundle([]))
    wizard["current_step"] = 3
    other = _bundle([])
    other["registration"]["id"] = "reg_2"
    wizard = _hydrate_registration_wizard_from_bundle(wizard, other, preserve_existing_progress=True)
    assert wizard["current_step"] == 1
    assert wizard["edit_registration_id"] == "reg_2"


def test_step1_edit_mode_next_uses_locked_wizard_email_not_disabled_widget_email():
    from jupr_app.ui.pages.tournament_registration import _advance_step1_registration_wizard

    wizard = {"edit_mode": True, "step1": {"email": "locked@example.com"}}
    advanced, error = _advance_step1_registration_wizard(
        wizard,
        tournament_id="t1",
        first_name="Ada",
        last_name="Lovelace",
        email_for_submit=wizard["step1"]["email"],
        phone="",
        gender="Female",
        age="35",
        notes="",
        find_existing_registration=lambda *_: {"id": "duplicate"},
    )
    assert advanced is True
    assert error == ""
    assert wizard["step1"]["email"] == "locked@example.com"


def test_step1_edit_mode_next_target_is_step3():
    from jupr_app.ui.pages.tournament_registration import _advance_step1_registration_wizard

    wizard = {"edit_mode": True, "step1": {"email": "locked@example.com"}}
    _advance_step1_registration_wizard(wizard, tournament_id="t1", first_name="Ada", last_name="Lovelace", email_for_submit="locked@example.com", phone="", gender="Female", age="35", notes="", find_existing_registration=None)
    assert wizard["current_step"] == 3


def test_step1_non_edit_mode_next_target_remains_step2():
    from jupr_app.ui.pages.tournament_registration import _advance_step1_registration_wizard

    wizard = {"edit_mode": False}
    _advance_step1_registration_wizard(wizard, tournament_id="t1", first_name="Ada", last_name="Lovelace", email_for_submit="ada@example.com", phone="", gender="Female", age="35", notes="", find_existing_registration=lambda *_: None)
    assert wizard["current_step"] == 2


def test_duplicate_detection_skipped_in_edit_mode():
    from jupr_app.ui.pages.tournament_registration import _advance_step1_registration_wizard

    calls = []
    wizard = {"edit_mode": True}
    _advance_step1_registration_wizard(wizard, tournament_id="t1", first_name="Ada", last_name="Lovelace", email_for_submit="ada@example.com", phone="", gender="Female", age="35", notes="", find_existing_registration=lambda *args: calls.append(args))
    assert calls == []
    assert wizard["current_step"] == 3


def test_duplicate_detection_runs_in_non_edit_mode():
    from jupr_app.ui.pages.tournament_registration import _advance_step1_registration_wizard

    calls = []
    wizard = {"edit_mode": False}
    _advance_step1_registration_wizard(wizard, tournament_id="t1", first_name="Ada", last_name="Lovelace", email_for_submit="ada@example.com", phone="", gender="Female", age="35", notes="", find_existing_registration=lambda *args: calls.append(args) or {"id": "reg_1"})
    assert calls == [("t1", "ada@example.com")]
    assert wizard["current_step"] == 0


def test_missing_first_last_with_display_name_split_best_effort():
    bundle = _bundle([])
    bundle["registration"]["first_name"] = ""
    bundle["registration"]["last_name"] = ""
    bundle["registration"]["display_name"] = "Ada King Lovelace"
    wizard = _hydrate_registration_wizard_from_bundle({}, bundle)
    assert wizard["step1"]["first_name"] == "Ada"
    assert wizard["step1"]["last_name"] == "King Lovelace"


def test_tournament_registration_page_imports_cleanly():
    import jupr_app.ui.pages.tournament_registration as page

    assert page is not None


def test_no_private_get_submission_result_reference():
    from pathlib import Path

    assert "_get_submission_result" not in Path("jupr_app/ui/pages/tournament_registration.py").read_text()


def test_registration_does_not_embed_partner_board_discovery():
    from pathlib import Path

    source = Path("jupr_app/ui/pages/tournament_registration.py").read_text(
        encoding="utf-8"
    )

    assert "_partner_board_targets_for_event" not in source
    assert "Players looking for partners in this division" not in source
    assert "wizard_partner_board_request" not in source


def test_partner_request_ready_requires_durable_target_not_legacy_text():
    from jupr_app.ui.pages.tournament_registration import _partner_request_ready

    assert _partner_request_ready({"partner_mode": "REQUEST_PARTNER", "partner_name": "Elizabeth whalen", "partner_email": "e@example.com"}) is False
    assert _partner_request_ready({"partner_mode": "REQUEST_PARTNER", "target_selection_id": "sel_elizabeth"}) is True
    assert _partner_request_ready({"partner_mode": "REQUEST_PARTNER", "target_player_id": 22}) is True


def test_registered_partner_target_for_player_reports_status_and_ids():
    from jupr_app.ui.pages.tournament_registration import _registered_partner_target_for_player

    state = {
        "event_rosters": [
            {"event_option_id": "event_1", "entries": [{"status": "NEEDS_PARTNER", "members": [{"player_id": 22, "selection_id": "sel_22", "registration_id": "reg_22"}]}]}
        ]
    }

    assert _registered_partner_target_for_player(state, "event_1", 22) == {
        "target_selection_id": "sel_22",
        "target_registration_id": "reg_22",
        "target_player_id": "22",
        "status": "NEEDS_PARTNER",
    }


def test_legacy_partner_reconciliation_suggests_same_event_match():
    from jupr_app.ui.pages.tournament_registration import (
        _legacy_partner_reconciliation_issues,
        _legacy_partner_suggestions,
    )

    mary = {
        "registration_id": "reg_mary",
        "selection_id": "sel_mary",
        "registration": {"display_name": "Mary Bauman", "email": "mary@example.com"},
        "selection": {"id": "sel_mary", "registration_id": "reg_mary", "event_option_id": "wd35", "partner_mode": "HAS_PARTNER", "partner_name": "Elizabeth whalen"},
    }
    elizabeth = {
        "registration_id": "reg_elizabeth",
        "selection_id": "sel_elizabeth",
        "registration": {"display_name": "Elizabeth Whelan", "email": "elizabeth@example.com"},
        "selection": {"id": "sel_elizabeth", "registration_id": "reg_elizabeth", "event_option_id": "wd35", "partner_mode": "NEEDS_PARTNER"},
    }

    issues = _legacy_partner_reconciliation_issues([mary, elizabeth], [], [])
    suggestions = _legacy_partner_suggestions(issues[0], [mary, elizabeth])

    assert issues == [mary]
    assert suggestions[0]["row"] == elizabeth
    assert suggestions[0]["scope"] == "same_event"


def test_legacy_partner_reconciliation_excludes_confirmed_and_pending_rows():
    from jupr_app.ui.pages.tournament_registration import _legacy_partner_reconciliation_issues

    row = {
        "selection_id": "sel_mary",
        "selection": {"id": "sel_mary", "partner_mode": "HAS_PARTNER", "partner_name": "Elizabeth whalen"},
        "registration": {"display_name": "Mary Bauman"},
    }

    assert _legacy_partner_reconciliation_issues([row], [], [{"selection_id": "sel_mary", "status": "ACTIVE"}]) == []
    assert _legacy_partner_reconciliation_issues([row], [{"requester_selection_id": "sel_mary", "status": "PENDING"}], []) == []
    assert _legacy_partner_reconciliation_issues([row], [], []) == [row]
