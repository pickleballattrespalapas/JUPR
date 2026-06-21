from jupr_app.ui.pages.tournament_registration import (
    _hydrate_registration_wizard_from_bundle,
    _mask_email,
    _partner_details_from_selections,
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
