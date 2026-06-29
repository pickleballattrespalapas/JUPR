from jupr_app.domain.tournament_registration_compiler import compile_tournament_registration_state


def test_needs_partner_registration_appears_on_public_partner_board_even_when_legacy_flags_are_false():
    state = compile_tournament_registration_state(
        tournament={"id": "t-1"},
        settings={},
        days=[{"id": "day-1", "label": "Day 1", "sort_order": 1}],
        event_options=[
            {
                "id": "wd-35",
                "registration_day_id": "day-1",
                "label": "Women's Doubles 3.5",
                "event_family_label": "Women's Doubles",
                "division_name": "3.5",
                "event_type": "GENDER_DOUBLES",
                "partner_required": True,
                "public_partner_board": False,
                "sort_order": 1,
            }
        ],
        registrations=[
            {
                "id": "reg-1",
                "display_name": "Ada Lovelace",
                "email": "ada@example.com",
                "submitted_at": "2026-06-01T10:00:00+00:00",
            }
        ],
        selections=[
            {
                "id": "sel-1",
                "registration_id": "reg-1",
                "registration_day_id": "day-1",
                "event_option_id": "wd-35",
                "partner_mode": "NEEDS_PARTNER",
                "show_on_partner_board": False,
                "partner_note": "Can play morning games",
            }
        ],
    )

    assert state["partner_board"] == [
        {
            "id": state["partner_board"][0]["id"],
            "tournament_id": "t-1",
            "event_day_id": "day-1",
            "event_day_label": "Day 1",
            "event_option_id": "wd-35",
            "event_label": "Women's Doubles 3.5",
            "selection_id": "sel-1",
            "registration_id": "reg-1",
            "player_id": None,
            "player": {
                "registration_id": "reg-1",
                "selection_id": "sel-1",
                "player_id": None,
                "display_name": "Ada Lovelace",
                "email": "ada@example.com",
                "phone": None,
                "dupr_id": None,
                "skill": None,
                "age": None,
                "gender": None,
                "age_bracket": None,
            },
            "note": "Can play morning games",
            "show_contact_email": True,
        }
    ]
