from jupr_app.domain import tournament_registration_repo as repo


def test_public_roster_includes_all_registered_statuses_and_needs_partner_only_list(monkeypatch):
    compiled_state = {
        "event_options": [
            {
                "id": "event-1",
                "event_family_label": "Men's Doubles",
                "division_name": "3.5",
            },
            {
                "id": "event-2",
                "event_family_label": "Mixed Doubles",
                "division_name": "4.0",
            },
        ],
        "event_rosters": [
            {
                "event_option_id": "event-1",
                "event_day_id": "day-3",
                "event_day_label": "Day 3",
                "event_label": "Men's Doubles 3.5",
                "entries": [
                    {
                        "status": "CONFIRMED",
                        "members": [{"display_name": "Confirmed Player"}],
                    },
                    {
                        "status": "WAITLIST",
                        "members": [{"display_name": "Waitlist Player"}],
                    },
                    {
                        "status": "NEEDS_PARTNER",
                        "show_on_partner_board": False,
                        "members": [
                            {
                                "registration_id": "reg-joe",
                                "selection_id": "sel-joe",
                                "player_id": 42,
                                "display_name": "Joe Baumann",
                                "skill": "3.5",
                                "age": 42,
                                "age_bracket": "40+",
                            }
                        ],
                        "notes": "Can play either side",
                    },
                    {
                        "status": "REVIEW",
                        "members": [{"display_name": "Review Player"}],
                    },
                    {
                        "status": "PARTNER_MISSING",
                        "members": [{"display_name": "Partner Missing Player"}],
                    },
                    {
                        "status": "CONFIRMED",
                        "members": [],
                    },
                ],
            },
            {
                "event_option_id": "event-2",
                "event_day_id": "day-4-5",
                "event_day_label": "Day 4/5",
                "event_label": "Mixed Doubles 4.0",
                "entries": [
                    {
                        "status": "NEEDS_PARTNER",
                        "members": [
                            {
                                "display_name": "Mixed Needs Partner",
                                "registration_id": "reg-mixed",
                                "selection_id": "sel-mixed",
                                "player_id": 88,
                            }
                        ],
                    },
                ],
            },
        ],
        "summary": {"total_registrations": 6, "waitlist_entries": 1},
    }

    monkeypatch.setattr(
        repo,
        "build_registration_state",
        lambda supabase, tournament, settings, days, event_options: compiled_state,
    )

    state = repo.build_public_tournament_roster_state(None, {"id": "t-1"}, {}, [], [])

    roster_rows = state["registrations_by_event"]
    roster_names = [
        member["display_name"]
        for row in roster_rows
        for member in row["members"]
    ]
    assert roster_names == [
        "Confirmed Player",
        "Waitlist Player",
        "Joe Baumann",
        "Review Player",
        "Partner Missing Player",
        "Mixed Needs Partner",
    ]
    assert [row["status"] for row in roster_rows] == [
        None,
        "Waitlist",
        "Needs Partner",
        None,
        None,
        "Needs Partner",
    ]
    assert "Pending Review" not in {row["status"] for row in roster_rows}

    needs_partner_row = next(
        row
        for row in roster_rows
        if row["members"][0]["display_name"] == "Joe Baumann"
    )
    assert needs_partner_row["event_day_id"] == "day-3"
    assert needs_partner_row["event_day_label"] == "Day 3"
    assert needs_partner_row["event_family"] == "Men's Doubles"
    assert needs_partner_row["division"] == "3.5"
    assert needs_partner_row["status"] == "Needs Partner"

    assert state["players_needing_partners"] == [
        {
            "player_name": "Joe Baumann",
            "selection_id": "sel-joe",
            "registration_id": "reg-joe",
            "player_id": 42,
            "event_option_id": "event-1",
            "event_day_label": "Day 3",
            "event_family": "Men's Doubles",
            "division": "3.5",
            "event_label": "Men's Doubles 3.5",
            "skill": "3.5",
            "age": 42,
            "age_bracket": "40+",
            "note": "Can play either side",
        },
        {
            "player_name": "Mixed Needs Partner",
            "selection_id": "sel-mixed",
            "registration_id": "reg-mixed",
            "player_id": 88,
            "event_option_id": "event-2",
            "event_day_label": "Day 4/5",
            "event_family": "Mixed Doubles",
            "division": "4.0",
            "event_label": "Mixed Doubles 4.0",
            "skill": None,
            "age": None,
            "age_bracket": None,
            "note": "",
        },
    ]
    assert state["summary"]["waitlist"] == 1
    assert [row["members"][0]["display_name"] for row in state["confirmed_teams"]] == ["Confirmed Player", "Waitlist Player"]
    assert state["pending_partner_requests"] == []
    assert [row["members"][0]["display_name"] for row in state["unresolved_partner_entries"]] == ["Review Player", "Partner Missing Player"]
