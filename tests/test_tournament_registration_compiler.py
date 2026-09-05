from jupr_app.domain.tournament_registration_compiler import compile_tournament_registration_state


def _state(*, selections):
    return compile_tournament_registration_state(
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
                "public_partner_board": True,
                "sort_order": 1,
            }
        ],
        registrations=[
            {
                "id": "reg-elizabeth",
                "display_name": "Elizabeth Whelan",
                "email": "elizabeth@example.com",
                "submitted_at": "2026-06-01T10:00:00+00:00",
            },
            {
                "id": "reg-mary",
                "display_name": "Mary Bauman",
                "email": "mary@example.com",
                "submitted_at": "2026-06-01T10:05:00+00:00",
            },
        ],
        selections=selections,
    )


def _linked_state(
    *,
    selections,
    partner_requests=None,
    partner_links=None,
    team_members=None,
    registrations=None,
):
    return compile_tournament_registration_state(
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
                "public_partner_board": True,
                "sort_order": 1,
            },
            {
                "id": "singles-35",
                "registration_day_id": "day-1",
                "label": "Singles 3.5",
                "event_family_label": "Singles",
                "division_name": "3.5",
                "event_type": "SINGLES",
                "partner_required": False,
                "sort_order": 2,
            },
        ],
        registrations=registrations
        or [
            {"id": "reg-mary", "display_name": "Mary Bauman", "email": "mary@example.com", "player_id": 101, "doubles_skill": 4.0, "singles_skill": 3.5, "submitted_at": "2026-06-01T10:05:00+00:00"},
            {"id": "reg-elizabeth", "display_name": "Elizabeth Whelan", "email": "elizabeth@example.com", "player_id": 102, "submitted_at": "2026-06-01T10:00:00+00:00"},
        ],
        selections=selections,
        partner_requests=partner_requests or [],
        partner_links=partner_links or [],
        team_members=team_members or [],
    )


def _entries(state):
    return state["event_rosters"][0]["entries"]


def test_free_text_partner_does_not_create_pseudo_team_when_needs_partner_exists():
    state = _state(
        selections=[
            {
                "id": "sel-elizabeth",
                "registration_id": "reg-elizabeth",
                "registration_day_id": "day-1",
                "event_option_id": "wd-35",
                "partner_mode": "NEEDS_PARTNER",
                "show_on_partner_board": True,
            },
            {
                "id": "sel-mary",
                "registration_id": "reg-mary",
                "registration_day_id": "day-1",
                "event_option_id": "wd-35",
                "partner_mode": "HAS_PARTNER",
                "partner_name": "Elizabeth whalen",
            },
        ],
    )

    entries = _entries(state)
    assert [entry["status"] for entry in entries] == ["NEEDS_PARTNER", "LEGACY_PARTNER_UNRESOLVED"]
    assert not any(entry["status"] == "CONFIRMED" for entry in entries)

    elizabeth_entry = entries[0]
    assert [member["display_name"] for member in elizabeth_entry["members"]] == ["Elizabeth Whelan"]

    mary_entry = entries[1]
    assert [member["display_name"] for member in mary_entry["members"]] == ["Mary Bauman"]
    assert mary_entry["legacy_partner"]["partner_name"] == "Elizabeth whalen"
    assert all(
        "Elizabeth whalen" not in [member.get("display_name") for member in entry["members"]]
        for entry in entries
    )


def test_needs_partner_registration_is_public_even_without_opt_in_flag():
    state = _state(
        selections=[
            {
                "id": "sel-elizabeth",
                "registration_id": "reg-elizabeth",
                "registration_day_id": "day-1",
                "event_option_id": "wd-35",
                "partner_mode": "NEEDS_PARTNER",
                "show_on_partner_board": False,
                "partner_note": "Available Saturday",
            }
        ],
    )

    assert len(state["partner_board"]) == 1
    assert state["partner_board"][0]["selection_id"] == "sel-elizabeth"
    assert state["partner_board"][0]["player"]["display_name"] == "Elizabeth Whelan"
    assert state["partner_board"][0]["note"] == "Available Saturday"


def test_mutual_free_text_partner_data_cannot_create_confirmed_team():
    state = _state(
        selections=[
            {
                "id": "sel-elizabeth",
                "registration_id": "reg-elizabeth",
                "registration_day_id": "day-1",
                "event_option_id": "wd-35",
                "partner_mode": "HAS_PARTNER",
                "partner_name": "Mary Bauman",
            },
            {
                "id": "sel-mary",
                "registration_id": "reg-mary",
                "registration_day_id": "day-1",
                "event_option_id": "wd-35",
                "partner_mode": "HAS_PARTNER",
                "partner_name": "Elizabeth Whelan",
            },
        ],
    )

    entries = _entries(state)
    assert [entry["status"] for entry in entries] == [
        "LEGACY_PARTNER_UNRESOLVED",
        "LEGACY_PARTNER_UNRESOLVED",
    ]
    assert not any(entry["status"] == "CONFIRMED" for entry in entries)
    assert all(len(entry["members"]) == 1 for entry in entries)
    assert state["summary"]["confirmed_entries"] == 0
    assert state["summary"]["legacy_partner_unresolved_entries"] == 2


def test_pending_request_appears_without_confirmed_team_and_no_pseudo_member():
    state = _linked_state(
        selections=[
            {"id": "sel-mary", "registration_id": "reg-mary", "registration_day_id": "day-1", "event_option_id": "wd-35", "partner_mode": "HAS_PARTNER", "partner_name": "Elizabeth whalen"},
            {"id": "sel-elizabeth", "registration_id": "reg-elizabeth", "registration_day_id": "day-1", "event_option_id": "wd-35", "partner_mode": "NEEDS_PARTNER", "show_on_partner_board": True},
        ],
        partner_requests=[
            {
                "id": "req-1",
                "tournament_id": "t-1",
                "event_option_id": "wd-35",
                "requester_selection_id": "sel-mary",
                "target_selection_id": "sel-elizabeth",
                "status": "PENDING",
                "created_at": "2026-06-01T10:06:00+00:00",
            }
        ],
    )

    entries = _entries(state)
    assert not any(entry["status"] == "CONFIRMED" for entry in entries)
    pending = [entry for entry in entries if entry["status"] == "PENDING_PARTNER_REQUEST"]
    assert len(pending) == 1
    assert [member["display_name"] for member in pending[0]["members"]] == ["Mary Bauman", "Elizabeth Whelan"]
    assert not any(entry["status"] == "LEGACY_PARTNER_UNRESOLVED" for entry in entries)
    assert any(entry["status"] == "NEEDS_PARTNER" for entry in entries)


def test_accepted_link_creates_single_confirmed_team_and_suppresses_needs_partner():
    state = _linked_state(
        selections=[
            {"id": "sel-mary", "registration_id": "reg-mary", "registration_day_id": "day-1", "event_option_id": "wd-35", "partner_mode": "HAS_PARTNER", "partner_name": "Elizabeth whalen"},
            {"id": "sel-elizabeth", "registration_id": "reg-elizabeth", "registration_day_id": "day-1", "event_option_id": "wd-35", "partner_mode": "NEEDS_PARTNER", "show_on_partner_board": True},
            {"id": "sel-mary-singles", "registration_id": "reg-mary", "registration_day_id": "day-1", "event_option_id": "singles-35", "partner_mode": "NONE"},
        ],
        partner_links=[
            {
                "id": "link-1",
                "tournament_id": "t-1",
                "event_option_id": "wd-35",
                "selection1_id": "sel-mary",
                "selection2_id": "sel-elizabeth",
                "status": "CONFIRMED",
                "accepted_request_id": "req-1",
            }
        ],
        team_members=[
            {"id": "tm-1", "team_link_id": "link-1", "tournament_id": "t-1", "event_option_id": "wd-35", "selection_id": "sel-mary", "registration_id": "reg-mary", "player_id": 101, "player_order": 1, "status": "ACTIVE"},
            {"id": "tm-2", "team_link_id": "link-1", "tournament_id": "t-1", "event_option_id": "wd-35", "selection_id": "sel-elizabeth", "registration_id": "reg-elizabeth", "player_id": 102, "player_order": 2, "status": "ACTIVE"},
        ],
    )

    doubles_entries = _entries(state)
    confirmed = [entry for entry in doubles_entries if entry["status"] == "CONFIRMED"]
    assert len(confirmed) == 1
    assert [member["display_name"] for member in confirmed[0]["members"]] == ["Mary Bauman", "Elizabeth Whelan"]
    assert not any(entry["status"] == "NEEDS_PARTNER" for entry in doubles_entries)
    assert not any(entry["status"] == "LEGACY_PARTNER_UNRESOLVED" for entry in doubles_entries)
    singles_roster = next(roster for roster in state["event_rosters"] if roster["event_option_id"] == "singles-35")
    assert singles_roster["entries"][0]["status"] == "CONFIRMED"
    assert singles_roster["entries"][0]["members"][0]["skill"] == 3.5


def test_rosters_do_not_borrow_skill_between_singles_and_doubles():
    singles_state = _linked_state(
        selections=[
            {
                "id": "sel-mary-singles",
                "registration_id": "reg-mary",
                "registration_day_id": "day-1",
                "event_option_id": "singles-35",
                "partner_mode": "NONE",
            }
        ],
        registrations=[
            {
                "id": "reg-mary",
                "display_name": "Mary Bauman",
                "email": "mary@example.com",
                "player_id": 101,
                "doubles_skill": 4.0,
                "singles_skill": None,
                "submitted_at": "2026-06-01T10:05:00+00:00",
            }
        ],
    )
    singles_roster = next(
        roster
        for roster in singles_state["event_rosters"]
        if roster["event_option_id"] == "singles-35"
    )
    assert singles_roster["entries"][0]["members"][0]["skill"] is None

    doubles_state = _linked_state(
        selections=[
            {
                "id": "sel-mary-doubles",
                "registration_id": "reg-mary",
                "registration_day_id": "day-1",
                "event_option_id": "wd-35",
                "partner_mode": "NEEDS_PARTNER",
            }
        ],
        registrations=[
            {
                "id": "reg-mary",
                "display_name": "Mary Bauman",
                "email": "mary@example.com",
                "player_id": 101,
                "doubles_skill": None,
                "singles_skill": 3.5,
                "submitted_at": "2026-06-01T10:05:00+00:00",
            }
        ],
    )
    doubles_roster = next(
        roster
        for roster in doubles_state["event_rosters"]
        if roster["event_option_id"] == "wd-35"
    )
    assert doubles_roster["entries"][0]["members"][0]["skill"] is None


def test_four_player_event_compiles_durable_team_instead_of_partner_missing():
    registrations = [
        {
            "id": f"reg-{index}",
            "display_name": name,
            "email": f"player{index}@example.com",
            "player_id": 200 + index,
            "status": "CONFIRMED",
            "submitted_at": f"2026-06-01T10:0{index}:00+00:00",
        }
        for index, name in enumerate(
            ["Captain One", "Player Two", "Player Three", "Player Four"],
            start=1,
        )
    ]
    selections = [
        {
            "id": f"sel-{index}",
            "registration_id": f"reg-{index}",
            "registration_day_id": "day-1",
            "event_option_id": "team-open",
            "partner_mode": "NONE",
            "player_id": 200 + index,
        }
        for index in range(1, 5)
    ]
    state = compile_tournament_registration_state(
        tournament={"id": "t-1"},
        settings={},
        days=[{"id": "day-1", "label": "Saturday", "sort_order": 1}],
        event_options=[
            {
                "id": "team-open",
                "registration_day_id": "day-1",
                "label": "Open Team",
                "event_type": "MIXED_DOUBLES",
                "partner_required": True,
                "competition_format": "FOUR_PLAYER_TEAM",
                "sort_order": 1,
            }
        ],
        registrations=registrations,
        selections=selections,
        four_player_teams=[
            {
                "id": "team-1",
                "event_option_id": "team-open",
                "name": "Kitchen Crew",
                "captain_registration_id": "reg-1",
                "status": "CONFIRMED",
                "eligibility_state": "NOT_REQUIRED",
                "created_at": "2026-06-01T10:05:00+00:00",
            }
        ],
        four_player_team_members=[
            {
                "id": f"member-{index}",
                "team_id": "team-1",
                "event_option_id": "team-open",
                "slot": slot,
                "registration_id": f"reg-{index}",
                "player_id": 200 + index,
                "display_name_snapshot": registrations[index - 1]["display_name"],
                "status": "ACCEPTED",
            }
            for index, slot in enumerate(
                ["MAN_1", "MAN_2", "WOMAN_1", "WOMAN_2"],
                start=1,
            )
        ],
    )

    entries = state["event_rosters"][0]["entries"]
    assert len(entries) == 1
    assert entries[0]["status"] == "CONFIRMED"
    assert entries[0]["entry_type"] == "four_player_team"
    assert entries[0]["team_name"] == "Kitchen Crew"
    assert len(entries[0]["members"]) == 4
    assert set(entries[0]["source_selection_ids"]) == {
        "sel-1",
        "sel-2",
        "sel-3",
        "sel-4",
    }
    assert not any(
        issue["issue_type"] == "MISSING_PARTNER_DETAILS"
        for issue in state["issues"]
    )


def test_four_player_forming_team_never_exposes_invited_members():
    state = compile_tournament_registration_state(
        tournament={"id": "t-1"},
        settings={},
        days=[{"id": "day-1", "label": "Saturday", "sort_order": 1}],
        event_options=[
            {
                "id": "team-open",
                "registration_day_id": "day-1",
                "label": "Open Team",
                "event_type": "MIXED_DOUBLES",
                "partner_required": True,
                "competition_format": "FOUR_PLAYER_TEAM",
                "sort_order": 1,
            }
        ],
        registrations=[
            {
                "id": "reg-1",
                "display_name": "Captain One",
                "email": "captain@example.com",
                "status": "CONFIRMED",
                "submitted_at": "2026-06-01T10:01:00+00:00",
            }
        ],
        selections=[
            {
                "id": "sel-1",
                "registration_id": "reg-1",
                "registration_day_id": "day-1",
                "event_option_id": "team-open",
                "partner_mode": "NONE",
            }
        ],
        four_player_teams=[
            {
                "id": "team-1",
                "event_option_id": "team-open",
                "name": "Kitchen Crew",
                "captain_registration_id": "reg-1",
                "status": "FORMING",
                "eligibility_state": "NOT_REQUIRED",
            }
        ],
        four_player_team_members=[
            {
                "id": "member-1",
                "team_id": "team-1",
                "event_option_id": "team-open",
                "slot": "MAN_1",
                "registration_id": "reg-1",
                "display_name_snapshot": "Captain One",
                "status": "ACCEPTED",
            },
            {
                "id": "member-2",
                "team_id": "team-1",
                "event_option_id": "team-open",
                "slot": "MAN_2",
                "display_name_snapshot": "Private Invitee",
                "status": "INVITED",
            },
        ],
    )

    entry = state["event_rosters"][0]["entries"][0]
    assert entry["status"] == "REVIEW"
    assert [member["display_name"] for member in entry["members"]] == [
        "Captain One"
    ]
