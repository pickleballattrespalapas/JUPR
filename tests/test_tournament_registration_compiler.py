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


def _linked_state(*, selections, partner_requests=None, partner_links=None, team_members=None):
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
        registrations=[
            {"id": "reg-mary", "display_name": "Mary Bauman", "email": "mary@example.com", "player_id": 101, "submitted_at": "2026-06-01T10:05:00+00:00"},
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
