from jupr_app.ui.pages import tournament_registration as registration


def test_public_empty_state_message_variants():
    assert (
        registration._public_empty_state_message(
            registration_open=False,
            selectable_count=3,
            hidden_draft_count=0,
        )
        == "Registration is closed."
    )

    assert (
        registration._public_empty_state_message(
            registration_open=True,
            selectable_count=0,
            hidden_draft_count=2,
        )
        == "Registration coming soon. Divisions are being finalized."
    )

    assert (
        registration._public_empty_state_message(
            registration_open=True,
            selectable_count=0,
            hidden_draft_count=0,
        )
        == "No open divisions are available right now."
    )

    assert (
        registration._public_empty_state_message(
            registration_open=True,
            selectable_count=1,
            hidden_draft_count=0,
        )
        is None
    )


def test_build_public_division_status_rows_shows_open_closed_and_full():
    days = [{"id": "day-1", "label": "Day 1", "enabled": True}]
    event_options = [
        {
            "id": "evt-open",
            "registration_day_id": "day-1",
            "event_family_label": "Mixed Doubles",
            "division_name": "4.0",
            "status": "open",
            "enabled": True,
            "capacity_teams": 2,
        },
        {
            "id": "evt-closed",
            "registration_day_id": "day-1",
            "event_family_label": "Mixed Doubles",
            "division_name": "4.5",
            "status": "closed",
            "enabled": True,
        },
        {
            "id": "evt-draft",
            "registration_day_id": "day-1",
            "event_family_label": "Mixed Doubles",
            "division_name": "5.0",
            "status": "draft",
            "enabled": True,
        },
    ]
    event_rosters = [
        {
            "event_option_id": "evt-open",
            "entries": [{"team": 1}, {"team": 2}],
        }
    ]

    rows = registration._build_public_division_status_rows(
        days=days,
        event_options=event_options,
        event_rosters=event_rosters,
    )

    assert rows == [
        {
            "day": "Day 1",
            "event": "Mixed Doubles",
            "division": "4.0",
            "status": "Full",
        },
        {
            "day": "Day 1",
            "event": "Mixed Doubles",
            "division": "4.5",
            "status": "Closed",
        },
    ]
