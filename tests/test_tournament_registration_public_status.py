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
