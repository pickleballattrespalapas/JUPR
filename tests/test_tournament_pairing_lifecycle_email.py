from jupr_app.domain.notifications.tournament_pairing_interest_email import (
    build_pairing_status_html,
    build_pairing_status_subject,
    build_pairing_status_text,
)


def _status_kwargs(action="accepted"):
    return {
        "tournament_name": "Tres Open",
        "division_name": "Mixed 4.0",
        "requester_name": "Alex Player",
        "target_name": "Casey Player",
        "board_url": "https://example.test/partner-board?edit_token=opaque",
        "action": action,
        "recipient_kind": "requester",
    }


def test_pairing_lifecycle_email_explains_acceptance_without_contact_disclosure():
    html = build_pairing_status_html(**_status_kwargs())
    text = build_pairing_status_text(**_status_kwargs())

    assert "now paired" in html
    assert "now paired" in text
    assert "Alex Player" in html
    assert "Casey Player" in text
    assert "@example.com" not in html
    assert "+1" not in text
    assert "Contact details remain private" in html


def test_pairing_lifecycle_subject_and_decline_copy_are_deterministic():
    assert build_pairing_status_subject(tournament_name="Tres Open", action="declined") == "Tournament partner request declined: Tres Open"
    assert "No team was created" in build_pairing_status_text(**_status_kwargs("declined"))
