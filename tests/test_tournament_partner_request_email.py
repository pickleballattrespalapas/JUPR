from jupr_app.domain.notifications.tournament_partner_request_email import (
    build_tournament_partner_request_email_html,
    build_tournament_partner_request_email_text,
    build_tournament_partner_request_subject,
)


def _email_kwargs():
    return {
        "tournament_name": "Baja Classic 2026",
        "target_name": "Elizabeth Whelan",
        "requester_name": "Ada Lovelace",
        "requester_email": "ada@example.com",
        "requester_phone": "555-0100",
        "event_label": "Women's Doubles",
        "division_label": "Women's Doubles 3.5",
        "day_label": "Day 3 · Fri Nov 20",
        "message": "Would you like to partner?",
    }


def test_partner_request_email_shares_requester_contact_but_not_recipient_email():
    html = build_tournament_partner_request_email_html(**_email_kwargs())
    text = build_tournament_partner_request_email_text(**_email_kwargs())

    assert "ada@example.com" in html
    assert "555-0100" in html
    assert "ada@example.com" in text
    assert "555-0100" in text
    assert "recipient@example.com" not in html
    assert "recipient@example.com" not in text
    assert "Your email address was not shared" in html
    assert "Your email address was not shared" in text


def test_partner_request_subject_identifies_requester_and_tournament():
    assert build_tournament_partner_request_subject(
        tournament_name="Baja Classic 2026",
        requester_name="Ada Lovelace",
    ) == "Ada Lovelace wants to partner with you for Baja Classic 2026"
