from jupr_app.domain.notifications.tournament_registrant_broadcast_email import (
    build_tournament_registrant_broadcast_email_html,
    build_tournament_registrant_broadcast_email_text,
    build_tournament_registrant_broadcast_subject,
)


def test_composer_message_does_not_add_a_second_greeting():
    options = dict(tournament_name="Baja Classic", recipient_name="Heather", subject="Women's Open",
        message="Hi Heather,\n\nSee you soon.\nJoe <organizer>", personalize_greeting=False)
    text = build_tournament_registrant_broadcast_email_text(**options)
    html = build_tournament_registrant_broadcast_email_html(**options)
    assert text.count("Hi Heather,") == 1
    assert html.count("Hi Heather,") == 1
    assert "Joe &lt;organizer&gt;" in html
    assert "Joe <organizer>" in text


def test_broadcast_subject_includes_tournament_when_needed():
    assert build_tournament_registrant_broadcast_subject(
        tournament_name="Baja Classic 2026",
        subject="Schedule update",
    ) == "Baja Classic 2026: Schedule update"


def test_broadcast_email_body_is_personal_and_does_not_expose_recipient_list():
    html = build_tournament_registrant_broadcast_email_html(
        tournament_name="Baja Classic 2026",
        recipient_name="Ada Lovelace",
        subject="Schedule update",
        message="Courts open at 8am.",
    )
    text = build_tournament_registrant_broadcast_email_text(
        tournament_name="Baja Classic 2026",
        recipient_name="Ada Lovelace",
        subject="Schedule update",
        message="Courts open at 8am.",
    )

    assert "Ada Lovelace" in html
    assert "Ada Lovelace" in text
    assert "Courts open at 8am." in html
    assert "Courts open at 8am." in text
    assert "all@example.com" not in html
    assert "all@example.com" not in text



def test_event_details_are_escaped_and_appear_between_message_and_supporting_sponsors():
    options = dict(tournament_name="Baja", recipient_name="Alex", subject="Schedule", message="Organizer message",
        personalize_greeting=False, registration_events=[{"name": "Alex <script>", "events": [
            {"division": "Mixed <Open>", "day": "Day 1", "event_date": "2026-11-19", "partner_name": 'Sam <img src=x onerror=alert(1)>'}]}],
        email_sponsors=[{"name": "Title Sponsor", "tier": "presenting"}, {"name": "Supporting Sponsor", "tier": "premier"}])
    html = build_tournament_registrant_broadcast_email_html(**options)
    text = build_tournament_registrant_broadcast_email_text(**options)
    assert "Alex &lt;script&gt;" in html and "Mixed &lt;Open&gt;" in html
    assert "<img src=x" not in html
    assert "Nov 19, 2026" in text and "Partner: Sam" in text
    assert html.index("Title Sponsor") < html.index("Organizer message") < html.index("Your registration events") < html.index("Supporting Sponsor")


def test_edit_buttons_escape_names_and_urls_and_preserve_sponsors():
    options = dict(tournament_name="Baja", recipient_name="Alex", subject="Update", message="Organizer message",
        registration_edit_links=[{"name": "Alex <script>", "edit_url": "https://example.test/edit?t=fixture&x=1"}],
        email_sponsors=[{"name": "Sponsor", "tier": "premier"}])
    html = build_tournament_registrant_broadcast_email_html(**options)
    text = build_tournament_registrant_broadcast_email_text(**options)
    assert "Alex &lt;script&gt;" in html
    assert 'href="https://example.test/edit?t=fixture&amp;x=1"' in html
    assert "https://example.test/edit?t=fixture&x=1" in text
    assert html.index("Organizer message") < html.index("Edit Registration") < html.index("Sponsor")
    assert "48 hours" in text
