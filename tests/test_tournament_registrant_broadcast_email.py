from jupr_app.domain.notifications.tournament_registrant_broadcast_email import (
    build_tournament_registrant_broadcast_email_html,
    build_tournament_registrant_broadcast_email_text,
    build_tournament_registrant_broadcast_subject,
)


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
