from pathlib import Path


def test_partner_board_dispatches_target_selection_to_request_form():
    source = Path("jupr_app/ui/pages/tournament_partner_board.py").read_text(encoding="utf-8")

    assert "target_selection_id" in source
    assert "tournament_partner_request.render(ctx)" in source
    assert "tournament_roster.render(ctx, focus_partners=True" in source


def test_partner_request_page_hides_recipient_email_and_requires_requester_contact():
    source = Path("jupr_app/ui/pages/tournament_partner_request.py").read_text(encoding="utf-8")

    assert "requested player's email address will not be shown or shared" in source
    assert "Your email" in source
    assert "Your phone / WhatsApp" in source
    assert "Enter your email or phone number" in source
    assert "target_email" in source


def test_partner_request_page_blocks_cancelled_registrations():
    source = Path("jupr_app/ui/pages/tournament_partner_request.py").read_text(encoding="utf-8")

    assert "_registration_is_cancelled" in source
    assert "registration was cancelled" in source
