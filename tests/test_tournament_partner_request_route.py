from pathlib import Path

from jupr_app.domain.tournament_public_references import build_public_tournament_reference
from jupr_app.ui.pages.tournament_partner_request import (
    _load_target_selection,
    _target_is_publicly_available,
)
from tests.test_public_tournament_registration_service import FakeSupabase, fake_storage


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


def test_partner_request_page_rechecks_public_visibility_and_consent():
    source = Path("jupr_app/ui/pages/tournament_partner_request.py").read_text(encoding="utf-8")

    assert "public_tournament_reference_matches" in source
    assert 'selection.get("show_on_partner_board")' in source
    assert 'settings.get("partner_board_enabled")' in source
    assert 'event.get("partner_board_enabled", event.get("public_partner_board"))' in source
    assert 'registration.get("wants_partner_board_contact")' in source
    assert "_registration_is_active" in source


def test_partner_request_resolves_opaque_board_entry_key_only():
    storage = fake_storage()
    storage["tournament_registration_selections"].append(
        {
            "id": "selection-private-id",
            "tournament_id": "t1",
            "registration_id": "registration-1",
            "event_option_id": "event1",
        }
    )
    supabase = FakeSupabase(storage)
    selection = storage["tournament_registration_selections"][0]
    selection_id = str(selection["id"])
    tournament_id = str(selection["tournament_id"])
    board_entry_key = build_public_tournament_reference(
        tournament_id=tournament_id,
        namespace="partner-board-selection",
        source_id=selection_id,
    )

    resolved = _load_target_selection(
        supabase,
        tournament_id=tournament_id,
        target_selection_id=board_entry_key,
    )

    assert resolved and resolved["id"] == selection_id
    assert _load_target_selection(
        supabase,
        tournament_id=tournament_id,
        target_selection_id=selection_id,
    ) is None


def test_partner_request_target_requires_every_public_board_gate():
    settings = {"partner_board_enabled": True}
    selection = {"partner_mode": "NEEDS_PARTNER", "show_on_partner_board": True}
    event = {"enabled": True, "partner_board_enabled": True, "status": "active"}
    registration = {"status": "CONFIRMED", "wants_partner_board_contact": True}

    assert _target_is_publicly_available(
        settings=settings,
        selection=selection,
        event=event,
        registration=registration,
    )

    gated_values = [
        ("settings", "partner_board_enabled", False),
        ("selection", "show_on_partner_board", False),
        ("selection", "partner_mode", "HAS_PARTNER"),
        ("event", "enabled", False),
        ("event", "partner_board_enabled", False),
        ("event", "status", "draft"),
        ("registration", "status", "WITHDRAWN"),
        ("registration", "wants_partner_board_contact", False),
    ]
    for group_name, field, value in gated_values:
        candidate_settings = dict(settings)
        candidate_selection = dict(selection)
        candidate_event = dict(event)
        candidate_registration = dict(registration)
        groups = {
            "settings": candidate_settings,
            "selection": candidate_selection,
            "event": candidate_event,
            "registration": candidate_registration,
        }
        groups[group_name][field] = value

        assert not _target_is_publicly_available(
            settings=candidate_settings,
            selection=candidate_selection,
            event=candidate_event,
            registration=candidate_registration,
        )
