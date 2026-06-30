from pathlib import Path

from jupr_app.ui.pages import tournament_registration_admin_streamlined as admin_ui


def test_streamlined_registration_admin_has_mass_edit_tools():
    source = Path("jupr_app/ui/pages/tournament_registration_admin_streamlined.py").read_text(encoding="utf-8")

    assert "Mass edit tools" in source
    assert "Select entries for mass edit" in source
    assert "Type APPLY" in source
    assert "Download filtered CSV" in source
    assert "Edit one entry" in source


def test_streamlined_registration_admin_bulk_actions_include_core_workflows():
    assert "Set registration status" in admin_ui.BULK_ACTIONS
    assert "Set payment status" in admin_ui.BULK_ACTIONS
    assert "Set partner mode" in admin_ui.BULK_ACTIONS
    assert "Move division" in admin_ui.BULK_ACTIONS
    assert "Append admin note" in admin_ui.BULK_ACTIONS
    assert "Cancel registrations" in admin_ui.BULK_ACTIONS


def test_needs_partner_selection_payload_stays_public_by_default():
    payload = admin_ui._selection_payload_from_existing(
        {
            "registration_day_id": "day-1",
            "event_option_id": "event-1",
            "partner_mode": "NONE",
            "show_on_partner_board": False,
        },
        partner_mode="NEEDS_PARTNER",
    )

    assert payload["partner_mode"] == "NEEDS_PARTNER"
    assert payload["show_on_partner_board"] is True


def test_tournament_registration_wrapper_routes_admin_to_streamlined_ui():
    source = Path("jupr_app/ui/pages/tournament_registration/__init__.py").read_text(encoding="utf-8")

    assert "tournament_registration_admin_streamlined" in source
    assert "tournament_registration_admin_streamlined.render(ctx)" in source
