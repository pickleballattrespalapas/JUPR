from pathlib import Path

from jupr_app.ui.pages import tournament_registration


def test_public_registration_wrapper_reexports_legacy_helpers():
    assert callable(tournament_registration._hydrate_registration_wizard_from_bundle)
    assert callable(tournament_registration._advance_step1_registration_wizard)
    assert callable(tournament_registration.render)


def test_public_registration_entry_choice_labels_are_present():
    source = Path("jupr_app/ui/pages/tournament_registration/__init__.py").read_text(encoding="utf-8")

    assert "Start a new registration" in source
    assert "Edit an existing registration" in source
    assert "Send secure edit link" in source


def test_new_registration_with_existing_email_routes_to_edit_flow():
    wizard = {"registration_flow_choice": "new"}

    advanced, error = tournament_registration._advance_step1_registration_wizard(
        wizard,
        tournament_id="tournament_1",
        first_name="Ada",
        last_name="Lovelace",
        email_for_submit="ADA@example.com",
        phone="",
        gender="Female",
        age="35",
        notes="",
        find_existing_registration=lambda *_: {"id": "reg_1"},
    )

    assert advanced is True
    assert error == ""
    assert wizard["registration_flow_choice"] == "edit"
    assert wizard["returning_registration_id"] == "reg_1"
    assert wizard["returning_email"] == "ada@example.com"
    assert wizard["current_step"] == 0


def test_new_registration_without_existing_email_proceeds_to_profile_step():
    wizard = {"registration_flow_choice": "new"}

    advanced, error = tournament_registration._advance_step1_registration_wizard(
        wizard,
        tournament_id="tournament_1",
        first_name="Ada",
        last_name="Lovelace",
        email_for_submit="ada@example.com",
        phone="",
        gender="Female",
        age="35",
        notes="",
        find_existing_registration=lambda *_: None,
    )

    assert advanced is True
    assert error == ""
    assert wizard["step1"]["email"] == "ada@example.com"
    assert wizard["current_step"] == 2
