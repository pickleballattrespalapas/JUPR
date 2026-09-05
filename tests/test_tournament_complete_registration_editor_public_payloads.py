from __future__ import annotations

import math
from pathlib import Path

import pytest

from jupr_app.services import public_tournament_registration_service as public_service
from jupr_app.services.admin_tournament_setup_service import (
    _event_option_payload,
    build_admin_tournament_setup_templates,
)
from jupr_app.services.public_tournament_registration_edit_service import (
    _selection_public_payload,
)


ROOT = Path(__file__).resolve().parent.parent


def _web(path: str) -> str:
    return (ROOT / "apps/web" / path).read_text(encoding="utf-8")


def _open_doubles_event() -> dict[str, object]:
    return {
        "id": "event-1",
        "registration_day_id": "day-1",
        "division_name": "Open Doubles",
        "event_family_label": "Doubles",
        "event_type": "DOUBLES",
        "gender_restriction": "ANY",
        "skill_label": "Open",
        "skill_mode": "OPEN",
        "eligibility_mode": "OPEN",
        "partner_required": True,
        "partner_board_enabled": True,
        "status": "open",
        "enabled": True,
        "selectable": True,
    }


def _manual_partner_selection(*, partner_mode: str = "HAS_PARTNER") -> dict[str, object]:
    return {
        "event_option_id": "event-1",
        "registration_day_id": "day-1",
        "partner_mode": partner_mode,
        "partner_name": "Partner Player",
        "partner_email": "partner@example.com",
        "partner_phone": "555-0100",
        "partner_dupr_id": "dupr-partner",
        "partner_skill": 3.75,
        "partner_age": 39,
        "partner_gender": "Women",
        "partner_note": "Manual partner",
        "show_on_partner_board": False,
    }


def _registration_payload() -> dict[str, object]:
    return {
        "first_name": "Primary",
        "last_name": "Player",
        "display_name": "Primary Player",
        "email": "primary@example.com",
        "phone": "555-0110",
        "doubles_skill": 3.5,
        "singles_skill": 3.5,
        "age": 41,
        "gender": "Men",
        "terms_accepted": True,
        "selections": [_manual_partner_selection()],
    }


def test_admin_and_public_event_payloads_round_trip_explicit_skill_bounds() -> None:
    row = {
        **_open_doubles_event(),
        "skill_min_rating": 3.25,
        "skill_max_rating": 4.5,
    }

    admin_event = _event_option_payload(row)
    public_event = public_service._public_event(row, registration_open=True)

    assert admin_event["skill_min_rating"] == 3.25
    assert admin_event["skill_max_rating"] == 4.5
    assert public_event["skill_min_rating"] == 3.25
    assert public_event["skill_max_rating"] == 4.5


def test_open_setup_template_explicitly_clears_skill_boundaries() -> None:
    template = build_admin_tournament_setup_templates(
        tournament={"id": "tournament-1", "start_date": "2026-08-20"},
        days=[],
    )[0]

    for event in template["event_options"]:
        assert event["eligibility_mode"] == "OPEN"
        assert event["skill_min_rating"] is None
        assert event["skill_max_rating"] is None
        assert event["combined_rating_cap"] is None


@pytest.mark.parametrize("locked", [False, True], ids=["new", "edit"])
def test_public_new_and_edit_save_payloads_preserve_manual_partner_gender(
    monkeypatch: pytest.MonkeyPatch,
    locked: bool,
) -> None:
    monkeypatch.setattr(
        public_service,
        "_registered_partner_profile",
        lambda *_args, **_kwargs: None,
    )
    locked_registration = (
        {
            "id": "registration-1",
            "email": "primary@example.com",
            "player_id": None,
            "status": "confirmed",
            "payment_status": "unpaid",
        }
        if locked
        else None
    )

    save_payload = public_service.build_validated_public_registration_save_payload(
        object(),
        club_id="club-1",
        tournament_id="tournament-1",
        page={
            "events": [_open_doubles_event()],
            "settings": {"partner_board_enabled": True},
        },
        payload=_registration_payload(),
        locked_registration=locked_registration,
    )

    assert save_payload["selections"][0]["partner_gender"] == "Women"


def test_public_edit_read_payload_round_trips_partner_gender() -> None:
    selection = {
        **_manual_partner_selection(),
        "id": "selection-1",
        "updated_at": "2026-08-08T12:00:00+00:00",
    }

    payload = _selection_public_payload(selection)

    assert payload["partner_gender"] == "Women"


def test_public_event_picker_defers_doubles_minimum_until_partner_is_known() -> None:
    source = _web("lib/tournamentRegistrationEligibility.ts")

    assert "!isDoubles && policy.minimum != null" in source
    assert 'policy.mode === "COMBINED_RATING_CAP"' in source
    assert "rating >= policy.combinedCap" in source
    assert "rating >= policy.maximumExclusive" in source


def test_public_skill_inputs_match_the_server_one_through_seven_contract() -> None:
    create = _web(
        "app/clubs/[clubSlug]/tournament-registration/TournamentRegistrationForm.tsx"
    )
    edit = _web(
        "app/clubs/[clubSlug]/tournament-registration/edit/"
        "EditTournamentRegistrationForm.tsx"
    )

    assert "must be between 1 and 7." in create
    assert 'aria-label="Doubles skill" type="number" min="1" max="7"' in create
    assert 'aria-label="Singles skill" type="number" min="1" max="7"' in create
    assert "Doubles not set" in create
    assert "Singles not set" in create
    assert "No JUPR singles rating yet?" in create
    assert "partner skill`} type=\"number\" min=\"1\" max=\"7\"" in create
    assert 'name="doubles_skill"' in edit and 'type="number" min="1" max="7"' in edit
    assert 'name="singles_skill"' in edit and edit.count('type="number" min="1" max="7"') >= 3
    assert "disabled={linkedPlayer?.doubles_skill != null}" in edit
    assert "disabled={linkedPlayer?.singles_skill != null}" in edit
    assert "No JUPR singles rating yet?" in edit


def test_legacy_registration_collects_missing_official_singles_rating() -> None:
    legacy = Path("jupr_app/ui/pages/tournament_registration.py").read_text(
        encoding="utf-8"
    )

    assert '"Singles skill (optional)"' in legacy
    assert '"singles_skill": clean_singles_self_rating' in legacy
    assert "does not create an official JUPR rating" in legacy
    assert "disabled=not singles_self_rating_valid" in legacy


def test_public_edit_prefills_and_requires_complete_manual_partner_details() -> None:
    source = _web(
        "app/clubs/[clubSlug]/tournament-registration/edit/"
        "EditTournamentRegistrationForm.tsx"
    )

    assert 'defaultValue={prior?.partner_gender || ""} required' in source
    assert '<option value="Non-binary">Non-binary</option>' in source
    assert 'defaultValue={prior?.partner_age ?? ""} type="number" min="1" max="120" required' in source


@pytest.mark.parametrize("partner_mode", ["NONE", "NEEDS_PARTNER"])
def test_non_partner_modes_clear_partner_gender(partner_mode: str) -> None:
    event = _open_doubles_event()
    event["partner_required"] = partner_mode == "NEEDS_PARTNER"
    raw_selection = _manual_partner_selection(partner_mode=partner_mode)

    cleaned = public_service.validate_and_clean_tournament_selection(
        object(),
        club_id="club-1",
        tournament_id="tournament-1",
        event=event,
        raw_selection=raw_selection,
        player_profile={
            "email": "primary@example.com",
            "player_id": None,
            "doubles_skill": 3.5,
            "singles_skill": 3.5,
            "age": 41,
            "gender": "Men",
        },
        settings={"partner_board_enabled": True},
    )

    assert cleaned["partner_gender"] == ""


def test_none_mode_clears_stale_nonfinite_partner_values_without_accepting_them() -> None:
    event = _open_doubles_event()
    event["partner_required"] = False
    raw_selection = _manual_partner_selection(partner_mode="NONE")
    raw_selection["partner_skill"] = math.nan
    raw_selection["partner_age"] = math.inf

    cleaned = public_service.validate_and_clean_tournament_selection(
        object(),
        club_id="club-1",
        tournament_id="tournament-1",
        event=event,
        raw_selection=raw_selection,
        player_profile={
            "email": "primary@example.com",
            "player_id": None,
            "doubles_skill": 3.5,
            "singles_skill": 3.5,
            "age": 41,
            "gender": "Men",
        },
        settings={"partner_board_enabled": True},
    )

    assert cleaned["partner_skill"] is None
    assert cleaned["partner_age"] is None
    assert cleaned["partner_gender"] == ""


@pytest.mark.parametrize("rating", [0.99, 7.01, math.nan, math.inf, -math.inf])
def test_public_ratings_require_finite_one_through_seven_values(rating: float) -> None:
    with pytest.raises(ValueError, match="between 1 and 7"):
        public_service._validated_rating(rating, label="Skill")


@pytest.mark.parametrize("age", [math.nan, math.inf, -math.inf, 39.5])
def test_public_ages_require_finite_whole_numbers(age: float) -> None:
    with pytest.raises(ValueError, match="finite whole number"):
        public_service._validated_age(age, label="Age")


def test_public_event_payload_does_not_emit_nonfinite_skill_bounds() -> None:
    event = {
        **_open_doubles_event(),
        "skill_min_rating": math.nan,
        "skill_max_rating": math.inf,
    }

    payload = public_service._public_event(event, registration_open=True)

    assert payload["skill_min_rating"] is None
    assert payload["skill_max_rating"] is None


def test_canonical_player_skills_ignore_nonfinite_and_out_of_range_values() -> None:
    assert public_service._canonical_player_skills(
        {
            "rating": math.inf,
            "doubles_skill": 0.5,
            "singles_skill": 7.5,
        }
    ) == (None, None)
