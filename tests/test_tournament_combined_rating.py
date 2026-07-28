from decimal import Decimal

import pytest

from jupr_app.domain.tournament_combined_rating import (
    evaluate_combined_rating,
    normalize_combined_rating_cap,
    resolve_tournament_rating,
    validate_combined_rating_division,
)
from jupr_app.domain.tournament_registration_compiler import (
    validate_selection_against_skill,
)


def test_combined_rating_cap_rejects_invalid_values():
    for value in (None, "", 0, -1, 14.01, "not-a-rating"):
        with pytest.raises(ValueError):
            normalize_combined_rating_cap(value)


def test_linked_rating_is_authoritative_over_organizer_fallback():
    assert resolve_tournament_rating(
        linked_rating=3.47,
        organizer_verified_rating=4.25,
    ) == {"rating": Decimal("3.47"), "source": "PCS_LINKED"}


def test_missing_partner_is_provisional_and_missing_rating_requires_review():
    provisional = evaluate_combined_rating(
        combined_rating_cap=8,
        player_linked_rating=3.5,
        partner_present=False,
    )
    review = evaluate_combined_rating(
        combined_rating_cap=8,
        player_linked_rating=3.5,
        partner_present=True,
    )

    assert provisional["state"] == "PROVISIONAL_NEEDS_PARTNER"
    assert review["state"] == "REVIEW_REQUIRED"


def test_347_and_433_are_strictly_eligible_under_800():
    result = evaluate_combined_rating(
        combined_rating_cap=8,
        player_linked_rating=3.47,
        partner_linked_rating=4.33,
    )

    assert result["combined_rating"] == Decimal("7.80")
    assert result["state"] == "ELIGIBLE"


def test_equality_is_rejected_but_playing_up_is_allowed():
    equal = validate_combined_rating_division(
        combined_rating=8,
        combined_rating_cap=8,
    )
    playing_up = validate_combined_rating_division(
        combined_rating=6.8,
        combined_rating_cap=8,
        lower_division_caps=[7],
    )

    assert equal["state"] == "INELIGIBLE"
    assert playing_up["state"] == "ELIGIBLE"
    assert playing_up["playing_up"] is True


def test_registration_validation_uses_combined_cap_before_skill_band():
    event = {
        "event_type": "GENDER_DOUBLES",
        "partner_required": True,
        "skill_label": "3.0",
        "eligibility_mode": "COMBINED_RATING_CAP",
        "combined_rating_cap": 8,
    }
    allowed, message = validate_selection_against_skill(
        event=event,
        selection={"partner_mode": "HAS_PARTNER"},
        player={"doubles_skill": 3.47},
        partner={"doubles_skill": 4.33},
    )
    blocked, blocked_message = validate_selection_against_skill(
        event=event,
        selection={"partner_mode": "HAS_PARTNER"},
        player={"doubles_skill": 4},
        partner={"doubles_skill": 4},
    )

    assert (allowed, message) == (True, None)
    assert blocked is False
    assert "strictly below" in str(blocked_message)
