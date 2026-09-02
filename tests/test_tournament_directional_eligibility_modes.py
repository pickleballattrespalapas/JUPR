from __future__ import annotations

from jupr_app.domain.tournament_registration_compiler import (
    canonical_skill_policy,
    evaluate_selection_skill_eligibility,
)


def _event(mode: str, *, doubles: bool = False, **overrides):
    event = {
        "eligibility_mode": mode,
        "skill_mode": mode,
        "skill_label": "3.5",
        "event_type": "DOUBLES" if doubles else "SINGLES",
        "partner_required": doubles,
    }
    event.update(overrides)
    return event


def _result(event, player_rating, partner_rating=None):
    return evaluate_selection_skill_eligibility(
        event=event,
        selection={"partner_mode": "HAS_PARTNER" if partner_rating is not None else "NONE"},
        player={"doubles_skill": player_rating, "singles_skill": player_rating},
        partner=(
            {"doubles_skill": partner_rating, "singles_skill": partner_rating}
            if partner_rating is not None
            else None
        ),
    )


def test_standard_ceiling_is_exclusive_and_still_allows_playing_up() -> None:
    event = _event("STANDARD")

    assert _result(event, 3.99)["status"] == "ELIGIBLE"
    assert _result(event, 4.0)["status"] == "INELIGIBLE"
    assert _result(event, 2.5)["status"] == "ELIGIBLE"


def test_minimum_is_inclusive_has_no_upper_ceiling_and_uses_controlling_rating() -> None:
    event = _event("MINIMUM", doubles=True, skill_min_rating=3.5)

    assert _result(event, 3.49, 3.49)["status"] == "INELIGIBLE"
    assert _result(event, 3.5, 3.5)["status"] == "ELIGIBLE"
    assert _result(event, 2.5, 5.75)["status"] == "ELIGIBLE"
    assert canonical_skill_policy(event)["skill_ceiling_exclusive"] is None


def test_open_has_no_rating_requirement_even_when_ratings_are_missing() -> None:
    event = _event("OPEN", skill_label="Open")

    assert _result(event, None)["status"] == "ELIGIBLE"
    assert _result(event, 7.0)["status"] == "ELIGIBLE"


def test_custom_supports_lower_upper_and_two_sided_boundaries() -> None:
    lower = _event("CUSTOM", skill_min_rating=3.0, skill_max_rating=None)
    upper = _event("CUSTOM", skill_min_rating=None, skill_max_rating=4.0)
    both = _event("CUSTOM", skill_min_rating=3.0, skill_max_rating=4.0)

    assert _result(lower, 2.99)["status"] == "INELIGIBLE"
    assert _result(lower, 6.0)["status"] == "ELIGIBLE"
    assert _result(upper, 3.99)["status"] == "ELIGIBLE"
    assert _result(upper, 4.0)["status"] == "INELIGIBLE"
    assert _result(both, 3.0)["status"] == "ELIGIBLE"
    assert _result(both, 4.0)["status"] == "INELIGIBLE"


def test_combined_cap_is_doubles_only_and_rejects_equality() -> None:
    doubles = _event(
        "COMBINED_RATING_CAP",
        doubles=True,
        skill_label="Open",
        combined_rating_cap=8.0,
    )
    singles = _event(
        "COMBINED_RATING_CAP",
        combined_rating_cap=8.0,
    )

    assert _result(doubles, 3.99, 4.0)["status"] == "ELIGIBLE"
    assert _result(doubles, 4.0, 4.0)["status"] == "INELIGIBLE"
    assert _result(singles, 4.0)["issue_type"] == "INVALID_SKILL_POLICY"


def test_legacy_plus_label_remains_open_until_minimum_mode_is_explicit() -> None:
    legacy = _event("STANDARD", skill_mode="OPEN", skill_label="3.5+")
    explicit = _event("MINIMUM", skill_label="3.5+", skill_min_rating=3.5)

    assert canonical_skill_policy(legacy)["mode"] == "OPEN"
    assert canonical_skill_policy(explicit)["mode"] == "MINIMUM"


def test_custom_policy_never_infers_a_boundary_from_its_display_label() -> None:
    policy = canonical_skill_policy(
        _event("CUSTOM", skill_label="3.5", skill_min_rating=None, skill_max_rating=4.5)
    )

    assert policy["mode"] == "CUSTOM"
    assert policy["skill_minimum_inclusive"] is None
    assert policy["skill_ceiling_exclusive"] == 4.5


def test_nonfinite_boundaries_and_ratings_are_treated_as_missing_not_valid() -> None:
    policy = canonical_skill_policy(
        _event("CUSTOM", skill_min_rating=float("nan"), skill_max_rating=float("inf"))
    )
    result = _result(_event("STANDARD"), float("nan"))

    assert policy["skill_minimum_inclusive"] is None
    assert policy["skill_ceiling_exclusive"] is None
    assert result["status"] == "MISSING_DATA"


def test_explicit_standard_skill_mode_wins_over_stale_open_display_text() -> None:
    policy = canonical_skill_policy(
        _event("STANDARD", skill_mode="STANDARD", skill_label="Open")
    )

    assert policy["mode"] == "STANDARD_CEILING"
    assert policy["skill_ceiling_exclusive"] is None
