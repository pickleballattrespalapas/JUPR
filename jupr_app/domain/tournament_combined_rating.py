from __future__ import annotations

from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

RATING_MIN = Decimal("0.00")
RATING_MAX = Decimal("7.00")
COMBINED_CAP_MAX = Decimal("14.00")
_HUNDREDTH = Decimal("0.01")


def normalize_combined_rating_cap(value: Any) -> Decimal:
    """Return a supported combined-rating cap at two-decimal precision."""

    try:
        cap = Decimal(str(value)).quantize(_HUNDREDTH, rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Combined-rating cap must be a number.") from exc
    if cap <= 0 or cap > COMBINED_CAP_MAX:
        raise ValueError("Combined-rating cap must be greater than 0 and at most 14.00.")
    return cap


def normalize_tournament_rating(value: Any) -> Decimal | None:
    """Normalize one player rating, returning None when no rating is available."""

    if value in (None, ""):
        return None
    try:
        rating = Decimal(str(value)).quantize(_HUNDREDTH, rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Player rating must be a number.") from exc
    if rating < RATING_MIN or rating > RATING_MAX:
        raise ValueError("Player rating must be between 0.00 and 7.00.")
    return rating


def resolve_tournament_rating(
    *,
    linked_rating: Any = None,
    organizer_verified_rating: Any = None,
) -> dict[str, Any]:
    """Resolve the authoritative rating without allowing an override of linked data."""

    linked = normalize_tournament_rating(linked_rating)
    verified = normalize_tournament_rating(organizer_verified_rating)
    if linked is not None:
        return {"rating": linked, "source": "PCS_LINKED"}
    if verified is not None:
        return {"rating": verified, "source": "ORGANIZER_VERIFIED"}
    return {"rating": None, "source": "MISSING"}


def evaluate_combined_rating(
    *,
    combined_rating_cap: Any,
    player_linked_rating: Any = None,
    player_verified_rating: Any = None,
    partner_linked_rating: Any = None,
    partner_verified_rating: Any = None,
    partner_present: bool = True,
) -> dict[str, Any]:
    """Evaluate strict-under-cap eligibility and retain source evidence.

    A missing partner is provisional. Once a partner exists, a missing rating
    requires review. Equality with the cap is ineligible.
    """

    cap = normalize_combined_rating_cap(combined_rating_cap)
    player = resolve_tournament_rating(
        linked_rating=player_linked_rating,
        organizer_verified_rating=player_verified_rating,
    )
    partner = resolve_tournament_rating(
        linked_rating=partner_linked_rating,
        organizer_verified_rating=partner_verified_rating,
    )
    if not partner_present:
        state = "PROVISIONAL_NEEDS_PARTNER"
        combined = None
    elif player["rating"] is None or partner["rating"] is None:
        state = "REVIEW_REQUIRED"
        combined = None
    else:
        combined = (player["rating"] + partner["rating"]).quantize(_HUNDREDTH)
        state = "ELIGIBLE" if combined < cap else "INELIGIBLE"
    return {
        "state": state,
        "player_rating": player["rating"],
        "partner_rating": partner["rating"],
        "combined_rating": combined,
        "combined_rating_cap": cap,
        "player_rating_source": player["source"],
        "partner_rating_source": partner["source"],
        "strictly_below_cap": state == "ELIGIBLE",
    }


def validate_combined_rating_division(
    *,
    combined_rating: Any,
    combined_rating_cap: Any,
    lower_division_caps: list[Any] | tuple[Any, ...] = (),
) -> dict[str, Any]:
    """Validate strict eligibility while permitting a team to play up.

    ``lower_division_caps`` describes smaller caps offered for the same event.
    Being eligible for one of those divisions never blocks entry in a larger
    cap; that is the supported "playing up" behavior.
    """

    cap = normalize_combined_rating_cap(combined_rating_cap)
    if combined_rating in (None, ""):
        raise ValueError("Combined rating is required.")
    try:
        rating = Decimal(str(combined_rating)).quantize(
            _HUNDREDTH,
            rounding=ROUND_HALF_UP,
        )
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("Combined rating must be a number.") from exc
    if rating < RATING_MIN or rating > COMBINED_CAP_MAX:
        raise ValueError("Combined rating must be between 0.00 and 14.00.")
    eligible = rating < cap
    lower_caps = sorted(normalize_combined_rating_cap(value) for value in lower_division_caps)
    return {
        "eligible": eligible,
        "state": "ELIGIBLE" if eligible else "INELIGIBLE",
        "combined_rating": rating,
        "combined_rating_cap": cap,
        "playing_up": eligible and any(rating < lower_cap for lower_cap in lower_caps),
    }


# Compatibility names used by earlier service prototypes and tests.
normalize_rating_cap = normalize_combined_rating_cap
evaluate_combined_rating_eligibility = evaluate_combined_rating
