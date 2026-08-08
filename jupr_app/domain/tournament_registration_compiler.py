from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any
import math
import re
import uuid


DoublesTypes = {"GENDER_DOUBLES", "MIXED_DOUBLES", "DOUBLES", "MIXED"}

def _coerce_skill(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except Exception:
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_skill_label(skill_label: str) -> tuple[float, bool] | None:
    text = str(skill_label or "").strip()
    if not text or text.lower() == "open":
        return None
    match = re.fullmatch(r"(?:skill\s*)?([0-9](?:\.[0-9]{1,2})?)\s*(\+)?", text, re.IGNORECASE)
    if not match:
        return None
    try:
        anchor = float(match.group(1))
    except Exception:
        return None
    if anchor < 1.0 or anchor > 7.0:
        return None
    return round(anchor, 2), bool(match.group(2))


def _parse_skill_anchor(skill_label: str) -> float | None:
    parsed = _parse_skill_label(skill_label)
    return parsed[0] if parsed else None


def _next_half_step(value: float) -> float:
    return round(value + 0.5, 2)


def _skill_band_for_label(skill_label: Any) -> tuple[float, float] | None:
    anchor = _parse_skill_anchor(str(skill_label or ""))
    if anchor is None:
        return None
    return anchor, _next_half_step(anchor)


def _rating_at_or_above_ceiling(rating: float | None, ceiling_exclusive: float) -> bool:
    return rating is not None and rating >= ceiling_exclusive


def _recommended_anchor_for_rating(rating: float | None) -> float | None:
    if rating is None:
        return None
    return round(math.floor(rating * 2.0) / 2.0, 1)


def _format_skill(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _effective_singles_skill(player: dict[str, Any]) -> float | None:
    singles = _coerce_skill(player.get("singles_skill"))
    if singles is not None:
        return singles
    return _coerce_skill(player.get("doubles_skill"))


def _effective_doubles_skill(player: dict[str, Any]) -> float | None:
    doubles = _coerce_skill(player.get("doubles_skill"))
    if doubles is not None:
        return doubles
    return _coerce_skill(player.get("singles_skill"))


def _explicit_skill_mode(event: dict[str, Any]) -> str:
    eligibility_mode = str(event.get("eligibility_mode") or "").strip().upper()
    skill_mode = str(event.get("skill_mode") or "").strip().upper()
    label = str(event.get("skill_label") or "").strip()
    parsed = _parse_skill_label(label)
    if eligibility_mode == "COMBINED_RATING_CAP" or skill_mode == "COMBINED_RATING_CAP":
        return "COMBINED_RATING_CAP"
    if eligibility_mode == "CUSTOM" or skill_mode == "CUSTOM":
        return "CUSTOM"
    if eligibility_mode == "MINIMUM" or skill_mode in {"MINIMUM", "MIN", "AT_LEAST"}:
        return "MINIMUM"
    if eligibility_mode in {"OPEN", "NONE"}:
        return "OPEN"
    # New editor writes an explicit skill_mode. Let it override stale label
    # text left by a mode transition; legacy Open rows used OPEN/NONE here.
    if skill_mode in {"STANDARD", "SKILL_BRACKET", "CEILING", "MAXIMUM"}:
        return "STANDARD"
    if label.lower() == "open":
        return "OPEN"
    # A trailing plus was historically an upward/open label. Only the new
    # explicit MINIMUM mode turns it into a lower-bound rule.
    if label.endswith("+"):
        return "OPEN"
    if eligibility_mode == "STANDARD":
        return "STANDARD"
    # Legacy rows sometimes retained skill_mode=OPEN after the organizer chose
    # a numeric label. The numeric label remains the authoritative boundary.
    if parsed is not None:
        return "STANDARD"
    if skill_mode in {"OPEN", "NONE"}:
        return "OPEN"
    # Legacy organizer-defined labels such as "Beginner" historically carried no
    # numeric rating boundary. Preserve that behavior unless the organizer selects
    # an explicit Standard or Custom policy.
    return "OPEN"


def skill_ceiling_exclusive(event: dict[str, Any]) -> float | None:
    """Return the hard upper boundary for the proposed Division.

    Standard 3.5 means a controlling rating strictly below 4.0. Minimum / Skill+
    and Open divisions have no upper ceiling. Custom divisions may provide an
    explicit exclusive upper boundary.
    """

    mode = _explicit_skill_mode(event)
    if mode in {"OPEN", "MINIMUM", "COMBINED_RATING_CAP"}:
        return None
    explicit = _coerce_skill(event.get("skill_max_rating"))
    if explicit is not None:
        return round(explicit, 2)
    if mode == "CUSTOM":
        return None
    parsed = _parse_skill_label(str(event.get("skill_label") or ""))
    if not parsed or parsed[1]:
        return None
    return _next_half_step(parsed[0])


def skill_minimum_inclusive(event: dict[str, Any]) -> float | None:
    mode = _explicit_skill_mode(event)
    if mode not in {"MINIMUM", "CUSTOM"}:
        return None
    explicit = _coerce_skill(event.get("skill_min_rating"))
    if explicit is not None:
        return round(explicit, 2)
    if mode == "MINIMUM":
        parsed = _parse_skill_label(str(event.get("skill_label") or ""))
        return parsed[0] if parsed else None
    return None


def canonical_skill_policy(event: dict[str, Any]) -> dict[str, Any]:
    mode = _explicit_skill_mode(event)
    if mode == "COMBINED_RATING_CAP":
        cap = _coerce_skill(event.get("combined_rating_cap"))
        return {
            "mode": "COMBINED_RATING_CAP",
            "combined_rating_cap": round(cap, 2) if cap is not None else None,
        }
    if mode == "OPEN":
        return {
            "mode": "OPEN",
            "skill_minimum_inclusive": None,
            "skill_ceiling_exclusive": None,
        }
    minimum = skill_minimum_inclusive(event)
    ceiling = skill_ceiling_exclusive(event)
    if mode == "MINIMUM":
        return {
            "mode": "MINIMUM",
            "skill_minimum_inclusive": minimum,
            "skill_ceiling_exclusive": None,
        }
    if mode == "CUSTOM":
        return {
            "mode": "CUSTOM",
            "skill_minimum_inclusive": minimum,
            "skill_ceiling_exclusive": ceiling,
        }
    return {
        "mode": "STANDARD_CEILING",
        "skill_minimum_inclusive": None,
        "skill_ceiling_exclusive": round(ceiling, 2) if ceiling is not None else None,
    }


def evaluate_selection_skill_eligibility(
    *,
    event: dict[str, Any],
    selection: dict[str, Any],
    player: dict[str, Any],
    partner: dict[str, Any] | None = None,
    allow_missing_partner_for_preview: bool = False,
) -> dict[str, Any]:
    """Return ELIGIBLE, MISSING_DATA, or INELIGIBLE for skill rules."""

    team_event = _is_doubles_event(event)
    partner_mode = str(selection.get("partner_mode") or "NONE").strip().upper()
    policy = canonical_skill_policy(event)
    player_rating = _effective_doubles_skill(player) if team_event else _effective_singles_skill(player)
    partner_rating = _effective_doubles_skill(partner or {}) if team_event and partner else None
    known_ratings = [value for value in (player_rating, partner_rating) if value is not None]
    controlling_rating = max(known_ratings) if known_ratings else None
    base: dict[str, Any] = {
        "policy": policy,
        "player_rating": player_rating,
        "partner_rating": partner_rating,
        "controlling_rating": controlling_rating,
        "pending_partner": team_event and partner_mode == "NEEDS_PARTNER",
    }

    if policy["mode"] == "OPEN":
        return {**base, "status": "ELIGIBLE", "issue_type": None, "issue": None}

    if policy["mode"] == "COMBINED_RATING_CAP":
        if not team_event:
            return {
                **base,
                "status": "INELIGIBLE",
                "issue_type": "INVALID_SKILL_POLICY",
                "issue": "Combined-rating divisions require a doubles or team event.",
            }
        cap = _coerce_skill(policy.get("combined_rating_cap"))
        if cap is None or cap <= 0 or cap > 14:
            return {
                **base,
                "status": "INELIGIBLE",
                "issue_type": "INVALID_SKILL_POLICY",
                "issue": "This combined-rating division is missing a valid rating cap.",
            }
        if any(value >= cap for value in known_ratings):
            highest = max(known_ratings)
            return {
                **base,
                "status": "INELIGIBLE",
                "issue_type": "SKILL_NOT_ELIGIBLE",
                "issue": (
                    f"Known rating {_format_skill(highest)} cannot fit a combined-rating cap "
                    f"strictly below {_format_skill(cap)}."
                ),
                "combined_rating_cap": cap,
            }
        missing_fields: list[str] = []
        if player_rating is None:
            missing_fields.append("player rating")
        if team_event and partner_mode in {"HAS_PARTNER", "NEEDS_PARTNER"} and partner_rating is None:
            missing_fields.append("partner rating")
        if missing_fields:
            if allow_missing_partner_for_preview and missing_fields == ["partner rating"] and partner_mode == "NEEDS_PARTNER":
                return {**base, "status": "ELIGIBLE", "issue_type": None, "issue": None, "combined_rating_cap": cap}
            return {
                **base,
                "status": "MISSING_DATA",
                "issue_type": "MISSING_SKILL_DATA",
                "issue": f"Complete {' and '.join(missing_fields)} before confirming combined-rating eligibility.",
                "missing_fields": missing_fields,
                "combined_rating_cap": cap,
            }
        combined = round(float(player_rating or 0) + float(partner_rating or 0), 2)
        if combined >= cap:
            return {
                **base,
                "status": "INELIGIBLE",
                "issue_type": "SKILL_NOT_ELIGIBLE",
                "issue": (
                    f"Combined rating {_format_skill(combined)} is not strictly below "
                    f"the {_format_skill(cap)} cap."
                ),
                "combined_rating_cap": cap,
                "combined_rating": combined,
            }
        return {
            **base,
            "status": "ELIGIBLE",
            "issue_type": None,
            "issue": None,
            "combined_rating_cap": cap,
            "combined_rating": combined,
        }

    minimum = _coerce_skill(policy.get("skill_minimum_inclusive"))
    ceiling = _coerce_skill(policy.get("skill_ceiling_exclusive"))
    if policy["mode"] == "MINIMUM" and minimum is None:
        return {
            **base,
            "status": "INELIGIBLE",
            "issue_type": "INVALID_SKILL_POLICY",
            "issue": "This Skill+ division is missing a valid minimum rating.",
        }
    if policy["mode"] == "STANDARD_CEILING" and ceiling is None:
        return {
            **base,
            "status": "INELIGIBLE",
            "issue_type": "INVALID_SKILL_POLICY",
            "issue": "This standard skill division is missing a numeric skill level.",
        }
    if policy["mode"] == "CUSTOM" and minimum is None and ceiling is None:
        return {
            **base,
            "status": "INELIGIBLE",
            "issue_type": "INVALID_SKILL_POLICY",
            "issue": "This custom skill division needs a minimum or maximum rating.",
        }

    # A known rating at or above an upper boundary is definitively ineligible,
    # even when a partner rating is still missing: adding another partner cannot
    # lower the controlling (higher) rating. Evaluate this before missing-data
    # handling so Review/public/admin surfaces agree.
    if ceiling is not None and _rating_at_or_above_ceiling(controlling_rating, ceiling):
        anchor = round(ceiling - 0.5, 2) if policy["mode"] == "STANDARD_CEILING" else None
        recommended = _recommended_anchor_for_rating(controlling_rating)
        subject = "Your team" if team_event else "Your rating"
        label = _format_skill(anchor) if anchor is not None else str(event.get("skill_label") or "this")
        return {
            **base,
            "status": "INELIGIBLE",
            "issue_type": "SKILL_NOT_ELIGIBLE",
            "issue": (
                f"{subject} is rated above the {label} division cap. "
                f"Please register for a {_format_skill(recommended)} or higher division."
            ),
            "skill_minimum_inclusive": minimum,
            "skill_ceiling_exclusive": ceiling,
        }

    missing_fields: list[str] = []
    if player_rating is None:
        missing_fields.append("player rating")
    if team_event and partner_mode in {"HAS_PARTNER", "NEEDS_PARTNER"} and partner_rating is None:
        missing_fields.append("partner rating")
    if missing_fields:
        if allow_missing_partner_for_preview and missing_fields == ["partner rating"] and partner_mode == "NEEDS_PARTNER":
            return {**base, "status": "ELIGIBLE", "issue_type": None, "issue": None}
        return {
            **base,
            "status": "MISSING_DATA",
            "issue_type": "MISSING_SKILL_DATA",
            "issue": f"Complete {' and '.join(missing_fields)} before confirming skill eligibility.",
            "missing_fields": missing_fields,
            "skill_minimum_inclusive": minimum,
            "skill_ceiling_exclusive": ceiling,
        }

    if minimum is not None and controlling_rating is not None and controlling_rating < minimum:
        subject = "Your team's controlling rating" if team_event else "Your rating"
        return {
            **base,
            "status": "INELIGIBLE",
            "issue_type": "SKILL_NOT_ELIGIBLE",
            "issue": f"{subject} is {_format_skill(controlling_rating)}. This division requires {_format_skill(minimum)} or higher.",
            "skill_minimum_inclusive": minimum,
            "skill_ceiling_exclusive": ceiling,
        }
    return {
        **base,
        "status": "ELIGIBLE",
        "issue_type": None,
        "issue": None,
        "skill_minimum_inclusive": minimum,
        "skill_ceiling_exclusive": ceiling,
    }

def normalize_tournament_gender(value: Any) -> str:
    text = re.sub(r"[^a-z]", "", str(value or "").strip().lower())
    if text in {"m", "male", "man", "men", "mens", "boy", "boys"}:
        return "MEN"
    if text in {"f", "female", "woman", "women", "womens", "girl", "girls"}:
        return "WOMEN"
    return "OTHER" if text else ""


def canonical_gender_restriction(event: dict[str, Any]) -> str:
    restriction = str(event.get("gender_restriction") or "ANY").strip().upper()
    if restriction in {"", "OPEN", "NONE"}:
        return "ANY"
    if restriction == "MALE":
        return "MEN"
    if restriction == "FEMALE":
        return "WOMEN"
    return restriction


def evaluate_selection_gender_eligibility(
    *,
    event: dict[str, Any],
    selection: dict[str, Any],
    player: dict[str, Any],
    partner: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate gender restrictions without treating missing data as denial."""

    restriction = canonical_gender_restriction(event)
    player_gender = normalize_tournament_gender(player.get("gender"))
    partner_gender = normalize_tournament_gender((partner or {}).get("gender")) if partner else ""
    partner_mode = str(selection.get("partner_mode") or "NONE").strip().upper()
    team_event = _is_doubles_event(event)
    base = {
        "restriction": restriction,
        "player_gender": player_gender,
        "partner_gender": partner_gender,
        "pending_partner": team_event and partner_mode == "NEEDS_PARTNER",
    }
    if restriction == "ANY":
        return {**base, "status": "ELIGIBLE", "issue_type": None, "issue": None}
    if not player_gender:
        return {
            **base,
            "status": "MISSING_DATA",
            "issue_type": "MISSING_GENDER_DATA",
            "issue": "Complete player gender before confirming eligibility.",
            "missing_fields": ["player gender"],
        }
    if restriction in {"MEN", "WOMEN"}:
        if player_gender != restriction:
            return {
                **base,
                "status": "INELIGIBLE",
                "issue_type": "GENDER_NOT_ELIGIBLE",
                "issue": f"Player does not meet the proposed {restriction.lower()} restriction.",
            }
        if team_event and partner_mode in {"HAS_PARTNER", "NEEDS_PARTNER"}:
            if not partner_gender:
                return {
                    **base,
                    "status": "MISSING_DATA",
                    "issue_type": "MISSING_GENDER_DATA",
                    "issue": "Complete partner gender before confirming eligibility.",
                    "missing_fields": ["partner gender"],
                }
            if partner_gender != restriction:
                return {
                    **base,
                    "status": "INELIGIBLE",
                    "issue_type": "GENDER_NOT_ELIGIBLE",
                    "issue": f"Partner does not meet the proposed {restriction.lower()} restriction.",
                }
        return {**base, "status": "ELIGIBLE", "issue_type": None, "issue": None}
    if restriction == "MIXED":
        if player_gender not in {"MEN", "WOMEN"}:
            return {
                **base,
                "status": "INELIGIBLE",
                "issue_type": "GENDER_NOT_ELIGIBLE",
                "issue": "Player does not meet the proposed Mixed Division rule.",
            }
        if partner_mode in {"HAS_PARTNER", "NEEDS_PARTNER"}:
            if not partner_gender:
                return {
                    **base,
                    "status": "MISSING_DATA",
                    "issue_type": "MISSING_GENDER_DATA",
                    "issue": "Complete partner gender before confirming Mixed Division eligibility.",
                    "missing_fields": ["partner gender"],
                }
            if partner_gender not in {"MEN", "WOMEN"} or partner_gender == player_gender:
                return {
                    **base,
                    "status": "INELIGIBLE",
                    "issue_type": "GENDER_NOT_ELIGIBLE",
                    "issue": "Mixed doubles requires one men's and one women's registrant.",
                }
        return {**base, "status": "ELIGIBLE", "issue_type": None, "issue": None}
    return {**base, "status": "ELIGIBLE", "issue_type": None, "issue": None}


def validate_selection_against_skill(
    *,
    event: dict[str, Any],
    selection: dict[str, Any],
    player: dict[str, Any],
    partner: dict[str, Any] | None = None,
    allow_missing_partner_for_preview: bool = False,
) -> tuple[bool, str | None]:
    result = evaluate_selection_skill_eligibility(
        event=event,
        selection=selection,
        player=player,
        partner=partner,
        allow_missing_partner_for_preview=allow_missing_partner_for_preview,
    )
    if result["status"] == "INELIGIBLE":
        return False, str(result.get("issue") or "This registration is outside the proposed skill ceiling.")
    return True, None


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _normalize_email(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_dt(value: Any) -> datetime:
    if not value:
        return datetime.min
    text = str(value).strip()
    for candidate in (text, text.replace("Z", "+00:00")):
        try:
            return datetime.fromisoformat(candidate)
        except Exception:
            continue
    return datetime.min


def _issue(
    tournament_id: str,
    issue_type: str,
    severity: str,
    message: str,
    *,
    registration_id: str | None = None,
    selection_id: str | None = None,
    event_option_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": _uid("issue"),
        "tournament_id": str(tournament_id),
        "registration_id": registration_id,
        "selection_id": selection_id,
        "event_option_id": event_option_id,
        "issue_type": issue_type,
        "severity": severity,
        "message": message,
    }


def _event_sort_key(event: dict[str, Any], day_lookup: dict[str, dict[str, Any]]) -> tuple:
    day = day_lookup.get(str(event.get("registration_day_id")), {})
    return (
        int(day.get("sort_order") or 0),
        int(event.get("sort_order") or 0),
        str(event.get("label") or ""),
    )


def _day_sort_key(day: dict[str, Any]) -> tuple:
    return (int(day.get("sort_order") or 0), str(day.get("label") or ""))


def _is_doubles_event(event: dict[str, Any]) -> bool:
    event_type = str(event.get("event_type") or event.get("participant_type") or "").upper()
    if event_type in DoublesTypes:
        return True
    try:
        if int(event.get("team_roster_size") or 1) > 1:
            return True
    except Exception:
        pass
    return bool(event.get("partner_required"))


def _to_member_from_registration(registration: dict[str, Any], selection: dict[str, Any] | None = None) -> dict[str, Any]:
    display_name = str(registration.get("display_name") or "").strip()
    if not display_name:
        display_name = " ".join(
            part for part in [str(registration.get("first_name") or "").strip(), str(registration.get("last_name") or "").strip()] if part
        ).strip()
    if not display_name:
        display_name = str(registration.get("email") or "Player")
    return {
        "registration_id": str(registration.get("id") or "").strip() or None,
        "selection_id": str((selection or {}).get("id") or "").strip() or None,
        "player_id": (selection or {}).get("player_id") or registration.get("player_id"),
        "display_name": display_name,
        "email": _normalize_email(registration.get("email")),
        "phone": str(registration.get("phone") or "").strip() or None,
        "dupr_id": str(registration.get("dupr_id") or "").strip() or None,
        "skill": registration.get("doubles_skill") or registration.get("singles_skill"),
        "age": registration.get("age"),
        "gender": registration.get("gender"),
        "age_bracket": registration.get("age_bracket"),
    }


def _entry_identity(selection: dict[str, Any], registration: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_registration_ids": [str(registration.get("id"))],
        "source_selection_ids": [str(selection.get("id"))],
        "source_player_ids": [selection.get("player_id") or registration.get("player_id")],
    }


def _legacy_partner_metadata(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "partner_name": str(selection.get("partner_name") or "").strip() or None,
        "partner_email": _normalize_email(selection.get("partner_email")) or None,
        "partner_phone": str(selection.get("partner_phone") or "").strip() or None,
        "partner_dupr_id": str(selection.get("partner_dupr_id") or "").strip() or None,
        "partner_skill": selection.get("partner_skill"),
        "partner_age": selection.get("partner_age"),
        "partner_note": selection.get("partner_note"),
    }


def collapse_duplicate_registrations(
    tournament_id: str,
    registrations: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Merge duplicate registrations by tournament+email, keeping the latest registration record and
    the latest event selection per event option.
    """
    reg_by_id = {str(row.get("id")): deepcopy(row) for row in registrations}
    selections_by_reg_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for selection in selections:
        selections_by_reg_id[str(selection.get("registration_id"))].append(deepcopy(selection))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for registration in reg_by_id.values():
        dedupe_key = _normalize_email(registration.get("email")) or str(registration.get("id"))
        grouped[dedupe_key].append(registration)

    merged_regs: list[dict[str, Any]] = []
    merged_selections: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for _, group in grouped.items():
        ordered = sorted(group, key=lambda row: _parse_dt(row.get("submitted_at") or row.get("updated_at") or row.get("created_at")))
        latest = deepcopy(ordered[-1])
        selection_by_event: dict[str, dict[str, Any]] = {}

        for index, registration in enumerate(ordered):
            reg_id = str(registration.get("id"))
            for selection in selections_by_reg_id.get(reg_id, []):
                event_id = str(selection.get("event_option_id") or selection.get("id"))
                selection_by_event[event_id] = deepcopy(selection)
            if index < len(ordered) - 1:
                issues.append(
                    _issue(
                        str(tournament_id),
                        "DUPLICATE_SUBMISSION",
                        "warning",
                        f"{latest.get('display_name') or latest.get('email')} submitted more than once. Latest player record and latest selection per division were used.",
                        registration_id=reg_id,
                    )
                )

        latest["_collapsed_from_ids"] = [str(row.get("id")) for row in ordered]
        latest["_selection_count"] = len(selection_by_event)
        merged_regs.append(latest)

        for selection in selection_by_event.values():
            selection["registration_id"] = str(latest.get("id"))
            merged_selections.append(selection)

    return merged_regs, merged_selections, issues


def _compile_singles_roster(
    tournament_id: str,
    day: dict[str, Any],
    event: dict[str, Any],
    selections: list[dict[str, Any]],
    reg_lookup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = sorted(
        selections,
        key=lambda row: _parse_dt(reg_lookup.get(str(row.get("registration_id")), {}).get("submitted_at")),
    )
    for selection in ordered:
        registration = reg_lookup.get(str(selection.get("registration_id")), {})
        rows.append(
            {
                "id": _uid("roster"),
                "tournament_id": str(tournament_id),
                "event_day_id": str(day.get("id")),
                "event_day_label": str(day.get("label") or ""),
                "event_option_id": str(event.get("id")),
                "event_label": str(event.get("label") or ""),
                "status": "CONFIRMED",
                "members": [_to_member_from_registration(registration, selection)],
                **_entry_identity(selection, registration),
                "submitted_at": registration.get("submitted_at"),
                "sort_key": _parse_dt(registration.get("submitted_at")),
            }
        )
    return rows


def _compile_doubles_roster(
    tournament_id: str,
    day: dict[str, Any],
    event: dict[str, Any],
    selections: list[dict[str, Any]],
    reg_lookup: dict[str, dict[str, Any]],
    *,
    partner_requests: list[dict[str, Any]] | None = None,
    partner_links: list[dict[str, Any]] | None = None,
    team_members: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    partner_board: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    confirmed_selection_ids: set[str] = set()
    pending_selection_ids: set[str] = set()

    selection_lookup = {str(row.get("id")): row for row in selections}
    event_id = str(event.get("id"))
    event_requests = [row for row in (partner_requests or []) if str(row.get("event_option_id")) == event_id]
    event_links = [row for row in (partner_links or []) if str(row.get("event_option_id")) == event_id]
    members_by_link: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in team_members or []:
        if str(member.get("event_option_id")) != event_id:
            continue
        members_by_link[str(member.get("team_link_id"))].append(member)

    def _member_for_selection_id(selection_id: str) -> dict[str, Any] | None:
        selection = selection_lookup.get(str(selection_id))
        if not selection:
            return None
        registration = reg_lookup.get(str(selection.get("registration_id")), {})
        return _to_member_from_registration(registration, selection)

    def _entry_sort_for_selection_ids(selection_ids: list[str]) -> datetime:
        sort_values = []
        for selection_id in selection_ids:
            selection = selection_lookup.get(str(selection_id))
            if not selection:
                continue
            registration = reg_lookup.get(str(selection.get("registration_id")), {})
            sort_values.append(_parse_dt(registration.get("submitted_at")))
        return min(sort_values) if sort_values else datetime.min

    for link in event_links:
        status = str(link.get("status") or "").upper()
        if status not in {"CONFIRMED", "ADMIN_CONFIRMED"}:
            continue
        link_id = str(link.get("id") or "")
        link_members = sorted(members_by_link.get(link_id, []), key=lambda row: int(row.get("player_order") or 0))
        selection_ids = [str(row.get("selection_id")) for row in link_members if str(row.get("selection_id") or "")]
        if not selection_ids:
            selection_ids = [str(link.get("selection1_id") or ""), str(link.get("selection2_id") or "")]
        selection_ids = [sid for sid in selection_ids if sid in selection_lookup]
        if len(selection_ids) < 2:
            issues.append(
                _issue(
                    str(tournament_id),
                    "CONFIRMED_LINK_MISSING_SELECTION",
                    "warning",
                    f"Confirmed partner link {link_id or 'unknown'} references missing selections for {event.get('label')}.",
                    event_option_id=event_id,
                )
            )
            continue
        members = [_member_for_selection_id(selection_id) for selection_id in selection_ids]
        members = [member for member in members if member]
        registrations = [reg_lookup.get(str(selection_lookup[sid].get("registration_id")), {}) for sid in selection_ids]
        rows.append(
            {
                "id": _uid("roster"),
                "tournament_id": str(tournament_id),
                "event_day_id": str(day.get("id")),
                "event_day_label": str(day.get("label") or ""),
                "event_option_id": event_id,
                "event_label": str(event.get("label") or ""),
                "status": status,
                "entry_type": "confirmed_team",
                "partner_link_id": link_id or None,
                "accepted_request_id": link.get("accepted_request_id"),
                "members": members,
                "source_registration_ids": [str(row.get("id")) for row in registrations if row],
                "source_selection_ids": selection_ids,
                "source_player_ids": [selection_lookup[sid].get("player_id") or reg_lookup.get(str(selection_lookup[sid].get("registration_id")), {}).get("player_id") for sid in selection_ids],
                "submitted_at": min((row.get("submitted_at") or "" for row in registrations if row), default=""),
                "sort_key": _entry_sort_for_selection_ids(selection_ids),
            }
        )
        confirmed_selection_ids.update(selection_ids)

    for request in event_requests:
        if str(request.get("status") or "").upper() != "PENDING":
            continue
        requester_id = str(request.get("requester_selection_id") or "")
        target_id = str(request.get("target_selection_id") or "")
        selection_ids = [sid for sid in [requester_id, target_id] if sid and sid in selection_lookup and sid not in confirmed_selection_ids]
        if not selection_ids:
            continue
        pending_selection_ids.update(selection_ids)
        members = [_member_for_selection_id(selection_id) for selection_id in selection_ids]
        rows.append(
            {
                "id": _uid("roster"),
                "tournament_id": str(tournament_id),
                "event_day_id": str(day.get("id")),
                "event_day_label": str(day.get("label") or ""),
                "event_option_id": event_id,
                "event_label": str(event.get("label") or ""),
                "status": "PENDING_PARTNER_REQUEST",
                "entry_type": "pending_partner_request",
                "partner_request_id": str(request.get("id") or "") or None,
                "members": [member for member in members if member],
                "source_registration_ids": [str(selection_lookup[sid].get("registration_id")) for sid in selection_ids],
                "source_selection_ids": selection_ids,
                "source_player_ids": [selection_lookup[sid].get("player_id") or reg_lookup.get(str(selection_lookup[sid].get("registration_id")), {}).get("player_id") for sid in selection_ids],
                "target_display_name_snapshot": request.get("target_display_name_snapshot"),
                "submitted_at": request.get("created_at"),
                "sort_key": _entry_sort_for_selection_ids(selection_ids),
            }
        )

    ordered = sorted(
        selections,
        key=lambda row: _parse_dt(reg_lookup.get(str(row.get("registration_id")), {}).get("submitted_at")),
    )

    for selection in ordered:
        selection_id = str(selection.get("id"))
        if selection_id in confirmed_selection_ids:
            continue

        registration = reg_lookup.get(str(selection.get("registration_id")), {})
        partner_mode = str(selection.get("partner_mode") or "NONE").upper()
        if partner_mode == "NEEDS_PARTNER":
            row = {
                "id": _uid("roster"),
                "tournament_id": str(tournament_id),
                "event_day_id": str(day.get("id")),
                "event_day_label": str(day.get("label") or ""),
                "event_option_id": event_id,
                "event_label": str(event.get("label") or ""),
                "status": "NEEDS_PARTNER",
                "entry_type": "needs_partner",
                "members": [_to_member_from_registration(registration, selection)],
                **_entry_identity(selection, registration),
                "submitted_at": registration.get("submitted_at"),
                "notes": selection.get("partner_note"),
                "sort_key": _parse_dt(registration.get("submitted_at")),
            }
            rows.append(row)
            if bool(selection.get("show_on_partner_board")) and bool(event.get("public_partner_board", True)):
                partner_board.append(
                    {
                        "id": _uid("partner"),
                        "tournament_id": str(tournament_id),
                        "event_day_id": str(day.get("id")),
                        "event_day_label": str(day.get("label") or ""),
                        "event_option_id": event_id,
                        "event_label": str(event.get("label") or ""),
                        "selection_id": selection_id,
                        "registration_id": str(registration.get("id") or ""),
                        "player_id": selection.get("player_id") or registration.get("player_id"),
                        "player": _to_member_from_registration(registration, selection),
                        "note": selection.get("partner_note"),
                        "show_contact_email": True,
                    }
                )
            continue

        if selection_id in pending_selection_ids:
            continue

        if partner_mode == "HAS_PARTNER":
            rows.append(
                {
                    "id": _uid("roster"),
                    "tournament_id": str(tournament_id),
                    "event_day_id": str(day.get("id")),
                    "event_day_label": str(day.get("label") or ""),
                    "event_option_id": event_id,
                    "event_label": str(event.get("label") or ""),
                    "status": "LEGACY_PARTNER_UNRESOLVED",
                    "entry_type": "unresolved_partner",
                    "members": [_to_member_from_registration(registration, selection)],
                    **_entry_identity(selection, registration),
                    "submitted_at": registration.get("submitted_at"),
                    "legacy_partner": _legacy_partner_metadata(selection),
                    "sort_key": _parse_dt(registration.get("submitted_at")),
                }
            )
            issues.append(
                _issue(
                    str(tournament_id),
                    "LEGACY_PARTNER_UNRESOLVED",
                    "warning",
                    f"{registration.get('display_name') or registration.get('email')} listed free-text partner details for {event.get('label')}. Admin review is required before this can become a team.",
                    registration_id=str(registration.get("id")),
                    selection_id=selection_id,
                    event_option_id=event_id,
                )
            )
            continue

        issues.append(
            _issue(
                str(tournament_id),
                "MISSING_PARTNER_DETAILS",
                "blocker",
                f"{registration.get('display_name') or registration.get('email')} selected a doubles event without partner information or a needs-partner request.",
                registration_id=str(registration.get("id")),
                selection_id=selection_id,
                event_option_id=event_id,
            )
        )
        rows.append(
            {
                "id": _uid("roster"),
                "tournament_id": str(tournament_id),
                "event_day_id": str(day.get("id")),
                "event_day_label": str(day.get("label") or ""),
                "event_option_id": event_id,
                "event_label": str(event.get("label") or ""),
                "status": "PARTNER_MISSING",
                "entry_type": "partner_missing",
                "members": [_to_member_from_registration(registration, selection)],
                **_entry_identity(selection, registration),
                "submitted_at": registration.get("submitted_at"),
                "sort_key": _parse_dt(registration.get("submitted_at")),
            }
        )

    return rows, partner_board, issues


def _compile_four_player_team_roster(
    tournament_id: str,
    day: dict[str, Any],
    event: dict[str, Any],
    selections: list[dict[str, Any]],
    reg_lookup: dict[str, dict[str, Any]],
    *,
    four_player_teams: list[dict[str, Any]] | None = None,
    four_player_team_members: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compile the durable four-player team model into the canonical roster.

    A four-player event is intentionally not sent through the legacy doubles
    partner compiler.  Only accepted members become public roster members;
    invited teammate snapshots remain private on the team-registration
    surface until those teammates accept.
    """

    event_id = str(event.get("id") or "")
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    registration_aliases: dict[str, dict[str, Any]] = {}
    for registration in reg_lookup.values():
        registration_aliases[str(registration.get("id") or "")] = registration
        for alias in registration.get("_collapsed_from_ids") or []:
            registration_aliases[str(alias)] = registration

    active_selections = [
        selection
        for selection in selections
        if str(
            registration_aliases.get(
                str(selection.get("registration_id") or ""), {}
            ).get("status")
            or ""
        ).strip().lower()
        not in {"cancelled", "canceled", "withdrawn"}
    ]
    selection_by_registration: dict[str, dict[str, Any]] = {}
    for selection in active_selections:
        registration = registration_aliases.get(
            str(selection.get("registration_id") or "")
        )
        if not registration:
            continue
        selection_by_registration[str(registration.get("id") or "")] = selection
        for alias in registration.get("_collapsed_from_ids") or []:
            selection_by_registration[str(alias)] = selection

    members_by_team: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in four_player_team_members or []:
        if str(member.get("event_option_id") or "") != event_id:
            continue
        if str(member.get("status") or "").upper() == "REMOVED":
            continue
        members_by_team[str(member.get("team_id") or "")].append(member)

    represented_selection_ids: set[str] = set()
    active_team_statuses = {
        "FORMING",
        "CONFIRMED",
        "WAITLIST",
        "REVIEW_REQUIRED",
        "INELIGIBLE",
    }
    event_teams = [
        team
        for team in (four_player_teams or [])
        if str(team.get("event_option_id") or "") == event_id
        and str(team.get("status") or "").upper() in active_team_statuses
    ]
    event_teams.sort(
        key=lambda team: (
            _parse_dt(team.get("created_at")),
            str(team.get("name") or ""),
            str(team.get("id") or ""),
        )
    )

    for team in event_teams:
        team_id = str(team.get("id") or "")
        durable_members = sorted(
            members_by_team.get(team_id, []),
            key=lambda member: (
                str(member.get("slot") or ""),
                str(member.get("id") or ""),
            ),
        )
        accepted_members: list[dict[str, Any]] = []
        source_registration_ids: list[str] = []
        source_selection_ids: list[str] = []
        source_player_ids: list[Any] = []
        submitted_values: list[str] = []

        for member in durable_members:
            if str(member.get("status") or "").upper() != "ACCEPTED":
                continue
            registration_id = str(member.get("registration_id") or "")
            registration = registration_aliases.get(registration_id)
            selection = selection_by_registration.get(registration_id)
            if (
                not registration
                or str(registration.get("status") or "").upper()
                != "CONFIRMED"
            ):
                issues.append(
                    _issue(
                        str(tournament_id),
                        "FOUR_PLAYER_TEAM_ACCEPTED_REGISTRATION_INVALID",
                        "blocker",
                        f"{team.get('name') or 'Four-player team'} has an "
                        "accepted member without a confirmed tournament "
                        "registration.",
                        registration_id=registration_id or None,
                        selection_id=(
                            str(selection.get("id"))
                            if selection
                            else None
                        ),
                        event_option_id=event_id,
                    )
                )
                continue
            compiled_member = _to_member_from_registration(
                registration, selection
            )
            submitted_values.append(str(registration.get("submitted_at") or ""))
            accepted_members.append(compiled_member)
            canonical_registration_id = str(registration.get("id") or "")
            if canonical_registration_id:
                source_registration_ids.append(canonical_registration_id)
            if selection and str(selection.get("id") or ""):
                selection_id = str(selection.get("id"))
                source_selection_ids.append(selection_id)
                represented_selection_ids.add(selection_id)
            player_id = member.get("player_id") or (
                registration.get("player_id") if registration else None
            )
            if player_id is not None:
                source_player_ids.append(player_id)

        captain_registration_id = str(team.get("captain_registration_id") or "")
        captain_selection = selection_by_registration.get(captain_registration_id)
        if captain_selection and str(captain_selection.get("id") or ""):
            captain_selection_id = str(captain_selection.get("id"))
            if captain_selection_id not in source_selection_ids:
                source_selection_ids.append(captain_selection_id)
            represented_selection_ids.add(captain_selection_id)
        captain_registration = registration_aliases.get(captain_registration_id)
        if captain_registration:
            canonical_captain_id = str(captain_registration.get("id") or "")
            if (
                canonical_captain_id
                and canonical_captain_id not in source_registration_ids
            ):
                source_registration_ids.append(canonical_captain_id)
            submitted_values.append(
                str(captain_registration.get("submitted_at") or "")
            )

        team_status = str(team.get("status") or "").upper()
        eligibility = str(team.get("eligibility_state") or "").upper()
        if team_status == "CONFIRMED" and eligibility in {
            "ELIGIBLE",
            "NOT_REQUIRED",
        }:
            roster_status = "CONFIRMED"
        elif team_status == "WAITLIST":
            roster_status = "WAITLIST"
        else:
            roster_status = "REVIEW"

        # A malformed durable team must stay visible to organizers as a
        # blocker, but never manufacture unaccepted public members.
        if roster_status == "CONFIRMED" and len(accepted_members) != 4:
            roster_status = "REVIEW"
            issues.append(
                _issue(
                    str(tournament_id),
                    "FOUR_PLAYER_TEAM_ROSTER_INCOMPLETE",
                    "blocker",
                    f"{team.get('name') or 'Four-player team'} is marked confirmed "
                    "without exactly four accepted members.",
                    registration_id=(
                        str(captain_registration.get("id"))
                        if captain_registration
                        else captain_registration_id or None
                    ),
                    selection_id=(
                        str(captain_selection.get("id"))
                        if captain_selection
                        else None
                    ),
                    event_option_id=event_id,
                )
            )

        # Captain acceptance is a database invariant for created teams, so an
        # empty member list signals corrupt/incomplete setup and must not emit a
        # browser entry with fabricated identity.
        if not accepted_members:
            issues.append(
                _issue(
                    str(tournament_id),
                    "FOUR_PLAYER_TEAM_ACCEPTED_MEMBER_MISSING",
                    "blocker",
                    f"{team.get('name') or 'Four-player team'} has no accepted "
                    "roster member.",
                    registration_id=(
                        str(captain_registration.get("id"))
                        if captain_registration
                        else captain_registration_id or None
                    ),
                    event_option_id=event_id,
                )
            )
            continue

        submitted_at = min((value for value in submitted_values if value), default="")
        rows.append(
            {
                "id": _uid("roster"),
                "tournament_id": str(tournament_id),
                "event_day_id": str(day.get("id") or ""),
                "event_day_label": str(day.get("label") or ""),
                "event_option_id": event_id,
                "event_label": str(event.get("label") or ""),
                "status": roster_status,
                "entry_type": "four_player_team",
                "four_player_team_id": team_id or None,
                "team_name": str(team.get("name") or "").strip() or None,
                "members": accepted_members,
                "source_registration_ids": list(
                    dict.fromkeys(source_registration_ids)
                ),
                "source_selection_ids": list(
                    dict.fromkeys(source_selection_ids)
                ),
                "source_player_ids": list(dict.fromkeys(source_player_ids)),
                "submitted_at": submitted_at,
                "sort_key": _parse_dt(team.get("created_at") or submitted_at),
            }
        )

    for selection in active_selections:
        selection_id = str(selection.get("id") or "")
        if selection_id in represented_selection_ids:
            continue
        registration = registration_aliases.get(
            str(selection.get("registration_id") or ""), {}
        )
        issues.append(
            _issue(
                str(tournament_id),
                "FOUR_PLAYER_TEAM_SETUP_REQUIRED",
                "blocker",
                f"{registration.get('display_name') or registration.get('email') or 'A captain'} "
                f"selected {event.get('label') or 'a four-player event'} but "
                "does not have a durable team setup.",
                registration_id=str(registration.get("id") or "") or None,
                selection_id=selection_id or None,
                event_option_id=event_id,
            )
        )
        rows.append(
            {
                "id": _uid("roster"),
                "tournament_id": str(tournament_id),
                "event_day_id": str(day.get("id") or ""),
                "event_day_label": str(day.get("label") or ""),
                "event_option_id": event_id,
                "event_label": str(event.get("label") or ""),
                "status": "REVIEW",
                "entry_type": "four_player_team_setup_required",
                "members": [_to_member_from_registration(registration, selection)],
                **_entry_identity(selection, registration),
                "submitted_at": registration.get("submitted_at"),
                "sort_key": _parse_dt(registration.get("submitted_at")),
            }
        )

    return rows, issues


def _apply_capacity(
    tournament_id: str,
    event: dict[str, Any],
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    capacity = event.get("capacity_teams")
    if not capacity:
        return entries, []

    issues: list[dict[str, Any]] = []
    waitlist_enabled = bool(event.get("waitlist_enabled", True))
    confirmed_slots = 0
    ordered = sorted(entries, key=lambda row: row.get("sort_key") or datetime.min)
    out: list[dict[str, Any]] = []
    for entry in ordered:
        status = str(entry.get("status") or "")
        if status in {"NEEDS_PARTNER", "PARTNER_MISSING", "LEGACY_PARTNER_UNRESOLVED", "PENDING_PARTNER_REQUEST"}:
            out.append(entry)
            continue
        confirmed_slots += 1
        if confirmed_slots > int(capacity):
            entry = dict(entry)
            entry["status"] = "WAITLIST" if waitlist_enabled else "REVIEW"
            overflow_status = "waitlist" if waitlist_enabled else "manual review"
            issues.append(
                _issue(
                    str(tournament_id),
                    "EVENT_AT_CAPACITY",
                    "warning",
                    f"{event.get('label')} exceeded its configured capacity. Later entry moved to {overflow_status}.",
                    registration_id=(entry.get("source_registration_ids") or [None])[0],
                    event_option_id=str(event.get("id")),
                )
            )
        out.append(entry)
    return out, issues


def compile_tournament_registration_state(
    *,
    tournament: dict[str, Any],
    settings: dict[str, Any] | None,
    days: list[dict[str, Any]],
    event_options: list[dict[str, Any]],
    registrations: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    partner_requests: list[dict[str, Any]] | None = None,
    partner_links: list[dict[str, Any]] | None = None,
    team_members: list[dict[str, Any]] | None = None,
    four_player_teams: list[dict[str, Any]] | None = None,
    four_player_team_members: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    tournament_id = str(tournament.get("id"))
    settings = settings or {}
    days = [deepcopy(row) for row in days]
    event_options = [deepcopy(row) for row in event_options]
    registrations = [deepcopy(row) for row in registrations]
    selections = [deepcopy(row) for row in selections]
    partner_requests = [deepcopy(row) for row in (partner_requests or [])]
    partner_links = [deepcopy(row) for row in (partner_links or [])]
    team_members = [deepcopy(row) for row in (team_members or [])]
    four_player_teams = [deepcopy(row) for row in (four_player_teams or [])]
    four_player_team_members = [
        deepcopy(row) for row in (four_player_team_members or [])
    ]

    day_lookup = {str(row.get("id")): row for row in days}
    event_lookup = {str(row.get("id")): row for row in event_options}

    merged_regs, merged_selections, issues = collapse_duplicate_registrations(
        tournament_id,
        registrations,
        selections,
    )
    reg_lookup = {str(row.get("id")): row for row in merged_regs}

    # Basic validation
    valid_selections: list[dict[str, Any]] = []
    for selection in merged_selections:
        event = event_lookup.get(str(selection.get("event_option_id")))
        day = day_lookup.get(str(selection.get("registration_day_id")))
        if not event or not day:
            issues.append(
                _issue(
                    tournament_id,
                    "UNKNOWN_EVENT",
                    "blocker",
                    f"Selection {selection.get('id')} references a missing day or event configuration.",
                    registration_id=str(selection.get("registration_id")),
                    selection_id=str(selection.get("id")),
                )
            )
            continue
        valid_selections.append(selection)

    selections_by_event: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for selection in valid_selections:
        selections_by_event[str(selection.get("event_option_id"))].append(selection)

    event_rosters: list[dict[str, Any]] = []
    partner_board: list[dict[str, Any]] = []

    for event in sorted(event_options, key=lambda row: _event_sort_key(row, day_lookup)):
        day = day_lookup.get(str(event.get("registration_day_id")), {})
        event_selections = selections_by_event.get(str(event.get("id")), [])
        if str(event.get("competition_format") or "").upper() == "FOUR_PLAYER_TEAM":
            entries, event_issues = _compile_four_player_team_roster(
                tournament_id,
                day,
                event,
                event_selections,
                reg_lookup,
                four_player_teams=four_player_teams,
                four_player_team_members=four_player_team_members,
            )
            issues.extend(event_issues)
        elif _is_doubles_event(event):
            entries, partner_rows, event_issues = _compile_doubles_roster(
                tournament_id,
                day,
                event,
                event_selections,
                reg_lookup,
                partner_requests=partner_requests,
                partner_links=partner_links,
                team_members=team_members,
            )
            issues.extend(event_issues)
            partner_board.extend(partner_rows)
        else:
            entries = _compile_singles_roster(
                tournament_id,
                day,
                event,
                event_selections,
                reg_lookup,
            )
        entries, cap_issues = _apply_capacity(tournament_id, event, entries)
        issues.extend(cap_issues)
        event_rosters.append(
            {
                "event_day_id": str(day.get("id")),
                "event_day_label": str(day.get("label") or ""),
                "event_option_id": str(event.get("id")),
                "event_label": str(event.get("label") or ""),
                "event_type": str(event.get("event_type") or ""),
                "entries": entries,
            }
        )

    summary_entries = [entry for roster in event_rosters for entry in roster.get("entries", [])]
    for entry in summary_entries:
        entry.pop("sort_key", None)

    return {
        "tournament": tournament,
        "settings": settings,
        "days": sorted(days, key=_day_sort_key),
        "event_options": sorted(event_options, key=lambda row: _event_sort_key(row, day_lookup)),
        "registrations": sorted(merged_regs, key=lambda row: _parse_dt(row.get("submitted_at")), reverse=True),
        "event_rosters": event_rosters,
        "partner_board": partner_board,
        "issues": issues,
        "summary": {
            "total_registrations": len(merged_regs),
            "total_selections": len(valid_selections),
            "confirmed_entries": sum(1 for row in summary_entries if row.get("status") == "CONFIRMED"),
            "admin_confirmed_entries": sum(1 for row in summary_entries if row.get("status") == "ADMIN_CONFIRMED"),
            "review_entries": sum(1 for row in summary_entries if row.get("status") == "REVIEW"),
            "waitlist_entries": sum(1 for row in summary_entries if row.get("status") == "WAITLIST"),
            "pending_partner_request_entries": sum(1 for row in summary_entries if row.get("status") == "PENDING_PARTNER_REQUEST"),
            "needs_partner_entries": sum(1 for row in summary_entries if row.get("status") == "NEEDS_PARTNER"),
            "partner_missing_entries": sum(1 for row in summary_entries if row.get("status") == "PARTNER_MISSING"),
            "legacy_partner_unresolved_entries": sum(1 for row in summary_entries if row.get("status") == "LEGACY_PARTNER_UNRESOLVED"),
            "issue_count": len(issues),
            "blocker_count": sum(1 for row in issues if row.get("severity") == "blocker"),
        },
    }
