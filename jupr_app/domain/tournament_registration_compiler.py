from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any
import uuid


DoublesTypes = {"GENDER_DOUBLES", "MIXED_DOUBLES", "DOUBLES", "MIXED"}


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _normalize_email(value: Any) -> str:
    return str(value or "").strip().lower()


def _normalize_name(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


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
    event_type = str(event.get("event_type") or "").upper()
    if event_type in DoublesTypes:
        return True
    return bool(event.get("partner_required"))


def _to_member_from_registration(registration: dict[str, Any]) -> dict[str, Any]:
    display_name = str(registration.get("display_name") or "").strip()
    if not display_name:
        display_name = " ".join(
            part for part in [str(registration.get("first_name") or "").strip(), str(registration.get("last_name") or "").strip()] if part
        ).strip()
    if not display_name:
        display_name = str(registration.get("email") or "Player")
    return {
        "display_name": display_name,
        "email": _normalize_email(registration.get("email")),
        "phone": str(registration.get("phone") or "").strip() or None,
        "dupr_id": str(registration.get("dupr_id") or "").strip() or None,
        "skill": registration.get("doubles_skill") or registration.get("singles_skill"),
        "age": registration.get("age"),
        "gender": registration.get("gender"),
        "age_bracket": registration.get("age_bracket"),
    }


def _to_member_from_partner(selection: dict[str, Any]) -> dict[str, Any] | None:
    partner_name = str(selection.get("partner_name") or "").strip()
    partner_email = _normalize_email(selection.get("partner_email"))
    partner_phone = str(selection.get("partner_phone") or "").strip() or None
    partner_dupr = str(selection.get("partner_dupr_id") or "").strip() or None
    partner_skill = selection.get("partner_skill")
    partner_age = selection.get("partner_age")

    if not any([partner_name, partner_email, partner_phone, partner_dupr, partner_skill, partner_age]):
        return None

    return {
        "display_name": partner_name or "Partner TBD",
        "email": partner_email or None,
        "phone": partner_phone,
        "dupr_id": partner_dupr,
        "skill": partner_skill,
        "age": partner_age,
    }


def _selection_references_registration(selection: dict[str, Any], registration: dict[str, Any]) -> bool:
    if str(selection.get("partner_mode") or "").upper() != "HAS_PARTNER":
        return False

    partner_email = _normalize_email(selection.get("partner_email"))
    partner_name = _normalize_name(selection.get("partner_name"))
    reg_email = _normalize_email(registration.get("email"))
    reg_name = _normalize_name(registration.get("display_name"))

    if partner_email and reg_email and partner_email == reg_email:
        return True
    if partner_name and reg_name and partner_name == reg_name:
        return True
    return False


def collapse_duplicate_registrations(
    tournament_id: str,
    registrations: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Merge duplicate registrations by tournament+email, keeping the latest registration record and
    the latest event selection per day.
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
        selection_by_day: dict[str, dict[str, Any]] = {}

        for index, registration in enumerate(ordered):
            reg_id = str(registration.get("id"))
            for selection in selections_by_reg_id.get(reg_id, []):
                day_id = str(selection.get("registration_day_id"))
                selection_by_day[day_id] = deepcopy(selection)
            if index < len(ordered) - 1:
                issues.append(
                    _issue(
                        str(tournament_id),
                        "DUPLICATE_SUBMISSION",
                        "warning",
                        f"{latest.get('display_name') or latest.get('email')} submitted more than once. Latest player record and latest selection per day were used.",
                        registration_id=reg_id,
                    )
                )

        latest["_collapsed_from_ids"] = [str(row.get("id")) for row in ordered]
        latest["_selection_count"] = len(selection_by_day)
        merged_regs.append(latest)

        for selection in selection_by_day.values():
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
                "members": [_to_member_from_registration(registration)],
                "source_registration_ids": [str(registration.get("id"))],
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    partner_board: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    processed: set[str] = set()

    ordered = sorted(
        selections,
        key=lambda row: _parse_dt(reg_lookup.get(str(row.get("registration_id")), {}).get("submitted_at")),
    )

    for selection in ordered:
        selection_id = str(selection.get("id"))
        if selection_id in processed:
            continue

        registration = reg_lookup.get(str(selection.get("registration_id")), {})
        partner_mode = str(selection.get("partner_mode") or "NONE").upper()
        if partner_mode == "NEEDS_PARTNER":
            row = {
                "id": _uid("roster"),
                "tournament_id": str(tournament_id),
                "event_day_id": str(day.get("id")),
                "event_day_label": str(day.get("label") or ""),
                "event_option_id": str(event.get("id")),
                "event_label": str(event.get("label") or ""),
                "status": "NEEDS_PARTNER",
                "members": [_to_member_from_registration(registration)],
                "source_registration_ids": [str(registration.get("id"))],
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
                        "event_option_id": str(event.get("id")),
                        "event_label": str(event.get("label") or ""),
                        "player": _to_member_from_registration(registration),
                        "note": selection.get("partner_note"),
                        "show_contact_email": True,
                    }
                )
            processed.add(selection_id)
            continue

        if partner_mode == "HAS_PARTNER":
            candidate = None
            for other in ordered:
                other_id = str(other.get("id"))
                if other_id == selection_id or other_id in processed:
                    continue
                other_reg = reg_lookup.get(str(other.get("registration_id")), {})
                if _selection_references_registration(selection, other_reg) or _selection_references_registration(other, registration):
                    candidate = other
                    break

            if candidate is None:
                partner_member = _to_member_from_partner(selection)
                rows.append(
                    {
                        "id": _uid("roster"),
                        "tournament_id": str(tournament_id),
                        "event_day_id": str(day.get("id")),
                        "event_day_label": str(day.get("label") or ""),
                        "event_option_id": str(event.get("id")),
                        "event_label": str(event.get("label") or ""),
                        "status": "PARTNER_MISSING",
                        "members": [
                            _to_member_from_registration(registration),
                            *( [partner_member] if partner_member else [] ),
                        ],
                        "source_registration_ids": [str(registration.get("id"))],
                        "submitted_at": registration.get("submitted_at"),
                        "sort_key": _parse_dt(registration.get("submitted_at")),
                    }
                )
                issues.append(
                    _issue(
                        str(tournament_id),
                        "PARTNER_NOT_REGISTERED",
                        "warning",
                        f"{registration.get('display_name') or registration.get('email')} listed a partner for {event.get('label')}, but the partner is not registered in the same event.",
                        registration_id=str(registration.get("id")),
                        selection_id=selection_id,
                        event_option_id=str(event.get("id")),
                    )
                )
                processed.add(selection_id)
                continue

            candidate_reg = reg_lookup.get(str(candidate.get("registration_id")), {})
            is_mutual = _selection_references_registration(selection, candidate_reg) and _selection_references_registration(candidate, registration)
            status = "CONFIRMED" if is_mutual else "REVIEW"
            if not is_mutual:
                issues.append(
                    _issue(
                        str(tournament_id),
                        "ONE_SIDED_PARTNER_MATCH",
                        "warning",
                        f"{registration.get('display_name') or registration.get('email')} and {candidate_reg.get('display_name') or candidate_reg.get('email')} appear paired in {event.get('label')}, but only one side declared the partner.",
                        registration_id=str(registration.get("id")),
                        selection_id=selection_id,
                        event_option_id=str(event.get("id")),
                    )
                )
            rows.append(
                {
                    "id": _uid("roster"),
                    "tournament_id": str(tournament_id),
                    "event_day_id": str(day.get("id")),
                    "event_day_label": str(day.get("label") or ""),
                    "event_option_id": str(event.get("id")),
                    "event_label": str(event.get("label") or ""),
                    "status": status,
                    "members": [
                        _to_member_from_registration(registration),
                        _to_member_from_registration(candidate_reg),
                    ],
                    "source_registration_ids": [str(registration.get("id")), str(candidate_reg.get("id"))],
                    "submitted_at": min(
                        registration.get("submitted_at") or "",
                        candidate_reg.get("submitted_at") or "",
                    ),
                    "sort_key": min(
                        _parse_dt(registration.get("submitted_at")),
                        _parse_dt(candidate_reg.get("submitted_at")),
                    ),
                }
            )
            processed.add(selection_id)
            processed.add(str(candidate.get("id")))
            continue

        issues.append(
            _issue(
                str(tournament_id),
                "MISSING_PARTNER_DETAILS",
                "blocker",
                f"{registration.get('display_name') or registration.get('email')} selected a doubles event without partner information or a needs-partner request.",
                registration_id=str(registration.get("id")),
                selection_id=selection_id,
                event_option_id=str(event.get("id")),
            )
        )
        rows.append(
            {
                "id": _uid("roster"),
                "tournament_id": str(tournament_id),
                "event_day_id": str(day.get("id")),
                "event_day_label": str(day.get("label") or ""),
                "event_option_id": str(event.get("id")),
                "event_label": str(event.get("label") or ""),
                "status": "PARTNER_MISSING",
                "members": [_to_member_from_registration(registration)],
                "source_registration_ids": [str(registration.get("id"))],
                "submitted_at": registration.get("submitted_at"),
                "sort_key": _parse_dt(registration.get("submitted_at")),
            }
        )
        processed.add(selection_id)

    return rows, partner_board, issues


def _apply_capacity(
    tournament_id: str,
    event: dict[str, Any],
    entries: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    capacity = event.get("capacity_teams")
    if not capacity:
        return entries, []

    issues: list[dict[str, Any]] = []
    confirmed_slots = 0
    ordered = sorted(entries, key=lambda row: row.get("sort_key") or datetime.min)
    out: list[dict[str, Any]] = []
    for entry in ordered:
        status = str(entry.get("status") or "")
        if status in {"NEEDS_PARTNER", "PARTNER_MISSING"}:
            out.append(entry)
            continue
        confirmed_slots += 1
        if confirmed_slots > int(capacity):
            entry = dict(entry)
            entry["status"] = "WAITLIST"
            issues.append(
                _issue(
                    str(tournament_id),
                    "EVENT_AT_CAPACITY",
                    "warning",
                    f"{event.get('label')} exceeded its configured capacity. Later entry moved to waitlist.",
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
) -> dict[str, Any]:
    tournament_id = str(tournament.get("id"))
    settings = settings or {}
    days = [deepcopy(row) for row in days]
    event_options = [deepcopy(row) for row in event_options]
    registrations = [deepcopy(row) for row in registrations]
    selections = [deepcopy(row) for row in selections]

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
        if _is_doubles_event(event):
            entries, partner_rows, event_issues = _compile_doubles_roster(
                tournament_id,
                day,
                event,
                event_selections,
                reg_lookup,
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
            "review_entries": sum(1 for row in summary_entries if row.get("status") == "REVIEW"),
            "waitlist_entries": sum(1 for row in summary_entries if row.get("status") == "WAITLIST"),
            "needs_partner_entries": sum(1 for row in summary_entries if row.get("status") == "NEEDS_PARTNER"),
            "partner_missing_entries": sum(1 for row in summary_entries if row.get("status") == "PARTNER_MISSING"),
            "issue_count": len(issues),
            "blocker_count": sum(1 for row in issues if row.get("severity") == "blocker"),
        },
    }
