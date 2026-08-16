from __future__ import annotations

import json
from typing import Any

from jupr_app.services.admin_tournament_guarded_operation import (
    require_tournament_admin_mutation_runtime,
    tournament_admin_mutation_status,
)


CHECK_IN_RPC = "admin_upsert_tournament_registration_check_in"
CHECK_IN_SURFACE = "tournament_live"
ACTIVE_REGISTRATION_STATUSES = {"ACTIVE", "APPROVED", "CONFIRMED", "REGISTERED"}
INACTIVE_REGISTRATION_STATUSES = {"CANCELLED", "CANCELED", "WITHDRAWN", "REJECTED"}
ACTIVE_TEAM_LINK_STATUSES = {"CONFIRMED", "ADMIN_CONFIRMED"}
PAYMENT_READY_STATUSES = {"PAID", "WAIVED"}


class StaleTournamentCheckInError(ValueError):
    """The event-day state changed after the operator loaded it."""


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _query_rows(query: Any, *, label: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(query.execute())
    except Exception as exc:
        raise RuntimeError(f"Could not load tournament check-in {label}.") from exc


def _optional_rows(query: Any) -> list[dict[str, Any]]:
    try:
        return _safe_rows(query.execute())
    except Exception:
        return []


def _clean(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _upper(value: Any, default: str = "") -> str:
    return _clean(value, limit=80).upper() or default


def _safe_int(value: Any) -> int | None:
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _display_name(row: dict[str, Any]) -> str:
    display = _clean(row.get("display_name"), limit=160)
    if display:
        return display
    joined = " ".join(
        part
        for part in (
            _clean(row.get("first_name"), limit=80),
            _clean(row.get("last_name"), limit=80),
        )
        if part
    )
    return joined or _clean(row.get("email"), limit=160) or "Unnamed registrant"


def _registration_status(row: dict[str, Any]) -> str:
    return _upper(row.get("status") or row.get("registration_status"), "CONFIRMED")


def _registration_is_active(row: dict[str, Any]) -> bool:
    status = _registration_status(row)
    if status in INACTIVE_REGISTRATION_STATUSES:
        return False
    return status in ACTIVE_REGISTRATION_STATUSES


def _registration_attendee_identity_key(
    registration: dict[str, Any],
    *,
    substitute_player_id: int | None = None,
) -> str:
    """Mirror the RPC's server-derived identity key for fail-closed reads."""

    if substitute_player_id is not None:
        return f"player:{int(substitute_player_id)}"
    registration_player_id = _safe_int(registration.get("player_id"))
    if registration_player_id is not None:
        return f"player:{registration_player_id}"
    registration_id = str(registration.get("id") or "")
    profile_parts = [
        str(registration.get(field) or "").strip().lower()
        for field in ("display_name", "first_name", "last_name", "email")
    ]
    return "registration:" + ":".join([registration_id, *profile_parts])


def _blocker(code: str, title: str, detail: str, *, severity: str = "BLOCKED") -> dict[str, str]:
    return {
        "code": str(code),
        "status": str(severity),
        "title": str(title),
        "detail": str(detail),
    }


def _substitution_policy(
    *,
    registration_id: str,
    selections: list[dict[str, Any]],
    events_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    selected_event_ids = [
        _clean(row.get("event_option_id"), limit=160)
        for row in selections
        if _clean(row.get("registration_id"), limit=160) == registration_id
        and _clean(row.get("event_option_id"), limit=160)
    ]
    selected_events = [events_by_id.get(event_id) for event_id in selected_event_ids]
    if not selected_event_ids or any(event is None for event in selected_events):
        return {
            "allowed": False,
            "event_policy_allows": False,
            "blocker": _blocker(
                "SUBSTITUTE_POLICY_UNAVAILABLE",
                "Substitute policy unavailable",
                "The registration's selected events cannot be resolved authoritatively, so substitute assignment is disabled.",
            ),
        }
    if any(event.get("team_allow_substitutes") is not True for event in selected_events):
        return {
            "allowed": False,
            "event_policy_allows": False,
            "blocker": _blocker(
                "SUBSTITUTE_POLICY_NOT_ALLOWED",
                "Selected event does not allow substitutes",
                "Every selected event must explicitly allow substitutes. This registration does not meet that policy.",
            ),
        }

    # Selection rows have no enforced registration foreign-key locking
    # contract. A concurrent insert can create a phantom selected event after
    # this read, so the present schema cannot prove atomic eligibility.
    return {
        "allowed": False,
        "event_policy_allows": True,
        "blocker": _blocker(
            "SUBSTITUTE_ASSIGNMENT_ATOMICITY_UNAVAILABLE",
            "Substitute assignment unavailable",
            "Selected events allow substitutes, but atomic eligibility and uniqueness cannot be proven by the current registration schema. New substitute assignment is disabled.",
        ),
    }


def _event_label(event: dict[str, Any]) -> str:
    family = _clean(event.get("event_family_label"), limit=120)
    division = _clean(event.get("division_name") or event.get("label"), limit=120)
    if family and division and family.lower() != division.lower():
        return f"{family} / {division}"
    return division or family or "Unlabeled division"


def _court_labels(value: Any) -> list[str]:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            parsed = [part.strip() for part in value.split(",")]
    if not isinstance(parsed, list):
        return []
    return [_clean(item, limit=80) for item in parsed if _clean(item, limit=80)]


def _schedule_readiness(
    *,
    tournament: dict[str, Any],
    settings: dict[str, Any],
    days: list[dict[str, Any]],
) -> dict[str, Any]:
    blockers: list[dict[str, str]] = []
    timezone_name = _clean(settings.get("timezone"), limit=120)
    if not timezone_name:
        blockers.append(
            _blocker(
                "TIMEZONE_MISSING",
                "Timezone missing",
                "Set the tournament registration timezone before event-day operations.",
            )
        )
    if not _clean(tournament.get("start_date"), limit=40):
        blockers.append(
            _blocker(
                "TOURNAMENT_DATE_MISSING",
                "Tournament date missing",
                "Set the tournament start date before calling players to court.",
            )
        )

    active_days = [row for row in days if row.get("enabled") is not False]
    if not active_days:
        blockers.append(
            _blocker(
                "SCHEDULE_DAY_MISSING",
                "No active event day",
                "Add at least one enabled registration day with its event date and court plan.",
            )
        )
    for day in active_days:
        label = _clean(day.get("label"), limit=120) or "Event day"
        if not _clean(day.get("event_date"), limit=40):
            blockers.append(
                _blocker(
                    "EVENT_DATE_MISSING",
                    f"{label} date missing",
                    "Every active event day needs an authoritative calendar date.",
                )
            )
        court_count = _safe_int(day.get("court_count")) or 0
        labels = _court_labels(day.get("court_labels"))
        if court_count <= 0 and not labels:
            blockers.append(
                _blocker(
                    "COURT_PLAN_MISSING",
                    f"{label} courts missing",
                    "Set a court count or named court list for this event day.",
                )
            )
        if not _clean(day.get("court_open_time"), limit=40) or not _clean(
            day.get("court_close_time"), limit=40
        ):
            blockers.append(
                _blocker(
                    "COURT_TIME_MISSING",
                    f"{label} court hours incomplete",
                    "Set both court opening and closing times for this event day.",
                )
            )

    return {
        "status": "COMPLETE" if not blockers else "BLOCKED",
        "timezone": timezone_name or None,
        "active_day_count": len(active_days),
        "blockers": blockers,
        "days": [
            {
                "id": _clean(day.get("id"), limit=160),
                "label": _clean(day.get("label"), limit=120) or "Event day",
                "event_date": day.get("event_date"),
                "court_count": _safe_int(day.get("court_count")),
                "court_labels": _court_labels(day.get("court_labels")),
                "court_open_time": day.get("court_open_time"),
                "court_close_time": day.get("court_close_time"),
            }
            for day in active_days
        ],
    }


def _draw_readiness(
    *,
    event_options: list[dict[str, Any]],
    draws: list[dict[str, Any]],
) -> dict[str, Any]:
    active_events = [row for row in event_options if row.get("enabled") is not False]
    draw_event_ids = {
        _clean(row.get("event_option_id"), limit=160)
        for row in draws
        if _upper(row.get("status")) not in {"CANCELLED", "CANCELED", "ARCHIVED"}
    }
    blockers = [
        _blocker(
            "DRAW_MISSING",
            f"{_event_label(event)} draw missing",
            "Create or assign a draw before this division can begin play.",
        )
        for event in active_events
        if _clean(event.get("id"), limit=160) not in draw_event_ids
    ]
    return {
        "status": "COMPLETE" if not blockers else "BLOCKED",
        "active_division_count": len(active_events),
        "draw_count": len(draws),
        "blockers": blockers,
    }


def _payment_map(
    registrations: list[dict[str, Any]], orders: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    by_registration: dict[str, dict[str, Any]] = {}
    for order in orders:
        registration_id = _clean(order.get("registration_id"), limit=160)
        if not registration_id or registration_id in by_registration:
            continue
        by_registration[registration_id] = {
            "status": _upper(order.get("payment_status"), "UNPAID"),
            "source": "offline_payment_tracking",
            "ready": _upper(order.get("payment_status"), "UNPAID")
            in PAYMENT_READY_STATUSES,
        }
    for registration in registrations:
        registration_id = _clean(registration.get("id"), limit=160)
        if registration_id not in by_registration:
            status = _upper(registration.get("payment_status"), "UNPAID")
            by_registration[registration_id] = {
                "status": status,
                "source": "offline_registration_record",
                "ready": status in PAYMENT_READY_STATUSES,
            }
    return by_registration


def _relationship_projection(
    *,
    registrations_by_id: dict[str, dict[str, Any]],
    selections: list[dict[str, Any]],
    events_by_id: dict[str, dict[str, Any]],
    links: list[dict[str, Any]],
    members: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    active_links = {
        _clean(row.get("id"), limit=160): row
        for row in links
        if _clean(row.get("id"), limit=160)
        and _upper(row.get("status")) in ACTIVE_TEAM_LINK_STATUSES
    }
    members_by_selection = {
        _clean(row.get("selection_id"), limit=160): row
        for row in members
        if _upper(row.get("status")) == "ACTIVE"
        and _clean(row.get("selection_id"), limit=160)
    }
    members_by_link: dict[str, list[dict[str, Any]]] = {}
    for member in members:
        if _upper(member.get("status")) != "ACTIVE":
            continue
        members_by_link.setdefault(
            _clean(member.get("team_link_id"), limit=160), []
        ).append(member)

    events_by_registration: dict[str, list[dict[str, Any]]] = {}
    unresolved: list[dict[str, Any]] = []
    for selection in selections:
        registration_id = _clean(selection.get("registration_id"), limit=160)
        registration = registrations_by_id.get(registration_id)
        if not registration or not _registration_is_active(registration):
            continue
        selection_id = _clean(selection.get("id"), limit=160)
        event_id = _clean(selection.get("event_option_id"), limit=160)
        event = events_by_id.get(event_id, {})
        partner_required = bool(event.get("partner_required")) or _upper(
            event.get("event_type")
        ) in {"DOUBLES", "MIXED_DOUBLES"}
        partner_mode = _upper(selection.get("partner_mode"), "NONE")
        entered_partner_name = _clean(selection.get("partner_name"), limit=160)
        event_blockers: list[dict[str, str]] = []
        team_state = "NOT_REQUIRED" if not partner_required else "UNRESOLVED"
        partner_name: str | None = None

        member = members_by_selection.get(selection_id)
        link = active_links.get(_clean((member or {}).get("team_link_id"), limit=160))
        if link is not None:
            link_members = members_by_link.get(_clean(link.get("id"), limit=160), [])
            linked_registration_ids = {
                _clean(row.get("registration_id"), limit=160) for row in link_members
            }
            linked_registrations = [
                registrations_by_id.get(linked_id)
                for linked_id in linked_registration_ids
                if linked_id
            ]
            relationship_complete = (
                len(link_members) == 2
                and len(linked_registration_ids) == 2
                and len(linked_registrations) == 2
                and all(
                    row is not None and _registration_is_active(row)
                    for row in linked_registrations
                )
            )
            other = next(
                (
                    row
                    for row in linked_registrations
                    if row is not None
                    and _clean(row.get("id"), limit=160) != registration_id
                ),
                None,
            )
            if relationship_complete and other is not None:
                team_state = "CONFIRMED_LINK"
                partner_name = _display_name(other)
            else:
                event_blockers.append(
                    _blocker(
                        "PARTNER_REGISTRATION_INACTIVE",
                        "Confirmed pair is no longer active",
                        "A linked partner registration is cancelled, withdrawn, missing, or otherwise not confirmed.",
                    )
                )

        if partner_required and team_state != "CONFIRMED_LINK":
            if not event_blockers:
                if partner_mode == "NEEDS_PARTNER":
                    kind = "NEEDS_PARTNER"
                    title = "Partner still needed"
                    detail = "This confirmed registrant has no canonical partner team."
                elif partner_mode == "HAS_PARTNER" and entered_partner_name:
                    kind = "UNLINKED_FREE_TEXT_PARTNER"
                    title = "Entered partner is not linked"
                    detail = (
                        f"{entered_partner_name} is free-text registration data, not a confirmed team link."
                    )
                else:
                    kind = "PARTNER_MISSING"
                    title = "Partner link missing"
                    detail = "This doubles entry has no confirmed canonical partner team."
                event_blockers.append(_blocker(kind, title, detail))
            else:
                kind = event_blockers[0]["code"]
                title = event_blockers[0]["title"]
                detail = event_blockers[0]["detail"]
            unresolved.append(
                {
                    "kind": kind,
                    "registration_id": registration_id,
                    "registration_name": _display_name(registration),
                    "selection_id": selection_id,
                    "event_label": _event_label(event),
                    "entered_partner_name": entered_partner_name or None,
                    "title": title,
                    "detail": detail,
                }
            )

        events_by_registration.setdefault(registration_id, []).append(
            {
                "selection_id": selection_id,
                "event_option_id": event_id,
                "event_label": _event_label(event),
                "team_state": team_state,
                "partner_name": partner_name,
                "entered_partner_name": entered_partner_name or None,
                "blockers": event_blockers,
            }
        )

    return events_by_registration, unresolved


def build_admin_tournament_checkin_snapshot(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
) -> dict[str, Any]:
    tournaments = _query_rows(
        supabase.table("tournaments")
        .select("id,club_id,name,status,start_date,end_date,updated_at")
        .eq("club_id", str(club_id))
        .eq("id", str(tournament_id))
        .limit(1),
        label="tournament",
    )
    if not tournaments:
        raise ValueError("Tournament was not found for this club.")
    tournament = tournaments[0]

    settings_rows = _query_rows(
        supabase.table("tournament_registration_settings")
        .select(
            "id,tournament_id,registration_status,timezone,location_name,venue_address,updated_at"
        )
        .eq("tournament_id", str(tournament_id))
        .limit(1),
        label="registration settings",
    )
    settings = settings_rows[0] if settings_rows else {}
    days = _query_rows(
        supabase.table("tournament_registration_days")
        .select(
            "id,tournament_id,label,event_date,enabled,sort_order,court_count,court_labels,available_court_ids,court_open_time,court_close_time,court_notes"
        )
        .eq("tournament_id", str(tournament_id))
        .order("sort_order"),
        label="schedule",
    )
    event_options = _query_rows(
        supabase.table("tournament_event_options")
        .select(
            "id,tournament_id,registration_day_id,label,event_family_label,division_name,event_type,partner_required,team_allow_substitutes,enabled,status,sort_order"
        )
        .eq("tournament_id", str(tournament_id))
        .order("sort_order"),
        label="event options",
    )
    registrations = _query_rows(
        supabase.table("tournament_registrations")
        .select(
            "id,tournament_id,player_id,first_name,last_name,display_name,email,status,payment_status,submitted_at,updated_at"
        )
        .eq("tournament_id", str(tournament_id))
        .order("display_name"),
        label="registrations",
    )
    selections = _query_rows(
        supabase.table("tournament_registration_selections")
        .select(
            "id,tournament_id,registration_id,registration_day_id,event_option_id,partner_mode,partner_name,updated_at"
        )
        .eq("tournament_id", str(tournament_id)),
        label="event entries",
    )
    links = _query_rows(
        supabase.table("tournament_registration_team_links")
        .select(
            "id,tournament_id,event_option_id,registration1_id,registration2_id,selection1_id,selection2_id,status,updated_at"
        )
        .eq("tournament_id", str(tournament_id)),
        label="partner teams",
    )
    members = _query_rows(
        supabase.table("tournament_registration_team_members")
        .select(
            "id,team_link_id,tournament_id,event_option_id,selection_id,registration_id,player_id,player_order,status"
        )
        .eq("tournament_id", str(tournament_id)),
        label="partner team members",
    )
    check_ins = _query_rows(
        supabase.table("tournament_registration_check_ins")
        .select(
            "id,tournament_id,registration_id,checked_in,waiver_verified,attendee_identity_key,approved_substitute_player_id,approved_substitute_name,notes,updated_by,created_at,updated_at"
        )
        .eq("tournament_id", str(tournament_id)),
        label="durable state",
    )
    players = _query_rows(
        supabase.table("players")
        .select("id,club_id,name,active")
        .eq("club_id", str(club_id)),
        label="players",
    )
    draws = _query_rows(
        supabase.table("tournament_event_draws")
        .select("id,tournament_id,event_option_id,name,status,updated_at")
        .eq("tournament_id", str(tournament_id)),
        label="draw readiness",
    )
    orders = _optional_rows(
        supabase.table("tournament_commerce_orders")
        .select("id,club_id,tournament_id,registration_id,status,payment_status,updated_at")
        .eq("club_id", str(club_id))
        .eq("tournament_id", str(tournament_id))
        .order("updated_at", desc=True)
    )

    registrations_by_id = {
        _clean(row.get("id"), limit=160): row
        for row in registrations
        if _clean(row.get("id"), limit=160)
    }
    events_by_id = {
        _clean(row.get("id"), limit=160): row
        for row in event_options
        if _clean(row.get("id"), limit=160)
    }
    players_by_id = {
        _safe_int(row.get("id")): row for row in players if _safe_int(row.get("id")) is not None
    }
    check_ins_by_registration = {
        _clean(row.get("registration_id"), limit=160): row
        for row in check_ins
        if _clean(row.get("registration_id"), limit=160)
    }
    payment_by_registration = _payment_map(registrations, orders)
    events_by_registration, unresolved_participants = _relationship_projection(
        registrations_by_id=registrations_by_id,
        selections=selections,
        events_by_id=events_by_id,
        links=links,
        members=members,
    )

    active_registrations = [row for row in registrations if _registration_is_active(row)]
    inactive_registrations = [row for row in registrations if not _registration_is_active(row)]
    active_registration_ids = {
        _clean(row.get("id"), limit=160) for row in active_registrations
    }
    active_attendee_identity_counts: dict[str, int] = {}
    for state in check_ins:
        if _clean(state.get("registration_id"), limit=160) not in active_registration_ids:
            continue
        identity_key = _clean(state.get("attendee_identity_key"), limit=1000)
        if identity_key:
            active_attendee_identity_counts[identity_key] = (
                active_attendee_identity_counts.get(identity_key, 0) + 1
            )
    registration_ids_by_player: dict[int, set[str]] = {}
    for registration in registrations:
        player_id = _safe_int(registration.get("player_id"))
        registration_id = _clean(registration.get("id"), limit=160)
        if player_id is not None and registration_id:
            registration_ids_by_player.setdefault(player_id, set()).add(registration_id)

    registrants: list[dict[str, Any]] = []
    for registration in active_registrations:
        registration_id = _clean(registration.get("id"), limit=160)
        substitution = _substitution_policy(
            registration_id=registration_id,
            selections=selections,
            events_by_id=events_by_id,
        )
        state = check_ins_by_registration.get(registration_id, {})
        has_saved_state = bool(_clean(state.get("id"), limit=160))
        substitute_player_id = _safe_int(state.get("approved_substitute_player_id"))
        substitute_player = players_by_id.get(substitute_player_id)
        saved_substitute_name = _clean(
            state.get("approved_substitute_name"), limit=160
        )
        substitute_name = _clean((substitute_player or {}).get("name"), limit=160)
        substitute_player_is_eligible = bool(
            substitute_player_id is not None
            and substitute_player is not None
            and str(substitute_player.get("club_id")) == str(club_id)
            and substitute_player.get("active") is True
            and substitute_name
            and substitute_player_id != _safe_int(registration.get("player_id"))
        )
        has_name_only_substitute = bool(
            substitute_player_id is None and saved_substitute_name
        )
        stored_identity_key = _clean(state.get("attendee_identity_key"), limit=1000)
        duplicate_attendee_identity = bool(
            stored_identity_key
            and active_attendee_identity_counts.get(stored_identity_key, 0) > 1
        )
        substitute_is_registered_elsewhere = bool(
            substitute_player_id is not None
            and any(
                other_registration_id != registration_id
                for other_registration_id in registration_ids_by_player.get(
                    substitute_player_id, set()
                )
            )
        )
        substitute_is_current = bool(
            substitute_player_is_eligible
            and not duplicate_attendee_identity
            and not substitute_is_registered_elsewhere
            and substitution["allowed"] is True
        )
        original_name = _display_name(registration)
        if substitute_player_id is not None:
            attendee_name = (
                substitute_name
                if substitute_player_is_eligible
                else saved_substitute_name or "Unavailable saved substitute"
            )
            attendee_player_id = substitute_player_id
            is_approved_substitute = True
            current_identity_key = (
                _registration_attendee_identity_key(
                    registration,
                    substitute_player_id=substitute_player_id,
                )
                if substitute_is_current
                else None
            )
        elif has_name_only_substitute:
            attendee_name = saved_substitute_name
            attendee_player_id = None
            is_approved_substitute = True
            current_identity_key = None
        else:
            attendee_name = original_name
            attendee_player_id = _safe_int(registration.get("player_id"))
            is_approved_substitute = False
            current_identity_key = _registration_attendee_identity_key(registration)

        identity_current = not has_saved_state or bool(
            current_identity_key
            and stored_identity_key
            and stored_identity_key == current_identity_key
            and not duplicate_attendee_identity
        )
        requires_reconfirmation = bool(has_saved_state and not identity_current)
        checked_in = bool(state.get("checked_in")) and identity_current
        waiver_verified = bool(state.get("waiver_verified")) and identity_current
        payment = payment_by_registration[registration_id]
        card_events = events_by_registration.get(registration_id, [])
        blockers = [
            blocker
            for event in card_events
            for blocker in event.get("blockers", [])
        ]
        if has_saved_state and not identity_current:
            if duplicate_attendee_identity:
                blockers.append(
                    _blocker(
                        "DUPLICATE_ATTENDEE_IDENTITY",
                        "Attendee is assigned more than once",
                        "The same attendee identity appears on multiple active registration check-ins. Attendance and waiver are not trusted until the conflict is resolved.",
                    )
                )
            elif substitute_is_registered_elsewhere:
                blockers.append(
                    _blocker(
                        "SUBSTITUTE_ALREADY_REGISTERED",
                        "Saved substitute is already registered",
                        "A registered tournament participant cannot also attend for another registration as a substitute.",
                    )
                )
            elif substitute_player_is_eligible:
                blockers.append(substitution["blocker"])
            elif substitute_player_id is not None or has_name_only_substitute:
                blockers.append(
                    _blocker(
                        "APPROVED_SUBSTITUTE_INVALID",
                        "Saved attendee is unavailable",
                        "The saved substitute no longer resolves to an active player in this club. Select the original registrant, save, then reconfirm check-in and waiver.",
                    )
                )
            else:
                blockers.append(
                    _blocker(
                        "ATTENDEE_IDENTITY_STALE",
                        "Attendee changed—reconfirmation required",
                        "The registration's attending-player identity changed after this check-in was saved. Check-in and waiver are not trusted until an operator saves and reconfirms them.",
                    )
                )
        if not payment["ready"]:
            blockers.append(
                _blocker(
                    "PAYMENT_UNRESOLVED",
                    "Offline payment needs review",
                    f"The authoritative offline payment status is {payment['status']}.",
                    severity="NEEDS_REVIEW",
                )
            )
        if not waiver_verified:
            blockers.append(
                _blocker(
                    "WAIVER_UNVERIFIED",
                    "Attending-player waiver not verified",
                    f"Verify the waiver for {attendee_name} before play.",
                    severity="NEEDS_REVIEW",
                )
            )
        registrants.append(
            {
                "registration_id": registration_id,
                "registration_status": _registration_status(registration),
                "registration_updated_at": registration.get("updated_at"),
                "original_registrant": {
                    "player_id": _safe_int(registration.get("player_id")),
                    "name": original_name,
                },
                "attendee": {
                    "player_id": attendee_player_id,
                    "name": attendee_name,
                    "is_approved_substitute": is_approved_substitute,
                },
                "substitution": substitution,
                "check_in": {
                    "checked_in": checked_in,
                    "notes": _clean(state.get("notes"), limit=1000) or None,
                    "updated_at": state.get("updated_at"),
                    "updated_by": _clean(state.get("updated_by"), limit=320) or None,
                    "identity_current": identity_current,
                    "requires_reconfirmation": requires_reconfirmation,
                },
                "waiver": {
                    "verified": waiver_verified,
                    "subject": "attending_player",
                    "subject_name": attendee_name,
                },
                "payment": payment,
                "events": card_events,
                "blockers": blockers,
            }
        )

    registrants.sort(
        key=lambda row: (
            not bool((row.get("check_in") or {}).get("checked_in")),
            str((row.get("attendee") or {}).get("name") or "").lower(),
        )
    )
    checked_in_count = sum(
        1 for row in registrants if bool((row.get("check_in") or {}).get("checked_in"))
    )
    unresolved_registration_ids = {
        _clean(row.get("registration_id"), limit=160)
        for row in unresolved_participants
    }
    schedule = _schedule_readiness(tournament=tournament, settings=settings, days=days)
    draw_readiness = _draw_readiness(event_options=event_options, draws=draws)
    staffing = {
        "status": "NEEDS_REVIEW",
        "source": "no_authoritative_staffing_record",
        "blockers": [
            _blocker(
                "STAFFING_REVIEW_REQUIRED",
                "Staffing needs operator review",
                "No authoritative event-day staffing record exists yet. Confirm desk, court, and escalation coverage outside this screen.",
                severity="NEEDS_REVIEW",
            )
        ],
    }
    all_readiness_blockers = [
        *schedule["blockers"],
        *draw_readiness["blockers"],
        *staffing["blockers"],
    ]
    completed_items = [
        {
            "code": "REGISTRATION_STATE_REVIEWED",
            "title": "Registration intake state",
            "status": "COMPLETE"
            if _upper(settings.get("registration_status")) == "CLOSED"
            else "NEEDS_REVIEW",
            "detail": f"Registration is {_clean(settings.get('registration_status'), limit=40) or 'not configured'}.",
        },
        {
            "code": "PARTNER_TEAMS",
            "title": "Partners and teams",
            "status": "COMPLETE" if not unresolved_participants else "BLOCKED",
            "detail": (
                "Every active partner-required entry has a confirmed canonical team link."
                if not unresolved_participants
                else f"{len(unresolved_registration_ids)} registrant(s) have unresolved partner/team state."
            ),
        },
        {
            "code": "SCHEDULE",
            "title": "Dates, courts, and times",
            "status": schedule["status"],
            "detail": (
                "Event-day schedule fields are complete."
                if schedule["status"] == "COMPLETE"
                else f"{len(schedule['blockers'])} schedule blocker(s) remain."
            ),
        },
    ]

    return {
        "ok": True,
        "mode": "tournament_registration_check_in",
        "authority": "python_fastapi_supabase",
        "tournament": {
            "id": _clean(tournament.get("id"), limit=160),
            "name": _clean(tournament.get("name"), limit=180) or "Tournament",
            "status": _upper(tournament.get("status")),
            "start_date": tournament.get("start_date"),
            "end_date": tournament.get("end_date"),
        },
        "summary": {
            "expected": len(registrants),
            "checked_in": checked_in_count,
            "absent": max(0, len(registrants) - checked_in_count),
            "unresolved": len(unresolved_registration_ids),
        },
        "registrants": registrants,
        "inactive_registrants": [
            {
                "registration_id": _clean(row.get("id"), limit=160),
                "name": _display_name(row),
                "registration_status": _registration_status(row),
            }
            for row in inactive_registrations
        ],
        "unresolved_participants": unresolved_participants,
        "player_options": [
            {
                "id": int(player_id),
                "name": _clean(player.get("name"), limit=160),
            }
            for player_id, player in sorted(
                players_by_id.items(),
                key=lambda item: _clean(item[1].get("name"), limit=160).lower(),
            )
            if player_id is not None
            and str(player.get("club_id")) == str(club_id)
            and player.get("active") is True
            and _clean(player.get("name"), limit=160)
        ],
        "readiness": {
            "schedule": schedule,
            "draws": draw_readiness,
            "staffing": staffing,
        },
        "completed_items": completed_items,
        "blockers": all_readiness_blockers,
        "runtime": tournament_admin_mutation_status(),
    }


def _rpc_payload(response: Any) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, dict):
        return dict(data)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return dict(data[0])
    return {}


def _public_check_in_row(row: dict[str, Any]) -> dict[str, Any]:
    """Keep internal identity fingerprints and creator metadata server-only."""

    return {
        "registration_id": _clean(row.get("registration_id"), limit=160),
        "checked_in": bool(row.get("checked_in")),
        "waiver_verified": bool(row.get("waiver_verified")),
        "approved_substitute_player_id": _safe_int(
            row.get("approved_substitute_player_id")
        ),
        "approved_substitute_name": _clean(
            row.get("approved_substitute_name"), limit=160
        )
        or None,
        "notes": _clean(row.get("notes"), limit=1000) or None,
        "updated_by": _clean(row.get("updated_by"), limit=320) or None,
        "updated_at": row.get("updated_at"),
    }


def update_admin_tournament_checkin(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str,
    expected_updated_at: str | None,
    checked_in: bool,
    waiver_verified: bool,
    approved_substitute_player_id: int | None,
    approved_substitute_name: str | None,
    notes: str | None,
    actor_email: str,
    actor_role: str,
) -> dict[str, Any]:
    del actor_role  # authorization is enforced by the API; the durable row records the actor.
    require_tournament_admin_mutation_runtime(CHECK_IN_SURFACE)

    tournaments = _query_rows(
        supabase.table("tournaments")
        .select("id,club_id")
        .eq("club_id", str(club_id))
        .eq("id", str(tournament_id))
        .limit(1),
        label="tournament scope",
    )
    if not tournaments:
        raise ValueError("Tournament was not found for this club.")
    registrations = _query_rows(
        supabase.table("tournament_registrations")
        .select("id,tournament_id,player_id,status")
        .eq("tournament_id", str(tournament_id))
        .eq("id", str(registration_id))
        .limit(1),
        label="registration scope",
    )
    if not registrations:
        raise ValueError("Registration was not found for this tournament.")
    if not _registration_is_active(registrations[0]):
        raise ValueError(
            "Only active, approved, confirmed, or registered entries can be checked in."
        )

    substitute_id = _safe_int(approved_substitute_player_id)
    requested_substitute_name = _clean(approved_substitute_name, limit=160) or None
    if approved_substitute_player_id is not None and substitute_id is None:
        raise ValueError("Approved substitute player id is invalid.")
    if substitute_id is None and requested_substitute_name:
        raise ValueError(
            "Select an active club player as the approved substitute; a typed name is not authoritative."
        )
    if substitute_id is not None:
        policy_selections = _query_rows(
            supabase.table("tournament_registration_selections")
            .select("id,tournament_id,registration_id,event_option_id")
            .eq("tournament_id", str(tournament_id))
            .eq("registration_id", str(registration_id)),
            label="substitute event selections",
        )
        policy_events = _query_rows(
            supabase.table("tournament_event_options")
            .select("id,tournament_id,team_allow_substitutes")
            .eq("tournament_id", str(tournament_id)),
            label="substitute event policy",
        )
        policy = _substitution_policy(
            registration_id=str(registration_id),
            selections=policy_selections,
            events_by_id={
                _clean(row.get("id"), limit=160): row for row in policy_events
            },
        )
        raise ValueError(str(policy["blocker"]["detail"]))

    params = {
        "p_club_id": str(club_id),
        "p_tournament_id": str(tournament_id),
        "p_registration_id": str(registration_id),
        "p_expected_updated_at": str(expected_updated_at)
        if expected_updated_at
        else None,
        "p_checked_in": bool(checked_in),
        "p_waiver_verified": bool(waiver_verified),
        "p_approved_substitute_player_id": substitute_id,
        # Kept as a null RPC argument for signature compatibility. PostgreSQL
        # rejects name-only substitutions and derives the stored name itself.
        "p_approved_substitute_name": None,
        "p_notes": _clean(notes, limit=1000) or None,
        "p_updated_by": _clean(actor_email, limit=320),
    }
    try:
        payload = _rpc_payload(supabase.rpc(CHECK_IN_RPC, params).execute())
    except Exception as exc:
        detail = str(exc)
        lowered = detail.lower()
        if "jupr_check_in_stale" in lowered or "40001" in lowered:
            raise StaleTournamentCheckInError(
                "Check-in changed after it was loaded. Reload the player before saving again."
            ) from exc
        if "jupr_check_in_not_found" in lowered:
            raise ValueError("Registration was not found for this tournament.") from exc
        if "jupr_check_in_inactive" in lowered:
            raise ValueError(
                "Only active, approved, confirmed, or registered entries can be checked in."
            ) from exc
        if "jupr_check_in_substitute" in lowered:
            if "atomicity" in lowered:
                raise ValueError(
                    "Selected events allow substitutes, but atomic eligibility and uniqueness cannot be proven by the current registration schema. New substitute assignment is disabled."
                ) from exc
            if "policy" in lowered:
                raise ValueError(
                    "Every selected event must explicitly allow substitutes before an assignment can be considered."
                ) from exc
            raise ValueError(
                "The approved substitute must be an active player in this club."
            ) from exc
        if "jupr_check_in_invalid" in lowered:
            raise ValueError(detail) from exc
        raise RuntimeError(
            "Tournament check-in storage is unavailable. No safe retry should be attempted until the authoritative state is reloaded."
        ) from exc

    if not payload.get("ok"):
        code = _upper(payload.get("code"))
        if code == "CHECK_IN_STALE":
            raise StaleTournamentCheckInError(
                "Check-in changed after it was loaded. Reload the player before saving again."
            )
        raise RuntimeError("Tournament check-in did not return a durable result.")
    check_in = payload.get("check_in")
    if not isinstance(check_in, dict) or not check_in.get("updated_at"):
        raise RuntimeError("Tournament check-in did not return an authoritative row version.")
    return {
        "ok": True,
        "mode": "tournament_registration_check_in_update",
        "check_in": _public_check_in_row(check_in),
        "attendee_identity_changed": bool(
            payload.get("attendee_identity_changed")
        ),
        "attendance_reset": bool(payload.get("attendance_reset")),
        "message": (
            "Attending player changed. Check-in and waiver verification were reset for safety."
            if payload.get("attendance_reset")
            else "Check-in saved for the reviewed attendee."
        ),
    }


__all__ = [
    "CHECK_IN_RPC",
    "StaleTournamentCheckInError",
    "build_admin_tournament_checkin_snapshot",
    "update_admin_tournament_checkin",
]
