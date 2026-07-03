from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any

from jupr_app.domain.tournament_registration_repo import (
    build_public_tournament_roster_state,
    get_public_tournament_bundle,
    get_registration_confirmation_bundle,
    is_day_enabled,
    list_open_public_tournaments,
    public_event_option_visibility,
    registration_feature_available,
    registration_is_open,
    save_registration,
)

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_PUBLIC_SELECTIONS = 8


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _clean_text(value: Any, *, limit: int = 240) -> str:
    text = str(value or "").replace("<", "").replace(">", "").strip()
    return text[:limit]


def _clean_email(value: Any) -> str:
    return _clean_text(value, limit=320).lower()


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _public_tournament(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "id": str(row.get("id") or ""),
        "name": _clean_text(row.get("name") or "Tournament"),
        "status": _clean_text(row.get("status")),
        "start_date": _json_safe(row.get("start_date")),
        "end_date": _json_safe(row.get("end_date")),
        "event_tags": row.get("event_tags") if isinstance(row.get("event_tags"), dict) else None,
    }


def _public_settings(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not row:
        return None
    return {
        "registration_slug": _clean_text(row.get("registration_slug"), limit=120),
        "registration_status": _clean_text(row.get("registration_status") or "draft", limit=40).lower(),
        "registration_open_at": _json_safe(row.get("registration_open_at")),
        "registration_close_at": _json_safe(row.get("registration_close_at")),
        "waitlist_enabled": _safe_bool(row.get("waitlist_enabled")),
        "partner_board_enabled": _safe_bool(row.get("partner_board_enabled")),
        "rules_markdown": _clean_text(row.get("rules_markdown"), limit=4000),
        "refund_policy_markdown": _clean_text(row.get("refund_policy_markdown"), limit=4000),
        "sponsor_markdown": _clean_text(row.get("sponsor_markdown"), limit=4000),
    }


def _public_day(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "label": _clean_text(row.get("label") or "Day", limit=120),
        "event_date": _json_safe(row.get("event_date")),
        "sort_order": _safe_int(row.get("sort_order")) or 0,
        "enabled": is_day_enabled(row),
    }


def _public_event(row: dict[str, Any], *, registration_open: bool) -> dict[str, Any]:
    visibility = public_event_option_visibility(row)
    event_format = row.get("event_format_override") or row.get("event_format_default")
    scoring = row.get("scoring_override") or row.get("scoring_default")
    return {
        "id": str(row.get("id") or ""),
        "registration_day_id": str(row.get("registration_day_id") or ""),
        "label": _clean_text(row.get("label") or row.get("division_name") or "Division", limit=160),
        "event_family_label": _clean_text(row.get("event_family_label") or row.get("label") or "Event", limit=160),
        "division_name": _clean_text(row.get("division_name") or row.get("label") or "Division", limit=160),
        "event_type": _clean_text(row.get("event_type"), limit=40),
        "gender_restriction": _clean_text(row.get("gender_restriction") or "ANY", limit=40),
        "skill_label": _clean_text(row.get("skill_label"), limit=80),
        "age_label": _clean_text(row.get("age_label"), limit=80),
        "skill_mode": _clean_text(row.get("skill_mode"), limit=80),
        "age_mode": _clean_text(row.get("age_mode"), limit=80),
        "event_format": _clean_text(event_format, limit=120),
        "scoring": _clean_text(scoring, limit=120),
        "capacity_teams": _safe_int(row.get("capacity_teams")),
        "price_usd": _safe_float(row.get("price_usd")),
        "partner_required": _safe_bool(row.get("partner_required")),
        "partner_board_enabled": _safe_bool(row.get("partner_board_enabled", row.get("public_partner_board"))),
        "waitlist_enabled": _safe_bool(row.get("waitlist_enabled", True)),
        "status": _clean_text(row.get("status") or "draft", limit=40).lower(),
        "visibility": visibility,
        "selectable": bool(registration_open and visibility == "selectable"),
    }


def _open_tournament_choices(supabase: Any, *, club_id: str) -> list[dict[str, Any]]:
    choices = []
    for item in list_open_public_tournaments(supabase, str(club_id)):
        tournament = _public_tournament(item.get("tournament") or {})
        settings = _public_settings(item.get("settings") or {})
        if tournament and settings:
            choices.append({"tournament": tournament, "settings": settings})
    return choices


def build_public_tournament_registration_page(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str | None = None,
    registration_slug: str | None = None,
) -> dict[str, Any]:
    available, detail = registration_feature_available(supabase)
    if not available:
        return {
            "available": False,
            "setup_error": detail,
            "tournaments": [],
            "tournament": None,
            "settings": None,
            "registration_open": False,
            "registration_closed_reason": "Registration is not configured.",
            "days": [],
            "events": [],
            "roster_summary": None,
        }

    slug = _clean_text(registration_slug, limit=120)
    tid = _clean_text(tournament_id, limit=120)
    open_choices = _open_tournament_choices(supabase, club_id=str(club_id))
    if not slug and not tid and open_choices:
        tid = str(open_choices[0]["tournament"].get("id") or "")

    tournament, settings, days_raw, events_raw = get_public_tournament_bundle(
        supabase,
        club_id=str(club_id),
        tournament_id=tid or None,
        registration_slug=slug or None,
    )
    if not tournament or not settings:
        return {
            "available": True,
            "setup_error": None,
            "tournaments": open_choices,
            "tournament": None,
            "settings": None,
            "registration_open": False,
            "registration_closed_reason": "No open tournament registration was found.",
            "days": [],
            "events": [],
            "roster_summary": None,
        }

    registration_open, closed_reason = registration_is_open(settings)
    public_days = [_public_day(row) for row in days_raw if is_day_enabled(row)]
    public_day_ids = {row["id"] for row in public_days}
    public_events = [
        _public_event(row, registration_open=registration_open)
        for row in events_raw
        if str(row.get("registration_day_id") or "") in public_day_ids and public_event_option_visibility(row) != "hidden"
    ]
    public_events.sort(key=lambda item: (item.get("registration_day_id") or "", str(item.get("event_family_label") or ""), str(item.get("division_name") or "")))

    roster_state = build_public_tournament_roster_state(supabase, tournament, settings, days_raw, events_raw)
    return {
        "available": True,
        "setup_error": None,
        "tournaments": open_choices,
        "tournament": _public_tournament(tournament),
        "settings": _public_settings(settings),
        "registration_open": bool(registration_open),
        "registration_closed_reason": closed_reason,
        "days": public_days,
        "events": public_events,
        "roster_summary": roster_state.get("summary") if isinstance(roster_state, dict) else None,
    }


def _validate_submit_payload(payload: dict[str, Any]) -> None:
    if _clean_text(payload.get("website")):
        raise ValueError("Unable to submit registration.")
    if not _safe_bool(payload.get("terms_accepted")):
        raise ValueError("Please confirm the tournament policies before submitting.")
    email = _clean_email(payload.get("email"))
    if not email or not _EMAIL_RE.match(email):
        raise ValueError("A valid email is required.")
    display_name = _clean_text(payload.get("display_name") or " ".join(part for part in [payload.get("first_name"), payload.get("last_name")] if _clean_text(part)))
    if not display_name:
        raise ValueError("Player name is required.")
    selections = payload.get("selections") or []
    if not isinstance(selections, list) or not selections:
        raise ValueError("Select at least one event.")
    if len(selections) > MAX_PUBLIC_SELECTIONS:
        raise ValueError(f"Select no more than {MAX_PUBLIC_SELECTIONS} events.")


def _clean_selection(selection: dict[str, Any]) -> dict[str, Any]:
    partner_mode = _clean_text(selection.get("partner_mode") or "NONE", limit=40).upper()
    if partner_mode not in {"NONE", "HAS_PARTNER", "NEEDS_PARTNER"}:
        partner_mode = "NONE"
    return {
        "event_option_id": _clean_text(selection.get("event_option_id"), limit=160),
        "registration_day_id": _clean_text(selection.get("registration_day_id"), limit=160),
        "partner_mode": partner_mode,
        "partner_name": _clean_text(selection.get("partner_name"), limit=160),
        "partner_email": _clean_email(selection.get("partner_email")),
        "partner_phone": _clean_text(selection.get("partner_phone"), limit=60),
        "partner_dupr_id": _clean_text(selection.get("partner_dupr_id"), limit=80),
        "partner_skill": _safe_float(selection.get("partner_skill")),
        "partner_age": _safe_int(selection.get("partner_age")),
        "partner_note": _clean_text(selection.get("partner_note"), limit=500),
        "show_on_partner_board": _safe_bool(selection.get("show_on_partner_board")),
    }


def submit_public_tournament_registration(
    supabase: Any,
    *,
    club_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    _validate_submit_payload(payload)
    page = build_public_tournament_registration_page(
        supabase,
        club_id=str(club_id),
        tournament_id=_clean_text(payload.get("tournament_id"), limit=120) or None,
        registration_slug=_clean_text(payload.get("registration_slug"), limit=120) or None,
    )
    if not page.get("available"):
        raise ValueError("Tournament registration is not configured.")
    if not page.get("registration_open"):
        raise ValueError(str(page.get("registration_closed_reason") or "Registration is not open."))
    tournament = page.get("tournament") or {}
    tournament_id = str(tournament.get("id") or "").strip()
    if not tournament_id:
        raise ValueError("Tournament registration was not found.")

    selectable = {str(event.get("id")): event for event in (page.get("events") or []) if event.get("selectable")}
    selections = []
    seen: set[str] = set()
    for raw_selection in payload.get("selections") or []:
        if not isinstance(raw_selection, dict):
            continue
        clean_selection = _clean_selection(raw_selection)
        event_option_id = str(clean_selection.get("event_option_id") or "").strip()
        if not event_option_id:
            continue
        if event_option_id in seen:
            continue
        event = selectable.get(event_option_id)
        if not event:
            raise ValueError("One or more selected events is no longer open for registration.")
        clean_selection["registration_day_id"] = str(event.get("registration_day_id") or clean_selection.get("registration_day_id") or "")
        selections.append(clean_selection)
        seen.add(event_option_id)
    if not selections:
        raise ValueError("Select at least one open event.")

    save_payload = {
        "first_name": _clean_text(payload.get("first_name"), limit=80),
        "last_name": _clean_text(payload.get("last_name"), limit=80),
        "display_name": _clean_text(payload.get("display_name"), limit=160),
        "email": _clean_email(payload.get("email")),
        "phone": _clean_text(payload.get("phone"), limit=60),
        "player_id": payload.get("player_id"),
        "dupr_id": _clean_text(payload.get("dupr_id"), limit=80),
        "doubles_skill": _safe_float(payload.get("doubles_skill")),
        "singles_skill": _safe_float(payload.get("singles_skill")),
        "age": _safe_int(payload.get("age")),
        "gender": _clean_text(payload.get("gender"), limit=40),
        "notes": _clean_text(payload.get("notes"), limit=800),
        "wants_partner_board_contact": _safe_bool(payload.get("wants_partner_board_contact")),
        "selections": selections,
    }
    result = save_registration(supabase, tournament_id=tournament_id, payload=save_payload)
    return {
        "ok": True,
        "tournament": tournament,
        "settings": page.get("settings"),
        "registration_id": result.get("registration_id"),
        "submitted_at": result.get("submitted_at"),
        "selection_count": result.get("selection_count"),
    }


def build_public_tournament_registration_confirmation(
    supabase: Any,
    *,
    club_id: str,
    registration_id: str,
    tournament_id: str | None = None,
    registration_slug: str | None = None,
) -> dict[str, Any] | None:
    page = build_public_tournament_registration_page(
        supabase,
        club_id=str(club_id),
        tournament_id=tournament_id,
        registration_slug=registration_slug,
    )
    tournament = page.get("tournament") or {}
    tid = str(tournament.get("id") or tournament_id or "").strip()
    if not tid:
        return None
    bundle = get_registration_confirmation_bundle(supabase, tid, str(registration_id))
    registration = bundle.get("registration") or None
    if not registration:
        return None
    if str((bundle.get("tournament") or {}).get("club_id") or club_id) != str(club_id):
        return None
    event_lookup = {str(row.get("id")): row for row in (bundle.get("event_options") or [])}
    day_lookup = {str(row.get("id")): row for row in (bundle.get("days") or [])}
    selections = []
    for selection in bundle.get("selections") or []:
        event = event_lookup.get(str(selection.get("event_option_id") or "")) or {}
        day = day_lookup.get(str(selection.get("registration_day_id") or "")) or {}
        selections.append(
            {
                "selection_id": str(selection.get("id") or ""),
                "event_label": _clean_text(event.get("division_name") or event.get("label") or "Division"),
                "event_family_label": _clean_text(event.get("event_family_label") or event.get("label") or "Event"),
                "day_label": _clean_text(day.get("label") or "Day"),
                "partner_mode": _clean_text(selection.get("partner_mode")),
                "partner_name": _clean_text(selection.get("partner_name")),
                "show_on_partner_board": _safe_bool(selection.get("show_on_partner_board")),
            }
        )
    return {
        "tournament": _public_tournament(bundle.get("tournament") or {}),
        "settings": _public_settings(bundle.get("settings") or {}),
        "registration": {
            "id": str(registration.get("id") or ""),
            "display_name": _clean_text(registration.get("display_name") or "Player"),
            "email": _clean_email(registration.get("email")),
            "status": _clean_text(registration.get("status")),
            "payment_status": _clean_text(registration.get("payment_status")),
            "submitted_at": _json_safe(registration.get("submitted_at")),
        },
        "selections": selections,
        "total_price_usd": float(bundle.get("total_price_usd") or 0),
    }
