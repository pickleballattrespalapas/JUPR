from __future__ import annotations

import os
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_registration_repo import (
    ADMIN_PAYMENT_STATUS_OPTIONS,
    ADMIN_REGISTRATION_STATUS_OPTIONS,
    PARTNER_MODE_OPTIONS,
    registration_is_imported_to_draw,
    update_admin_registration,
    update_admin_registration_selection,
)

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
TOURNAMENT_SELECT = "id,club_id,name,status,start_date,end_date,event_tags,created_at,updated_at"
TOURNAMENT_MINIMAL_SELECT = "id,club_id,name,status"
REGISTRATION_SETTINGS_SELECT = "id,tournament_id,registration_slug,registration_status,registration_open_at,registration_close_at,waitlist_enabled,partner_board_enabled,updated_at"
REGISTRATION_SELECT = (
    "id,tournament_id,player_id,first_name,last_name,display_name,email,phone,"
    "status,payment_status,notes,wants_partner_board_contact,created_at,updated_at"
)
REGISTRATION_LEGACY_SELECT = (
    "id,tournament_id,player_id,first_name,last_name,display_name,email,phone,"
    "registration_status,payment_status,wants_partner_board_contact,created_at,updated_at"
)
REGISTRATION_MINIMAL_SELECT = "id,tournament_id,display_name,email,status,payment_status,created_at,updated_at"
REGISTRATION_LEGACY_MINIMAL_SELECT = "id,tournament_id,display_name,email,registration_status,payment_status,created_at,updated_at"
SELECTION_SELECT = (
    "id,tournament_id,registration_id,registration_day_id,event_option_id,partner_mode,"
    "partner_name,partner_email,partner_phone,partner_note,show_on_partner_board,created_at,updated_at"
)
EVENT_OPTION_SELECT = (
    "id,tournament_id,registration_day_id,event_family_label,division_name,event_format_default,"
    "scoring_default,skill_mode,age_mode,status,enabled,waitlist_enabled,partner_board_enabled,sort_order"
)
DAY_SELECT = "id,tournament_id,label,date,start_date,end_date,enabled,sort_order"
CONFIRM_REGISTRATION_UPDATE = "SAVE REGISTRATION"
CONFIRM_SELECTION_UPDATE = "SAVE SELECTION"


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_tournament_admin_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS")


def is_api_audit_log_required() -> bool:
    return _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG")


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _query_rows(query: Any) -> list[dict[str, Any]]:
    return _safe_rows(query.execute())


def _fetch_tournament_rows(supabase: Any, *, club_id: str, include_archived: bool = False) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    try:
        query = supabase.table("tournaments").select(TOURNAMENT_SELECT).eq("club_id", str(club_id)).order("created_at", desc=True)
        rows = _query_rows(query)
    except Exception as exc:
        warnings.append(f"Fell back to minimal tournament columns: {exc.__class__.__name__}")
        try:
            rows = _query_rows(
                supabase.table("tournaments")
                .select(TOURNAMENT_MINIMAL_SELECT)
                .eq("club_id", str(club_id))
                .order("name", desc=False)
            )
        except Exception as fallback_exc:
            return [], [*warnings, f"Could not load tournaments: {fallback_exc.__class__.__name__}"]
    if not include_archived:
        rows = [row for row in rows if str(row.get("status") or "").upper() != "ARCHIVED"]
    return rows, warnings


def _first_row(supabase: Any, table_name: str, select_expr: str, *, key: str, value: Any) -> dict[str, Any] | None:
    try:
        rows = _query_rows(supabase.table(table_name).select(select_expr).eq(key, value).limit(1))
    except Exception:
        return None
    return rows[0] if rows else None


def _table_rows_for_tournament(supabase: Any, table_name: str, select_expr: str, *, tournament_id: str) -> list[dict[str, Any]]:
    try:
        return _query_rows(supabase.table(table_name).select(select_expr).eq("tournament_id", str(tournament_id)))
    except Exception:
        return []


def _registration_rows(supabase: Any, *, tournament_id: str, limit: int = 500) -> list[dict[str, Any]]:
    for select_expr in (REGISTRATION_SELECT, REGISTRATION_LEGACY_SELECT, REGISTRATION_MINIMAL_SELECT, REGISTRATION_LEGACY_MINIMAL_SELECT):
        try:
            return _query_rows(
                supabase.table("tournament_registrations")
                .select(select_expr)
                .eq("tournament_id", str(tournament_id))
                .order("created_at", desc=True)
                .limit(int(limit))
            )
        except Exception:
            continue
    return []


def _registration_status(row: dict[str, Any]) -> str:
    return _clean_text(row.get("status") or row.get("registration_status") or "confirmed", limit=40) or "confirmed"


def _display_name(row: dict[str, Any]) -> str:
    display = _clean_text(row.get("display_name"), limit=160)
    if display:
        return display
    name = " ".join(part for part in [_clean_text(row.get("first_name"), limit=80), _clean_text(row.get("last_name"), limit=80)] if part)
    return name or _clean_text(row.get("email"), limit=160) or "Unnamed registrant"


def _registration_payload(row: dict[str, Any], *, selection_count: int = 0) -> dict[str, Any]:
    return {
        "id": _clean_text(row.get("id"), limit=120),
        "player_id": row.get("player_id"),
        "display_name": _display_name(row),
        "email": _clean_text(row.get("email"), limit=180),
        "phone": _clean_text(row.get("phone"), limit=80),
        "registration_status": _registration_status(row),
        "payment_status": _clean_text(row.get("payment_status") or "unpaid", limit=40),
        "notes": _clean_text(row.get("notes"), limit=1000),
        "wants_partner_board_contact": _safe_bool(row.get("wants_partner_board_contact"), default=False),
        "selection_count": int(selection_count),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _event_label(row: dict[str, Any] | None) -> str:
    row = row or {}
    family = _clean_text(row.get("event_family_label"), limit=120)
    division = _clean_text(row.get("division_name") or row.get("label"), limit=120)
    if family and division and family != division:
        return f"{family} / {division}"
    return division or family or _clean_text(row.get("id"), limit=80) or "Event"


def _event_option_map(event_options: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_clean_text(row.get("id"), limit=120): row for row in event_options if _clean_text(row.get("id"), limit=120)}


def _selection_payload(row: dict[str, Any], *, event_options: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    event_options_by_id = _event_option_map(event_options or [])
    event_option_id = _clean_text(row.get("event_option_id"), limit=120)
    event_option = event_options_by_id.get(event_option_id, {})
    return {
        "id": _clean_text(row.get("id"), limit=120),
        "registration_id": _clean_text(row.get("registration_id"), limit=120),
        "registration_day_id": _clean_text(row.get("registration_day_id"), limit=120),
        "event_option_id": event_option_id,
        "event_label": _event_label(event_option),
        "partner_mode": _clean_text(row.get("partner_mode") or "NONE", limit=40).upper(),
        "partner_name": _clean_text(row.get("partner_name"), limit=160),
        "partner_email": _clean_text(row.get("partner_email"), limit=180),
        "partner_phone": _clean_text(row.get("partner_phone"), limit=80),
        "partner_note": _clean_text(row.get("partner_note"), limit=500),
        "show_on_partner_board": _safe_bool(row.get("show_on_partner_board"), default=False),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _summary_counts(registrations: list[dict[str, Any]], selections: list[dict[str, Any]]) -> dict[str, Any]:
    by_registration_status: dict[str, int] = {}
    by_payment_status: dict[str, int] = {}
    for row in registrations:
        registration_status = _registration_status(row)
        payment_status = _clean_text(row.get("payment_status") or "unpaid", limit=40) or "unpaid"
        by_registration_status[registration_status] = by_registration_status.get(registration_status, 0) + 1
        by_payment_status[payment_status] = by_payment_status.get(payment_status, 0) + 1
    return {
        "registrations": len(registrations),
        "selections": len(selections),
        "by_registration_status": by_registration_status,
        "by_payment_status": by_payment_status,
    }


def _tournament_payload(row: dict[str, Any], *, registration_count: int | None = None, selection_count: int | None = None, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    tournament_id = _clean_text(row.get("id"), limit=120)
    return {
        "id": tournament_id,
        "name": _clean_text(row.get("name") or tournament_id, limit=180),
        "status": _clean_text(row.get("status") or "DRAFT", limit=40).upper(),
        "start_date": row.get("start_date"),
        "end_date": row.get("end_date"),
        "registration_slug": _clean_text((settings or {}).get("registration_slug"), limit=160),
        "registration_status": _clean_text((settings or {}).get("registration_status"), limit=60),
        "registration_count": int(registration_count or 0),
        "selection_count": int(selection_count or 0),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _fetch_registration_by_id(supabase: Any, *, tournament_id: str, registration_id: str) -> dict[str, Any] | None:
    for row in _registration_rows(supabase, tournament_id=str(tournament_id), limit=1000):
        if _clean_text(row.get("id"), limit=120) == str(registration_id):
            return row
    return None


def _fetch_selection_by_id(supabase: Any, *, tournament_id: str, selection_id: str) -> dict[str, Any] | None:
    rows = _table_rows_for_tournament(supabase, "tournament_registration_selections", SELECTION_SELECT, tournament_id=str(tournament_id))
    for row in rows:
        if _clean_text(row.get("id"), limit=120) == str(selection_id):
            return row
    return None


def _selection_count_for_registration(supabase: Any, *, tournament_id: str, registration_id: str) -> int:
    rows = _table_rows_for_tournament(supabase, "tournament_registration_selections", SELECTION_SELECT, tournament_id=str(tournament_id))
    return len([row for row in rows if _clean_text(row.get("registration_id"), limit=120) == str(registration_id)])


def _event_option_by_id(supabase: Any, *, tournament_id: str, event_option_id: str) -> dict[str, Any] | None:
    for row in _table_rows_for_tournament(supabase, "tournament_event_options", EVENT_OPTION_SELECT, tournament_id=str(tournament_id)):
        if _clean_text(row.get("id"), limit=120) == str(event_option_id):
            return row
    return None


def build_admin_tournament_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "tournaments_endpoint": None,
            "tournament_detail_endpoint": None,
            "registration_update_endpoint": None,
            "selection_update_endpoint": None,
            "warnings": ["Next Tournament Admin is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS on FastAPI for a closed-club pilot."],
        }
    tournament_count = None
    warnings: list[str] = []
    if supabase is not None:
        rows, warnings = _fetch_tournament_rows(supabase, club_id=str(club_id), include_archived=False)
        tournament_count = len(rows)
    return {
        "enabled": True,
        "status": "ready_for_tournament_registration_admin",
        "tournaments_endpoint": "/admin/clubs/{club_id}/tournaments/admin/tournaments",
        "tournament_detail_endpoint": "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}",
        "registration_update_endpoint": "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/{registration_id}",
        "selection_update_endpoint": "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/selections/{selection_id}",
        "tournament_count": tournament_count,
        "warnings": warnings,
    }


def list_admin_tournaments(supabase: Any, *, club_id: str, include_archived: bool = False) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    rows, warnings = _fetch_tournament_rows(supabase, club_id=str(club_id), include_archived=include_archived)
    tournaments: list[dict[str, Any]] = []
    for row in rows:
        tournament_id = _clean_text(row.get("id"), limit=120)
        if not tournament_id:
            continue
        settings = _first_row(supabase, "tournament_registration_settings", REGISTRATION_SETTINGS_SELECT, key="tournament_id", value=tournament_id) or {}
        registrations = _registration_rows(supabase, tournament_id=tournament_id, limit=1000)
        selections = _table_rows_for_tournament(supabase, "tournament_registration_selections", "id,tournament_id", tournament_id=tournament_id)
        tournaments.append(_tournament_payload(row, registration_count=len(registrations), selection_count=len(selections), settings=settings))
    return {"ok": True, "mode": "tournament_admin_list", "tournaments": tournaments, "count": len(tournaments), "warnings": warnings}


def get_admin_tournament_detail(supabase: Any, *, club_id: str, tournament_id: str) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    clean_id = _clean_text(tournament_id, limit=120)
    if not clean_id:
        raise ValueError("tournament_id is required")
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    settings = _first_row(supabase, "tournament_registration_settings", REGISTRATION_SETTINGS_SELECT, key="tournament_id", value=clean_id) or {}
    days = _table_rows_for_tournament(supabase, "tournament_registration_days", DAY_SELECT, tournament_id=clean_id)
    event_options = _table_rows_for_tournament(supabase, "tournament_event_options", EVENT_OPTION_SELECT, tournament_id=clean_id)
    registrations_raw = _registration_rows(supabase, tournament_id=clean_id, limit=1000)
    selections_raw = _table_rows_for_tournament(supabase, "tournament_registration_selections", SELECTION_SELECT, tournament_id=clean_id)
    selections_by_registration: dict[str, int] = {}
    for row in selections_raw:
        registration_id = _clean_text(row.get("registration_id"), limit=120)
        if registration_id:
            selections_by_registration[registration_id] = selections_by_registration.get(registration_id, 0) + 1
    registrations = [_registration_payload(row, selection_count=selections_by_registration.get(_clean_text(row.get("id"), limit=120), 0)) for row in registrations_raw]
    selections = [_selection_payload(row, event_options=event_options) for row in selections_raw]
    return {
        "ok": True,
        "mode": "tournament_admin_detail",
        "tournament": _tournament_payload(tournament, registration_count=len(registrations_raw), selection_count=len(selections_raw), settings=settings),
        "settings": settings,
        "days": days,
        "event_options": event_options,
        "registrations": registrations,
        "selections": selections,
        "summary": _summary_counts(registrations_raw, selections_raw),
        "warnings": [],
    }


def update_admin_tournament_registration(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str,
    patch: dict[str, Any],
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_registration_update",
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_REGISTRATION_UPDATE:
        raise ValueError(f"Type {CONFIRM_REGISTRATION_UPDATE} to confirm registration changes.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_registration_id = _clean_text(registration_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    before = _fetch_registration_by_id(supabase, tournament_id=clean_tournament_id, registration_id=clean_registration_id)
    if before is None:
        raise ValueError("registration not found")

    update_payload: dict[str, Any] = {}
    if "registration_status" in patch:
        next_status = _clean_text(patch.get("registration_status"), limit=40).lower()
        if next_status not in ADMIN_REGISTRATION_STATUS_OPTIONS:
            raise ValueError(f"Invalid registration status: {patch.get('registration_status')}")
        update_payload["status"] = next_status
    if "payment_status" in patch:
        next_payment = _clean_text(patch.get("payment_status"), limit=40).lower()
        if next_payment not in ADMIN_PAYMENT_STATUS_OPTIONS:
            raise ValueError(f"Invalid payment status: {patch.get('payment_status')}")
        update_payload["payment_status"] = next_payment
    if "notes" in patch:
        update_payload["notes"] = _clean_text(patch.get("notes"), limit=2000)
    if not update_payload:
        raise ValueError("No supported registration fields were provided.")

    updated = update_admin_registration(
        supabase,
        tournament_id=clean_tournament_id,
        registration_id=clean_registration_id,
        payload=update_payload,
    )
    selection_count = _selection_count_for_registration(supabase, tournament_id=clean_tournament_id, registration_id=clean_registration_id)
    registration = _registration_payload(updated, selection_count=selection_count)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_tournament_registration_admin",
        entity_type="tournament_registration",
        entity_id=clean_registration_id,
        before_json={"registration": _registration_payload(before, selection_count=selection_count)},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "patch": update_payload, "registration": registration},
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "tournament_registration_update", "registration": registration, "warnings": warnings}


def update_admin_tournament_selection(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    selection_id: str,
    patch: dict[str, Any],
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_selection_update",
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_SELECTION_UPDATE:
        raise ValueError(f"Type {CONFIRM_SELECTION_UPDATE} to confirm event-entry changes.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_selection_id = _clean_text(selection_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    before = _fetch_selection_by_id(supabase, tournament_id=clean_tournament_id, selection_id=clean_selection_id)
    if before is None:
        raise ValueError("selection not found")

    update_payload: dict[str, Any] = {}
    if "event_option_id" in patch and _clean_text(patch.get("event_option_id"), limit=120):
        next_event_id = _clean_text(patch.get("event_option_id"), limit=120)
        current_event_id = _clean_text(before.get("event_option_id"), limit=120)
        event = _event_option_by_id(supabase, tournament_id=clean_tournament_id, event_option_id=next_event_id)
        if not event:
            raise ValueError("event option not found for this tournament")
        if next_event_id != current_event_id and registration_is_imported_to_draw(supabase, tournament_id=clean_tournament_id, selection_id=clean_selection_id):
            raise ValueError("This event entry is already imported into a draw. Remove the draw team before moving divisions.")
        update_payload["event_option_id"] = next_event_id
        update_payload["registration_day_id"] = _clean_text(event.get("registration_day_id"), limit=120)
    if "partner_mode" in patch:
        partner_mode = _clean_text(patch.get("partner_mode"), limit=40).upper() or "NONE"
        if partner_mode not in PARTNER_MODE_OPTIONS:
            raise ValueError(f"Invalid partner mode: {patch.get('partner_mode')}")
        update_payload["partner_mode"] = partner_mode
        update_payload["show_on_partner_board"] = partner_mode == "NEEDS_PARTNER"
    for field, limit in [
        ("partner_name", 160),
        ("partner_email", 180),
        ("partner_phone", 80),
        ("partner_note", 500),
    ]:
        if field in patch:
            update_payload[field] = _clean_text(patch.get(field), limit=limit)
    if not update_payload:
        raise ValueError("No supported event-entry fields were provided.")

    updated = update_admin_registration_selection(
        supabase,
        tournament_id=clean_tournament_id,
        selection_id=clean_selection_id,
        payload=update_payload,
    )
    event_options = _table_rows_for_tournament(supabase, "tournament_event_options", EVENT_OPTION_SELECT, tournament_id=clean_tournament_id)
    selection = _selection_payload(updated, event_options=event_options)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_tournament_registration_selection_admin",
        entity_type="tournament_registration_selection",
        entity_id=clean_selection_id,
        before_json={"selection": _selection_payload(before, event_options=event_options)},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "patch": update_payload, "selection": selection},
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "tournament_selection_update", "selection": selection, "warnings": warnings}
