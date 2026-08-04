from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_registration_repo import (
    ADMIN_PAYMENT_STATUS_OPTIONS,
    ADMIN_REGISTRATION_STATUS_OPTIONS,
    PARTNER_MODE_OPTIONS,
    is_day_enabled,
    public_event_option_visibility,
    registration_is_imported_to_draw,
    update_admin_registration,
    update_admin_registration_selection,
)
from jupr_app.domain.tournament_admin_operations import stable_tournament_admin_fingerprint
from jupr_app.domain.tournament_partner_service import admin_replace_partner_link
from jupr_app.services.public_tournament_registration_service import (
    build_tournament_registration_player_profile,
    validate_and_clean_tournament_selection,
)

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
TOURNAMENT_SELECT = "id,club_id,name,status,start_date,end_date,event_tags,created_at,updated_at"
TOURNAMENT_MINIMAL_SELECT = "id,club_id,name,status"
REGISTRATION_SETTINGS_SELECT = "id,tournament_id,registration_slug,registration_status,registration_open_at,registration_close_at,waitlist_enabled,partner_board_enabled,updated_at"
REGISTRATION_SELECT = (
    "id,tournament_id,player_id,first_name,last_name,display_name,email,phone,"
    "dupr_id,doubles_skill,singles_skill,age,age_bracket,gender,"
    "status,payment_status,notes,wants_partner_board_contact,submitted_at,updated_at"
)
REGISTRATION_LEGACY_SELECT = (
    "id,tournament_id,player_id,first_name,last_name,display_name,email,phone,"
    "registration_status,payment_status,wants_partner_board_contact,submitted_at,updated_at"
)
REGISTRATION_MINIMAL_SELECT = "id,tournament_id,display_name,email,status,payment_status,submitted_at,updated_at"
REGISTRATION_LEGACY_MINIMAL_SELECT = "id,tournament_id,display_name,email,registration_status,payment_status,submitted_at,updated_at"
SELECTION_SELECT = (
    "id,tournament_id,registration_id,registration_day_id,event_option_id,partner_mode,"
    "partner_name,partner_email,partner_phone,partner_dupr_id,partner_skill,partner_age,"
    "partner_note,show_on_partner_board,created_at,updated_at"
)
EVENT_OPTION_SELECT = (
    "id,tournament_id,registration_day_id,label,event_family_label,division_name,event_type,"
    "gender_restriction,skill_label,age_label,partner_required,capacity_teams,public_partner_board,"
    "event_format_default,scoring_default,skill_mode,age_mode,status,enabled,waitlist_enabled,"
    "partner_board_enabled,sort_order"
)
DAY_SELECT = "id,tournament_id,label,event_date,enabled,sort_order,court_count,court_labels,court_open_time,court_close_time,court_notes,created_at"
CONFIRM_REGISTRATION_UPDATE = "SAVE REGISTRATION"
CONFIRM_SELECTION_UPDATE = "SAVE SELECTION"
CONFIRM_TOURNAMENT_UPDATE = "SAVE TOURNAMENT"


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
                .order("submitted_at", desc=True)
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
        "created_at": row.get("submitted_at") or row.get("created_at"),
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


def _event_family_key(event: dict[str, Any]) -> tuple[str, str]:
    day_id = _clean_text(event.get("registration_day_id"), limit=120)
    family = " ".join(
        _clean_text(event.get("event_family_label") or event.get("label") or "Event", limit=160)
        .lower()
        .split()
    )
    return day_id, family


def _selection_payload(
    row: dict[str, Any],
    *,
    event_options: list[dict[str, Any]] | None = None,
    partner: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "partner_team_link_id": (partner or {}).get("team_link_id"),
        "partner_team_status": (partner or {}).get("team_status"),
        "partner_selection_id": (partner or {}).get("selection_id"),
        "partner_registration_id": (partner or {}).get("registration_id"),
        "partner_display_name": (partner or {}).get("display_name"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _partner_relationships_for_tournament(
    supabase: Any,
    *,
    tournament_id: str,
    selections: list[dict[str, Any]],
    registrations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    try:
        links = _query_rows(
            supabase.table("tournament_registration_team_links")
            .select("id,tournament_id,event_option_id,selection1_id,selection2_id,status,updated_at")
            .eq("tournament_id", str(tournament_id))
        )
    except Exception:
        return {}
    selection_by_id = {
        _clean_text(row.get("id"), limit=120): row
        for row in selections
        if _clean_text(row.get("id"), limit=120)
    }
    registration_by_id = {
        _clean_text(row.get("id"), limit=120): row
        for row in registrations
        if _clean_text(row.get("id"), limit=120)
    }
    result: dict[str, dict[str, Any]] = {}
    for link in links:
        status = _clean_text(link.get("status"), limit=40).upper()
        if status not in {"CONFIRMED", "ADMIN_CONFIRMED"}:
            continue
        left_id = _clean_text(link.get("selection1_id"), limit=120)
        right_id = _clean_text(link.get("selection2_id"), limit=120)
        for selection_id, partner_selection_id in (
            (left_id, right_id),
            (right_id, left_id),
        ):
            partner_selection = selection_by_id.get(partner_selection_id) or {}
            partner_registration_id = _clean_text(
                partner_selection.get("registration_id"), limit=120
            )
            partner_registration = registration_by_id.get(partner_registration_id) or {}
            result[selection_id] = {
                "team_link_id": _clean_text(link.get("id"), limit=120),
                "team_status": status,
                "selection_id": partner_selection_id or None,
                "registration_id": partner_registration_id or None,
                "display_name": _display_name(partner_registration)
                if partner_registration
                else None,
            }
    return result


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


def _admin_detail_state_fingerprint(
    *,
    tournament: dict[str, Any],
    settings: dict[str, Any],
    registrations: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> str:
    return stable_tournament_admin_fingerprint(
        {
            "tournament": {
                "id": tournament.get("id"),
                "status": tournament.get("status"),
                "updated_at": tournament.get("updated_at"),
            },
            "settings_updated_at": settings.get("updated_at"),
            "registrations": sorted(
                (str(row.get("id") or ""), str(row.get("updated_at") or ""))
                for row in registrations
            ),
            "selections": sorted(
                (str(row.get("id") or ""), str(row.get("updated_at") or ""))
                for row in selections
            ),
        }
    )


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


def _required_relation_rows(
    supabase: Any,
    table_name: str,
    select_expr: str,
    *,
    tournament_id: str,
    field_name: str,
    field_value: str,
) -> list[dict[str, Any]]:
    try:
        return _query_rows(
            supabase.table(table_name)
            .select(select_expr)
            .eq("tournament_id", str(tournament_id))
            .eq(str(field_name), str(field_value))
            .limit(100)
        )
    except Exception as exc:
        raise RuntimeError(f"Could not verify tournament selection relationships: {table_name}") from exc


def _selection_relationship_lock_reason(
    supabase: Any,
    *,
    tournament_id: str,
    selection_id: str,
) -> str | None:
    active_members = _required_relation_rows(
        supabase,
        "tournament_registration_team_members",
        "id,selection_id,status",
        tournament_id=tournament_id,
        field_name="selection_id",
        field_value=selection_id,
    )
    if any(
        _clean_text(row.get("status"), limit=40).upper() == "ACTIVE"
        for row in active_members
    ):
        return "This event entry belongs to a confirmed partner team. Change the canonical team link first."

    team_links: list[dict[str, Any]] = []
    for field_name in ("selection1_id", "selection2_id"):
        team_links.extend(
            _required_relation_rows(
                supabase,
                "tournament_registration_team_links",
                "id,selection1_id,selection2_id,status",
                tournament_id=tournament_id,
                field_name=field_name,
                field_value=selection_id,
            )
        )
    if any(
        _clean_text(row.get("status"), limit=40).upper() in {"CONFIRMED", "ADMIN_CONFIRMED"}
        for row in team_links
    ):
        return "This event entry belongs to a confirmed partner team. Change the canonical team link first."

    partner_requests: list[dict[str, Any]] = []
    for field_name in ("requester_selection_id", "target_selection_id"):
        partner_requests.extend(
            _required_relation_rows(
                supabase,
                "tournament_registration_partner_requests",
                "id,requester_selection_id,target_selection_id,status",
                tournament_id=tournament_id,
                field_name=field_name,
                field_value=selection_id,
            )
        )
    if any(
        _clean_text(row.get("status"), limit=40).upper() == "PENDING"
        for row in partner_requests
    ):
        return "This event entry has a pending partner request. Resolve or cancel the request first."
    return None


def _required_registration_settings(supabase: Any, *, tournament_id: str) -> dict[str, Any]:
    try:
        rows = _query_rows(
            supabase.table("tournament_registration_settings")
            .select(REGISTRATION_SETTINGS_SELECT)
            .eq("tournament_id", str(tournament_id))
            .limit(1)
        )
    except Exception as exc:
        raise RuntimeError("Could not verify tournament registration settings.") from exc
    if not rows:
        raise RuntimeError("Tournament registration settings are required before editing event entries.")
    return rows[0]


def build_admin_tournament_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    from jupr_app.services.admin_tournament_guarded_operation import tournament_admin_mutation_status
    from jupr_app.services.admin_tournament_ops_service import build_admin_tournament_ops_runtime_status

    if not is_admin_tournament_admin_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "tournaments_endpoint": None,
            "tournament_detail_endpoint": None,
            "registration_update_endpoint": None,
            "selection_update_endpoint": None,
            "registration_export_endpoint": None,
            "broadcast_preview_endpoint": None,
            "tournament_update_endpoint": None,
            "import_handoff_endpoint": None,
            "warnings": ["Next Tournament Admin is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS on FastAPI for a closed-club pilot."],
            "mutation_runtime": tournament_admin_mutation_status(),
            "operations_runtime": build_admin_tournament_ops_runtime_status(),
            "streamlit_fallback_url": os.getenv("JUPR_STREAMLIT_FALLBACK_URL", "").strip() or "https://juprtrespalapas.streamlit.app",
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
        "registration_export_endpoint": "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/export.csv",
        "broadcast_preview_endpoint": "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/broadcast-preview",
        "tournament_update_endpoint": "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}",
        "import_handoff_endpoint": "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/registrations/import-handoff",
        "tournament_count": tournament_count,
        "warnings": warnings,
        "mutation_runtime": tournament_admin_mutation_status(),
        "operations_runtime": build_admin_tournament_ops_runtime_status(),
        "streamlit_fallback_url": os.getenv("JUPR_STREAMLIT_FALLBACK_URL", "").strip() or "https://juprtrespalapas.streamlit.app",
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
    partner_relationships = _partner_relationships_for_tournament(
        supabase,
        tournament_id=clean_id,
        selections=selections_raw,
        registrations=registrations_raw,
    )
    selections = [
        _selection_payload(
            row,
            event_options=event_options,
            partner=partner_relationships.get(_clean_text(row.get("id"), limit=120)),
        )
        for row in selections_raw
    ]
    state_fingerprint = _admin_detail_state_fingerprint(
        tournament=tournament,
        settings=settings,
        registrations=registrations_raw,
        selections=selections_raw,
    )
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
        "state_fingerprint": state_fingerprint,
        "warnings": [],
    }


def build_admin_tournament_registration_import_handoff(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
) -> dict[str, Any]:
    """Build a read-only handoff; registration admin never mutates draw teams."""

    detail = get_admin_tournament_detail(
        supabase,
        club_id=str(club_id),
        tournament_id=str(tournament_id),
    )
    imported_selection_ids = [
        str(selection.get("id") or "")
        for selection in detail.get("selections") or []
        if registration_is_imported_to_draw(
            supabase,
            tournament_id=str(tournament_id),
            selection_id=str(selection.get("id") or ""),
        )
    ]
    confirmed = [
        row
        for row in detail.get("registrations") or []
        if str(row.get("registration_status") or "").lower() == "confirmed"
    ]
    return {
        "ok": True,
        "mode": "tournament_registration_import_handoff",
        "dry_run": True,
        "write_count": 0,
        "tournament": detail["tournament"],
        "state_fingerprint": detail["state_fingerprint"],
        "confirmed_registration_count": len(confirmed),
        "imported_selection_count": len(imported_selection_ids),
        "imported_selection_ids": imported_selection_ids,
        "direct_import_available": False,
        "ops_path": f"/admin/tournaments/ops?tournament_id={str(tournament_id)}",
        "required_ops_confirmation": "IMPORT REGISTRATIONS",
        "integrity_notice": "Tournament Ops rechecks draw scope, confirmed player links, duplicates, and existing games. Registration Admin cannot bypass those draw-integrity guards.",
    }


def replace_admin_tournament_selection_partner(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    selection_id: str,
    partner_selection_id: str | None,
    unpaired_mode: str,
    expected_updated_at: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_registration_detail",
    dry_run: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != "SAVE PARTNER":
        raise ValueError("Type SAVE PARTNER to confirm the partner assignment.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_selection_id = _clean_text(selection_id, limit=120)
    clean_partner_id = _clean_text(partner_selection_id, limit=120) or None
    expected = _clean_text(expected_updated_at, limit=120)
    if not expected:
        raise ValueError("expected_updated_at is required for partner changes.")
    tournament = _first_row(
        supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id
    )
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    before = _fetch_selection_by_id(
        supabase, tournament_id=clean_tournament_id, selection_id=clean_selection_id
    )
    if before is None:
        raise ValueError("selection not found")
    if str(before.get("updated_at") or "") != expected:
        from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError

        raise StaleTournamentAdminStateError(
            "This event entry changed after it was loaded. Reload before changing its partner."
        )
    if registration_is_imported_to_draw(
        supabase,
        tournament_id=clean_tournament_id,
        selection_id=clean_selection_id,
    ):
        raise ValueError(
            "This event entry is already represented in a draw. Correct the team in Tournament Ops."
        )
    mode = _clean_text(unpaired_mode, limit=40).upper() or "NEEDS_PARTNER"
    if mode not in {"NONE", "NEEDS_PARTNER"}:
        raise ValueError("Unpaired mode must be NONE or NEEDS_PARTNER.")
    event_option_id = _clean_text(before.get("event_option_id"), limit=120)
    if clean_partner_id:
        partner = _fetch_selection_by_id(
            supabase,
            tournament_id=clean_tournament_id,
            selection_id=clean_partner_id,
        )
        if partner is None:
            raise ValueError("partner event entry not found")
        if _clean_text(partner.get("event_option_id"), limit=120) != event_option_id:
            raise ValueError("Partners must be registered in the same division.")
        partner_registration = _fetch_registration_by_id(
            supabase,
            tournament_id=clean_tournament_id,
            registration_id=_clean_text(partner.get("registration_id"), limit=120),
        )
        if partner_registration is None:
            raise ValueError("partner registration not found")
        if _registration_status(partner_registration).lower() == "cancelled":
            raise ValueError("A cancelled registration cannot be assigned as a partner.")
        if registration_is_imported_to_draw(
            supabase,
            tournament_id=clean_tournament_id,
            selection_id=clean_partner_id,
        ):
            raise ValueError(
                "The selected partner is already represented in a draw. Correct the team in Tournament Ops."
            )
    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_registration_partner_preflight",
            "dry_run": True,
            "write_count": 0,
            "selection_id": clean_selection_id,
            "partner_selection_id": clean_partner_id,
            "unpaired_mode": mode,
        }

    result = admin_replace_partner_link(
        supabase,
        tournament_id=clean_tournament_id,
        event_option_id=event_option_id,
        selection_id=clean_selection_id,
        partner_selection_id=clean_partner_id,
        unpaired_mode=mode,
        admin_user_id=actor_email,
        source="ADMIN_RECONCILIATION",
    )
    detail = get_admin_tournament_detail(
        supabase, club_id=str(club_id), tournament_id=clean_tournament_id
    )
    selection = next(
        (row for row in detail.get("selections") or [] if str(row.get("id") or "") == clean_selection_id),
        None,
    )
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_tournament_registration_partner_admin",
        entity_type="tournament_registration_selection",
        entity_id=clean_selection_id,
        before_json={"selection": _selection_payload(before)},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "partner_selection_id": clean_partner_id,
            "unpaired_mode": mode,
            "result": result,
            "selection": selection,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings = [audit_write.warning] if audit_write.warning else []
    if not audit_write.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "tournament_registration_partner_update",
        "selection": selection,
        "partner_result": result,
        "warnings": warnings,
    }


def update_admin_tournament(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    patch: dict[str, Any],
    expected_updated_at: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_tournament_update",
    dry_run: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_TOURNAMENT_UPDATE:
        raise ValueError(f"Type {CONFIRM_TOURNAMENT_UPDATE} to confirm tournament changes.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    before = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not before or str(before.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    update_payload: dict[str, Any] = {}
    if "name" in patch:
        name = _clean_text(patch.get("name"), limit=180)
        if not name:
            raise ValueError("Tournament name is required.")
        update_payload["name"] = name
    for field in ("start_date", "end_date"):
        if field in patch:
            update_payload[field] = _clean_text(patch.get(field), limit=40) or None
    if not update_payload:
        raise ValueError("No supported tournament fields were provided.")
    next_start_date = update_payload.get("start_date", before.get("start_date"))
    next_end_date = update_payload.get("end_date", before.get("end_date"))
    if next_start_date and next_end_date and str(next_end_date) < str(next_start_date):
        raise ValueError("Tournament end date cannot be before its start date.")
    if dry_run:
        return {"ok": True, "mode": "tournament_update_preflight", "dry_run": True, "write_count": 0, "patch": update_payload}
    update_payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    query = (
        supabase.table("tournaments")
        .update(update_payload)
        .eq("club_id", str(club_id))
        .eq("id", clean_tournament_id)
    )
    if expected_updated_at:
        query = query.eq("updated_at", str(expected_updated_at))
    updated_rows = _safe_rows(query.execute())
    if not updated_rows:
        if expected_updated_at:
            from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError

            raise StaleTournamentAdminStateError("Tournament changed after it was loaded. Reload and try again.")
        raise ValueError("tournament not found")
    updated = updated_rows[0]
    tournament = _tournament_payload(updated)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_tournament_admin",
        entity_type="tournament",
        entity_id=clean_tournament_id,
        before_json={"tournament": _tournament_payload(before)},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "patch": update_payload, "tournament": tournament},
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings = [audit_write.warning] if audit_write.warning else []
    if not audit_write.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "tournament_update", "tournament": tournament, "warnings": warnings}


def update_admin_tournament_registration(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_id: str,
    patch: dict[str, Any],
    expected_updated_at: str | None = None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_registration_update",
    dry_run: bool = False,
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

    if "status" in update_payload and registration_is_imported_to_draw(
        supabase,
        tournament_id=clean_tournament_id,
        registration_id=clean_registration_id,
    ):
        raise ValueError(
            "This registration is already imported into a draw. Change draw membership in Tournament Ops before changing registration status."
        )
    if dry_run:
        return {"ok": True, "mode": "tournament_registration_update_preflight", "dry_run": True, "write_count": 0, "patch": update_payload}

    updated = update_admin_registration(
        supabase,
        tournament_id=clean_tournament_id,
        registration_id=clean_registration_id,
        payload=update_payload,
        expected_updated_at=expected_updated_at,
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
    expected_updated_at: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_selection_update",
    dry_run: bool = False,
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_SELECTION_UPDATE:
        raise ValueError(f"Type {CONFIRM_SELECTION_UPDATE} to confirm event-entry changes.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    clean_selection_id = _clean_text(selection_id, limit=120)
    clean_expected_updated_at = _clean_text(expected_updated_at, limit=120)
    if not clean_expected_updated_at:
        raise ValueError("expected_updated_at is required for event-entry changes.")
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    before = _fetch_selection_by_id(supabase, tournament_id=clean_tournament_id, selection_id=clean_selection_id)
    if before is None:
        raise ValueError("selection not found")

    registration_id = _clean_text(before.get("registration_id"), limit=120)
    registration = _fetch_registration_by_id(
        supabase,
        tournament_id=clean_tournament_id,
        registration_id=registration_id,
    )
    if registration is None:
        raise ValueError("registration not found for this event entry")

    event_options = _table_rows_for_tournament(
        supabase,
        "tournament_event_options",
        EVENT_OPTION_SELECT,
        tournament_id=clean_tournament_id,
    )
    events_by_id = _event_option_map(event_options)
    current_event_id = _clean_text(before.get("event_option_id"), limit=120)
    next_event_id = _clean_text(patch.get("event_option_id"), limit=120) if "event_option_id" in patch else current_event_id
    if not next_event_id:
        raise ValueError("Each event entry must identify a division.")
    event = events_by_id.get(next_event_id)
    if not event:
        raise ValueError("event option not found for this tournament")
    event_changed = next_event_id != current_event_id

    days = _table_rows_for_tournament(
        supabase,
        "tournament_registration_days",
        DAY_SELECT,
        tournament_id=clean_tournament_id,
    )
    day_id = _clean_text(event.get("registration_day_id"), limit=120)
    day = next((row for row in days if _clean_text(row.get("id"), limit=120) == day_id), None)
    if day is None:
        raise ValueError("Selected division is not attached to a registration day for this tournament.")
    if event_changed:
        if not is_day_enabled(day):
            raise ValueError("Selected division is not on an enabled registration day.")
        if public_event_option_visibility(event) != "selectable":
            raise ValueError("Selected division is not open for registration.")

        target_family = _event_family_key(event)
        sibling_rows = _table_rows_for_tournament(
            supabase,
            "tournament_registration_selections",
            SELECTION_SELECT,
            tournament_id=clean_tournament_id,
        )
        for sibling in sibling_rows:
            if _clean_text(sibling.get("id"), limit=120) == clean_selection_id:
                continue
            if _clean_text(sibling.get("registration_id"), limit=120) != registration_id:
                continue
            sibling_event_id = _clean_text(sibling.get("event_option_id"), limit=120)
            if sibling_event_id == next_event_id:
                raise ValueError("The same division cannot be selected more than once.")
            sibling_event = events_by_id.get(sibling_event_id)
            if sibling_event and _event_family_key(sibling_event) == target_family:
                family = _clean_text(event.get("event_family_label") or event.get("label") or "Event", limit=160)
                raise ValueError(f"Choose only one division for {family} on the same registration day.")

    current_partner_mode = _clean_text(before.get("partner_mode") or "NONE", limit=40).upper() or "NONE"
    next_partner_mode = (
        _clean_text(patch.get("partner_mode"), limit=40).upper() or "NONE"
        if "partner_mode" in patch
        else current_partner_mode
    )
    if next_partner_mode not in PARTNER_MODE_OPTIONS:
        raise ValueError(f"Invalid partner mode: {patch.get('partner_mode')}")

    for field, limit in [("partner_name", 160), ("partner_email", 180), ("partner_phone", 80)]:
        if field not in patch:
            continue
        before_value = _clean_text(before.get(field), limit=limit)
        next_value = _clean_text(patch.get(field), limit=limit)
        if field == "partner_email":
            before_value = before_value.lower()
            next_value = next_value.lower()
        if next_value != before_value:
            raise ValueError("Partner identity is read-only in this editor. Use the canonical partner-link workflow.")

    partner_mode_changed = next_partner_mode != current_partner_mode
    relationship_sensitive_change = event_changed or partner_mode_changed
    if relationship_sensitive_change:
        relationship_lock = _selection_relationship_lock_reason(
            supabase,
            tournament_id=clean_tournament_id,
            selection_id=clean_selection_id,
        )
        if relationship_lock:
            raise ValueError(relationship_lock)
    if registration_is_imported_to_draw(
        supabase,
        tournament_id=clean_tournament_id,
        selection_id=clean_selection_id,
    ):
        raise ValueError("This event entry is already imported into a draw. Make the corresponding team change in Tournament Ops; this registration editor will not bypass draw integrity.")
    if next_partner_mode == "HAS_PARTNER" and current_partner_mode != "HAS_PARTNER":
        raise ValueError("Creating a partner link requires the canonical partner-link workflow.")
    if event_changed and next_partner_mode == "HAS_PARTNER":
        raise ValueError("Move or remove the canonical partner link before changing this division.")

    settings = _required_registration_settings(supabase, tournament_id=clean_tournament_id)
    candidate = dict(before)
    candidate["event_option_id"] = next_event_id
    candidate["registration_day_id"] = day_id
    candidate["partner_mode"] = next_partner_mode
    if "partner_note" in patch:
        candidate["partner_note"] = _clean_text(patch.get("partner_note"), limit=500)
    if next_partner_mode == "NEEDS_PARTNER":
        event_board_enabled = _safe_bool(
            event.get("partner_board_enabled", event.get("public_partner_board")),
            default=True,
        )
        candidate["show_on_partner_board"] = _safe_bool(settings.get("partner_board_enabled"), default=False) and event_board_enabled
    elif next_partner_mode == "NONE":
        candidate["show_on_partner_board"] = False

    validated: dict[str, Any] | None = None
    if event_changed or partner_mode_changed or ("partner_mode" in patch and next_partner_mode != "HAS_PARTNER"):
        player_profile = build_tournament_registration_player_profile(
            supabase,
            club_id=str(club_id),
            registration=registration,
            require_active_link=False,
        )
        validated = validate_and_clean_tournament_selection(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            event=event,
            raw_selection=candidate,
            player_profile=player_profile,
            settings=settings,
            primary_registration_id=registration_id,
        )

    update_payload: dict[str, Any] = {}
    if "event_option_id" in patch or event_changed:
        update_payload["event_option_id"] = next_event_id
        update_payload["registration_day_id"] = day_id
    if "partner_mode" in patch or event_changed or partner_mode_changed:
        canonical = validated or candidate
        for field in [
            "partner_mode",
            "partner_name",
            "partner_email",
            "partner_phone",
            "partner_dupr_id",
            "partner_skill",
            "partner_age",
            "show_on_partner_board",
        ]:
            update_payload[field] = canonical.get(field)
    if "partner_note" in patch:
        update_payload["partner_note"] = _clean_text(patch.get("partner_note"), limit=500)
    if not update_payload:
        raise ValueError("No supported event-entry fields were provided.")
    if dry_run:
        return {"ok": True, "mode": "tournament_selection_update_preflight", "dry_run": True, "write_count": 0, "patch": update_payload}

    updated = update_admin_registration_selection(
        supabase,
        tournament_id=clean_tournament_id,
        selection_id=clean_selection_id,
        payload=update_payload,
        expected_updated_at=clean_expected_updated_at,
    )
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
