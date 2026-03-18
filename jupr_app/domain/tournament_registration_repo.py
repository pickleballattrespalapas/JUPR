from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import re
import uuid

from .tournament_registration_compiler import compile_tournament_registration_state

REGISTRATION_STATUS_OPTIONS = ["draft", "open", "closed"]
EVENT_TYPE_OPTIONS = ["SINGLES", "GENDER_DOUBLES", "MIXED_DOUBLES"]
GENDER_RESTRICTION_OPTIONS = ["ANY", "MEN", "WOMEN", "MIXED"]
PARTNER_MODE_OPTIONS = ["NONE", "HAS_PARTNER", "NEEDS_PARTNER"]


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _safe_data(resp: Any) -> list[dict[str, Any]]:
    try:
        return list(resp.data or [])
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_data(resp)
    return rows[0] if rows else None


def _normalize_email(value: Any) -> str:
    return str(value or "").strip().lower()


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(value or "").strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or f"tournament-{uuid.uuid4().hex[:6]}"


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def registration_feature_available(supabase) -> tuple[bool, str | None]:
    try:
        supabase.table("tournament_registration_settings").select("id").limit(1).execute()
        return True, None
    except Exception as exc:
        return False, str(exc)


def list_existing_tournaments(supabase, club_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournaments")
        .select("*")
        .eq("club_id", str(club_id))
        .order("created_at", desc=True)
        .execute()
    )
    return _safe_data(resp)


def get_tournament_record(supabase, tournament_id: str) -> dict[str, Any] | None:
    resp = (
        supabase.table("tournaments")
        .select("*")
        .eq("id", str(tournament_id))
        .limit(1)
        .execute()
    )
    return _safe_first(resp)


def get_registration_settings(supabase, tournament_id: str, *, tournament_name: str | None = None) -> dict[str, Any]:
    resp = (
        supabase.table("tournament_registration_settings")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .limit(1)
        .execute()
    )
    row = _safe_first(resp)
    if row:
        return row
    return {
        "id": _uid("regset"),
        "tournament_id": str(tournament_id),
        "registration_slug": _slugify(tournament_name or str(tournament_id)),
        "locale": "en",
        "registration_status": "draft",
        "registration_open_at": None,
        "registration_close_at": None,
        "waitlist_enabled": True,
        "partner_board_enabled": True,
        "rules_markdown": "",
        "refund_policy_markdown": "",
        "sponsor_markdown": "",
    }


def upsert_registration_settings(supabase, payload: dict[str, Any]) -> dict[str, Any]:
    clean = {
        "id": str(payload.get("id") or _uid("regset")),
        "tournament_id": str(payload.get("tournament_id")),
        "registration_slug": _slugify(str(payload.get("registration_slug") or payload.get("tournament_id") or "tournament")),
        "locale": str(payload.get("locale") or "en"),
        "registration_status": str(payload.get("registration_status") or "draft").lower(),
        "registration_open_at": payload.get("registration_open_at") or None,
        "registration_close_at": payload.get("registration_close_at") or None,
        "waitlist_enabled": _coerce_bool(payload.get("waitlist_enabled", True)),
        "partner_board_enabled": _coerce_bool(payload.get("partner_board_enabled", True)),
        "rules_markdown": str(payload.get("rules_markdown") or ""),
        "refund_policy_markdown": str(payload.get("refund_policy_markdown") or ""),
        "sponsor_markdown": str(payload.get("sponsor_markdown") or ""),
        "updated_at": _now_iso(),
    }
    resp = (
        supabase.table("tournament_registration_settings")
        .upsert(clean, on_conflict="tournament_id")
        .execute()
    )
    return _safe_first(resp) or clean


def list_registration_days(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_registration_days")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("sort_order")
        .execute()
    )
    return _safe_data(resp)


def list_event_options(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_event_options")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("sort_order")
        .execute()
    )
    return _safe_data(resp)


def count_tournament_registrations(supabase, tournament_id: str) -> int:
    resp = (
        supabase.table("tournament_registrations")
        .select("id", count="exact")
        .eq("tournament_id", str(tournament_id))
        .execute()
    )
    try:
        return int(resp.count or 0)
    except Exception:
        rows = _safe_data(resp)
        return len(rows)


def replace_registration_configuration(
    supabase,
    *,
    tournament_id: str,
    days: list[dict[str, Any]],
    event_options: list[dict[str, Any]],
    allow_replace_with_registrations: bool = False,
) -> None:
    registration_count = count_tournament_registrations(supabase, tournament_id)
    if registration_count and not allow_replace_with_registrations:
        raise ValueError(
            "This tournament already has registrations. Freeze the current registration form or create a new tournament before replacing days and events."
        )

    (
        supabase.table("tournament_event_options")
        .delete()
        .eq("tournament_id", str(tournament_id))
        .execute()
    )
    (
        supabase.table("tournament_registration_days")
        .delete()
        .eq("tournament_id", str(tournament_id))
        .execute()
    )

    if days:
        (
            supabase.table("tournament_registration_days")
            .insert(days)
            .execute()
        )
    if event_options:
        (
            supabase.table("tournament_event_options")
            .insert(event_options)
            .execute()
        )


def list_registrations(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_registrations")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("submitted_at", desc=True)
        .execute()
    )
    return _safe_data(resp)


def list_registration_selections(supabase, tournament_id: str) -> list[dict[str, Any]]:
    resp = (
        supabase.table("tournament_registration_selections")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .order("created_at")
        .execute()
    )
    return _safe_data(resp)


def _get_existing_registration_by_email(supabase, tournament_id: str, email: str) -> dict[str, Any] | None:
    resp = (
        supabase.table("tournament_registrations")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .eq("email", _normalize_email(email))
        .limit(1)
        .execute()
    )
    return _safe_first(resp)


def save_registration(supabase, *, tournament_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    email = _normalize_email(payload.get("email"))
    if not email:
        raise ValueError("Email is required.")

    display_name = str(payload.get("display_name") or "").strip()
    if not display_name:
        first_name = str(payload.get("first_name") or "").strip()
        last_name = str(payload.get("last_name") or "").strip()
        display_name = " ".join(part for part in [first_name, last_name] if part).strip()
    if not display_name:
        raise ValueError("Player name is required.")

    existing = _get_existing_registration_by_email(supabase, tournament_id, email)
    registration_id = str(existing.get("id")) if existing else _uid("reg")
    submitted_at = _now_iso()

    reg_row = {
        "id": registration_id,
        "tournament_id": str(tournament_id),
        "submitted_at": submitted_at,
        "status": str(payload.get("status") or "pending").lower(),
        "payment_status": str(payload.get("payment_status") or "unpaid").lower(),
        "first_name": str(payload.get("first_name") or "").strip() or None,
        "last_name": str(payload.get("last_name") or "").strip() or None,
        "display_name": display_name,
        "email": email,
        "phone": str(payload.get("phone") or "").strip() or None,
        "dupr_id": str(payload.get("dupr_id") or "").strip() or None,
        "doubles_skill": payload.get("doubles_skill"),
        "singles_skill": payload.get("singles_skill"),
        "age": payload.get("age"),
        "age_bracket": str(payload.get("age_bracket") or "").strip() or None,
        "gender": str(payload.get("gender") or "").strip() or None,
        "notes": str(payload.get("notes") or "").strip() or None,
        "wants_partner_board_contact": _coerce_bool(payload.get("wants_partner_board_contact", False)),
        "updated_at": submitted_at,
    }

    (
        supabase.table("tournament_registrations")
        .upsert(reg_row, on_conflict="id")
        .execute()
    )

    (
        supabase.table("tournament_registration_selections")
        .delete()
        .eq("registration_id", registration_id)
        .execute()
    )

    rows: list[dict[str, Any]] = []
    for index, selection in enumerate(payload.get("selections") or []):
        if not selection.get("event_option_id"):
            continue
        rows.append(
            {
                "id": str(selection.get("id") or _uid("sel")),
                "tournament_id": str(tournament_id),
                "registration_id": registration_id,
                "registration_day_id": str(selection.get("registration_day_id")),
                "event_option_id": str(selection.get("event_option_id")),
                "partner_mode": str(selection.get("partner_mode") or "NONE").upper(),
                "partner_name": str(selection.get("partner_name") or "").strip() or None,
                "partner_email": _normalize_email(selection.get("partner_email")) or None,
                "partner_phone": str(selection.get("partner_phone") or "").strip() or None,
                "partner_dupr_id": str(selection.get("partner_dupr_id") or "").strip() or None,
                "partner_skill": selection.get("partner_skill"),
                "partner_age": selection.get("partner_age"),
                "partner_note": str(selection.get("partner_note") or "").strip() or None,
                "show_on_partner_board": _coerce_bool(selection.get("show_on_partner_board", False)),
                "sort_order": index,
                "created_at": submitted_at,
            }
        )

    if rows:
        (
            supabase.table("tournament_registration_selections")
            .insert(rows)
            .execute()
        )

    return {
        "registration_id": registration_id,
        "submitted_at": submitted_at,
        "selection_count": len(rows),
    }


def get_public_tournament_bundle(
    supabase,
    *,
    club_id: str,
    tournament_id: str | None = None,
    registration_slug: str | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    if registration_slug:
        settings_resp = (
            supabase.table("tournament_registration_settings")
            .select("*")
            .eq("registration_slug", str(registration_slug))
            .limit(1)
            .execute()
        )
        settings = _safe_first(settings_resp)
        if not settings:
            return None, None, [], []
        tournament_id = str(settings.get("tournament_id"))
    elif tournament_id:
        settings = get_registration_settings(supabase, str(tournament_id))
    else:
        return None, None, [], []

    tournament = get_tournament_record(supabase, str(tournament_id))
    if not tournament or str(tournament.get("club_id")) != str(club_id):
        return None, None, [], []

    days = list_registration_days(supabase, str(tournament_id))
    event_options = list_event_options(supabase, str(tournament_id))
    return tournament, settings, days, event_options


def list_open_public_tournaments(supabase, club_id: str) -> list[dict[str, Any]]:
    settings_rows = _safe_data(
        supabase.table("tournament_registration_settings").select("*").eq("registration_status", "open").execute()
    )
    out: list[dict[str, Any]] = []
    for settings in settings_rows:
        tournament = get_tournament_record(supabase, str(settings.get("tournament_id")))
        if not tournament:
            continue
        if str(tournament.get("club_id")) != str(club_id):
            continue
        out.append({"tournament": tournament, "settings": settings})
    out.sort(key=lambda row: str(row.get("tournament", {}).get("created_at") or ""), reverse=True)
    return out


def build_registration_state(supabase, tournament: dict[str, Any], settings: dict[str, Any], days: list[dict[str, Any]], event_options: list[dict[str, Any]]) -> dict[str, Any]:
    registrations = list_registrations(supabase, str(tournament.get("id")))
    selections = list_registration_selections(supabase, str(tournament.get("id")))
    return compile_tournament_registration_state(
        tournament=tournament,
        settings=settings,
        days=days,
        event_options=event_options,
        registrations=registrations,
        selections=selections,
    )


def build_public_urls(*, base_url: str, tournament_id: str, registration_slug: str | None = None) -> dict[str, str]:
    tournament_id = str(tournament_id)
    base_url = str(base_url or "").rstrip("/")
    reg_q = f"tournament={registration_slug}" if registration_slug else f"tournament_id={tournament_id}"
    board_q = reg_q
    return {
        "registration": f"{base_url}/?public=1&page=tournament_registration&{reg_q}",
        "partner_board": f"{base_url}/?public=1&page=tournament_partner_board&{board_q}",
        "admin_manager": f"{base_url}/?page=tournament_manager&tournament_id={tournament_id}",
        "admin_operations": f"{base_url}/?page=tournaments&tournament_id={tournament_id}",
    }


def registration_is_open(settings: dict[str, Any]) -> tuple[bool, str | None]:
    status = str(settings.get("registration_status") or "draft").lower()
    if status != "open":
        return False, "Registration is not currently open."

    now = datetime.now(timezone.utc)
    open_at = settings.get("registration_open_at")
    close_at = settings.get("registration_close_at")

    if open_at:
        try:
            if datetime.fromisoformat(str(open_at).replace("Z", "+00:00")) > now:
                return False, "Registration has not opened yet."
        except Exception:
            pass
    if close_at:
        try:
            if datetime.fromisoformat(str(close_at).replace("Z", "+00:00")) < now:
                return False, "Registration is closed."
        except Exception:
            pass
    return True, None
