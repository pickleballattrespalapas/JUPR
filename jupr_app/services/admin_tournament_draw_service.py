from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_tournament_service import (
    EVENT_OPTION_SELECT,
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    _table_rows_for_tournament,
    is_admin_tournament_admin_enabled,
)

CONFIRM_CREATE_DRAW = "CREATE DRAW"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _event_label(row: dict[str, Any] | None) -> str:
    row = row or {}
    family = _clean_text(row.get("event_family_label"), limit=120)
    division = _clean_text(row.get("division_name") or row.get("label"), limit=120)
    if family and division and family != division:
        return f"{family} / {division}"
    return division or family or "Tournament Draw"


def _fetch_event_option(supabase: Any, *, tournament_id: str, event_option_id: str) -> dict[str, Any] | None:
    if not event_option_id:
        return None
    rows = _table_rows_for_tournament(
        supabase,
        "tournament_event_options",
        EVENT_OPTION_SELECT,
        tournament_id=str(tournament_id),
    )
    for row in rows:
        if _clean_text(row.get("id"), limit=120) == str(event_option_id):
            return row
    return None


def _existing_draws(supabase: Any, *, tournament_id: str) -> list[dict[str, Any]]:
    try:
        return _safe_rows(
            supabase.table("tournament_event_draws")
            .select("*")
            .eq("tournament_id", str(tournament_id))
            .execute()
        )
    except Exception:
        return []


def _draw_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": _clean_text(row.get("id"), limit=120),
        "tournament_id": _clean_text(row.get("tournament_id"), limit=120),
        "registration_day_id": _clean_text(row.get("registration_day_id"), limit=120) or None,
        "event_option_id": _clean_text(row.get("event_option_id"), limit=120) or None,
        "name": _clean_text(row.get("name"), limit=180),
        "status": _clean_text(row.get("status") or "draft", limit=40).lower(),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def create_admin_tournament_draw(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    registration_day_id: str | None = None,
    event_option_id: str | None = None,
    name: str | None = None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_tournament_admin_create_draw",
) -> dict[str, Any]:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_CREATE_DRAW:
        raise ValueError(f"Type {CONFIRM_CREATE_DRAW} to create a tournament draw.")

    clean_tournament_id = _clean_text(tournament_id, limit=120)
    tournament = _first_row(supabase, "tournaments", TOURNAMENT_SELECT, key="id", value=clean_tournament_id)
    if not tournament or str(tournament.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")

    clean_event_option_id = _clean_text(event_option_id, limit=120) or None
    event_option = _fetch_event_option(supabase, tournament_id=clean_tournament_id, event_option_id=clean_event_option_id or "")
    if clean_event_option_id and not event_option:
        raise ValueError("event option not found for this tournament")

    clean_day_id = _clean_text(registration_day_id, limit=120) or None
    if event_option:
        event_day_id = _clean_text(event_option.get("registration_day_id"), limit=120) or None
        if clean_day_id and event_day_id and clean_day_id != event_day_id:
            raise ValueError("registration_day_id does not match the selected event option")
        clean_day_id = event_day_id or clean_day_id

    clean_name = _clean_text(name, limit=180)
    if not clean_name:
        clean_name = f"{_event_label(event_option)} Ops Draw" if event_option else "Tournament Ops Draw"

    for existing in _existing_draws(supabase, tournament_id=clean_tournament_id):
        same_event = _clean_text(existing.get("event_option_id"), limit=120) == (clean_event_option_id or "")
        same_day = _clean_text(existing.get("registration_day_id"), limit=120) == (clean_day_id or "")
        same_name = _clean_text(existing.get("name"), limit=180).lower() == clean_name.lower()
        if same_event and same_day and same_name:
            raise ValueError("A draw with this day, event, and name already exists.")

    now = _now_iso()
    insert_payload = {
        "id": str(uuid.uuid4()),
        "tournament_id": clean_tournament_id,
        "registration_day_id": clean_day_id,
        "event_option_id": clean_event_option_id,
        "name": clean_name,
        "status": "draft",
        "created_at": now,
        "updated_at": now,
    }
    rows = _safe_rows(supabase.table("tournament_event_draws").insert(insert_payload).execute())
    draw = _draw_payload(rows[0] if rows else insert_payload)

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="create_tournament_event_draw_admin",
        entity_type="tournament_event_draw",
        entity_id=str(draw.get("id") or clean_name),
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "draw": draw,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and str(__import__("os").getenv("JUPR_REQUIRE_API_AUDIT_LOG", "")).strip().lower() in {"1", "true", "yes", "y", "on"}:
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "tournament_draw_create", "draw": draw, "warnings": warnings}
