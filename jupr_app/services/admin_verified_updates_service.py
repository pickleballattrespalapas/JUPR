from __future__ import annotations

import os
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.notifications.player_profile_update_repo import (
    REQUEST_STATUS_ACTIVE,
    REQUEST_STATUS_PENDING,
    approve_request,
    list_subscriptions_by_status,
    mark_unsubscribed,
    reject_request,
)
from jupr_app.services.staging_write_guard import (
    require_staging_communications_mutations,
    staging_communications_mutations_enabled,
)

TRUTHY = {"1", "true", "yes", "y", "on"}
ACTIONS = {"approve", "reject", "unsubscribe"}
CONFIRM = "SAVE VERIFIED REQUEST"


def is_admin_verified_updates_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "").strip().lower() in TRUTHY


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _clean_text(value: Any, *, limit: int = 240) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _mask_email(email: Any) -> str:
    text = str(email or "").strip()
    if "@" not in text:
        return ""
    local, domain = text.split("@", 1)
    return f"{local[:2]}***@{domain}"


def _player_names(supabase: Any, *, club_id: str) -> dict[int, str]:
    try:
        rows = _safe_rows(supabase.table("players").select("id,name").eq("club_id", str(club_id)).execute())
    except Exception:
        rows = []
    result: dict[int, str] = {}
    for row in rows:
        try:
            result[int(row.get("id"))] = _clean_text(row.get("name"), limit=160)
        except Exception:
            continue
    return result


def _row_payload(row: dict[str, Any], names: dict[int, str]) -> dict[str, Any]:
    try:
        pid = int(row.get("player_id"))
    except Exception:
        pid = None
    return {
        "id": str(row.get("id") or ""),
        "player_id": pid,
        "player_name": names.get(int(pid), f"Player {pid}") if pid is not None else "Player",
        "email_masked": _mask_email(row.get("email") or row.get("email_normalized")),
        "request_status": str(row.get("request_status") or ""),
        "request_note": _clean_text(row.get("request_note"), limit=1000),
        "admin_note": _clean_text(row.get("admin_note"), limit=1000),
        "verified_by": row.get("verified_by"),
        "verified_at": row.get("verified_at"),
        "unsubscribed_at": row.get("unsubscribed_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "preferences_json": row.get("preferences_json") if isinstance(row.get("preferences_json"), dict) else {},
    }


def build_admin_verified_updates_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_verified_updates_enabled():
        return {"enabled": False, "mutations_enabled": False, "status": "guarded_off", "warnings": ["Enable JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES to review verified update requests in Next."]}
    counts = {"pending": 0, "active": 0}
    if supabase is not None:
        try:
            pending = list_subscriptions_by_status(supabase, str(club_id), statuses=[REQUEST_STATUS_PENDING], limit=500)
            active = list_subscriptions_by_status(supabase, str(club_id), statuses=[REQUEST_STATUS_ACTIVE], limit=500)
            counts = {"pending": len(pending), "active": len(active)}
        except Exception:
            pass
    mutations_enabled = staging_communications_mutations_enabled()
    warnings = [] if mutations_enabled else [
        "Read-only mode is active. Open the isolated communications write wave "
        "before reviewing verified update requests."
    ]
    return {"enabled": True, "mutations_enabled": mutations_enabled, "status": "ready_for_verified_updates_review", "counts": counts, "warnings": warnings}


def list_admin_verified_update_requests(supabase: Any, *, club_id: str, status: str = "pending", limit: int = 100) -> dict[str, Any]:
    if not is_admin_verified_updates_enabled():
        raise PermissionError("Next verified update requests are disabled.")
    clean_status = _clean_text(status, limit=40).lower()
    statuses = [REQUEST_STATUS_PENDING] if clean_status in {"", "pending"} else [clean_status]
    rows = list_subscriptions_by_status(supabase, str(club_id), statuses=statuses, limit=max(1, min(int(limit or 100), 500)))
    names = _player_names(supabase, club_id=str(club_id))
    return {"ok": True, "mode": "verified_updates_requests_list", "requests": [_row_payload(row, names) for row in rows], "count": len(rows)}


def update_admin_verified_update_request(
    supabase: Any,
    *,
    club_id: str,
    subscription_id: str,
    action: str,
    admin_note: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_verified_updates_request_review",
) -> dict[str, Any]:
    if not is_admin_verified_updates_enabled():
        raise PermissionError("Next verified update requests are disabled.")
    require_staging_communications_mutations()
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM:
        raise ValueError(f"Type {CONFIRM} to update the verified update request.")
    clean_action = _clean_text(action, limit=40).lower()
    if clean_action not in ACTIONS:
        raise ValueError("unsupported verified update request action")
    before = _safe_first(supabase.table("player_profile_update_subscriptions").select("*").eq("club_id", str(club_id)).eq("id", str(subscription_id)).limit(1).execute())
    if before is None:
        raise ValueError("verified update request not found")
    if clean_action == "approve":
        updated = approve_request(supabase, str(subscription_id), verified_by=actor_email, admin_note=admin_note)
    elif clean_action == "reject":
        updated = reject_request(supabase, str(subscription_id), admin_note=admin_note, verified_by=actor_email)
    else:
        updated = mark_unsubscribed(supabase, str(subscription_id))
    names = _player_names(supabase, club_id=str(club_id))
    audit = build_activity_payload(
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="verified_updates_request_review",
        entity_type="player_profile_update_subscription",
        entity_id=str(subscription_id),
        before_json={"request_status": before.get("request_status"), "player_id": before.get("player_id")},
        after_json={"source_client": "fastapi/nextjs", "action": clean_action, "request_status": updated.get("request_status"), "player_id": updated.get("player_id")},
        source_page=source,
        flagged_for_review=True,
    )
    write = write_admin_activity_log(supabase, audit)
    if not write.ok and os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").strip().lower() in TRUTHY:
        raise RuntimeError("audit log write required but unavailable")
    return {"ok": True, "mode": "verified_updates_request_update", "action": clean_action, "request": _row_payload(updated, names), "warnings": [write.warning] if write.warning else []}
