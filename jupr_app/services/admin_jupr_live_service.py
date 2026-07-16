from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.live_session_repo import abandon_expired_live_sessions

TRUTHY = {"1", "true", "yes", "y", "on"}
SESSION_STATUSES = {"active", "completed", "abandoned", "archived"}
CONFIRM_CREATE = "CREATE LIVE SESSION"
CONFIRM_STATUS = "SAVE LIVE SESSION"


def is_admin_jupr_live_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE", "").strip().lower() in TRUTHY


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _session_payload(row: dict[str, Any]) -> dict[str, Any]:
    state = _as_dict(row.get("state"))
    return {
        "id": str(row.get("id") or ""),
        "club_id": str(row.get("club_id") or ""),
        "session_key": str(row.get("session_key") or ""),
        "title": _clean_text(row.get("title"), limit=160),
        "status": str(row.get("status") or "active"),
        "source": str(row.get("source") or "jupr_live_admin"),
        "created_by_email": row.get("created_by_email"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "last_seen_at": row.get("last_seen_at"),
        "expires_at": row.get("expires_at"),
        "event_type": state.get("event_type") or state.get("eventType"),
        "state": state,
    }


def build_admin_jupr_live_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        return {"enabled": False, "status": "guarded_off", "warnings": ["Enable JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE to manage JUPR Live sessions in Next."]}
    counts = {"active": 0, "completed": 0, "abandoned": 0, "archived": 0}
    if supabase is not None:
        try:
            rows = _safe_rows(supabase.table("live_sessions").select("status").eq("club_id", str(club_id)).execute())
            for row in rows:
                status = str(row.get("status") or "").lower()
                if status in counts:
                    counts[status] += 1
        except Exception:
            pass
    return {"enabled": True, "status": "ready_for_jupr_live_admin", "counts": counts, "warnings": []}


def list_admin_jupr_live_sessions(supabase: Any, *, club_id: str, status: str | None = None, limit: int = 100) -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        raise PermissionError("Next JUPR Live Admin is disabled.")
    try:
        abandon_expired_live_sessions(supabase)
    except Exception:
        pass
    query = supabase.table("live_sessions").select("*").eq("club_id", str(club_id))
    clean_status = _clean_text(status, limit=40).lower()
    if clean_status:
        query = query.eq("status", clean_status)
    try:
        rows = _safe_rows(query.order("updated_at", desc=True).limit(max(1, min(int(limit or 100), 300))).execute())
    except Exception:
        rows = _safe_rows(query.execute())
    sessions = [_session_payload(row) for row in rows]
    return {"ok": True, "mode": "jupr_live_admin_sessions", "sessions": sessions, "count": len(sessions)}


def get_admin_jupr_live_session(supabase: Any, *, club_id: str, session_key: str) -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        raise PermissionError("Next JUPR Live Admin is disabled.")
    row = _safe_first(
        supabase.table("live_sessions")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("session_key", str(session_key))
        .limit(1)
        .execute()
    )
    if row is None:
        raise ValueError("live session not found")
    return {"ok": True, "mode": "jupr_live_admin_session_detail", "session": _session_payload(row)}


def create_admin_jupr_live_session(
    supabase: Any,
    *,
    club_id: str,
    title: str,
    event_type: str,
    participant_names: list[str] | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_jupr_live_admin_create",
) -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        raise PermissionError("Next JUPR Live Admin is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_CREATE:
        raise ValueError(f"Type {CONFIRM_CREATE} to create a durable JUPR Live session.")
    clean_event_type = _clean_text(event_type or "round_robin", limit=60).lower().replace(" ", "_")
    if clean_event_type not in {"round_robin", "league_ladder", "league", "ladder", "tournament"}:
        raise ValueError("unsupported JUPR Live event type")
    session_key = uuid4().hex
    now = _now_iso()
    names = [_clean_text(name, limit=160) for name in (participant_names or []) if _clean_text(name, limit=160)]
    state = {"event_type": clean_event_type, "participant_names": names, "source": source, "created_from_next_admin": True}
    payload = {
        "club_id": str(club_id),
        "session_key": session_key,
        "status": "active",
        "title": _clean_text(title, limit=160) or "JUPR Live Session",
        "state": state,
        "source": "jupr_live_admin",
        "created_by_email": str(actor_email or "").strip().lower() or None,
        "created_at": now,
        "updated_at": now,
        "last_seen_at": now,
        "expires_at": (datetime.now(timezone.utc) + timedelta(hours=18)).isoformat(),
    }
    inserted = _safe_first(supabase.table("live_sessions").insert(payload).execute()) or payload
    _audit(supabase, club_id=club_id, actor_email=actor_email, actor_role=actor_role, action_type="create_jupr_live_session_admin", entity_id=session_key, before_json={}, after_json={"session": _session_payload(inserted)}, source=source)
    return {"ok": True, "mode": "jupr_live_admin_session_create", "session": _session_payload(inserted)}


def update_admin_jupr_live_session_status(
    supabase: Any,
    *,
    club_id: str,
    session_key: str,
    status: str,
    title: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_jupr_live_admin_status",
) -> dict[str, Any]:
    if not is_admin_jupr_live_enabled():
        raise PermissionError("Next JUPR Live Admin is disabled.")
    if _clean_text(confirmation_text, limit=80).upper() != CONFIRM_STATUS:
        raise ValueError(f"Type {CONFIRM_STATUS} to update the JUPR Live session.")
    clean_status = _clean_text(status, limit=40).lower()
    if clean_status not in SESSION_STATUSES:
        raise ValueError("unsupported live session status")
    before = _safe_first(supabase.table("live_sessions").select("*").eq("club_id", str(club_id)).eq("session_key", str(session_key)).limit(1).execute())
    if before is None:
        raise ValueError("live session not found")
    patch: dict[str, Any] = {"status": clean_status, "updated_at": _now_iso(), "last_seen_at": _now_iso()}
    if title is not None:
        patch["title"] = _clean_text(title, limit=160) or None
    updated = _safe_first(supabase.table("live_sessions").update(patch).eq("club_id", str(club_id)).eq("session_key", str(session_key)).execute()) or {**before, **patch}
    _audit(supabase, club_id=club_id, actor_email=actor_email, actor_role=actor_role, action_type="update_jupr_live_session_admin", entity_id=session_key, before_json={"session": _session_payload(before)}, after_json={"session": _session_payload(updated)}, source=source)
    return {"ok": True, "mode": "jupr_live_admin_session_update", "session": _session_payload(updated)}


def _audit(supabase: Any, *, club_id: str, actor_email: str, actor_role: str, action_type: str, entity_id: str, before_json: dict[str, Any], after_json: dict[str, Any], source: str) -> None:
    payload = build_activity_payload(
        club_id=str(club_id), actor_email=actor_email, actor_role=actor_role, action_type=action_type, entity_type="live_session", entity_id=str(entity_id),
        before_json=before_json, after_json={"source_client": "fastapi/nextjs", **after_json}, source_page=source, flagged_for_review=True,
    )
    write = write_admin_activity_log(supabase, payload)
    if not write.ok and os.getenv("JUPR_REQUIRE_API_AUDIT_LOG", "").strip().lower() in TRUTHY:
        raise RuntimeError("audit log write required but unavailable")
