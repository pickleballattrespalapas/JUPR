from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

LIVE_SESSIONS_TABLE = "live_sessions"
LIVE_SESSIONS_INSTALL_MESSAGE = (
    "Durable JUPR Live sessions are not installed in Supabase yet. "
    "Apply supabase/migrations/20260702080000_live_sessions.sql."
)
_RESTORABLE_STATUSES = {"active"}


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value or "").strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _error_payload_text(exc: Exception) -> str:
    pieces = [str(exc)]
    for attr in ("code", "message", "details", "hint"):
        value = getattr(exc, attr, None)
        if value:
            pieces.append(str(value))
    response = getattr(exc, "response", None)
    if response is not None:
        text = getattr(response, "text", None)
        if text:
            pieces.append(str(text))
        json_fn = getattr(response, "json", None)
        if callable(json_fn):
            try:
                payload = json_fn()
            except Exception:
                payload = None
            if payload:
                pieces.append(str(payload))
    return " | ".join(pieces).lower()


def is_missing_live_sessions_table_error(exc: Exception) -> bool:
    payload = _error_payload_text(exc)
    if not payload:
        return False
    mentions_table = LIVE_SESSIONS_TABLE in payload
    has_missing_marker = any(
        marker in payload
        for marker in (
            "pgrst205",
            "42p01",
            "does not exist",
            "undefined table",
            "relation",
            "schema cache",
            "could not find",
        )
    )
    return mentions_table and has_missing_marker


def is_live_session_expired(row: dict[str, Any], *, now: datetime | None = None) -> bool:
    expires_at = _parse_datetime((row or {}).get("expires_at"))
    if expires_at is None:
        return False
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    return expires_at <= current.astimezone(timezone.utc)


def is_restorable_live_session(row: dict[str, Any] | None, *, now: datetime | None = None) -> bool:
    if not row:
        return False
    status = str(row.get("status") or "").strip().lower()
    if status not in _RESTORABLE_STATUSES:
        return False
    return not is_live_session_expired(row, now=now)


def get_live_session(
    supabase,
    *,
    club_id: str,
    session_key: str,
) -> dict[str, Any] | None:
    """Fetch one durable JUPR Live session by club/session key."""
    row = (
        supabase.table(LIVE_SESSIONS_TABLE)
        .select("*")
        .eq("club_id", str(club_id))
        .eq("session_key", str(session_key))
        .limit(1)
        .execute()
    )
    return _safe_first(row)


def upsert_live_session(
    supabase,
    *,
    club_id: str,
    session_key: str,
    state: dict[str, Any],
    title: str | None = None,
    created_by: str | None = None,
    created_by_email: str | None = None,
    expires_at: str | None = None,
    source: str = "jupr_live_admin",
) -> dict[str, Any]:
    """Create or update the recoverable JUPR Live state row."""
    now_iso = _now_iso()
    payload: dict[str, Any] = {
        "club_id": str(club_id),
        "session_key": str(session_key),
        "status": "active",
        "title": str(title).strip() if title else None,
        "state": dict(state or {}),
        "source": str(source or "jupr_live_admin"),
        "updated_at": now_iso,
        "last_seen_at": now_iso,
        "expires_at": expires_at,
    }
    if created_by:
        payload["created_by"] = str(created_by)
    if created_by_email:
        payload["created_by_email"] = str(created_by_email).strip().lower()

    resp = (
        supabase.table(LIVE_SESSIONS_TABLE)
        .upsert(payload, on_conflict="club_id,session_key")
        .execute()
    )
    return _safe_first(resp) or payload


def touch_live_session(
    supabase,
    *,
    club_id: str,
    session_key: str,
    expires_at: str | None = None,
) -> None:
    now_iso = _now_iso()
    patch = {"last_seen_at": now_iso, "updated_at": now_iso}
    if expires_at is not None:
        patch["expires_at"] = expires_at
    (
        supabase.table(LIVE_SESSIONS_TABLE)
        .update(patch)
        .eq("club_id", str(club_id))
        .eq("session_key", str(session_key))
        .execute()
    )


def mark_live_session_abandoned(
    supabase,
    *,
    club_id: str,
    session_key: str,
) -> bool:
    resp = (
        supabase.table(LIVE_SESSIONS_TABLE)
        .update({"status": "abandoned", "updated_at": _now_iso()})
        .eq("club_id", str(club_id))
        .eq("session_key", str(session_key))
        .eq("status", "active")
        .execute()
    )
    return bool(_safe_rows(resp))


def abandon_expired_live_sessions(
    supabase,
    *,
    now_iso: str | None = None,
) -> int:
    """Mark active rows past expires_at as abandoned. Intended for opportunistic cleanup."""
    cutoff = now_iso or _now_iso()
    resp = (
        supabase.table(LIVE_SESSIONS_TABLE)
        .update({"status": "abandoned", "updated_at": cutoff})
        .eq("status", "active")
        .lt("expires_at", cutoff)
        .execute()
    )
    return len(_safe_rows(resp))
