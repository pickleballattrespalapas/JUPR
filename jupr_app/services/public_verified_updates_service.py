from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
from typing import Any

from jupr_app.domain.notifications.player_profile_update_repo import (
    REQUEST_STATUS_ACTIVE,
    REQUEST_STATUS_PENDING,
    create_public_request,
    get_open_or_active_subscription,
    normalize_email,
)


PUBLIC_VERIFIED_UPDATE_REQUESTS_PER_EMAIL_PER_HOUR = 5
_EMAIL_PATTERN = re.compile(
    r"^[A-Z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?"
    r"(?:\.[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?)+$",
    re.IGNORECASE,
)


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _validated_email(value: Any) -> str:
    email = str(value or "").strip()
    if len(email) > 320 or any(char.isspace() for char in email):
        raise ValueError("Enter a valid email address.")
    if not _EMAIL_PATTERN.fullmatch(email):
        raise ValueError("Enter a valid email address.")
    local, domain = email.rsplit("@", 1)
    if len(local) > 64 or len(domain) > 253 or local.startswith(".") or local.endswith(".") or ".." in local:
        raise ValueError("Enter a valid email address.")
    return email


def _open_subscriptions_by_player(
    supabase: Any,
    *,
    club_id: str,
    player_ids: list[int],
) -> dict[int, dict[str, Any]]:
    if not player_ids:
        return {}
    rows = _safe_rows(
        supabase.table("player_profile_update_subscriptions")
        .select("player_id,request_status,created_at")
        .eq("club_id", club_id)
        .in_("player_id", player_ids)
        .in_("request_status", [REQUEST_STATUS_PENDING, REQUEST_STATUS_ACTIVE])
        .order("created_at", desc=True)
        .execute()
    )
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        player_id = _safe_int(row.get("player_id"))
        if player_id is not None and player_id not in result:
            result[player_id] = row
    return result


def _recent_email_request_count(supabase: Any, *, club_id: str, normalized_email: str) -> int:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    rows = _safe_rows(
        supabase.table("player_profile_update_subscriptions")
        .select("id")
        .eq("club_id", club_id)
        .eq("email_normalized", normalized_email)
        .gte("created_at", cutoff)
        .limit(PUBLIC_VERIFIED_UPDATE_REQUESTS_PER_EMAIL_PER_HOUR + 1)
        .execute()
    )
    return len(rows)


def _player_payload(row: dict[str, Any], *, open_request: dict[str, Any] | None = None) -> dict[str, Any]:
    pid = _safe_int(row.get("id"))
    status = str((open_request or {}).get("request_status") or "").strip().lower() or None
    return {
        "id": pid,
        "name": _clean_text(row.get("name"), limit=160) or (f"Player {pid}" if pid is not None else "Player"),
        "is_active": bool(row.get("active", row.get("is_active", True))),
        "already_requested": status in {"pending_admin_review", "active"},
        "request_status": status,
    }


def list_public_verified_update_player_options(supabase: Any, *, club_id: str, q: str | None = None, limit: int = 500) -> dict[str, Any]:
    clean_club = _clean_text(club_id, limit=80)
    if not clean_club:
        raise ValueError("club_id is required")
    try:
        query = supabase.table("players").select("id,name,active,is_active").eq("club_id", clean_club).order("name", desc=False)
        rows = _safe_rows(query.limit(max(1, min(int(limit or 500), 1000))).execute())
    except Exception:
        rows = _safe_rows(supabase.table("players").select("id,name").eq("club_id", clean_club).execute())
    open_by_player = _open_subscriptions_by_player(
        supabase,
        club_id=clean_club,
        player_ids=[pid for row in rows if (pid := _safe_int(row.get("id"))) is not None],
    )
    search = _clean_text(q, limit=80).lower()
    options: list[dict[str, Any]] = []
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is None:
            continue
        name = _clean_text(row.get("name"), limit=160)
        if search and search not in name.lower():
            continue
        options.append(_player_payload(row, open_request=open_by_player.get(pid)))
    return {"ok": True, "mode": "verified_updates_player_options", "players": options[: max(1, min(int(limit or 500), 1000))], "count": len(options)}


def _fetch_player(supabase: Any, *, club_id: str, player_id: int) -> dict[str, Any] | None:
    return _safe_first(
        supabase.table("players")
        .select("id,name,active,is_active")
        .eq("club_id", str(club_id))
        .eq("id", int(player_id))
        .limit(1)
        .execute()
    )


def create_public_verified_update_request(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    email: str,
    request_note: str | None = None,
    honeypot: str | None = None,
) -> dict[str, Any]:
    clean_club = _clean_text(club_id, limit=80)
    if not clean_club:
        raise ValueError("club_id is required")
    if _clean_text(honeypot, limit=100):
        # Bot-safe no-op. Do not reveal the honeypot decision to the requester.
        return {"ok": True, "mode": "verified_updates_request_accepted", "status": "accepted", "bot_trap": True}
    pid = _safe_int(player_id)
    if pid is None:
        raise ValueError("player_id is required")
    player = _fetch_player(supabase, club_id=clean_club, player_id=pid)
    if player is None:
        raise ValueError("player not found")
    clean_email = _validated_email(email)
    normalized_email = normalize_email(clean_email)
    existing = get_open_or_active_subscription(supabase, clean_club, pid)
    if existing is not None:
        if normalize_email(str(existing.get("email_normalized") or existing.get("email") or "")) == normalized_email:
            return {
                "ok": True,
                "mode": "verified_updates_request_create",
                "subscription_id": str(existing.get("id") or ""),
                "request_status": existing.get("request_status"),
                "deduplicated": True,
                "player": _player_payload(player, open_request=existing),
            }
        existing_status = str(existing.get("request_status") or "").strip().lower()
        if existing_status == REQUEST_STATUS_ACTIVE:
            raise ValueError("This player already has an active verified subscriber.")
        raise ValueError("A verified updates request is already pending for this player.")

    if _recent_email_request_count(supabase, club_id=clean_club, normalized_email=normalized_email) >= PUBLIC_VERIFIED_UPDATE_REQUESTS_PER_EMAIL_PER_HOUR:
        raise ValueError("Too many recent verified update requests. Try again later.")

    try:
        row = create_public_request(
            supabase,
            club_id=clean_club,
            player_id=pid,
            email=clean_email,
            request_note=_clean_text(request_note, limit=1000) or None,
        )
    except ValueError:
        # The unique open-request index is the final race-safe dedupe gate.
        latest = get_open_or_active_subscription(supabase, clean_club, pid)
        if latest is not None and normalize_email(str(latest.get("email_normalized") or latest.get("email") or "")) == normalized_email:
            row = latest
            deduplicated = True
        else:
            raise
    else:
        deduplicated = False
    return {
        "ok": True,
        "mode": "verified_updates_request_create",
        "subscription_id": str(row.get("id") or ""),
        "request_status": row.get("request_status"),
        "deduplicated": deduplicated,
        "player": _player_payload(player, open_request=row),
    }


def get_public_verified_update_request_status(supabase: Any, *, club_id: str, player_id: int) -> dict[str, Any]:
    clean_club = _clean_text(club_id, limit=80)
    pid = _safe_int(player_id)
    if not clean_club or pid is None:
        raise ValueError("club_id and player_id are required")
    player = _fetch_player(supabase, club_id=clean_club, player_id=pid)
    if player is None:
        raise ValueError("player not found")
    open_row = get_open_or_active_subscription(supabase, clean_club, pid)
    return {"ok": True, "mode": "verified_updates_request_status", "player": _player_payload(player, open_request=open_row)}
