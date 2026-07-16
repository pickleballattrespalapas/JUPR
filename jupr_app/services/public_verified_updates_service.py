from __future__ import annotations

from typing import Any

from jupr_app.domain.notifications.player_profile_update_repo import create_public_request, get_open_or_active_subscription


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
    search = _clean_text(q, limit=80).lower()
    options: list[dict[str, Any]] = []
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is None:
            continue
        name = _clean_text(row.get("name"), limit=160)
        if search and search not in name.lower():
            continue
        options.append(_player_payload(row))
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
    row = create_public_request(
        supabase,
        club_id=clean_club,
        player_id=pid,
        email=email,
        request_note=request_note,
    )
    return {
        "ok": True,
        "mode": "verified_updates_request_create",
        "subscription_id": str(row.get("id") or ""),
        "request_status": row.get("request_status"),
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
