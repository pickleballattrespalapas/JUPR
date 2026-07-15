from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_league_manager_service import (
    get_admin_league_manager_detail,
    is_admin_league_manager_enabled,
)

CONFIRM_SAVE_ROSTER = "SAVE ROSTER"
TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
ACTIONS = {"activate", "deactivate"}


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _clean_text(value: Any, *, limit: int = 200) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _safe_int(value: Any, *, field: str = "value") -> int:
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"{field} must be a whole number.") from exc


def _safe_float(value: Any, *, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception as exc:
        raise ValueError("rating must be a number.") from exc


def _fetch_player(supabase: Any, *, club_id: str, player_id: int) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("players")
        .select("id,name,rating")
        .eq("club_id", str(club_id))
        .eq("id", int(player_id))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _fetch_league_rating(supabase: Any, *, club_id: str, league_name: str, player_id: int) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("league_ratings")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("league_name", str(league_name))
        .eq("player_id", int(player_id))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def _starting_elo(value: Any, *, player: dict[str, Any]) -> float:
    parsed = _safe_float(value, default=None)
    if parsed is None:
        parsed = _safe_float(player.get("rating"), default=1200.0)
    assert parsed is not None
    if parsed <= 20:
        parsed *= 400.0
    if parsed < 400 or parsed > 2800:
        raise ValueError("rating must be an Elo value from 400-2800 or a JUPR value from 1.0-7.0.")
    return float(parsed)


def update_admin_league_manager_roster_membership(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    player_id: Any,
    action: str,
    starting_rating: Any = None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_league_manager_roster_update",
) -> dict[str, Any]:
    if not is_admin_league_manager_enabled():
        raise PermissionError("Next League Manager is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_SAVE_ROSTER:
        raise ValueError(f"Type {CONFIRM_SAVE_ROSTER} to save league roster membership.")
    clean_league = _clean_text(league_name, limit=120)
    if not clean_league:
        raise ValueError("league_name is required")
    clean_action = _clean_text(action, limit=40).lower()
    if clean_action not in ACTIONS:
        raise ValueError("action must be activate or deactivate.")
    pid = _safe_int(player_id, field="player_id")
    player = _fetch_player(supabase, club_id=str(club_id), player_id=pid)
    if not player:
        raise ValueError("player not found")
    before = _fetch_league_rating(supabase, club_id=str(club_id), league_name=clean_league, player_id=pid)

    if clean_action == "activate":
        start = _starting_elo(starting_rating, player=player)
        patch = {
            "club_id": str(club_id),
            "player_id": pid,
            "league_name": clean_league,
            "rating": start,
            "starting_rating": start,
            "wins": int((before or {}).get("wins", 0) or 0),
            "losses": int((before or {}).get("losses", 0) or 0),
            "matches_played": int((before or {}).get("matches_played", 0) or 0),
            "is_active": True,
            "inactive_at": None,
        }
        if before:
            updated = _safe_rows(supabase.table("league_ratings").update(patch).eq("id", before.get("id")).execute())
            after = updated[0] if updated else {**before, **patch}
        else:
            inserted = _safe_rows(supabase.table("league_ratings").insert(patch).execute())
            after = inserted[0] if inserted else patch
    else:
        if not before:
            raise ValueError("player is not currently in this league")
        patch = {"is_active": False, "inactive_at": _now_iso()}
        updated = _safe_rows(supabase.table("league_ratings").update(patch).eq("id", before.get("id")).execute())
        after = updated[0] if updated else {**before, **patch}

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_league_manager_roster_membership_admin",
        entity_type="league_ratings",
        entity_id=f"{clean_league}:{pid}",
        before_json=before or {},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "league_name": clean_league,
            "player_id": pid,
            "player_name": player.get("name"),
            "action": clean_action,
            "league_rating": after,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")

    detail = get_admin_league_manager_detail(supabase, club_id=str(club_id), league_name=clean_league)
    return {
        "ok": True,
        "mode": "league_manager_roster_membership_update",
        "league_name": clean_league,
        "player_id": pid,
        "action": clean_action,
        "league_rating": after,
        "detail": detail,
        "warnings": warnings,
    }
