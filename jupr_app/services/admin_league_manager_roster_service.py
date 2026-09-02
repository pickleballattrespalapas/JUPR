from __future__ import annotations

from typing import Any
from uuid import uuid4

from jupr_app.services.admin_league_manager_service import (
    get_admin_league_manager_detail,
    is_admin_league_manager_enabled,
)
from jupr_app.services.admin_league_manager_roster_batch_service import (
    CONFIRM_ROSTER_BATCH,
    update_admin_league_manager_roster_batch,
)

CONFIRM_SAVE_ROSTER = "SAVE ROSTER"
ACTIONS = {"activate", "deactivate"}


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


def _fetch_league_rating(
    supabase: Any,
    *,
    club_id: str,
    league_name: str,
    player_id: int,
) -> dict[str, Any] | None:
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
    idempotency_key: str = "",
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
    operation_key = str(idempotency_key or "").strip()
    if not operation_key:
        operation_key = f"legacy-roster:{uuid4()}"
    receipt = update_admin_league_manager_roster_batch(
        supabase,
        club_id=str(club_id),
        league_name=clean_league,
        action=clean_action,
        player_ids=[pid],
        starting_rating=starting_rating if clean_action == "activate" else None,
        idempotency_key=operation_key,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        confirmation_text=CONFIRM_ROSTER_BATCH,
        source=str(source or ""),
    )
    receipt_rows = receipt.get("league_ratings")
    after = (
        next(
            (
                dict(row)
                for row in receipt_rows
                if isinstance(row, dict)
                and int(row.get("player_id") or 0) == pid
            ),
            None,
        )
        if isinstance(receipt_rows, list)
        else None
    )
    warnings: list[str] = []
    if after is None:
        try:
            after = _fetch_league_rating(
                supabase,
                club_id=str(club_id),
                league_name=clean_league,
                player_id=pid,
            )
        except Exception:
            after = None
    if not after:
        after = {}
        warnings.append(
            "Roster membership committed, but its player row could not be read back."
        )

    try:
        detail = get_admin_league_manager_detail(
            supabase,
            club_id=str(club_id),
            league_name=clean_league,
        )
    except Exception:
        detail = {}
        warnings.append(
            "Roster membership committed, but refreshed league detail is unavailable."
        )
    return {
        "ok": True,
        "mode": "league_manager_roster_membership_update",
        "league_name": clean_league,
        "player_id": pid,
        "action": clean_action,
        "league_rating": after,
        "detail": detail,
        "warnings": warnings,
        "operation_id": receipt.get("operation_id"),
        "idempotent": bool(receipt.get("idempotent")),
    }
