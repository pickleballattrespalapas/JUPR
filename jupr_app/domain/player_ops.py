# jupr_app/domain/player_ops.py
from __future__ import annotations

from typing import Tuple

from postgrest.exceptions import APIError


def _normalized_player_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())

def safe_add_player(
    *,
    supabase,
    club_id: str,
    name: str,
    rating_jupr: float,
) -> Tuple[bool, str]:
    """
    Idempotent player creation for JUPR.

    Behavior:
    - Inserts player using ON CONFLICT (club_id, normalized_name)
    - Returns the upserted row id directly from the upsert response
    - Deterministic and race-safe
    """

    clean_name = str(name or "").strip()
    if not clean_name:
        return False, "Blank name."

    try:
        elo = float(rating_jupr) * 400.0
    except Exception:
        return False, "Invalid rating."

    normalized_name = _normalized_player_name(clean_name)

    payload = {
        "club_id": str(club_id),
        "name": clean_name,
        "normalized_name": normalized_name,
        "active": True,
        "rating": float(elo),
        "starting_rating": float(elo),
        "wins": 0,
        "losses": 0,
        "matches_played": 0,
        "last_game_at": None,
        "inactive_at": None,
    }

    try:
        resp = (
            supabase
            .table("players")
            .upsert(payload, on_conflict="club_id,normalized_name")
            .execute()
        )

        if resp.data and len(resp.data) > 0 and resp.data[0].get("id"):
            return True, resp.data[0]["id"]

        existing = (
            supabase
            .table("players")
            .select("id")
            .eq("club_id", str(club_id))
            .eq("normalized_name", normalized_name)
            .limit(1)
            .execute()
            .data
            or []
        )
        if existing and existing[0].get("id"):
            return True, existing[0]["id"]

        return False, "Upsert did not return an id"

    except APIError as e:
        if getattr(e, "code", "") == "42P10":
            return False, (
                "Schema mismatch: players upsert requires a unique index or constraint on "
                "(club_id, normalized_name)."
            )
        return False, str(e)
