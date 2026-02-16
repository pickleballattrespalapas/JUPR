# jupr_app/domain/player_ops.py
from __future__ import annotations

from typing import Tuple

from postgrest.exceptions import APIError

from jupr_app.data.sb_write import sb_upsert


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
    - If player already exists (active=true), returns existing id
    - Never throws duplicate key error
    - Deterministic and race-safe
    """

    clean_name = str(name or "").strip()
    if not clean_name:
        return False, "Blank name."

    try:
        elo = float(rating_jupr) * 400.0
    except Exception:
        return False, "Invalid rating."

    payload = {
        "club_id": str(club_id),
        "name": clean_name,
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
        resp = sb_upsert(
            supabase,
            "players",
            payload,
            conflict="club_id,normalized_name",
        )

        if resp.data and len(resp.data) > 0:
            return True, resp.data[0]["id"]

        existing = (
            supabase
            .table("players")
            .select("id")
            .eq("club_id", str(club_id))
            .eq("normalized_name", clean_name.lower())
            .eq("active", True)
            .limit(1)
            .execute()
        )

        if existing.data:
            return True, existing.data[0]["id"]

        return False, "Unknown insertion state"

    except APIError as e:
        if getattr(e, "code", None) == "23505":
            existing = (
                supabase
                .table("players")
                .select("id")
                .eq("club_id", str(club_id))
                .eq("normalized_name", clean_name.lower())
                .eq("active", True)
                .limit(1)
                .execute()
            )

            if existing.data:
                return True, existing.data[0]["id"]

        return False, str(e)
