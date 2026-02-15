# jupr_app/domain/player_ops.py
from __future__ import annotations

from typing import Tuple


def safe_add_player(
    *,
    supabase,
    club_id: str,
    name: str,
    rating_jupr: float,
) -> Tuple[bool, str]:
    """
    Ensures a player exists in `players`.

    - Attempts insert (rating_jupr stored as ELO x400)
    - If unique constraint indicates the name already exists for this club, treat as success
      and rely on a re-fetch to get the id.

    Returns (ok, error_message).
    """
    nm = str(name or "").strip()
    if not nm:
        return False, "Blank name."

    try:
        elo = float(rating_jupr) * 400.0
    except Exception:
        return False, "Invalid rating."

    payload = {
        "club_id": str(club_id),
        "name": nm,
        "rating": float(elo),
        "starting_rating": float(elo),
        "wins": 0,
        "losses": 0,
        "matches_played": 0,
        "active": True,
        "last_game_at": None,
        "inactive_at": None,
    }

    resp = supabase.table("players").insert(payload).execute()

    # Supabase does not raise exceptions for PostgREST errors.
    # Errors must be inspected on the response object.
    error = getattr(resp, "error", None)

    if error:
        msg = str(error)

        # If duplicate key (23505) on normalized name, the player already exists — treat as OK.
        # supabase-py surfaces the code in the string; we match conservatively.
        if "23505" in msg or "uq_players_club_name_active" in msg:
            return True, ""

        import logging

        logging.warning(f"safe_add_player insert failed: {msg}")
        return False, msg

    return True, ""
