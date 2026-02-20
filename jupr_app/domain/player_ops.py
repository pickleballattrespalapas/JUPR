# jupr_app/domain/player_ops.py
from __future__ import annotations

from typing import Tuple


def _normalized_player_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def safe_add_player(
    *,
    supabase,
    club_id: str,
    name: str,
    rating_jupr: float,
) -> Tuple[bool, str]:
    raise Exception("SAFE_ADD_PLAYER WAS CALLED")
