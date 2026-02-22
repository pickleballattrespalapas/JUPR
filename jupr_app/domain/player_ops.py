# jupr_app/domain/player_ops.py
from __future__ import annotations

import logging

from postgrest.exceptions import APIError

logger = logging.getLogger(__name__)


def _normalized_player_name(name: str) -> str:
    return " ".join(str(name or "").strip().lower().split())


def safe_add_player(
    *,
    supabase,
    club_id: str,
    name: str,
    rating_jupr: float,
) -> tuple[bool, str | None]:
    try:
        normalized_name = _normalized_player_name(name)
        if not club_id:
            return False, "club_id is required"
        if not normalized_name:
            return False, "Player name is required"

        payload = {
            "club_id": str(club_id),
            "name": str(name or "").strip(),
            "normalized_name": normalized_name,
            "rating": float(rating_jupr),
        }

        upsert_resp = (
            supabase.table("players")
            .upsert(
                payload,
                on_conflict="club_id,normalized_name",
                returning="representation",
            )
            .execute()
        )
        upsert_rows = upsert_resp.data or []
        if upsert_rows:
            return True, None

        lookup_resp = (
            supabase.table("players")
            .select("id")
            .eq("club_id", str(club_id))
            .eq("normalized_name", normalized_name)
            .limit(1)
            .execute()
        )
        lookup_rows = lookup_resp.data or []
        if lookup_rows:
            return True, None

        return False, "Player create succeeded but no player row was returned."

    except APIError as exc:
        code = str(getattr(exc, "code", "") or "")
        message = str(getattr(exc, "message", "") or str(exc))
        if code == "42P10" or "ON CONFLICT" in message.upper():
            return False, (
                "Schema mismatch: missing unique constraint for "
                "players(club_id, normalized_name)."
            )
        return False, message or "Failed to add player."
    except Exception as exc:
        logger.exception("safe_add_player failed unexpectedly")
        return False, str(exc) or "Failed to add player."
