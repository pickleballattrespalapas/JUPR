"""Small public sharing read: never load lineups, scores, tokens, or standings."""
from typing import Any

from jupr_app.services.public_play_generator_service import (
    PUBLIC_GENERATOR_MODE, PublicPlayGeneratorError, _clean, _not_expired,
)


def get_generator_share_metadata(supabase: Any, *, club_id: str, session_key: str) -> dict:
    rows = (supabase.table("live_sessions")
            .select("session_key,title,status,expires_at,mode:state->>mode")
            .eq("club_id", str(club_id)).eq("session_key", str(session_key))
            .limit(1).execute().data or [])
    row = rows[0] if rows else None
    if not row or row.get("mode") != PUBLIC_GENERATOR_MODE or not _not_expired(row):
        raise PublicPlayGeneratorError("We couldn’t find this session.")
    return {"ok": True, "session": {
        "session_key": str(row["session_key"]),
        "title": _clean(row.get("title")) or "Play session",
    }}
