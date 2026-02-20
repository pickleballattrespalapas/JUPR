from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Iterable

from jupr_app.data.sb_write import sb_rpc, sb_update, sb_upsert


BADGE_QUEUE_TABLE = "badge_eval_queue"


def _build_event_key(*, event_type: str, match_id: str | None, context_id: str | None, player_ids: list[int], payload: dict[str, Any]) -> str:
    normalized_match_id = str(match_id or "").strip()
    if normalized_match_id:
        return f"match:{normalized_match_id}"

    seed = {
        "event_type": str(event_type or "").strip(),
        "context_id": str(context_id or "").strip(),
        "player_ids": [int(pid) for pid in sorted(player_ids)],
        "payload": payload or {},
    }
    encoded = json.dumps(seed, sort_keys=True, separators=(",", ":"), default=str)
    return f"event:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"



def enqueue_badge_eval(
    supabase: Any,
    *,
    club_id: str,
    event_type: str,
    player_ids: Iterable[int] | None = None,
    context_id: str | None = None,
    match_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    if supabase is None or not club_id or not event_type:
        return
    payload_json = dict(payload or {})
    normalized_player_ids = [int(pid) for pid in (player_ids or [])]
    row = {
        "club_id": str(club_id),
        "event_type": str(event_type),
        "event_key": _build_event_key(
            event_type=str(event_type),
            match_id=match_id,
            context_id=context_id,
            player_ids=normalized_player_ids,
            payload=payload_json,
        ),
        "player_ids": normalized_player_ids,
        "context_id": context_id,
        "match_id": match_id,
        "payload_json": payload_json,
        "status": "pending",
    }
    sb_upsert(supabase, BADGE_QUEUE_TABLE, row, conflict="club_id,event_type,event_key")


def dequeue_badge_eval(supabase: Any, *, club_id: str, max_jobs: int = 1) -> dict[str, Any] | None:
    if supabase is None:
        return None
    if not str(club_id or "").strip():
        raise ValueError("club_id is required for dequeue_badge_eval")

    resp = sb_rpc(
        supabase,
        "dequeue_badge_eval_jobs",
        {
            "p_club_id": str(club_id),
            "p_limit": int(max(1, max_jobs)),
        },
    )
    rows = getattr(resp, "data", None) or []
    if not rows:
        return None
    return dict(rows[0])

def ack_badge_eval(
    supabase: Any,
    *,
    job_id: str,
    status: str,
    error: str | None = None,
) -> None:
    if supabase is None or not job_id:
        return
    payload: dict[str, Any] = {
        "status": status,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }
    if error:
        payload["last_error"] = error
    sb_update(supabase, BADGE_QUEUE_TABLE, payload, filters={"id": job_id})


