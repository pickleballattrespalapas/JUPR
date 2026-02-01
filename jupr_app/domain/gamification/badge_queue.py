from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable


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
    row = {
        "club_id": str(club_id),
        "event_type": str(event_type),
        "player_ids": [int(pid) for pid in (player_ids or [])],
        "context_id": context_id,
        "match_id": match_id,
        "payload_json": payload_json,
        "status": "pending",
    }
    table = supabase.table("badge_eval_queue")
    if match_id:
        table.upsert(row, on_conflict="event_type,match_id").execute()
    else:
        table.insert(row).execute()


def dequeue_badge_eval(supabase: Any) -> dict[str, Any] | None:
    if supabase is None:
        return None
    resp = (
        supabase.table("badge_eval_queue")
        .select("*")
        .eq("status", "pending")
        .order("created_at", desc=False)
        .limit(1)
        .execute()
    )
    rows = resp.data or []
    if not rows:
        return None
    job = rows[0]
    attempts = int(job.get("attempts") or 0) + 1
    supabase.table("badge_eval_queue").update(
        {"status": "processing", "attempts": attempts}
    ).eq("id", job.get("id")).execute()
    job["attempts"] = attempts
    job["status"] = "processing"
    return job


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
    supabase.table("badge_eval_queue").update(payload).eq("id", job_id).execute()
