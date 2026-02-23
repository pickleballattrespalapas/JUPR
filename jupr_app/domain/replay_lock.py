from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import sleep
from typing import Any

from postgrest.exceptions import APIError


class ReplayAlreadyRunningError(RuntimeError):
    """Raised when a replay is already running for a club."""


@dataclass(frozen=True)
class ReplayLockInfo:
    club_id: str
    started_at: str | None
    status: str | None


def _get_api_error_code(exc: APIError) -> str | None:
    code = getattr(exc, "code", None)
    if code:
        return code
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("code")
    return None


def acquire_replay_lock(supabase: Any, club_id: str) -> None:
    payload = {
        "club_id": str(club_id),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    try:
        supabase.table("replay_lock").insert(payload).execute()
    except APIError as exc:
        if _get_api_error_code(exc) == "23505":
            raise ReplayAlreadyRunningError(f"Replay already running for club_id={club_id}.") from exc
        raise


def acquire_replay_lock_with_wait(supabase: Any, club_id: str, *, wait: bool = False, poll_seconds: float = 1.0) -> None:
    while True:
        try:
            acquire_replay_lock(supabase, club_id)
            return
        except ReplayAlreadyRunningError:
            if not wait:
                raise
            sleep(max(poll_seconds, 0.1))


def release_replay_lock(supabase: Any, club_id: str) -> None:
    supabase.table("replay_lock").delete().eq("club_id", str(club_id)).execute()


def is_replay_running(supabase: Any, club_id: str) -> ReplayLockInfo | None:
    response = (
        supabase.table("replay_lock")
        .select("club_id,started_at,status")
        .eq("club_id", str(club_id))
        .limit(1)
        .execute()
    )
    rows = response.data or []
    if not rows:
        return None
    row = rows[0]
    return ReplayLockInfo(
        club_id=str(row.get("club_id") or club_id),
        started_at=row.get("started_at"),
        status=row.get("status"),
    )
