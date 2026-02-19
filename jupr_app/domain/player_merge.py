from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jupr_app.data.retry import sb_retry
from jupr_app.data.sb_write import sb_insert, sb_update
from jupr_app.domain.audit_logger import log_event

_SLOT_KEYS = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")


class PlayerMergeError(RuntimeError):
    """Raised when a player merge cannot be completed safely."""


def _require_club_id(club_id: str) -> str:
    scoped = str(club_id or "").strip()
    if not scoped:
        raise ValueError("club_id is required")
    return scoped


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _queue_replay_suggestion(*, supabase: Any, club_id: str) -> bool:
    try:
        sb_insert(
            supabase,
            "jobs",
            {
                "club_id": str(club_id),
                "job_type": "replay_ratings",
                "status": "pending",
            },
        )
        return True
    except Exception:
        return False


def merge_player_into(
    supabase,
    club_id: str,
    source_player_id: int,
    destination_player_id: int,
    actor: str,
):
    """Merge one player into another without running replay automatically."""

    scoped_club_id = _require_club_id(club_id)
    src = int(source_player_id)
    dst = int(destination_player_id)
    actor_name = str(actor or "system").strip() or "system"

    if src == dst:
        raise ValueError("source_player_id and destination_player_id must differ")

    summary: dict[str, Any] = {
        "success": False,
        "club_id": scoped_club_id,
        "source_player_id": src,
        "destination_player_id": dst,
        "matches_reassigned": 0,
        "league_ratings_reassigned": 0,
        "source_deactivated": False,
        "audit_logged": False,
        "replay_suggested": False,
        "consistency_guard_passed": False,
        "warnings": [],
        "error": None,
    }

    def _run_once() -> dict[str, Any]:
        destination_rows = (
            supabase.table("players")
            .select("id")
            .eq("club_id", scoped_club_id)
            .eq("id", dst)
            .limit(1)
            .execute()
            .data
            or []
        )
        if not destination_rows:
            raise PlayerMergeError("destination_player_id not found in club scope")

        source_rows = (
            supabase.table("players")
            .select("id,active,inactive_at")
            .eq("club_id", scoped_club_id)
            .eq("id", src)
            .limit(1)
            .execute()
            .data
            or []
        )
        if not source_rows:
            raise PlayerMergeError("source_player_id not found in club scope")

        matches = (
            supabase.table("matches")
            .select("id,t1_p1,t1_p2,t2_p1,t2_p2")
            .eq("club_id", scoped_club_id)
            .or_(f"t1_p1.eq.{src},t1_p2.eq.{src},t2_p1.eq.{src},t2_p2.eq.{src}")
            .execute()
            .data
            or []
        )

        matches_reassigned = 0
        for row in matches:
            match_id = int(row["id"])
            patch: dict[str, int] = {}
            for slot in _SLOT_KEYS:
                if _as_int(row.get(slot)) == src:
                    patch[slot] = dst
            if not patch:
                continue
            sb_update(
                supabase,
                "matches",
                patch,
                filters={"club_id": scoped_club_id, "id": match_id},
            )
            matches_reassigned += 1

        source_lr_rows = (
            supabase.table("league_ratings")
            .select("id")
            .eq("club_id", scoped_club_id)
            .eq("player_id", src)
            .execute()
            .data
            or []
        )
        league_ratings_reassigned = len(source_lr_rows)
        if league_ratings_reassigned:
            sb_update(
                supabase,
                "league_ratings",
                {"player_id": dst},
                filters={"club_id": scoped_club_id, "player_id": src},
            )

        sb_update(
            supabase,
            "players",
            {
                "active": False,
                "inactive_at": datetime.now(timezone.utc).isoformat(),
            },
            filters={"club_id": scoped_club_id, "id": src},
        )

        remaining_matches = (
            supabase.table("matches")
            .select("id", count="exact")
            .eq("club_id", scoped_club_id)
            .or_(f"t1_p1.eq.{src},t1_p2.eq.{src},t2_p1.eq.{src},t2_p2.eq.{src}")
            .limit(1)
            .execute()
        )
        remaining_match_refs = int(getattr(remaining_matches, "count", 0) or 0)

        remaining_league = (
            supabase.table("league_ratings")
            .select("id", count="exact")
            .eq("club_id", scoped_club_id)
            .eq("player_id", src)
            .limit(1)
            .execute()
        )
        remaining_league_refs = int(getattr(remaining_league, "count", 0) or 0)

        source_after = (
            supabase.table("players")
            .select("active,inactive_at")
            .eq("club_id", scoped_club_id)
            .eq("id", src)
            .limit(1)
            .execute()
            .data
            or []
        )
        source_after_row = dict(source_after[0]) if source_after else {}
        source_deactivated = (
            bool(source_after_row)
            and source_after_row.get("active") is False
            and bool(source_after_row.get("inactive_at"))
        )

        if remaining_match_refs or remaining_league_refs or not source_deactivated:
            raise PlayerMergeError(
                "consistency guard failed "
                f"(match_refs={remaining_match_refs}, league_refs={remaining_league_refs}, source_deactivated={source_deactivated})"
            )

        return {
            "matches_reassigned": matches_reassigned,
            "league_ratings_reassigned": league_ratings_reassigned,
            "source_deactivated": True,
            "consistency_guard_passed": True,
        }

    try:
        operation = sb_retry(_run_once)
        summary.update(operation)
        summary["success"] = True
    except Exception as exc:
        summary["error"] = str(exc)

    log_event(
        supabase=supabase,
        club_id=scoped_club_id,
        actor=actor_name,
        action_type="merge_player",
        payload={
            "source_player_id": src,
            "destination_player_id": dst,
            "matches_reassigned": int(summary.get("matches_reassigned") or 0),
            "league_ratings_reassigned": int(summary.get("league_ratings_reassigned") or 0),
            "source_deactivated": bool(summary.get("source_deactivated")),
            "consistency_guard_passed": bool(summary.get("consistency_guard_passed")),
            "success": bool(summary.get("success")),
            "error": summary.get("error"),
        },
    )
    summary["audit_logged"] = True

    replay_suggested = _queue_replay_suggestion(supabase=supabase, club_id=scoped_club_id)
    summary["replay_suggested"] = replay_suggested
    if not replay_suggested:
        summary["warnings"].append("Unable to queue replay suggestion.")

    return summary


__all__ = ["PlayerMergeError", "merge_player_into"]
