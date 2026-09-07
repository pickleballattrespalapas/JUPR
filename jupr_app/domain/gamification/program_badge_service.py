"""Durable program evaluation, shared by the API worker and historical previews."""
from __future__ import annotations

import logging
import os
from typing import Any

from jupr_app.domain.gamification.badge_types import BadgeCandidate
from jupr_app.domain.gamification.program_badges import evaluate_program_badges
from jupr_app.services.staging_write_guard import staging_write_wave_allows

logger = logging.getLogger(__name__)


def program_badge_preview(supabase: Any, club_id: str) -> dict:
    response = supabase.rpc("badge_program_snapshot_v1", {"p_club_id": str(club_id)}).execute()
    if not isinstance(response.data, dict) or response.data.get("club_id") != club_id:
        raise RuntimeError("Program badge history could not be read completely.")
    return evaluate_program_badges(response.data)


def reconcile_program_badges(supabase: Any, club_id: str, *, dry_run: bool = False) -> dict:
    preview = program_badge_preview(supabase, club_id)
    if dry_run:
        return preview
    if not program_badge_writes_enabled():
        raise PermissionError("Program badge awarding is paused.")
    result = supabase.rpc("apply_program_badges_v1", {"p_club_id": club_id, "p_revision": preview["revision"],
        "p_awards": preview["awards"], "p_pending_ties": preview["pending_ties"], "p_review": preview["review"]}).execute().data
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError("Program badge reconciliation was not confirmed.")
    return result


def program_badge_writes_enabled() -> bool:
    # Production activation remains a separate release decision. This release
    # runs only in the authorized staging environment and honors the stop switch.
    return os.getenv("JUPR_ENV", "").lower() == "staging" and staging_write_wave_allows("badge-diagnostics")


def process_pending_program_badges(supabase: Any, *, limit: int = 5) -> dict:
    if not program_badge_writes_enabled():
        return {"processed": 0, "errors": 0}
    rows = supabase.rpc("pending_program_badge_clubs_v1", {"p_limit": limit}).execute().data
    if not isinstance(rows, list):
        raise RuntimeError("Unable to read pending program badge clubs.")
    counts = {"processed": 0, "errors": 0}
    for row in rows:
        club = str(row["club_id"])
        try:
            reconcile_program_badges(supabase, club)
            counts["processed"] += 1
        except Exception:
            counts["errors"] += 1
            logger.exception("Program badge evaluation failed for club %s", club)
            # Errors remain durable and retryable. Do not acknowledge the revision.
            supabase.rpc("program_badge_failure_v1", {"p_club_id": club}).execute()
    return counts


def evaluate_program_badge(context: Any, *, badge_id: str):
    """Explicit program-timing evaluation for diagnostics; no legacy match fallback."""
    cache = getattr(context.ctx, "program_badge_preview", None)
    if cache is None:
        cache = program_badge_preview(context.ctx.supabase, context.club_id)
        context.ctx.program_badge_preview = cache
    for row in cache["awards"]:
        if row["badge_id"] == badge_id:
            yield BadgeCandidate(badge_id=badge_id, player_id=row["player_id"], club_id=context.club_id,
                                 context_type=row["context_type"], context_id=row["context_id"], match_id=None,
                                 value_json={**row["value_json"], "earned_at": row["earned_at"]}, value_num=row["value_num"])
