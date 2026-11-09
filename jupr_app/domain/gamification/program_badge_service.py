"""Durable program evaluation, shared by the API worker and historical previews."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from postgrest.exceptions import APIError

from jupr_app.domain.gamification.badge_types import BadgeCandidate
from jupr_app.domain.gamification.badge_lease import processing_deferred
from jupr_app.domain.gamification.program_badges import evaluate_program_badges
from jupr_app.services.staging_write_guard import staging_write_wave_allows

logger = logging.getLogger("jupr.badge.worker")
STALE_PROGRAM_REVISION_MESSAGE = "Program results changed. Read a fresh snapshot."


def program_badge_preview(supabase: Any, club_id: str) -> dict:
    response = supabase.rpc("badge_program_snapshot_v1", {"p_club_id": str(club_id)}).execute()
    if not isinstance(response.data, dict) or response.data.get("club_id") != club_id:
        raise RuntimeError("Program badge history could not be read completely.")
    return evaluate_program_badges(response.data)


def reconcile_program_badges(supabase: Any, club_id: str, *, dry_run: bool = False) -> dict:
    preview = program_badge_preview(supabase, club_id)
    if dry_run:
        return preview
    # A stale source revision needs a new snapshot, never a replay of the same
    # payload. Refresh at most once; continued churn stays pending for the
    # worker's durable one-minute backoff instead of a tight retry loop.
    for attempt in range(2):
        if not program_badge_writes_enabled():
            raise PermissionError("Program badge awarding is paused.")
        try:
            result = supabase.rpc("apply_program_badges_v1", {
                "p_club_id": club_id, "p_revision": preview["revision"],
                "p_awards": preview["awards"], "p_pending_ties": preview["pending_ties"],
                "p_review": preview["review"],
            }).execute().data
        except APIError as exc:
            if attempt or exc.code != "PT409" or exc.message != STALE_PROGRAM_REVISION_MESSAGE:
                raise
            preview = program_badge_preview(supabase, club_id)
            continue
        if not isinstance(result, dict) or not result.get("ok"):
            raise RuntimeError("Program badge reconciliation was not confirmed.")
        return result
    raise RuntimeError("Program badge reconciliation was not confirmed.")


def program_badge_writes_enabled() -> bool:
    # Production activation remains a separate release decision. This release
    # runs only in the authorized staging environment and honors the stop switch.
    return os.getenv("JUPR_ENV", "").lower() == "staging" and staging_write_wave_allows("badge-diagnostics")


def process_pending_program_badges(supabase: Any, *, limit: int = 5, club_id: str | None = None) -> dict:
    if processing_deferred(club_id):
        return {"processed": 0, "errors": 0}
    if not program_badge_writes_enabled():
        return {"processed": 0, "errors": 0}
    if club_id is None:
        # The pilot can occupy the first pending row. Read one extra candidate
        # before excluding it, so limit=1 still advances another club.
        effective_limit = max(1, min(int(limit), 20))
        pilot_deferred = processing_deferred("tres_palapas")
        query_limit = min(effective_limit + int(pilot_deferred), 20)
        rows = supabase.rpc("pending_program_badge_clubs_v1", {"p_limit": query_limit}).execute().data
    else:
        # Scope before limiting. Filtering the existing global RPC's result can
        # starve this club behind another club's pending revision indefinitely.
        club_id = str(club_id).strip()
        if not club_id:
            raise ValueError("club_id is required for scoped program processing")
        rows = (supabase.table("badge_program_state")
                .select("club_id,revision,applied_revision,retry_after")
                .eq("club_id", club_id)
                .lte("retry_after", datetime.now(timezone.utc).isoformat())
                .limit(1).execute().data)
    if not isinstance(rows, list):
        raise RuntimeError("Unable to read pending program badge clubs.")
    if club_id is not None:
        for row in rows:
            if (not isinstance(row, dict) or row.get("club_id") != club_id
                    or type(row.get("revision")) is not int
                    or type(row.get("applied_revision")) is not int):
                raise RuntimeError("Scoped program badge state could not be read completely.")
        rows = [row for row in rows if row["applied_revision"] < row["revision"]]
    else:
        rows = [row for row in rows if not processing_deferred(str(row["club_id"]))][:effective_limit]
    counts = {"processed": 0, "errors": 0}
    for row in rows:
        club = str(row["club_id"])
        try:
            reconcile_program_badges(supabase, club)
            counts["processed"] += 1
        except Exception as exc:
            counts["errors"] += 1
            # Provider exception messages/tracebacks may contain submitted
            # values or credentials; retain only identity and exception type.
            logger.error("badge.program.failed", extra={"club_id": club, "exception_type": type(exc).__name__})
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
