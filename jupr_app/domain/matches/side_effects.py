from __future__ import annotations

import logging
from typing import Any

from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue
from jupr_app.domain.gamification.live_awards import run_live_badge_awards
from jupr_app.domain.notifications.player_profile_update_repo import queue_player_updates_for_affected_subscribers

logger = logging.getLogger(__name__)


def ensure_notification_badge_job(*, supabase, club_id: str, match_ids: list[int | str], player_ids: list[int]) -> str:
    """Recover a missing exact badge job without resetting an existing job."""
    reference = str(match_ids[0])
    rows = (supabase.table("badge_eval_queue").select("id,status")
        .eq("club_id", club_id).eq("event_type", "match_recorded")
        .eq("match_id", reference).limit(1).execute().data or [])
    if not rows:
        queued = enqueue_badge_eval(supabase, club_id=club_id, event_type="match_recorded",
            player_ids=player_ids, match_id=reference,
            payload={"match_count": len(match_ids), "match_ids": list(match_ids)})
        if not queued.get("queued"):
            raise RuntimeError("The games are saved, but their award evaluation could not be queued. Resume approval to retry.")
    elif rows[0].get("status") == "error":
        # A resume may retry this exact failed job, while another worker's
        # processing/done state remains protected by the status predicate.
        (supabase.table("badge_eval_queue").update({"status": "pending", "last_error": None,
            "processed_at": None}).eq("club_id", club_id).eq("event_type", "match_recorded")
            .eq("match_id", reference).eq("status", "error").execute())
    return reference


def run_badge_side_effects(
    *,
    supabase,
    club_id: str,
    has_badge_eligible_match: bool,
    affected_players: set[int],
    db_matches: list[dict[str, Any]],
    match_payloads: list[dict[str, Any]],
    dedupe_match_id: str | None = None,
) -> dict[str, Any]:
    badge_summary: dict[str, Any] = {"mode": "skipped", "awarded_count": 0, "candidate_count": 0, "badge_ids": []}
    if not (supabase is not None and has_badge_eligible_match):
        return badge_summary
    enqueue_result = enqueue_badge_eval(
        supabase, club_id=str(club_id), event_type="match_recorded",
        player_ids=sorted(affected_players),
        match_id=str(dedupe_match_id) if dedupe_match_id else None,
        payload={"match_count": len(db_matches), "matches": match_payloads[:10]},
    )
    should_fallback = not bool(enqueue_result.get("queued"))
    if not should_fallback:
        try:
            worker_result = process_badge_eval_queue(
                supabase,
                str(club_id),
                max_jobs=1,
                time_budget_seconds=2,
            )
            should_fallback = bool(worker_result.get("errored")) or (int(worker_result.get("processed") or 0) == 0 and int(worker_result.get("errored") or 0) > 0)
            badge_summary = {"mode": "queue", **worker_result}
        except Exception as exc:  # noqa: BLE001
            should_fallback = True
            logger.warning("Badge queue worker failed during match processing: %s", exc)
    else:
        logger.warning("Badge queue enqueue unavailable during match processing; falling back inline. reason=%s", enqueue_result.get("reason"))
    if should_fallback:
        try:
            badge_summary = run_live_badge_awards(supabase, club_id=str(club_id), player_ids=sorted(affected_players), event_type="match_recorded")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Inline live badge fallback failed after match processing: %s", exc)
            badge_summary = {"mode": "inline_error", "error": str(exc)}
    if bool(enqueue_result.get("queued")) and dedupe_match_id and badge_summary.get("mode") != "inline":
        badge_summary["notification_badge_refs"] = [str(dedupe_match_id)]
    return badge_summary


def queue_player_updates(*, supabase, club_id: str, db_matches: list[dict[str, Any]], affected_players: set[int], successful_match_dates: list[str], match_ids: list[int | str] | None = None, session_evidence: dict[str, Any] | None = None, badge_refs: list[str] | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"mode": "skipped", "affected_players": len(affected_players), "week_windows": 0, "queued": 0, "already_queued": 0, "no_active_subscription": 0, "failed": 0}
    if not (db_matches and affected_players):
        return summary
    try:
        queue_summary = queue_player_updates_for_affected_subscribers(
            supabase,
            club_id=str(club_id),
            affected_player_ids=sorted(affected_players),
            match_dates=successful_match_dates,
            match_ids=match_ids,
            session_evidence=session_evidence,
            badge_refs=badge_refs,
        )
        return {"mode": "queued", **queue_summary}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Player update queueing failed after match processing: %s", exc)
        return {**summary, "mode": "error", "error": str(exc)}
