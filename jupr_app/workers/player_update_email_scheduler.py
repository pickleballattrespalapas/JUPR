"""Deliver queued player updates on the existing production API machine."""
from __future__ import annotations

import asyncio
import logging
import os
from threading import Event
from datetime import datetime, timezone
from typing import Any, Callable

from jupr_app.config import get_email_mode
from jupr_app.data.client import make_supabase
from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status
from jupr_app.domain.notifications.player_profile_update_repo import session_player_update_operation_key
from jupr_app.workers.player_update_email_worker import run_player_update_email_worker

logger = logging.getLogger(__name__)
POLL_SECONDS = 60
BATCH_SIZE = 25
SCAN_SIZE = 250
_delivery_status: dict[str, Any] = {}


def scheduler_enabled() -> bool:
    truthy = {"1", "true", "yes", "y", "on"}
    return (
        os.getenv("JUPR_ENV") == "production"
        and os.getenv("FLY_APP_NAME") == "juprleagues-api"
        and os.getenv("SUPABASE_URL", "").rstrip("/")
        == "https://dnoockbwfenunhcibwfn.supabase.co"
        and os.getenv("JUPR_PRODUCTION_WRITE_POLICY") == "enabled"
        and os.getenv("JUPR_STAGING_WRITE_WAVE") == "none"
        and all(os.getenv(name, "").strip().lower() in truthy for name in (
            "JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS",
            "JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL",
            "JUPR_REQUIRE_WORKER_RUN_LOG",
        ))
        and bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip())
        and get_email_mode() == "live"
        and bool(get_smtp_config_status().get("ok"))
    )


def _automatic_session(row: dict[str, Any]) -> bool:
    snapshot = row.get("digest_snapshot_json")
    if not isinstance(snapshot, dict) or snapshot.get("update_scope") != "session":
        return False
    try:
        expected_key = session_player_update_operation_key(row.get("club_id"), snapshot.get("match_ids"))
    except (ValueError, TypeError):
        return False
    return str(row.get("queue_operation_key")) == expected_key


def delivery_status() -> dict[str, Any]:
    """Small, address-free heartbeat; health reads never query business data."""
    return {**_delivery_status, "summary": dict(_delivery_status.get("summary") or {})}


def deliver_pending_updates(
    *, scan_after: str | None = None, should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    totals = {name: 0 for name in (
        "clubs", "examined", "selected", "ignored", "attempted", "sent", "skipped",
        "errors", "stale", "uncertain", "waiting_for_awards", "badge_jobs_processed", "badge_job_errors",
    )}
    totals["next_cursor"] = None
    if not scheduler_enabled() or (should_stop is not None and should_stop()):
        return totals
    client = make_supabase(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    try:
        query = (client.table("player_profile_update_outbox")
            .select("id,club_id,row_version,queue_operation_key,digest_snapshot_json")
            .eq("send_status", "pending")
            .eq("digest_snapshot_json->>update_scope", "session")
            .order("id")
            .limit(SCAN_SIZE))
        if scan_after:
            query = query.gt("id", scan_after)
        rows = query.execute().data or []
    finally:
        transport = getattr(getattr(client, "options", None), "httpx_client", None)
        if transport is not None:
            transport.close()

    clubs: dict[str, list[dict]] = {}
    last_id = None
    for row in rows:
        totals["examined"] += 1
        last_id = str(row["id"])
        if _automatic_session(row):
            clubs.setdefault(str(row["club_id"]), []).append({
                "id": row["id"], "expected_row_version": row["row_version"],
            })
            totals["selected"] += 1
        else:
            # Manual previews, malformed rows and legacy unscoped rows never
            # gain permission to send just because the worker restarts.
            totals["ignored"] += 1
        if totals["selected"] >= BATCH_SIZE:
            break
    if totals["examined"] < len(rows) or len(rows) == SCAN_SIZE:
        totals["next_cursor"] = last_id

    # Move through the queue even when an early session is waiting for awards.
    # Database row claims, not this in-memory scan cursor, prevent duplicates.
    for club_id, items in clubs.items():
        if not scheduler_enabled() or (should_stop is not None and should_stop()):
            break
        try:
            result = run_player_update_email_worker(
                club_id, limit=len(items), outbox_items=items, advance_waiting_awards=True,
                **({"should_stop": should_stop} if should_stop is not None else {}),
            )
            totals["clubs"] += 1
            for key in ("attempted", "sent", "skipped", "errors", "stale", "uncertain",
                        "waiting_for_awards", "badge_jobs_processed", "badge_job_errors"):
                totals[key] += int(result.get(key) or 0)
        except Exception as exc:
            totals["errors"] += 1
            logger.error("Player email worker failed (%s)", type(exc).__name__)
    return totals


async def run_scheduler(*, stop_event: Event | None = None) -> None:
    stop_event = stop_event or Event()
    scan_after = None
    while not stop_event.is_set():
        _delivery_status["last_started_at"] = datetime.now(timezone.utc).isoformat()
        try:
            result = await asyncio.to_thread(
                deliver_pending_updates, scan_after=scan_after, should_stop=stop_event.is_set,
            )
            scan_after = result.pop("next_cursor")
            _delivery_status.update(
                last_completed_at=datetime.now(timezone.utc).isoformat(),
                last_error_type=None, summary=result,
            )
            if result["sent"]:
                _delivery_status["last_sent_at"] = _delivery_status["last_completed_at"]
            if result["examined"] or result["errors"]:
                logger.info("Player email worker: %s", result)
        except Exception as exc:
            _delivery_status["last_error_type"] = type(exc).__name__
            logger.error("Player email polling failed (%s)", type(exc).__name__)
        # Wake promptly for shutdown without cancelling an in-flight SMTP call.
        for _ in range(POLL_SECONDS):
            if stop_event.is_set():
                break
            await asyncio.sleep(1)
