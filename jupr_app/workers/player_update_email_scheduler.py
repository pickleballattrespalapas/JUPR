"""Deliver queued player updates on the existing production API machine."""
from __future__ import annotations

import asyncio
import logging
import os
from uuid import NAMESPACE_URL, uuid5

from jupr_app.config import get_email_mode
from jupr_app.data.client import make_supabase
from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status
from jupr_app.workers.player_update_email_worker import run_player_update_email_worker

logger = logging.getLogger(__name__)
POLL_SECONDS = 60
BATCH_SIZE = 25


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


def deliver_pending_updates() -> dict[str, int]:
    totals = {"clubs": 0, "sent": 0, "errors": 0}
    if not scheduler_enabled():
        return totals
    client = make_supabase(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    try:
        rows = (
            client.table("player_profile_update_outbox")
            .select("id,club_id,subscription_id,week_start,week_end,row_version,queue_operation_key")
            .eq("send_status", "pending")
            .eq("digest_snapshot_json", "{}")
            .order("created_at")
            .limit(250)
            .execute().data or []
        )
    finally:
        transport = getattr(getattr(client, "options", None), "httpx_client", None)
        if transport is not None:
            transport.close()
    # Use the existing atomic row claims; API sends and other replicas cannot
    # deliver the same pending row. Errors and uncertain sends require review.
    clubs: dict[str, list[dict]] = {}
    for row in rows:
        club_id = str(row.get("club_id") or "")
        expected_key = str(uuid5(
            NAMESPACE_URL,
            f"jupr:auto-player-update:{club_id}:{row.get('subscription_id')}:{row.get('week_start')}:{row.get('week_end')}",
        ))
        if club_id and str(row.get("queue_operation_key")) == expected_key:
            clubs.setdefault(club_id, []).append({
                "id": row["id"], "expected_row_version": row["row_version"],
            })
    # Explicitly queued admin previews remain pending until the operator sends.
    for club_id, items in clubs.items():
        if not scheduler_enabled():
            break
        try:
            result = run_player_update_email_worker(club_id, limit=BATCH_SIZE, outbox_items=items[:BATCH_SIZE])
            totals["clubs"] += 1
            totals["sent"] += int(result.get("sent") or 0)
            totals["errors"] += int(result.get("errors") or 0)
        except Exception as exc:
            totals["errors"] += 1
            logger.error("Player email worker failed (%s)", type(exc).__name__)
    return totals


async def run_scheduler() -> None:
    while True:
        await asyncio.sleep(POLL_SECONDS)
        try:
            result = await asyncio.to_thread(deliver_pending_updates)
            if result["clubs"] or result["errors"]:
                logger.info("Player email worker: %s", result)
        except Exception as exc:
            logger.error("Player email polling failed (%s)", type(exc).__name__)
