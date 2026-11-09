"""Small durable worker: source revisions survive requests and API restarts."""
import asyncio
import logging
import os

from jupr_app.domain.gamification.badge_policy import badges_enabled
from jupr_app.domain.gamification.program_badge_service import process_pending_program_badges
from jupr_app.domain.gamification.badge_lease import processing_deferred

logger = logging.getLogger("jupr.badge.worker")


def install_program_badge_worker(app, *, get_supabase_client):
    @app.on_event("startup")
    async def start_program_badge_worker():
        # Enable only after a separately supervised worker is deployed and
        # verified. The default retains the current processing topology.
        if not badges_enabled() or processing_deferred():
            return
        if os.getenv("JUPR_ENV") != "staging" or os.getenv("FLY_APP_NAME") != "juprleagues-api-staging":
            return

        async def run():
            while True:
                try:
                    await asyncio.to_thread(lambda: process_pending_program_badges(get_supabase_client()))
                except Exception as exc:
                    logger.error("badge.program.poll_failed", extra={"exception_type": type(exc).__name__})
                await asyncio.sleep(15)

        app.state.program_badge_worker = asyncio.create_task(run())

    @app.on_event("shutdown")
    async def stop_program_badge_worker():
        task = getattr(app.state, "program_badge_worker", None)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
