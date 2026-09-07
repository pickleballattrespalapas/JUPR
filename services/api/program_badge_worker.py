"""Small durable worker: source revisions survive requests and API restarts."""
import asyncio
import logging
import os

from jupr_app.domain.gamification.program_badge_service import process_pending_program_badges

logger = logging.getLogger(__name__)


def install_program_badge_worker(app, *, get_supabase_client):
    @app.on_event("startup")
    async def start_program_badge_worker():
        if os.getenv("JUPR_ENV") != "staging" or os.getenv("FLY_APP_NAME") != "juprleagues-api-staging":
            return

        async def run():
            while True:
                try:
                    await asyncio.to_thread(lambda: process_pending_program_badges(get_supabase_client()))
                except Exception:
                    logger.exception("Unable to check pending program badges")
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
