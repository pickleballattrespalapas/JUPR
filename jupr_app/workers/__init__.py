"""Background worker entrypoints for JUPr."""

from jupr_app.workers.badge_queue_worker import run_badge_queue_worker

__all__ = ["run_badge_queue_worker"]
