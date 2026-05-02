"""Background worker entrypoints for JUPr."""

from jupr_app.workers.badge_queue_worker import run_badge_queue_worker
from jupr_app.workers.player_update_email_worker import run_player_update_email_worker

__all__ = ["run_badge_queue_worker", "run_player_update_email_worker"]
