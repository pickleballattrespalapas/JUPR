"""Dedicated badge process; activation is separate from deploying its code."""
from __future__ import annotations

import argparse
import logging
import logging.config
import json
import os
import signal
import threading
from pathlib import Path
from urllib.parse import urlsplit

from jupr_app.data.client import make_supabase
from jupr_app.domain.gamification.badge_lease import isolated_worker_context, leases_enabled
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue_until_empty
from jupr_app.domain.gamification.program_badge_service import process_pending_program_badges
from jupr_app.workers.badge_queue_worker import _resolve_supabase_config

logger = logging.getLogger("jupr.badge.worker")


def validate_worker_target(club_id: str) -> None:
    if os.getenv("JUPR_BADGE_WORKER_MODE") != "external" or not leases_enabled():
        raise ValueError("Dedicated badge worker requires external mode and queue leases.")
    environment, app = os.getenv("JUPR_ENV"), os.getenv("FLY_APP_NAME")
    expected = {"staging": "juprleagues-api-staging", "production": "juprleagues-api"}
    if environment not in expected or app != expected[environment] or club_id != "tres_palapas":
        raise ValueError("Dedicated badge worker target is not authorized by this configuration.")
    expected_hosts = {"staging": "sijpxjxvdtrehmqvirfi.supabase.co", "production": "dnoockbwfenunhcibwfn.supabase.co"}
    origin = urlsplit(os.getenv("SUPABASE_URL", "").strip())
    if (origin.scheme != "https" or origin.hostname != expected_hosts[environment]
            or origin.username or origin.password or origin.port is not None
            or origin.path not in {"", "/"} or origin.query or origin.fragment):
        raise ValueError("Dedicated badge worker database origin does not match its environment.")
    if environment == "production":
        # This policy exists in the production release. A staging image cannot
        # become a production worker just by changing its environment variables.
        from jupr_app.services.production_feature_policy import production_feature_enabled
        if not production_feature_enabled("badges", club_id):
            raise ValueError("Production badge processing is disabled.")


def run_cycle(db, club_id: str) -> dict:
    validate_worker_target(club_id)
    with isolated_worker_context():
        program = process_pending_program_badges(db, limit=1)
        queue = process_badge_eval_queue_until_empty(
            db, club_id, max_total_jobs=1, batch_max_jobs=1,
            per_batch_time_budget_seconds=5, max_wall_clock_seconds=5,
        )
    return {"program": program, "queue": queue}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the separately supervised badge worker.")
    parser.add_argument("--club-id", required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    validate_worker_target(args.club_id)
    logging.config.dictConfig(json.loads(Path("services/api/logging.json").read_text()))
    url, key, _ = _resolve_supabase_config()
    db = make_supabase(url, key)
    stop = threading.Event()
    for name in (signal.SIGTERM, signal.SIGINT):
        signal.signal(name, lambda *_: stop.set())
    while not stop.is_set():
        try:
            run_cycle(db, args.club_id)
        except Exception as exc:
            # No exception text: provider errors can contain inputs or keys.
            logger.error("badge.worker.cycle_failed", extra={"exception_type": type(exc).__name__})
            if args.once:
                return 1
        if args.once:
            return 0
        stop.wait(15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
