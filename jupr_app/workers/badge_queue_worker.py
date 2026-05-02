from __future__ import annotations

import argparse
import json
import os
from typing import Any

from jupr_app.data.client import make_supabase
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue_until_empty


def _resolve_supabase_config() -> tuple[str, str, str]:
    url = str(os.getenv("SUPABASE_URL") or "").strip()
    service_role = str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    anon = str(os.getenv("SUPABASE_ANON_KEY") or "").strip()
    key = service_role or anon
    key_source = "SUPABASE_SERVICE_ROLE_KEY" if service_role else "SUPABASE_ANON_KEY"
    if not url:
        raise ValueError("Missing required environment variable: SUPABASE_URL")
    if not key:
        raise ValueError(
            "Missing Supabase key. Set SUPABASE_SERVICE_ROLE_KEY (preferred) or SUPABASE_ANON_KEY."
        )
    return url, key, key_source


def run_badge_queue_worker(
    club_id: str,
    *,
    max_total_jobs: int = 500,
    batch_max_jobs: int = 10,
    per_batch_time_budget_seconds: float = 2.0,
    max_wall_clock_seconds: float = 90.0,
    max_errors: int = 10,
) -> dict[str, Any]:
    url, key, key_source = _resolve_supabase_config()
    supabase = make_supabase(url, key)
    worker_summary = process_badge_eval_queue_until_empty(
        supabase,
        club_id,
        max_total_jobs=max_total_jobs,
        batch_max_jobs=batch_max_jobs,
        per_batch_time_budget_seconds=per_batch_time_budget_seconds,
        max_wall_clock_seconds=max_wall_clock_seconds,
        max_errors=max_errors,
    )
    return {
        "ok": True,
        "club_id": club_id,
        "key_source": key_source,
        **worker_summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Process badge evaluation queue jobs.")
    parser.add_argument("--club-id", required=True, help="Club ID for queue jobs.")
    parser.add_argument("--max-total-jobs", type=int, default=500)
    parser.add_argument("--batch-max-jobs", type=int, default=10)
    parser.add_argument("--per-batch-time-budget-seconds", type=float, default=2.0)
    parser.add_argument("--max-wall-clock-seconds", type=float, default=90.0)
    parser.add_argument("--max-errors", type=int, default=10)
    args = parser.parse_args(argv)

    try:
        summary = run_badge_queue_worker(
            args.club_id,
            max_total_jobs=args.max_total_jobs,
            batch_max_jobs=args.batch_max_jobs,
            per_batch_time_budget_seconds=args.per_batch_time_budget_seconds,
            max_wall_clock_seconds=args.max_wall_clock_seconds,
            max_errors=args.max_errors,
        )
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1

    if int(summary.get("total_errored") or 0) >= int(args.max_errors):
        summary["ok"] = False
        print(json.dumps(summary, sort_keys=True, default=str))
        return 1

    print(json.dumps(summary, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
