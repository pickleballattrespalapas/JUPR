from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any

from jupr_app.data.client import make_supabase
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue_until_empty


def _resolve_supabase_config() -> tuple[str, str, str]:
    url = str(os.getenv("SUPABASE_URL") or "").strip()
    service_role = str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    if not url:
        raise ValueError("Missing required environment variable: SUPABASE_URL")
    if not service_role:
        raise ValueError(
            "Missing required environment variable: SUPABASE_SERVICE_ROLE_KEY. "
            "Badge queue claims are server-only and cannot use SUPABASE_ANON_KEY."
        )
    return url, service_role, "SUPABASE_SERVICE_ROLE_KEY"


def _require_worker_run_log() -> bool:
    return str(os.getenv("JUPR_REQUIRE_WORKER_RUN_LOG") or "").strip().lower() in {"1", "true", "yes"}


def _is_missing_table_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    return "worker_run_log" in detail and ("does not exist" in detail or "undefined table" in detail or "relation" in detail)


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
    run_id: str | None = None

    try:
        created = supabase.table("worker_run_log").insert(
            {"worker_name": "badge_queue_worker", "club_id": club_id, "status": "started", "summary_json": {}}
        ).execute().data or []
        run_id = str((created[0] or {}).get("id")) if created else None
    except Exception as exc:  # noqa: BLE001
        if not (_is_missing_table_error(exc) and not _require_worker_run_log()):
            raise

    try:
        worker_summary = process_badge_eval_queue_until_empty(
            supabase,
            club_id,
            max_total_jobs=max_total_jobs,
            batch_max_jobs=batch_max_jobs,
            per_batch_time_budget_seconds=per_batch_time_budget_seconds,
            max_wall_clock_seconds=max_wall_clock_seconds,
            max_errors=max_errors,
        )
        result = {"ok": True, "club_id": club_id, "key_source": key_source, **worker_summary}
        if run_id:
            supabase.table("worker_run_log").update(
                {"status": "success", "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "summary_json": result}
            ).eq("id", run_id).execute()
        return result
    except Exception as exc:  # noqa: BLE001
        if run_id:
            try:
                supabase.table("worker_run_log").update(
                    {"status": "failed", "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(), "error_text": f"{type(exc).__name__}: {exc}"}
                ).eq("id", run_id).execute()
            except Exception as log_exc:  # noqa: BLE001
                if not (_is_missing_table_error(log_exc) and not _require_worker_run_log()):
                    raise
        raise


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
