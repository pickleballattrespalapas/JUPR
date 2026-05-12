from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from typing import Any

from jupr_app.config import get_public_base_url
from jupr_app.data.client import make_supabase
from jupr_app.domain.notifications.player_update_sender import send_pending_player_update_emails
from jupr_app.services.context import ServiceContext


def _resolve_supabase_config() -> tuple[str, str, str]:
    url = str(os.getenv("SUPABASE_URL") or "").strip()
    service_role = str(os.getenv("SUPABASE_SERVICE_ROLE_KEY") or "").strip()
    anon = str(os.getenv("SUPABASE_ANON_KEY") or "").strip()
    key = service_role or anon
    key_source = "SUPABASE_SERVICE_ROLE_KEY" if service_role else "SUPABASE_ANON_KEY"
    if not url:
        raise ValueError("Missing required environment variable: SUPABASE_URL")
    if not key:
        raise ValueError("Missing Supabase key. Set SUPABASE_SERVICE_ROLE_KEY (preferred) or SUPABASE_ANON_KEY.")
    return url, key, key_source


def _require_worker_run_log() -> bool:
    return str(os.getenv("JUPR_REQUIRE_WORKER_RUN_LOG") or "").strip().lower() in {"1", "true", "yes"}


def _is_missing_table_error(exc: Exception) -> bool:
    detail = str(exc).lower()
    return "worker_run_log" in detail and ("does not exist" in detail or "undefined table" in detail or "relation" in detail)


def run_player_update_email_worker(club_id: str, *, limit: int = 250) -> dict[str, Any]:
    url, key, key_source = _resolve_supabase_config()
    supabase = make_supabase(url, key)
    run_id: str | None = None

    try:
        created = supabase.table("worker_run_log").insert(
            {"worker_name": "player_update_email_worker", "club_id": club_id, "status": "started", "summary_json": {}}
        ).execute().data or []
        run_id = str((created[0] or {}).get("id")) if created else None
    except Exception as exc:  # noqa: BLE001
        if not (_is_missing_table_error(exc) and not _require_worker_run_log()):
            raise

    public_base_url = get_public_base_url()
    ctx = ServiceContext(supabase=supabase, club_id=club_id, public_base_url=public_base_url)

    try:
        summary = send_pending_player_update_emails(ctx, limit=max(1, int(limit)), public_base_url=public_base_url)
        result = {
            "ok": True,
            "club_id": club_id,
            "key_source": key_source,
            "attempted": int(summary.get("attempted") or 0),
            "sent": int(summary.get("sent") or 0),
            "skipped": int(summary.get("skipped") or 0),
            "errors": int(summary.get("errors") or 0),
            "email_mode": str(summary.get("email_mode") or "unknown"),
        }
        if run_id:
            supabase.table("worker_run_log").update(
                {
                    "status": "success",
                    "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "summary_json": {
                        "attempted": result["attempted"],
                        "sent": result["sent"],
                        "skipped": result["skipped"],
                        "errors": result["errors"],
                        "email_mode": result["email_mode"],
                    },
                }
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
    parser = argparse.ArgumentParser(description="Send pending player update emails from the outbox.")
    parser.add_argument("--club-id", required=True, help="Club ID for player update outbox sends.")
    parser.add_argument("--limit", type=int, default=250)
    parser.add_argument("--fail-on-errors", action="store_true")
    args = parser.parse_args(argv)

    try:
        summary = run_player_update_email_worker(args.club_id, limit=args.limit)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1

    if args.fail_on_errors and int(summary.get("errors") or 0) > 0:
        summary["ok"] = False
        print(json.dumps(summary, sort_keys=True, default=str))
        return 1

    print(json.dumps(summary, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
