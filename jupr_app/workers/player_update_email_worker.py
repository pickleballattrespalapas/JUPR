from __future__ import annotations

import argparse
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
        raise ValueError(
            "Missing Supabase key. Set SUPABASE_SERVICE_ROLE_KEY (preferred) or SUPABASE_ANON_KEY."
        )
    return url, key, key_source


def run_player_update_email_worker(club_id: str, *, limit: int = 250) -> dict[str, Any]:
    url, key, key_source = _resolve_supabase_config()
    supabase = make_supabase(url, key)
    public_base_url = get_public_base_url()
    ctx = ServiceContext(supabase=supabase, club_id=club_id, public_base_url=public_base_url)
    summary = send_pending_player_update_emails(
        ctx,
        limit=max(1, int(limit)),
        public_base_url=public_base_url,
    )
    return {
        "ok": True,
        "club_id": club_id,
        "key_source": key_source,
        "attempted": int(summary.get("attempted") or 0),
        "sent": int(summary.get("sent") or 0),
        "skipped": int(summary.get("skipped") or 0),
        "errors": int(summary.get("errors") or 0),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send pending player update emails from the outbox.")
    parser.add_argument("--club-id", required=True, help="Club ID for player update outbox sends.")
    parser.add_argument("--limit", type=int, default=250)
    args = parser.parse_args(argv)

    try:
        summary = run_player_update_email_worker(args.club_id, limit=args.limit)
    except ValueError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        return 2
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1

    print(json.dumps(summary, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
