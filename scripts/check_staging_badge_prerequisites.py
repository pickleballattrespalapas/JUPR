"""Read-only deployment gate: staging schema, idle queue, no credentials in output."""
from __future__ import annotations

import json
import os
from urllib.request import Request, urlopen

STAGING_URL = "https://sijpxjxvdtrehmqvirfi.supabase.co"
RPCS = {"claim_badge_eval_queue_job_leased", "heartbeat_badge_eval_queue_job",
        "finish_badge_eval_queue_job"}


def check(url: str, key: str, *, fetch=None) -> None:
    if url.rstrip("/") != STAGING_URL or not key:
        raise ValueError("Staging worker database configuration is invalid.")
    if fetch is None:
        def fetch(path, accept="application/json"):
            request = Request(STAGING_URL + "/rest/v1/" + path, headers={
                "apikey": key, "Authorization": "Bearer " + key, "Accept": accept})
            with urlopen(request, timeout=20) as response:
                return json.load(response)
    schema = fetch("", "application/openapi+json")
    if not all("/rpc/" + rpc in schema.get("paths", {}) for rpc in RPCS):
        raise ValueError("Staging badge lease RPCs are unavailable.")
    fetch("badge_eval_queue?select=id,claim_token,worker_id,heartbeat_at,lease_expires_at&limit=0")
    # A preexisting claim is evidence for review, never permission to replay it.
    if fetch("badge_eval_queue?select=id&club_id=eq.tres_palapas&status=eq.processing&limit=1"):
        raise ValueError("A badge job is processing; defer the worker rollout until reviewed or finished.")


def main() -> int:
    try:
        check(os.getenv("SUPABASE_URL", ""), os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""))
    except Exception as exc:
        # Provider text/response bodies may contain credentials or record values.
        print(json.dumps({"ready": False, "exception_type": type(exc).__name__}))
        return 1
    print(json.dumps({"ready": True, "project": "sijpxjxvdtrehmqvirfi"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
