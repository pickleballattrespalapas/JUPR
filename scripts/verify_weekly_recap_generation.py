"""Exercise real weekly-report computation through a GET-only database client.

No recap is saved or published. Only aggregate verification results are printed.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import os
from zoneinfo import ZoneInfo

import httpx
from supabase import ClientOptions, create_client

from jupr_app.domain.recaps.weekly_recap import compute_weekly_recap, get_spotlight_candidates
from jupr_app.services.admin_weekly_recap_service import _recap_context


def require_read(request: httpx.Request) -> None:
    if request.method not in {"GET", "HEAD"}:
        raise RuntimeError("Weekly recap verification permits database reads only.")


def verify(db, *, club_id: str, start: date, end: date, timezone: str) -> dict:
    context = _recap_context(db, club_id=club_id)
    report = compute_weekly_recap(context, start_date=start, end_date=end,
                                 include_tournaments=True, tz_name=timezone)
    candidates = get_spotlight_candidates(context, start_date=start, end_date=end,
                                         include_tournaments=True, tz_name=timezone)
    # The same results must be serializable when an administrator saves a draft.
    json.dumps({"report": report, "candidates": candidates}, allow_nan=False)
    assert report["report"]["version"] == 1
    return {"ok": True, "start_date": start.isoformat(), "end_date": end.isoformat(),
            "games": report["numbers"]["matches"], "players": report["numbers"]["players"],
            "events": len(report["report"]["event_results"]),
            "coverage_warnings": len(report["report"]["coverage"]["warnings"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--club-id", required=True)
    parser.add_argument("--timezone", default="America/Mazatlan")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    args = parser.parse_args()
    end = args.end_date or datetime.now(ZoneInfo(args.timezone)).date()
    start = args.start_date or end - timedelta(days=6)
    with httpx.Client(http2=False, timeout=60, event_hooks={"request": [require_read]}) as transport:
        db = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"],
                           options=ClientOptions(httpx_client=transport))
        try:
            result = verify(db, club_id=args.club_id, start=start, end=end, timezone=args.timezone)
        except Exception as exc:
            print(json.dumps({"ok": False, "error_type": type(exc).__name__}), flush=True)
            return 1
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
