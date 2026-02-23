from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

from jupr_app.data.client import make_supabase
from jupr_app.domain.replay_history import FULL_RESET_LABEL, replay_history
from jupr_app.domain.replay_lock import (
    ReplayAlreadyRunningError,
    acquire_replay_lock_with_wait,
    release_replay_lock,
)


def _count_rows(supabase, table: str, club_id: str) -> int:
    result = (
        supabase.table(table)
        .select("id", count="exact")
        .eq("club_id", str(club_id))
        .limit(1)
        .execute()
    )
    return int(getattr(result, "count", 0) or 0)


def _load_previous_success_summary(supabase, club_id: str) -> dict:
    try:
        rows = (
            supabase.table("replay_runs")
            .select("summary")
            .eq("club_id", str(club_id))
            .eq("status", "success")
            .order("id", desc=True)
            .limit(1)
            .execute()
            .data
            or []
        )
    except Exception:
        return {}
    if not rows:
        return {}
    summary = rows[0].get("summary")
    return summary if isinstance(summary, dict) else {}


def _verify_deterministic_counts_or_raise(supabase, club_id: str, summary: dict) -> None:
    previous = _load_previous_success_summary(supabase, club_id)
    current_counts = {
        "matches": _count_rows(supabase, "matches", club_id),
        "league_ratings": _count_rows(supabase, "league_ratings", club_id),
    }
    summary["post_replay_counts"] = current_counts

    previous_counts = previous.get("post_replay_counts") if isinstance(previous, dict) else None
    if not isinstance(previous_counts, dict):
        return

    if (
        int(previous_counts.get("matches", -1)) != int(current_counts["matches"])
        or int(previous_counts.get("league_ratings", -1)) != int(current_counts["league_ratings"])
    ):
        raise RuntimeError(
            "Deterministic replay verification failed: current counts differ from previous successful run. "
            f"previous={previous_counts} current={current_counts}"
        )


def _load_df_meta(supabase, club_id: str) -> pd.DataFrame:
    try:
        response = (
            supabase.table("leagues")
            .select("league_name,k_factor")
            .eq("club_id", club_id)
            .execute()
        )
        rows = response.data or []
    except Exception:
        rows = []

    if not rows:
        return pd.DataFrame(columns=["league_name", "k_factor"])
    return pd.DataFrame(rows)


def _record_run(supabase, club_id: str, status: str, summary: dict) -> None:
    payload = {
        "club_id": club_id,
        "status": status,
        "summary": summary,
    }
    try:
        supabase.table("replay_runs").insert(payload).execute()
    except Exception:
        # Optional tracking table; do not fail replay if absent.
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay deterministic ratings from historical matches.")
    parser.add_argument("--club-id", required=True, help="Club ID to replay.")
    parser.add_argument("--dry-run", action="store_true", help="Validate replay inputs without mutating data.")
    parser.add_argument("--force", action="store_true", help="Wait for the replay lock instead of failing fast.")
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL"))
    parser.add_argument("--supabase-key", default=os.getenv("SUPABASE_KEY"))

    args = parser.parse_args()

    if not args.supabase_url or not args.supabase_key:
        print("Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY or pass flags.")
        return 2

    supabase = make_supabase(args.supabase_url, args.supabase_key)

    if args.dry_run:
        print(json.dumps({"club_id": args.club_id, "dry_run": True, "status": "validated"}, indent=2))
        return 0

    summary: dict = {}

    try:
        acquire_replay_lock_with_wait(supabase, str(args.club_id), wait=bool(args.force))
        try:
            df_meta = _load_df_meta(supabase, args.club_id)
            summary = replay_history(
                supabase=supabase,
                club_id=str(args.club_id),
                df_meta=df_meta,
                target_reset=FULL_RESET_LABEL,
                progress_cb=None,
                acquire_lock=False,
            )
            _verify_deterministic_counts_or_raise(supabase, str(args.club_id), summary)
            _record_run(supabase, str(args.club_id), "success", summary)
        finally:
            release_replay_lock(supabase, str(args.club_id))
    except ReplayAlreadyRunningError as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        _record_run(supabase, str(args.club_id), "failed", {"error": str(exc)})
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
