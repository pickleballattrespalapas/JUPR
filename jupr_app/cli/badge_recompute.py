from __future__ import annotations

import argparse
import json
import os
import sys

from jupr_app.data.client import make_supabase
from jupr_app.domain.gamification.recompute import run_badge_recompute


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute/backfill badge awards.")
    parser.add_argument("--club_id", required=True, help="Club ID to recompute.")
    parser.add_argument("--league_id", help="Limit to a league ID.")
    parser.add_argument("--context_id", help="Limit to a specific context_id.")
    parser.add_argument("--player_id", type=int, help="Limit to a single player ID.")
    parser.add_argument("--badge_id", help="Limit to a badge ID.")
    parser.add_argument("--since", help="Only evaluate matches on/after this ISO timestamp.")
    parser.add_argument("--until", help="Only evaluate matches on/before this ISO timestamp.")
    parser.add_argument("--mode", choices=["dry-run", "append-only", "strict"], default="append-only")
    parser.add_argument("--match-limit", type=int, default=5000, help="Max matches to load.")
    parser.add_argument("--created-by", help="Actor identifier for audit.")
    parser.add_argument("--rule-version", help="Override rule version hash.")
    parser.add_argument("--revoke-reason", help="Reason for revocation in strict mode.")
    parser.add_argument("--allow-strict-global", action="store_true", help="Allow strict mode without badge/player/context filters.")
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL"))
    parser.add_argument("--supabase-key", default=os.getenv("SUPABASE_KEY"))

    args = parser.parse_args()

    if not args.supabase_url or not args.supabase_key:
        print("Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY or pass flags.")
        return 2

    supabase = make_supabase(args.supabase_url, args.supabase_key)

    summary = run_badge_recompute(
        supabase,
        club_id=args.club_id,
        mode=args.mode,
        league_id=args.league_id,
        context_id=args.context_id,
        player_id=args.player_id,
        badge_id=args.badge_id,
        since=args.since,
        until=args.until,
        created_by=args.created_by,
        rule_version=args.rule_version,
        revoke_reason=args.revoke_reason,
        allow_strict_global=args.allow_strict_global,
        match_limit=args.match_limit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
