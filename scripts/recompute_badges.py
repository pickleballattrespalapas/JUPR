from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import pandas as pd

from jupr_app.data.client import make_supabase
from jupr_app.data.load import load_data
from jupr_app.domain.gamification.match_facts import build_player_match_facts
from jupr_app.domain.gamification.recompute import run_badge_recompute


DEFAULT_BADGES = ["dominant_run", "high_output", "above_expectations", "breakthrough"]


def _load_ctx(supabase, club_id: str, match_limit: int) -> SimpleNamespace:
    (
        df_players_all,
        df_players_active,
        df_leagues,
        df_matches,
        df_meta,
        df_badges,
        df_player_badges,
        name_to_id,
        id_to_name,
        schema_degraded,
        schema_degraded_reason,
    ) = load_data(supabase, club_id, match_limit=match_limit)

    return SimpleNamespace(
        supabase=supabase,
        club_id=club_id,
        df_players_all=df_players_all,
        df_players_active=df_players_active,
        df_leagues=df_leagues,
        df_matches=df_matches,
        df_meta=df_meta,
        df_badges=df_badges,
        df_player_badges=df_player_badges,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        public_mode=False,
        admin_logged_in=True,
        schema_degraded=schema_degraded,
        schema_degraded_reason=schema_degraded_reason,
    )


def _filtered_facts(ctx: SimpleNamespace, *, since: str | None, until: str | None) -> pd.DataFrame:
    df_matches = getattr(ctx, "df_matches", pd.DataFrame()).copy()
    if df_matches.empty:
        return pd.DataFrame()
    df_matches["date_dt"] = pd.to_datetime(df_matches.get("date"), utc=True, errors="coerce")
    if since:
        since_dt = pd.to_datetime(since, utc=True, errors="coerce")
        df_matches = df_matches[df_matches["date_dt"] >= since_dt]
    if until:
        until_dt = pd.to_datetime(until, utc=True, errors="coerce")
        df_matches = df_matches[df_matches["date_dt"] <= until_dt]
    df_matches = df_matches.drop(columns=["date_dt"])
    return build_player_match_facts(ctx, df_matches_override=df_matches, club_id_override=ctx.club_id)


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute/backfill placeholder badges.")
    parser.add_argument("--club_id", required=True, help="Club ID to recompute.")
    parser.add_argument("--league_id", help="Limit to a league ID.")
    parser.add_argument("--badge_ids", default=",".join(DEFAULT_BADGES), help="Comma-separated badge IDs.")
    parser.add_argument("--mode", choices=["dry-run", "append-only", "strict"], default="append-only")
    parser.add_argument("--since", help="Only evaluate matches on/after this ISO timestamp.")
    parser.add_argument("--until", help="Only evaluate matches on/before this ISO timestamp.")
    parser.add_argument("--created-by", help="Actor identifier for audit.")
    parser.add_argument("--revoke-reason", help="Reason for revocation in strict mode.")
    parser.add_argument("--allow-strict-global", action="store_true", help="Allow strict mode without badge/player/context filters.")
    parser.add_argument("--match-limit", type=int, default=5000, help="Max matches to load.")
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL"))
    parser.add_argument("--supabase-key", default=os.getenv("SUPABASE_KEY"))

    args = parser.parse_args()

    if not args.supabase_url or not args.supabase_key:
        print("Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY or pass flags.")
        return 2

    supabase = make_supabase(args.supabase_url, args.supabase_key)
    ctx = _load_ctx(supabase, args.club_id, args.match_limit)
    facts = _filtered_facts(ctx, since=args.since, until=args.until)

    badge_ids = [bid.strip() for bid in args.badge_ids.split(",") if bid.strip()]
    per_badge = {}
    totals = {"new_awards": 0, "revoked": 0, "unrevoked": 0}

    for badge_id in badge_ids:
        summary = run_badge_recompute(
            supabase,
            club_id=args.club_id,
            mode=args.mode,
            league_id=args.league_id,
            badge_id=badge_id,
            since=args.since,
            until=args.until,
            created_by=args.created_by,
            revoke_reason=args.revoke_reason,
            allow_strict_global=args.allow_strict_global,
            match_limit=args.match_limit,
            ctx=ctx,
        )
        per_badge[badge_id] = summary
        totals["new_awards"] += int(summary.get("new_awards_count", 0))
        totals["revoked"] += int(summary.get("revoked_count", 0))
        totals["unrevoked"] += int(summary.get("unrevoked_count", 0))

    report = {
        "badge_ids": badge_ids,
        "players_scanned": int(facts["player_id"].nunique()) if not facts.empty else 0,
        "matches_scanned": int(facts["match_id"].nunique()) if not facts.empty else 0,
        "totals": totals,
        "per_badge": per_badge,
    }
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
