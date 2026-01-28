from __future__ import annotations

from datetime import datetime, timezone
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _build_league_standings(df_leagues: pd.DataFrame, league_id: str) -> pd.DataFrame:
    if df_leagues is None or df_leagues.empty or "league_name" not in df_leagues.columns:
        return pd.DataFrame()

    df = df_leagues.copy()
    df["league_name"] = df["league_name"].fillna("").astype(str).str.strip()
    df = df[df["league_name"] == str(league_id).strip()].copy()
    if df.empty:
        return pd.DataFrame()

    df["player_id"] = pd.to_numeric(df.get("player_id"), errors="coerce").fillna(-1).astype(int)
    df = df[df["player_id"] > 0].copy()
    if df.empty:
        return pd.DataFrame()

    df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0.0)
    df["matches_played"] = pd.to_numeric(df.get("matches_played", 0), errors="coerce").fillna(0).astype(int)
    df = df.sort_values(["rating", "matches_played"], ascending=[False, False]).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


def award_league_podium_badges(ctx, league_id: str) -> None:
    if bool(getattr(ctx, "public_mode", False)):
        return

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "")
    if supabase is None or not club_id or not league_id:
        return

    df_leagues = getattr(ctx, "df_leagues", None)
    standings = _build_league_standings(df_leagues, league_id)
    if standings.empty:
        return

    badge_map = {
        1: "league_champion",
        2: "league_runner_up",
        3: "league_third_place",
    }
    podium = standings.head(3)
    now = datetime.now(timezone.utc).isoformat()
    payload = []
    for _, row in podium.iterrows():
        rank = int(row.get("rank", 0) or 0)
        badge_id = badge_map.get(rank)
        if not badge_id:
            continue
        player_id = int(row.get("player_id"))
        payload.append(
            {
                "club_id": club_id,
                "player_id": player_id,
                "badge_id": badge_id,
                "earned_at": now,
                "context_type": "league",
                "context_id": f"{league_id}:podium:{rank}",
                "value_json": {"league_id": league_id, "type": "podium", "rank": rank},
            }
        )

    if not payload:
        return

    try:
        supabase.table("player_badges").upsert(
            payload,
            on_conflict="club_id,player_id,badge_id,context_id",
        ).execute()
    except Exception:
        logger.exception("Failed to award podium badges", extra={"league_id": league_id})
