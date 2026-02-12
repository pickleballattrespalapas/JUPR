from __future__ import annotations

import logging

import pandas as pd

from jupr_app.ui.helpers import sanitize_story_text

logger = logging.getLogger(__name__)


def fetch_player_stories(supabase, club_id: str, pid: int, limit: int = 6) -> pd.DataFrame:
    try:
        resp = (
            supabase.table("player_stories")
            .select("story_type,context_id,created_at,title,body,importance,match_id")
            .eq("club_id", str(club_id))
            .eq("player_id", int(pid))
            .order("created_at", desc=True)
            .limit(int(limit) * 3)
            .execute()
        )
        return pd.DataFrame(resp.data or [])
    except Exception:
        logger.exception("Failed to load player stories")
        return pd.DataFrame()


def normalize_stories(df: pd.DataFrame, limit: int = 6) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    normalized = df.copy()
    normalized = normalized.drop_duplicates(subset=["story_type", "context_id"], keep="first")
    normalized = normalized.sort_values("created_at", ascending=False)
    return normalized.head(int(limit) * 2)


def safe_story_fields(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    sanitized = df.copy()
    if "title" in sanitized.columns:
        sanitized["title"] = sanitized["title"].apply(sanitize_story_text)
    if "body" in sanitized.columns:
        sanitized["body"] = sanitized["body"].apply(sanitize_story_text)
    return sanitized
