from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Iterable

import pandas as pd

INACTIVITY_DAYS = 14
INACTIVITY_THRESHOLD = timedelta(days=INACTIVITY_DAYS)


def coerce_utc_datetime(value) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def max_activity_time(current, candidate) -> datetime | None:
    cur_dt = coerce_utc_datetime(current)
    cand_dt = coerce_utc_datetime(candidate)
    if cur_dt is None:
        return cand_dt
    if cand_dt is None:
        return cur_dt
    return max(cur_dt, cand_dt)


def should_mark_inactive(
    last_game_at,
    created_at,
    *,
    now_utc: datetime | None = None,
    threshold: timedelta = INACTIVITY_THRESHOLD,
) -> bool:
    """Use created_at when the player has never logged a recorded game."""
    now = now_utc or datetime.now(timezone.utc)
    baseline = coerce_utc_datetime(last_game_at) or coerce_utc_datetime(created_at)
    if baseline is None:
        return False
    return (now - baseline) >= threshold


def add_activity_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    data = df.copy()
    if "inactive_at" in data.columns:
        inactive_at = pd.to_datetime(data["inactive_at"], utc=True, errors="coerce")
        data["active"] = inactive_at.isna()
    return data


def build_player_activity_update(existing_last_game_at, match_time) -> dict:
    latest = max_activity_time(existing_last_game_at, match_time)
    if latest is None:
        return {}
    return {"last_game_at": latest.isoformat(), "inactive_at": None}


def recompute_last_game_at_for_players(
    *,
    supabase,
    club_id: str,
    player_ids: Iterable[int],
) -> None:
    """Recompute last_game_at for players after match deletions/voids."""
    for pid in {int(pid) for pid in player_ids if pid is not None}:
        resp = (
            supabase.table("matches")
            .select("date,score_t1,score_t2")
            .eq("club_id", str(club_id))
            .or_(f"t1_p1.eq.{pid},t1_p2.eq.{pid},t2_p1.eq.{pid},t2_p2.eq.{pid}")
            .order("date", desc=True)
            .limit(50)
            .execute()
        )
        rows = resp.data or []
        df = pd.DataFrame(rows)
        if df.empty:
            latest = None
        else:
            df["score_t1"] = pd.to_numeric(df.get("score_t1", 0), errors="coerce").fillna(0).astype(int)
            df["score_t2"] = pd.to_numeric(df.get("score_t2", 0), errors="coerce").fillna(0).astype(int)
            df = df[(df["score_t1"] + df["score_t2"]) > 0].copy()
            if df.empty:
                latest = None
            else:
                df["date_dt"] = pd.to_datetime(df.get("date", None), errors="coerce", utc=True)
                latest = df["date_dt"].max()
                if pd.isna(latest):
                    latest = None
        payload = {"last_game_at": latest.isoformat()} if latest is not None else {"last_game_at": None}
        supabase.table("players").update(payload).eq("club_id", str(club_id)).eq("id", pid).execute()
