from __future__ import annotations

from typing import Iterable

import pandas as pd


def dedupe_player_badges_rows(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    required = ["club_id", "player_id", "badge_id", "context_id"]
    for col in required:
        if col not in df.columns:
            df[col] = None
    df["earned_at_dt"] = pd.to_datetime(df.get("earned_at", None), utc=True, errors="coerce")
    df = df.sort_values(["earned_at_dt", "id"], ascending=[True, True], na_position="last")
    deduped = df.drop_duplicates(subset=required, keep="first").copy()
    return deduped.drop(columns=["earned_at_dt"], errors="ignore")
