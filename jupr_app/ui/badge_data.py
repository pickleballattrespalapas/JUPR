from __future__ import annotations

import pandas as pd


def normalize_player_ids(player_ids: list[object] | tuple[object, ...] | set[object]) -> list[int]:
    if not player_ids:
        return []
    ids_series = pd.to_numeric(pd.Series(list(player_ids)), errors="coerce").dropna()
    if ids_series.empty:
        return []
    return ids_series.astype(int).tolist()


def normalize_player_badges_frame(pb_df: pd.DataFrame | None) -> pd.DataFrame:
    if pb_df is None or pb_df.empty:
        return pd.DataFrame()
    if "player_id" not in pb_df.columns:
        return pd.DataFrame()

    normalized = pb_df.copy()
    normalized["player_id"] = pd.to_numeric(normalized["player_id"], errors="coerce")
    normalized = normalized.dropna(subset=["player_id"]).copy()
    if normalized.empty:
        return pd.DataFrame(columns=pb_df.columns)
    normalized["player_id"] = normalized["player_id"].astype(int)

    if "badge_id" in normalized.columns:
        normalized["badge_id"] = normalized["badge_id"].fillna("").astype(str).str.strip()
    if "club_id" in normalized.columns:
        normalized["club_id"] = normalized["club_id"].fillna("").astype(str).str.strip()
    return normalized

