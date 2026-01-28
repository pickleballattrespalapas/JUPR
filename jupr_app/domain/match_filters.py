from __future__ import annotations

import pandas as pd


def apply_match_filters(df_matches: pd.DataFrame, context_filters: dict | None) -> pd.DataFrame:
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()

    df = df_matches.copy()
    context_filters = context_filters or {}

    club_id = context_filters.get("club_id")
    if club_id is not None and "club_id" in df.columns:
        df = df[df["club_id"].astype(str) == str(club_id)].copy()

    league_name = context_filters.get("league_name")
    if league_name:
        df["league"] = df.get("league", "").fillna("").astype(str).str.strip()
        df = df[df["league"] == str(league_name).strip()].copy()

    if context_filters.get("exclude_popups", False) and "match_type" in df.columns:
        df = df[df["match_type"].fillna("") != "PopUp"].copy()

    if "is_valid" in df.columns:
        df = df[df["is_valid"].fillna(True).astype(bool)].copy()

    if "context_type" in df.columns:
        df = df[df["context_type"].fillna("").astype(str).str.upper() != "TOURNAMENT"].copy()

    if "tournament_id" in df.columns:
        df = df[df["tournament_id"].isna()].copy()

    if "match_type" in df.columns:
        df = df[df["match_type"].fillna("").astype(str) != "Tournament"].copy()

    for col in [
        "is_void",
        "voided",
        "is_voided",
        "invalid",
        "is_invalid",
        "deleted",
        "is_deleted",
    ]:
        if col in df.columns:
            df = df[~df[col].fillna(False).astype(bool)].copy()

    start_date = context_filters.get("start_date")
    end_date = context_filters.get("end_date")
    df["date_dt"] = pd.to_datetime(df.get("date", None), utc=True, errors="coerce")
    if start_date is not None:
        start_dt = pd.to_datetime(start_date, utc=True, errors="coerce")
        if pd.notna(start_dt):
            df = df[df["date_dt"] >= start_dt].copy()
    if end_date is not None:
        end_dt = pd.to_datetime(end_date, utc=True, errors="coerce")
        if pd.notna(end_dt):
            df = df[df["date_dt"] <= end_dt].copy()

    eligible_ids = context_filters.get("eligible_player_ids")
    if eligible_ids:
        eligible_set = {int(pid) for pid in eligible_ids}
        player_cols = [c for c in ["t1_p1", "t1_p2", "t2_p1", "t2_p2"] if c in df.columns]
        if player_cols:
            mask = pd.Series(True, index=df.index)
            for col in player_cols:
                values = df[col]
                keep = values.isna() | values.map(lambda x: normalize_player_id(x) in eligible_set)
                mask &= keep
            df = df[mask].copy()

    return df


def normalize_player_id(value) -> int | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        pid = int(value)
        if pid <= 0:
            return None
        return pid
    except Exception:
        return None


def normalize_score(value) -> int:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return 0
        return int(value)
    except Exception:
        return 0
