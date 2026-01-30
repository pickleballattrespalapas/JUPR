from __future__ import annotations

import pandas as pd


def compute_lifetime_games(ctx) -> dict[int, int]:
    df_matches = getattr(ctx, "df_matches", None)
    if df_matches is None or df_matches.empty:
        return {}

    df = df_matches.copy()
    club_id = str(getattr(ctx, "club_id", "") or "")
    if club_id and "club_id" in df.columns:
        df = df[df["club_id"].astype(str) == club_id]

    if df.empty:
        return {}

    score_cols = _score_columns(df)
    if "player_id" in df.columns:
        pid = pd.to_numeric(df["player_id"], errors="coerce")
        valid = pid.notna()
        if score_cols:
            score_total = pd.to_numeric(df[score_cols[0]], errors="coerce").fillna(0) + pd.to_numeric(
                df[score_cols[1]], errors="coerce"
            ).fillna(0)
            valid &= score_total > 0
        pid = pid[valid].astype(int)
        if pid.empty:
            return {}
        return pid.value_counts().to_dict()

    player_cols = [c for c in ("t1_p1", "t1_p2", "t2_p1", "t2_p2") if c in df.columns]
    if len(player_cols) < 1:
        return {}

    players = df[player_cols].apply(pd.to_numeric, errors="coerce")
    valid = players.notna().all(axis=1)
    if score_cols:
        score_total = pd.to_numeric(df[score_cols[0]], errors="coerce").fillna(0) + pd.to_numeric(
            df[score_cols[1]], errors="coerce"
        ).fillna(0)
        valid &= score_total > 0

    players = players[valid]
    counts: dict[int, int] = {}
    for _, row in players.iterrows():
        ids = {int(pid) for pid in row.tolist() if pd.notna(pid)}
        for pid in ids:
            counts[pid] = counts.get(pid, 0) + 1
    return counts


def _score_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    if "score_t1" in df.columns and "score_t2" in df.columns:
        return "score_t1", "score_t2"
    if "s1" in df.columns and "s2" in df.columns:
        return "s1", "s2"
    return None
