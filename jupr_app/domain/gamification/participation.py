from __future__ import annotations

import pandas as pd

from jupr_app.domain.match_filters import apply_match_filters, normalize_player_id, normalize_score


def get_participation_player_match_pairs(ctx) -> pd.DataFrame:
    """Return unique (player_id, match_id) pairs after canonical match filtering."""
    df_matches = getattr(ctx, "df_matches", None)
    if df_matches is None or df_matches.empty:
        return pd.DataFrame(columns=["player_id", "match_id"])

    filters = {"club_id": getattr(ctx, "club_id", None), "exclude_popups": True}
    filtered = apply_match_filters(df_matches, filters)
    if filtered.empty:
        return pd.DataFrame(columns=["player_id", "match_id"])

    filtered = filtered.copy()
    match_id_col = _ensure_match_id_column(filtered)

    if "player_id" in filtered.columns:
        pairs = _pairs_from_player_rows(filtered, match_id_col)
    else:
        pairs = _pairs_from_match_rows(filtered, match_id_col)

    if pairs.empty:
        return pairs

    pairs = pairs.dropna(subset=["player_id", "match_id"]).drop_duplicates(["player_id", "match_id"])
    return pairs


def compute_lifetime_games(ctx) -> dict[int, int]:
    pairs = get_participation_player_match_pairs(ctx)
    if pairs.empty:
        return {}
    counts = pairs.groupby("player_id")["match_id"].nunique()
    return counts.to_dict()


def _ensure_match_id_column(df: pd.DataFrame) -> str:
    for col in ("match_id", "id", "matchId", "matchID"):
        if col in df.columns:
            return col
    df["match_id"] = range(1, len(df) + 1)
    return "match_id"


def _pairs_from_player_rows(df: pd.DataFrame, match_id_col: str) -> pd.DataFrame:
    match_ids = df[match_id_col].where(df[match_id_col].notna(), pd.NA).astype("string")
    player_ids = df["player_id"].map(normalize_player_id)
    pairs = pd.DataFrame({"player_id": player_ids, "match_id": match_ids})
    score_total = _score_total(df)
    if score_total is not None:
        pairs = pairs[score_total > 0]
    return pairs


def _pairs_from_match_rows(df: pd.DataFrame, match_id_col: str) -> pd.DataFrame:
    records: list[dict[str, str | int]] = []
    score_cols = _score_columns(df)
    for row in df.itertuples(index=False):
        match_id = getattr(row, match_id_col, None)
        if match_id is None or (isinstance(match_id, float) and pd.isna(match_id)):
            continue

        if score_cols:
            s1 = normalize_score(getattr(row, score_cols[0], None))
            s2 = normalize_score(getattr(row, score_cols[1], None))
            if (s1 + s2) <= 0:
                continue

        p1 = normalize_player_id(getattr(row, "t1_p1", None))
        p2 = normalize_player_id(getattr(row, "t1_p2", None))
        p3 = normalize_player_id(getattr(row, "t2_p1", None))
        p4 = normalize_player_id(getattr(row, "t2_p2", None))
        if not p1 or not p3:
            continue

        for pid in {pid for pid in (p1, p2, p3, p4) if pid}:
            records.append({"player_id": pid, "match_id": str(match_id)})

    if not records:
        return pd.DataFrame(columns=["player_id", "match_id"])
    return pd.DataFrame.from_records(records)


def _score_columns(df: pd.DataFrame) -> tuple[str, str] | None:
    if "score_t1" in df.columns and "score_t2" in df.columns:
        return "score_t1", "score_t2"
    if "s1" in df.columns and "s2" in df.columns:
        return "s1", "s2"
    return None


def _score_total(df: pd.DataFrame) -> pd.Series | None:
    score_cols = _score_columns(df)
    if not score_cols:
        return None
    s1 = pd.to_numeric(df[score_cols[0]], errors="coerce").fillna(0)
    s2 = pd.to_numeric(df[score_cols[1]], errors="coerce").fillna(0)
    return s1 + s2
