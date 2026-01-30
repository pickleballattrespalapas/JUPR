from __future__ import annotations

"""DEPRECATED: top performer awards are computed via the badge engine."""

import pandas as pd


TOP_PERFORMER_BADGE_IDS = {
    "highest_rating": "top_performer_highest_rating",
    "most_improved": "top_performer_most_improved",
    "best_win_pct": "top_performer_best_win_pct",
    "most_wins": "top_performer_most_wins",
}


def _min_games_for_league(df_meta: pd.DataFrame | None, league_id: str) -> int:
    if df_meta is None or df_meta.empty or "league_name" not in df_meta.columns:
        return 0
    try:
        cfg = df_meta.copy()
        cfg["league_name"] = cfg["league_name"].fillna("").astype(str).str.strip()
        hit = cfg[cfg["league_name"] == str(league_id).strip()]
        if hit.empty:
            return 0
        return int(hit.iloc[0].get("min_games", 0) or 0)
    except Exception:
        return 0


def _build_league_standings(df_leagues: pd.DataFrame | None, league_id: str, id_to_name: dict[int, str]) -> pd.DataFrame:
    if df_leagues is None or df_leagues.empty or "league_name" not in df_leagues.columns:
        return pd.DataFrame()
    df = df_leagues.copy()
    df["league_name"] = df["league_name"].fillna("").astype(str).str.strip()
    df = df[df["league_name"] == str(league_id).strip()].copy()
    if df.empty:
        return pd.DataFrame()
    if "name" not in df.columns:
        df["name"] = df["player_id"].map(id_to_name)
    for col in ["wins", "losses", "matches_played", "rating"]:
        if col not in df.columns:
            df[col] = 0
    if "starting_rating" not in df.columns:
        df["starting_rating"] = df.get("rating", 1200.0)
    df["_pid"] = pd.to_numeric(df.get("player_id"), errors="coerce").fillna(-1).astype(int)
    df = df[df["_pid"] > 0].copy()
    if df.empty:
        return pd.DataFrame()
    df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0.0)
    df["starting_rating"] = pd.to_numeric(
        df.get("starting_rating", df["rating"]), errors="coerce"
    ).fillna(df["rating"])
    df["wins"] = pd.to_numeric(df.get("wins", 0), errors="coerce").fillna(0).astype(int)
    df["losses"] = pd.to_numeric(df.get("losses", 0), errors="coerce").fillna(0).astype(int)
    df["matches_played"] = (
        pd.to_numeric(df.get("matches_played", 0), errors="coerce").fillna(0).astype(int)
    )
    df["JUPR"] = df["rating"].astype(float) / 400.0
    df["rating_gain"] = (df["rating"] - df["starting_rating"]).astype(float)
    df["Gain"] = df["rating_gain"].astype(float) / 400.0
    df["Win %"] = df.apply(
        lambda r: (
            (float(r["wins"]) / float(r["matches_played"]) * 100.0)
            if int(r["matches_played"]) > 0
            else pd.NA
        ),
        axis=1,
    )
    return df


def ensure_league_top_performer_awards(ctx, league_id: str) -> None:
    """DEPRECATED: use ensure_badges (badge engine) instead."""
    from jupr_app.domain.gamification.ensure_badges import ensure_badges

    ensure_badges(ctx, league_id=league_id)
