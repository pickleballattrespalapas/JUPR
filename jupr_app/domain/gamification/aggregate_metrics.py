from __future__ import annotations

import pandas as pd


def _resolve_players_source(ctx) -> pd.DataFrame:
    df_players_all = getattr(ctx, "df_players_all", None)
    if isinstance(df_players_all, pd.DataFrame) and not df_players_all.empty:
        return df_players_all.copy()
    df_players_active = getattr(ctx, "df_players_active", None)
    if isinstance(df_players_active, pd.DataFrame) and not df_players_active.empty:
        return df_players_active.copy()
    return pd.DataFrame()


def compute_overall_standings_totals(ctx) -> pd.DataFrame:
    """Return standings-backed aggregate totals keyed by player_id."""
    df_players = _resolve_players_source(ctx)
    if df_players.empty:
        return pd.DataFrame(columns=["player_id", "wins", "losses", "matches_played"])

    player_col = "id" if "id" in df_players.columns else "player_id"
    if player_col not in df_players.columns:
        return pd.DataFrame(columns=["player_id", "wins", "losses", "matches_played"])

    totals = pd.DataFrame()
    totals["player_id"] = pd.to_numeric(df_players[player_col], errors="coerce")
    totals = totals.dropna(subset=["player_id"])

    wins = pd.to_numeric(df_players.get("wins", 0), errors="coerce").fillna(0)
    losses = pd.to_numeric(df_players.get("losses", 0), errors="coerce").fillna(0)
    if "matches_played" in df_players.columns:
        matches = pd.to_numeric(df_players.get("matches_played"), errors="coerce")
    else:
        matches = pd.Series([pd.NA] * len(df_players), index=df_players.index)
    matches = matches.fillna(wins + losses)

    totals["wins"] = wins
    totals["losses"] = losses
    totals["matches_played"] = matches

    totals["player_id"] = totals["player_id"].astype(int)
    totals["wins"] = pd.to_numeric(totals["wins"], errors="coerce").fillna(0).astype(int)
    totals["losses"] = pd.to_numeric(totals["losses"], errors="coerce").fillna(0).astype(int)
    totals["matches_played"] = pd.to_numeric(totals["matches_played"], errors="coerce").fillna(0).astype(int)

    totals = totals.groupby("player_id", as_index=False)[["wins", "losses", "matches_played"]].max()
    return totals


def compute_high_roller_counts_from_standings(ctx) -> pd.Series:
    totals = compute_overall_standings_totals(ctx)
    if totals.empty:
        return pd.Series(dtype="int64")
    return totals.set_index("player_id")["wins"].sort_values(ascending=False)


def compute_participation_counts_from_standings(ctx) -> pd.Series:
    totals = compute_overall_standings_totals(ctx)
    if totals.empty:
        return pd.Series(dtype="int64")
    return totals.set_index("player_id")["matches_played"].sort_values(ascending=False)
