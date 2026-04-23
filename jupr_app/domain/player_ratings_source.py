from __future__ import annotations

from typing import Any

import pandas as pd


def _normalize_league_name(value: object) -> str:
    return str(value or "").strip()


def build_seed_rating_maps(
    *,
    supabase,
    club_id: str,
    player_ids: set[int] | None = None,
    league_names: set[str] | None = None,
    df_players_all: pd.DataFrame | None = None,
    df_leagues: pd.DataFrame | None = None,
) -> tuple[dict[int, float], dict[tuple[int, str], float], bool]:
    """
    Return (overall_map, league_map, from_live_tables).

    - overall_map keys are player_id -> overall ELO rating from players.rating
    - league_map keys are (player_id, league_name) -> league ELO from league_ratings.rating
    - from_live_tables indicates whether DB reads succeeded (True) or fallback DataFrame data was used (False)
    """
    player_ids = {int(pid) for pid in (player_ids or set()) if pid is not None}
    league_names = {_normalize_league_name(name) for name in (league_names or set()) if _normalize_league_name(name)}

    overall_map: dict[int, float] = {}
    league_map: dict[tuple[int, str], float] = {}

    def _fill_from_fallback() -> tuple[dict[int, float], dict[tuple[int, str], float]]:
        local_overall: dict[int, float] = {}
        local_league: dict[tuple[int, str], float] = {}

        if isinstance(df_players_all, pd.DataFrame) and not df_players_all.empty:
            frame = df_players_all.copy()
            if "id" in frame.columns and "rating" in frame.columns:
                if player_ids:
                    frame = frame[frame["id"].astype(int).isin(player_ids)]
                for _, row in frame.iterrows():
                    try:
                        local_overall[int(row["id"])] = float(row.get("rating", 1200.0) or 1200.0)
                    except Exception:
                        continue

        if isinstance(df_leagues, pd.DataFrame) and not df_leagues.empty:
            frame = df_leagues.copy()
            if {"player_id", "league_name", "rating"}.issubset(set(frame.columns)):
                if player_ids:
                    frame = frame[frame["player_id"].astype(int).isin(player_ids)]
                if league_names:
                    frame = frame[frame["league_name"].astype(str).str.strip().isin(league_names)]
                for _, row in frame.iterrows():
                    try:
                        key = (int(row["player_id"]), _normalize_league_name(row.get("league_name")))
                        if key[1]:
                            local_league[key] = float(row.get("rating", 1200.0) or 1200.0)
                    except Exception:
                        continue

        return local_overall, local_league

    from_live_tables = True
    try:
        players_query = (
            supabase.table("players")
            .select("id,rating")
            .eq("club_id", str(club_id))
        )
        if player_ids:
            players_query = players_query.in_("id", sorted(player_ids))
        players_resp = players_query.execute()
        for row in (players_resp.data or []):
            try:
                overall_map[int(row["id"])] = float(row.get("rating", 1200.0) or 1200.0)
            except Exception:
                continue

        leagues_query = (
            supabase.table("league_ratings")
            .select("player_id,league_name,rating")
            .eq("club_id", str(club_id))
        )
        if player_ids:
            leagues_query = leagues_query.in_("player_id", sorted(player_ids))
        leagues_resp = leagues_query.execute()
        for row in (leagues_resp.data or []):
            try:
                league_name = _normalize_league_name(row.get("league_name"))
                if league_names and league_name not in league_names:
                    continue
                key = (int(row["player_id"]), league_name)
                if key[1]:
                    league_map[key] = float(row.get("rating", 1200.0) or 1200.0)
            except Exception:
                continue
    except Exception:
        from_live_tables = False
        overall_map, league_map = _fill_from_fallback()

    return overall_map, league_map, from_live_tables


def current_seed_rating(
    *,
    player_id: int,
    league_name: str,
    overall_map: dict[int, float],
    league_map: dict[tuple[int, str], float],
    default_rating: float = 1200.0,
) -> float:
    normalized_league = _normalize_league_name(league_name)
    league_key = (int(player_id), normalized_league)
    if normalized_league and league_key in league_map:
        return float(league_map[league_key])
    if int(player_id) in overall_map:
        return float(overall_map[int(player_id)])
    return float(default_rating)
