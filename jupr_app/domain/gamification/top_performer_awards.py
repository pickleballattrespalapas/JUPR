from __future__ import annotations

import logging

import pandas as pd

from jupr_app.domain.awards import compute_top_performer_awards
from jupr_app.domain.gamification.badge_types import BadgeCandidate
from jupr_app.domain.gamification.badges_repo import upsert_player_badges


logger = logging.getLogger(__name__)


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
    if bool(getattr(ctx, "public_mode", False)):
        return

    supabase = getattr(ctx, "supabase", None)
    club_id = str(getattr(ctx, "club_id", "") or "")
    if supabase is None or not club_id or not league_id:
        return

    df_leagues = getattr(ctx, "df_leagues", None)
    df_meta = getattr(ctx, "df_meta", None)
    id_to_name = getattr(ctx, "id_to_name", {}) or {}

    min_games = _min_games_for_league(df_meta, league_id)
    if min_games <= 0:
        return

    standings = _build_league_standings(df_leagues, league_id, id_to_name)
    if standings.empty:
        return

    qualified = standings[standings["matches_played"] >= int(min_games)].copy()
    if qualified.empty:
        return

    awards = compute_top_performer_awards(qualified, min_games=min_games, winners_per_category=1)
    if not awards:
        return

    candidates: list[BadgeCandidate] = []
    for award in awards:
        category_key = award.get("category_key")
        badge_id = TOP_PERFORMER_BADGE_IDS.get(str(category_key))
        if not badge_id:
            continue
        rank = int(award.get("rank", 1))
        context_id = f"{league_id}:top_performer:{category_key}:{rank}"
        value_json = {
            "league_id": league_id,
            "category_key": category_key,
            "category_label": award.get("category_label"),
            "metric_value": award.get("metric_value"),
            "metric_display": award.get("metric_display"),
            "min_games": int(min_games),
            "rank": rank,
        }
        candidates.append(
            BadgeCandidate(
                badge_id=badge_id,
                player_id=int(award["player_id"]),
                club_id=club_id,
                context_type="league",
                context_id=context_id,
                match_id=None,
                value_json=value_json,
                value_num=award.get("metric_value") if award.get("metric_value") is not None else None,
            )
        )

    if not candidates:
        return

    try:
        created = upsert_player_badges(supabase, club_id, candidates)
        if created:
            logger.info(
                "Awarded top performer badges",
                extra={"league_id": league_id, "count": len(created)},
            )
    except Exception:
        logger.exception("Failed to award top performer badges", extra={"league_id": league_id})
