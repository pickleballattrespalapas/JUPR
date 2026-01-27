from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from jupr_app.domain.constants import (
    MIN_GAMES_PARTNER,
    MIN_GAMES_RIVAL,
    RIVAL_BALANCE_THRESHOLD,
)
from jupr_app.domain.match_filters import apply_match_filters, normalize_player_id, normalize_score


@dataclass(frozen=True)
class StoryStatsConfig:
    min_games_rival: int = MIN_GAMES_RIVAL
    rival_balance_threshold: float = RIVAL_BALANCE_THRESHOLD
    min_games_partner: int = MIN_GAMES_PARTNER


def compute_rival(
    player_id: int,
    context_filters: dict | None,
    df_matches: pd.DataFrame,
    config: StoryStatsConfig | None = None,
) -> dict | None:
    config = config or StoryStatsConfig()
    stats = _build_opponent_stats(df_matches, context_filters)
    return _select_rival(stats, int(player_id), config)


def compute_best_partner(
    player_id: int,
    context_filters: dict | None,
    df_matches: pd.DataFrame,
    config: StoryStatsConfig | None = None,
) -> dict | None:
    config = config or StoryStatsConfig()
    stats = _build_partner_stats(df_matches, context_filters)
    return _select_partner(stats, int(player_id), config)


def build_rival_map(
    player_ids: Iterable[int],
    context_filters: dict | None,
    df_matches: pd.DataFrame,
    config: StoryStatsConfig | None = None,
) -> dict[int, dict]:
    config = config or StoryStatsConfig()
    stats = _build_opponent_stats(df_matches, context_filters)
    rivals = {}
    for pid in player_ids:
        try:
            pid_int = int(pid)
        except Exception:
            continue
        rival = _select_rival(stats, pid_int, config)
        if rival:
            rivals[pid_int] = rival
    return rivals


def build_best_partner_map(
    player_ids: Iterable[int],
    context_filters: dict | None,
    df_matches: pd.DataFrame,
    config: StoryStatsConfig | None = None,
) -> dict[int, dict]:
    config = config or StoryStatsConfig()
    stats = _build_partner_stats(df_matches, context_filters)
    partners = {}
    for pid in player_ids:
        try:
            pid_int = int(pid)
        except Exception:
            continue
        partner = _select_partner(stats, pid_int, config)
        if partner:
            partners[pid_int] = partner
    return partners


def _build_opponent_stats(df_matches: pd.DataFrame, context_filters: dict | None) -> pd.DataFrame:
    filtered = apply_match_filters(df_matches, context_filters)
    if filtered.empty:
        return pd.DataFrame()

    records = []
    for row in filtered.itertuples(index=False):
        try:
            p1 = normalize_player_id(getattr(row, "t1_p1", None))
            p2 = normalize_player_id(getattr(row, "t1_p2", None))
            p3 = normalize_player_id(getattr(row, "t2_p1", None))
            p4 = normalize_player_id(getattr(row, "t2_p2", None))
            s1 = normalize_score(getattr(row, "score_t1", None))
            s2 = normalize_score(getattr(row, "score_t2", None))
            date_dt = getattr(row, "date_dt", pd.NaT)
        except Exception:
            continue

        if (s1 + s2) <= 0:
            continue

        team1 = [pid for pid in (p1, p2) if pid is not None]
        team2 = [pid for pid in (p3, p4) if pid is not None]
        if not team1 or not team2:
            continue

        winner = _winner_team(s1, s2)
        for pid in team1:
            for opp in team2:
                records.append(
                    {
                        "player_id": pid,
                        "opponent_id": opp,
                        "win": 1 if winner == 1 else 0,
                        "date_dt": date_dt,
                    }
                )
        for pid in team2:
            for opp in team1:
                records.append(
                    {
                        "player_id": pid,
                        "opponent_id": opp,
                        "win": 1 if winner == 2 else 0,
                        "date_dt": date_dt,
                    }
                )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    grouped = (
        df.groupby(["player_id", "opponent_id"], as_index=False)
        .agg(
            games=("win", "size"),
            wins=("win", "sum"),
            last_date=("date_dt", "max"),
        )
    )
    grouped["win_pct"] = grouped.apply(
        lambda r: (float(r["wins"]) / float(r["games"])) if r["games"] else 0.0,
        axis=1,
    )
    return grouped


def _build_partner_stats(df_matches: pd.DataFrame, context_filters: dict | None) -> pd.DataFrame:
    filtered = apply_match_filters(df_matches, context_filters)
    if filtered.empty:
        return pd.DataFrame()

    records = []
    for row in filtered.itertuples(index=False):
        try:
            p1 = normalize_player_id(getattr(row, "t1_p1", None))
            p2 = normalize_player_id(getattr(row, "t1_p2", None))
            p3 = normalize_player_id(getattr(row, "t2_p1", None))
            p4 = normalize_player_id(getattr(row, "t2_p2", None))
            s1 = normalize_score(getattr(row, "score_t1", None))
            s2 = normalize_score(getattr(row, "score_t2", None))
            date_dt = getattr(row, "date_dt", pd.NaT)
        except Exception:
            continue

        if (s1 + s2) <= 0:
            continue

        winner = _winner_team(s1, s2)
        if p1 is not None and p2 is not None and p1 != p2:
            records.append(
                {
                    "player_id": p1,
                    "partner_id": p2,
                    "win": 1 if winner == 1 else 0,
                    "date_dt": date_dt,
                }
            )
            records.append(
                {
                    "player_id": p2,
                    "partner_id": p1,
                    "win": 1 if winner == 1 else 0,
                    "date_dt": date_dt,
                }
            )
        if p3 is not None and p4 is not None and p3 != p4:
            records.append(
                {
                    "player_id": p3,
                    "partner_id": p4,
                    "win": 1 if winner == 2 else 0,
                    "date_dt": date_dt,
                }
            )
            records.append(
                {
                    "player_id": p4,
                    "partner_id": p3,
                    "win": 1 if winner == 2 else 0,
                    "date_dt": date_dt,
                }
            )

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)
    grouped = (
        df.groupby(["player_id", "partner_id"], as_index=False)
        .agg(
            games=("win", "size"),
            wins=("win", "sum"),
            last_date=("date_dt", "max"),
        )
    )
    grouped["win_pct"] = grouped.apply(
        lambda r: (float(r["wins"]) / float(r["games"])) if r["games"] else 0.0,
        axis=1,
    )
    return grouped


def _select_rival(stats: pd.DataFrame, player_id: int, config: StoryStatsConfig) -> dict | None:
    if stats is None or stats.empty:
        return None
    df = stats[stats["player_id"] == int(player_id)].copy()
    if df.empty:
        return None

    df = df[df["games"] >= int(config.min_games_rival)].copy()
    if df.empty:
        return None

    df["balance"] = (df["win_pct"] - 0.5).abs()
    within = df[df["balance"] <= float(config.rival_balance_threshold)].copy()
    candidates = within if not within.empty else df

    candidates = candidates.sort_values(
        ["games", "balance", "last_date"],
        ascending=[False, True, False],
        kind="mergesort",
    )
    if candidates.empty:
        return None
    row = candidates.iloc[0]
    games = int(row["games"])
    wins = int(row["wins"])
    win_pct = float(row["win_pct"]) if games else 0.0
    return {
        "player_id": int(player_id),
        "opponent_id": int(row["opponent_id"]),
        "games": games,
        "wins": wins,
        "losses": games - wins,
        "win_pct": win_pct,
        "last_date": row.get("last_date"),
    }


def _select_partner(stats: pd.DataFrame, player_id: int, config: StoryStatsConfig) -> dict | None:
    if stats is None or stats.empty:
        return None
    df = stats[stats["player_id"] == int(player_id)].copy()
    if df.empty:
        return None

    df = df[df["games"] >= int(config.min_games_partner)].copy()
    if df.empty:
        return None

    df = df.sort_values(
        ["win_pct", "games", "last_date"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    row = df.iloc[0]
    games = int(row["games"])
    wins = int(row["wins"])
    win_pct = float(row["win_pct"]) if games else 0.0
    return {
        "player_id": int(player_id),
        "partner_id": int(row["partner_id"]),
        "games": games,
        "wins": wins,
        "losses": games - wins,
        "win_pct": win_pct,
        "last_date": row.get("last_date"),
    }




def _winner_team(score_t1: int, score_t2: int) -> int:
    if score_t1 > score_t2:
        return 1
    if score_t2 > score_t1:
        return 2
    return 0
