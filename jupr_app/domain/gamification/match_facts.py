from __future__ import annotations

"""This module is the single source of truth for badge match-facts.

Match facts are generated after applying match filters (club_id scoping,
exclude_popups, invalid/void/deleted flags, tournament exclusions, and
standard score normalization) to ensure badge evaluations stay consistent.
"""

from collections.abc import Iterable
from typing import Any

import pandas as pd

from jupr_app.domain.match_filters import apply_match_filters, normalize_player_id, normalize_score


_FACT_COLUMNS = [
    "club_id",
    "player_id",
    "match_id",
    "league",
    "date_dt",
    "week_key",
    "month_key",
    "season_key",
    "win",
    "points_for",
    "points_against",
    "margin",
    "partner_id",
    "opponent_ids",
    "expected_win_prob",
    "elo_delta_signed",
    "abs_elo_delta",
    "opp_max_rating",
    "lobby_avg_rating",
    "rating_pre",
    "rating_post",
]


def _empty_facts() -> pd.DataFrame:
    return pd.DataFrame(columns=_FACT_COLUMNS)


def build_player_match_facts(
    ctx, df_matches_override: pd.DataFrame | None = None, club_id_override: str | None = None
) -> pd.DataFrame:
    df_matches = df_matches_override
    if df_matches is None:
        df_matches = getattr(ctx, "df_matches", None)
    if df_matches is None or df_matches.empty:
        return _empty_facts()

    df_players = getattr(ctx, "df_players_all", None)
    rating_map: dict[int, float] = {}
    if df_players is not None and not df_players.empty:
        try:
            rating_map = dict(zip(df_players["id"].astype(int), df_players["rating"].astype(float)))
        except Exception:
            rating_map = {}

    club_id = club_id_override if club_id_override is not None else getattr(ctx, "club_id", None)
    filters = {"club_id": club_id, "exclude_popups": True}
    filtered = apply_match_filters(df_matches, filters)
    if filtered.empty:
        return _empty_facts()

    filtered = filtered.copy()
    if "id" not in filtered.columns:
        filtered["id"] = range(1, len(filtered) + 1)
    filtered["date_dt"] = pd.to_datetime(filtered.get("date", None), utc=True, errors="coerce")
    filtered = filtered.dropna(subset=["date_dt"]).sort_values(["date_dt", "id"], ascending=[True, True])

    records: list[dict[str, Any]] = []
    for row in filtered.itertuples(index=False):
        match_id = str(getattr(row, "id", "") or "")
        league = str(getattr(row, "league", "") or "").strip() or "OVERALL"
        club_id = str(getattr(row, "club_id", "") or "")
        date_dt = getattr(row, "date_dt", pd.NaT)
        if not match_id or pd.isna(date_dt):
            continue

        s1 = normalize_score(getattr(row, "score_t1", None))
        s2 = normalize_score(getattr(row, "score_t2", None))
        if (s1 + s2) <= 0:
            continue

        p1 = normalize_player_id(getattr(row, "t1_p1", None))
        p2 = normalize_player_id(getattr(row, "t1_p2", None))
        p3 = normalize_player_id(getattr(row, "t2_p1", None))
        p4 = normalize_player_id(getattr(row, "t2_p2", None))
        if not p1 or not p3:
            continue

        r1 = _safe_rating(getattr(row, "t1_p1_r", None), rating_map.get(p1))
        r2 = _safe_rating(getattr(row, "t1_p2_r", None), rating_map.get(p2))
        r3 = _safe_rating(getattr(row, "t2_p1_r", None), rating_map.get(p3))
        r4 = _safe_rating(getattr(row, "t2_p2_r", None), rating_map.get(p4))
        e1 = _safe_float(getattr(row, "t1_p1_r_end", None))
        e2 = _safe_float(getattr(row, "t1_p2_r_end", None))
        e3 = _safe_float(getattr(row, "t2_p1_r_end", None))
        e4 = _safe_float(getattr(row, "t2_p2_r_end", None))

        team1 = [pid for pid in (p1, p2) if pid]
        team2 = [pid for pid in (p3, p4) if pid]
        if not team1 or not team2:
            continue

        t1_avg = _avg([r1, r2] if p2 else [r1])
        t2_avg = _avg([r3, r4] if p4 else [r3])
        expected_t1 = _expected_share(t1_avg, t2_avg)

        winner_team = 1 if s1 > s2 else 2 if s2 > s1 else 0
        delta_abs = _safe_float(getattr(row, "elo_delta", None))

        lobby_avg_rating = _avg([r for r in [r1, r2, r3, r4] if r is not None])

        for pid, team, partner, opp_ids, my_score, opp_score, opp_avg, opp_max, expected_win, pre, post in (
            (p1, 1, p2, team2, s1, s2, _avg([r3, r4] if p4 else [r3]), _max_rating([r3, r4]), expected_t1, r1, e1),
            (p2, 1, p1, team2, s1, s2, _avg([r3, r4] if p4 else [r3]), _max_rating([r3, r4]), expected_t1, r2, e2),
            (p3, 2, p4, team1, s2, s1, _avg([r1, r2] if p2 else [r1]), _max_rating([r1, r2]), 1.0 - expected_t1, r3, e3),
            (p4, 2, p3, team1, s2, s1, _avg([r1, r2] if p2 else [r1]), _max_rating([r1, r2]), 1.0 - expected_t1, r4, e4),
        ):
            if not pid:
                continue
            win = winner_team == team
            signed_delta = None
            if delta_abs is not None:
                signed_delta = float(delta_abs) if win else -float(delta_abs)
            records.append(
                {
                    "club_id": club_id,
                    "player_id": int(pid),
                    "match_id": match_id,
                    "league": league,
                    "date_dt": date_dt,
                    "week_key": _week_key(date_dt),
                    "month_key": _month_key(date_dt),
                    "season_key": _season_key(date_dt),
                    "win": bool(win),
                    "points_for": int(my_score),
                    "points_against": int(opp_score),
                    "margin": int(my_score - opp_score),
                    "partner_id": int(partner) if partner else None,
                    "opponent_ids": [int(x) for x in opp_ids if x],
                    "expected_win_prob": float(expected_win),
                    "elo_delta_signed": signed_delta,
                    "abs_elo_delta": abs(delta_abs) if delta_abs is not None else None,
                    "opp_max_rating": float(opp_max) if opp_max is not None else None,
                    "lobby_avg_rating": float(lobby_avg_rating) if lobby_avg_rating is not None else None,
                    "rating_pre": float(pre) if pre is not None else None,
                    "rating_post": float(post) if post is not None else None,
                }
            )

    return pd.DataFrame.from_records(records)


def _avg(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return float(sum(nums) / len(nums))


def _max_rating(values: Iterable[float | None]) -> float | None:
    nums = [v for v in values if v is not None]
    if not nums:
        return None
    return float(max(nums))


def _safe_float(value) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except Exception:
        return None


def _safe_rating(value, fallback: float | None) -> float | None:
    v = _safe_float(value)
    if v is not None:
        return v
    if fallback is not None:
        return float(fallback)
    return None


def _expected_share(team_avg: float | None, opp_avg: float | None) -> float:
    try:
        if team_avg is None or opp_avg is None:
            return 0.5
        return 1.0 / (1.0 + 10 ** ((float(opp_avg) - float(team_avg)) / 400.0))
    except Exception:
        return 0.5


def _week_key(date_dt: pd.Timestamp) -> str:
    iso = date_dt.isocalendar()
    return f"{iso.year}-W{int(iso.week):02d}"


def _month_key(date_dt: pd.Timestamp) -> str:
    return date_dt.strftime("%Y-%m")


def _season_key(date_dt: pd.Timestamp) -> str:
    return date_dt.strftime("%Y")
