from __future__ import annotations

"""This module is the single source of truth for badge match-facts.

Match facts are generated after applying match filters (club_id scoping,
exclude_popups, invalid/void/deleted flags, tournament exclusions, and
standard score normalization) to ensure badge evaluations stay consistent.
"""

from collections.abc import Iterable
from typing import Any

import pandas as pd

from jupr_app.domain.match_filters import (
    MatchFilterAudit,
    apply_match_filters,
    apply_match_filters_with_audit,
    normalize_player_id,
    normalize_score,
)


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
]


def _empty_facts() -> pd.DataFrame:
    return pd.DataFrame(columns=_FACT_COLUMNS)


def build_player_match_facts(
    ctx, df_matches_override: pd.DataFrame | None = None, club_id_override: str | None = None
) -> pd.DataFrame:
    """Backward-compatible canonical fact builder."""
    return build_canonical_player_match_facts(
        ctx,
        df_matches_override=df_matches_override,
        club_id_override=club_id_override,
    )


def build_canonical_player_match_facts(
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
    filtered = filter_matches_for_badge_facts(df_matches, filters)
    if filtered.empty:
        return _empty_facts()

    filtered = filtered.copy()
    if "id" not in filtered.columns:
        filtered["id"] = range(1, len(filtered) + 1)
    filtered["date_dt"] = pd.to_datetime(filtered.get("date", None), utc=True, errors="coerce", format="mixed")
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

        for pid, team, partner, opp_ids, my_score, opp_score, opp_avg, opp_max, expected_win in (
            (p1, 1, p2, team2, s1, s2, _avg([r3, r4] if p4 else [r3]), _max_rating([r3, r4]), expected_t1),
            (p2, 1, p1, team2, s1, s2, _avg([r3, r4] if p4 else [r3]), _max_rating([r3, r4]), expected_t1),
            (p3, 2, p4, team1, s2, s1, _avg([r1, r2] if p2 else [r1]), _max_rating([r1, r2]), 1.0 - expected_t1),
            (p4, 2, p3, team1, s2, s1, _avg([r1, r2] if p2 else [r1]), _max_rating([r1, r2]), 1.0 - expected_t1),
        ):
            if not pid:
                continue
            win = winner_team == team
            slot = {p1: "t1_p1", p2: "t1_p2", p3: "t2_p1", p4: "t2_p2"}.get(pid)
            before = _safe_float(getattr(row, f"{slot}_r", None))
            after = _safe_float(getattr(row, f"{slot}_r_end", None))
            signed_delta = _safe_float(getattr(row, f"elo_delta_t{team}", None))
            if before is not None and after is not None:
                signed_delta = after - before
            elif signed_delta is None and delta_abs is not None:
                signed_delta = abs(delta_abs) * (1 if win else -1 if winner_team else 0)
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
                    "abs_elo_delta": abs(signed_delta) if signed_delta is not None else None,
                    "opp_max_rating": float(opp_max) if opp_max is not None else None,
                    "lobby_avg_rating": float(lobby_avg_rating) if lobby_avg_rating is not None else None,
                }
            )

    return pd.DataFrame.from_records(records, columns=_FACT_COLUMNS)


def build_legacy_safe_player_match_facts(
    ctx, df_matches_override: pd.DataFrame | None = None, club_id_override: str | None = None
) -> pd.DataFrame:
    df_matches = df_matches_override
    if df_matches is None:
        df_matches = getattr(ctx, "df_matches", None)
    if df_matches is None or df_matches.empty:
        return _empty_facts_with_source()

    club_id = club_id_override if club_id_override is not None else getattr(ctx, "club_id", None)
    filters = {"club_id": club_id, "exclude_popups": True}
    filtered = filter_matches_for_badge_facts(df_matches, filters)
    if filtered.empty:
        return _empty_facts_with_source()

    filtered = filtered.copy()
    filtered["__match_id"] = filtered.apply(_resolve_match_id, axis=1)
    filtered["__date_dt"] = filtered.apply(_resolve_date, axis=1)
    filtered = filtered.dropna(subset=["__match_id", "__date_dt"]).sort_values(["__date_dt", "__match_id"], ascending=[True, True])

    records: list[dict[str, Any]] = []
    for row in filtered.to_dict(orient="records"):
        match_id = str(row.get("__match_id") or "").strip()
        date_dt = pd.to_datetime(row.get("__date_dt"), utc=True, errors="coerce")
        if not match_id or pd.isna(date_dt):
            continue

        s1 = _resolve_score(row, "score_t1")
        s2 = _resolve_score(row, "score_t2")
        if (s1 + s2) <= 0:
            continue

        p1 = _resolve_player(row, "t1_p1")
        p2 = _resolve_player(row, "t1_p2")
        p3 = _resolve_player(row, "t2_p1")
        p4 = _resolve_player(row, "t2_p2")
        if not p1 or not p3:
            continue

        team1 = [pid for pid in (p1, p2) if pid]
        team2 = [pid for pid in (p3, p4) if pid]
        if not team1 or not team2:
            continue

        league = _resolve_league(row)
        row_club_id = str(row.get("club_id") or club_id or "")
        winner_team = 1 if s1 > s2 else 2 if s2 > s1 else 0
        for pid, team, partner, opp_ids, my_score, opp_score in (
            (p1, 1, p2, team2, s1, s2),
            (p2, 1, p1, team2, s1, s2),
            (p3, 2, p4, team1, s2, s1),
            (p4, 2, p3, team1, s2, s1),
        ):
            if not pid:
                continue
            records.append(
                {
                    "club_id": row_club_id,
                    "player_id": int(pid),
                    "match_id": match_id,
                    "league": league,
                    "date_dt": date_dt,
                    "week_key": _week_key(date_dt),
                    "month_key": _month_key(date_dt),
                    "season_key": _season_key(date_dt),
                    "win": bool(winner_team == team),
                    "points_for": int(my_score),
                    "points_against": int(opp_score),
                    "margin": int(my_score - opp_score),
                    "partner_id": int(partner) if partner else None,
                    "opponent_ids": [int(x) for x in opp_ids if x],
                    "expected_win_prob": None,
                    "elo_delta_signed": None,
                    "abs_elo_delta": None,
                    "opp_max_rating": None,
                    "lobby_avg_rating": None,
                    "fact_source": "legacy",
                }
            )
    facts = pd.DataFrame.from_records(records, columns=[*_FACT_COLUMNS, "fact_source"])
    if facts.empty:
        return facts
    return facts.drop_duplicates(subset=["player_id", "match_id"], keep="first").reset_index(drop=True)


def build_hybrid_player_match_facts(
    ctx, df_matches_override: pd.DataFrame | None = None, club_id_override: str | None = None
) -> pd.DataFrame:
    canonical = build_canonical_player_match_facts(
        ctx,
        df_matches_override=df_matches_override,
        club_id_override=club_id_override,
    )
    legacy = build_legacy_safe_player_match_facts(
        ctx,
        df_matches_override=df_matches_override,
        club_id_override=club_id_override,
    )
    canonical = canonical.copy()
    canonical["fact_source"] = "canonical"
    if canonical.empty:
        return legacy
    if legacy.empty:
        return canonical

    canonical_keys = set(
        zip(
            canonical["player_id"].astype(int),
            canonical["match_id"].astype(str),
        )
    )
    missing_legacy = legacy[
        ~legacy.apply(lambda r: (int(r["player_id"]), str(r["match_id"])) in canonical_keys, axis=1)
    ].copy()
    combined = pd.concat([canonical, missing_legacy], ignore_index=True)
    return combined


def _empty_facts_with_source() -> pd.DataFrame:
    return pd.DataFrame(columns=[*_FACT_COLUMNS, "fact_source"])


def filter_matches_for_badge_facts(
    df_matches: pd.DataFrame,
    context_filters: dict | None,
    *,
    include_audit: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, MatchFilterAudit]:
    if "deleted_at" in df_matches.columns:
        df_matches = df_matches[df_matches["deleted_at"].isna()].copy()
    if include_audit:
        return apply_match_filters_with_audit(df_matches, context_filters)
    return apply_match_filters(df_matches, context_filters)


def _resolve_player(row: dict[str, Any], target: str) -> int | None:
    player_aliases, _ = _legacy_alias_maps()
    pid = normalize_player_id(row.get(target))
    if pid:
        return pid
    for alias in player_aliases.get(target, ()):
        pid = normalize_player_id(row.get(alias))
        if pid:
            return pid
    return None


def _resolve_score(row: dict[str, Any], target: str) -> int:
    _, score_aliases = _legacy_alias_maps()
    score = normalize_score(row.get(target))
    if score > 0:
        return score
    for alias in score_aliases.get(target, ()):
        score = normalize_score(row.get(alias))
        if score > 0:
            return score
    return normalize_score(row.get(target))


def _resolve_match_id(row: pd.Series) -> str | None:
    for key in ("id", "match_id", "matchId", "matchID"):
        raw = row.get(key)
        if raw is None:
            continue
        if isinstance(raw, float) and pd.isna(raw):
            continue
        txt = str(raw).strip()
        if txt:
            return txt
    return None


def _resolve_date(row: pd.Series) -> pd.Timestamp | None:
    for key in ("date", "played_on", "created_at"):
        if key not in row.index:
            continue
        dt = pd.to_datetime(row.get(key), utc=True, errors="coerce")
        if not pd.isna(dt):
            return dt
    return None


def _resolve_league(row: dict[str, Any]) -> str:
    league = str(row.get("league") or "").strip()
    if league:
        return league
    for key in ("league_name", "league_id"):
        val = str(row.get(key) or "").strip()
        if val:
            return val
    context_type = str(row.get("context_type") or "").strip().upper()
    context_id = str(row.get("context_id") or "").strip()
    if context_type == "LEAGUE" and context_id:
        return context_id.split(":")[0]
    return "OVERALL"


def _legacy_alias_maps() -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    try:
        from jupr_app.domain.match_canonical_migration import PLAYER_ALIASES as player_aliases
        from jupr_app.domain.match_canonical_migration import SCORE_ALIASES as score_aliases

        return player_aliases, score_aliases
    except Exception:
        return (
            {
                "t1_p1": ("team1_player1", "player1_id", "p1_id"),
                "t1_p2": ("team1_player2", "player2_id", "p2_id"),
                "t2_p1": ("team2_player1", "player3_id", "p3_id"),
                "t2_p2": ("team2_player2", "player4_id", "p4_id"),
            },
            {
                "score_t1": ("team1_score", "score1", "score_a"),
                "score_t2": ("team2_score", "score2", "score_b"),
            },
        )


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
