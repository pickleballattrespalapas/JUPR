from __future__ import annotations

from collections import Counter
from typing import Any

import pandas as pd

from jupr_app.domain.gamification.match_facts import build_player_match_facts
from jupr_app.domain.match_filters import normalize_player_id, normalize_score

PLAYER_SLOT_COLUMNS = ("t1_p1", "t1_p2", "t2_p1", "t2_p2")
SNAPSHOT_COLUMNS = (
    "t1_p1_r",
    "t1_p1_r_end",
    "t1_p2_r",
    "t1_p2_r_end",
    "t2_p1_r",
    "t2_p1_r_end",
    "t2_p2_r",
    "t2_p2_r_end",
)


def build_match_canonical_audit(
    ctx,
    *,
    club_id: str,
    player_id: int | None = None,
    league_id: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    scoped = _profile_visible_rows(
        getattr(ctx, "df_matches", pd.DataFrame()),
        club_id=str(club_id),
        player_id=player_id,
        league_id=league_id,
        limit=limit,
    )

    canonical_facts = build_player_match_facts(
        ctx,
        df_matches_override=scoped,
        club_id_override=str(club_id),
    )
    if player_id is not None and not canonical_facts.empty:
        canonical_facts = canonical_facts[
            pd.to_numeric(canonical_facts.get("player_id"), errors="coerce").fillna(-1).astype(int) == int(player_id)
        ].copy()

    profile_ids = _id_set(scoped)
    canonical_ids = set(canonical_facts.get("match_id", pd.Series(dtype=str)).dropna().astype(str).tolist())

    only_profile = sorted(profile_ids - canonical_ids, key=_sort_intish)
    only_canonical = sorted(canonical_ids - profile_ids, key=_sort_intish)
    shared = sorted(profile_ids & canonical_ids, key=_sort_intish)

    excluded_rows = _build_excluded_diagnostics(scoped, only_profile, player_id=player_id, league_id=league_id)
    reason_counts = Counter()
    for row in excluded_rows:
        for reason in row.get("exclusion_reasons", []):
            reason_counts[reason] += 1

    return {
        "scope": {
            "club_id": str(club_id),
            "player_id": int(player_id) if player_id is not None else None,
            "league_id": str(league_id).strip() if league_id else None,
            "limit": int(limit) if limit else None,
        },
        "profile_visible_match_ids": sorted(profile_ids, key=_sort_intish),
        "canonical_visible_match_ids": sorted(canonical_ids, key=_sort_intish),
        "only_in_profile": only_profile,
        "only_in_canonical": only_canonical,
        "shared_ids": shared,
        "excluded_only_in_profile": excluded_rows,
        "exclusion_reasons_summary": [
            {"reason": reason, "count": int(count)} for reason, count in sorted(reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        "counts": {
            "profile_visible": len(profile_ids),
            "canonical_visible": len(canonical_ids),
            "only_in_profile": len(only_profile),
            "only_in_canonical": len(only_canonical),
            "shared": len(shared),
        },
    }


def _profile_visible_rows(
    df_matches: pd.DataFrame | None,
    *,
    club_id: str,
    player_id: int | None,
    league_id: str | None,
    limit: int | None,
) -> pd.DataFrame:
    if df_matches is None or df_matches.empty:
        return pd.DataFrame()
    df = df_matches.copy()
    if "club_id" in df.columns:
        df = df[df["club_id"].astype(str) == str(club_id)].copy()

    if player_id is not None:
        pid = int(player_id)
        mask = pd.Series(False, index=df.index)
        for col in [c for c in PLAYER_SLOT_COLUMNS if c in df.columns]:
            mask = mask | df[col].map(lambda value: normalize_player_id(value) == pid)
        df = df[mask].copy()

    if league_id:
        league_key = str(league_id).strip()
        df["league"] = df.get("league", "").fillna("").astype(str).str.strip()
        df = df[df["league"] == league_key].copy()

    if "id" not in df.columns:
        df["id"] = range(1, len(df) + 1)

    df["date_dt"] = pd.to_datetime(df.get("date", None), utc=True, errors="coerce")
    df = df.sort_values(["date_dt", "id"], ascending=[False, False]).reset_index(drop=True)
    if limit:
        df = df.head(int(limit)).copy()
    df = df.dropna(subset=["date_dt"]).copy()

    return df


def _id_set(df: pd.DataFrame) -> set[str]:
    if df is None or df.empty:
        return set()
    return set(df.get("id", pd.Series(dtype=object)).dropna().astype(str).tolist())


def _sort_intish(value: str) -> tuple[int, str]:
    raw = str(value)
    try:
        return (0, f"{int(raw):020d}")
    except Exception:
        return (1, raw)


def _build_excluded_diagnostics(
    scoped_rows: pd.DataFrame,
    excluded_ids: list[str],
    *,
    player_id: int | None,
    league_id: str | None,
) -> list[dict[str, Any]]:
    if scoped_rows is None or scoped_rows.empty or not excluded_ids:
        return []

    id_set = set(str(mid) for mid in excluded_ids)
    subset = scoped_rows[scoped_rows["id"].astype(str).isin(id_set)].copy()
    duplicate_ids = _duplicate_match_ids(subset)

    rows: list[dict[str, Any]] = []
    for _, row in subset.iterrows():
        reasons = _classify_exclusion_reasons(row, player_id=player_id, league_id=league_id)
        if str(row.get("id")) in duplicate_ids:
            reasons.append("duplicate / superseded row")
        reasons = sorted(set(reasons)) or ["unknown/unclassified"]

        rows.append(
            {
                "match_id": _safe_int(row.get("id")),
                "date": _safe_date(row.get("date")),
                "league": _as_clean_str(row.get("league")),
                "match_type": _as_clean_str(row.get("match_type")),
                "context_type": _as_clean_str(row.get("context_type")),
                "score_t1": normalize_score(row.get("score_t1")),
                "score_t2": normalize_score(row.get("score_t2")),
                "t1_p1": normalize_player_id(row.get("t1_p1")),
                "t1_p2": normalize_player_id(row.get("t1_p2")),
                "t2_p1": normalize_player_id(row.get("t2_p1")),
                "t2_p2": normalize_player_id(row.get("t2_p2")),
                "snapshot": {col: row.get(col) for col in SNAPSHOT_COLUMNS if col in subset.columns},
                "exclusion_reasons": reasons,
                "suggested_normalization_action": _suggested_action(reasons),
            }
        )

    rows.sort(key=lambda r: (str(r.get("date") or ""), int(r.get("match_id") or 0)), reverse=True)
    return rows


def _classify_exclusion_reasons(row: pd.Series, *, player_id: int | None, league_id: str | None) -> list[str]:
    reasons: list[str] = []

    match_type = _as_clean_str(row.get("match_type"))
    context_type = _as_clean_str(row.get("context_type"))
    league = _as_clean_str(row.get("league"))

    if match_type.lower() == "popup":
        reasons.append("popup")

    if context_type.upper() == "TOURNAMENT" or pd.notna(row.get("tournament_id")):
        reasons.append("tournament context")

    if match_type.upper() == "TOURNAMENT":
        reasons.append("tournament match_type")

    s1 = normalize_score(row.get("score_t1"))
    s2 = normalize_score(row.get("score_t2"))
    if (s1 + s2) <= 0:
        reasons.append("invalid/zero score")

    p1 = normalize_player_id(row.get("t1_p1"))
    p3 = normalize_player_id(row.get("t2_p1"))
    if not p1 or not p3:
        reasons.append("missing required player slots")

    if player_id is not None:
        pid = int(player_id)
        participants = {
            normalize_player_id(row.get("t1_p1")),
            normalize_player_id(row.get("t1_p2")),
            normalize_player_id(row.get("t2_p1")),
            normalize_player_id(row.get("t2_p2")),
        }
        if pid not in participants:
            reasons.append("missing/legacy fields needed by canonical facts")

    if league_id:
        if not league or league != str(league_id).strip():
            reasons.append("malformed league/context values")
    elif not league:
        reasons.append("missing/legacy fields needed by canonical facts")

    return reasons


def _duplicate_match_ids(df: pd.DataFrame) -> set[str]:
    if df is None or df.empty:
        return set()

    keys: list[str] = []
    for _, row in df.iterrows():
        day = _safe_date(row.get("date"))
        players = sorted([normalize_player_id(row.get(col)) or -1 for col in PLAYER_SLOT_COLUMNS])
        score = (normalize_score(row.get("score_t1")), normalize_score(row.get("score_t2")))
        keys.append(f"{day}|{players}|{score}")

    key_series = pd.Series(keys)
    dup_mask = key_series.duplicated(keep=False)
    if not dup_mask.any():
        return set()
    dup_ids = df.loc[dup_mask, "id"].astype(str).tolist()
    return set(dup_ids)


def _suggested_action(reasons: list[str]) -> str:
    reason_set = set(reasons)
    if "invalid/zero score" in reason_set or "missing required player slots" in reason_set:
        return "manual-review-needed"
    if "tournament context" in reason_set or "tournament match_type" in reason_set:
        return "review_tournament_markers_then_normalize_context_if_legacy"
    if "popup" in reason_set:
        return "confirm_popup_semantics_no_change_if_intentional"
    if "malformed league/context values" in reason_set or "missing/legacy fields needed by canonical facts" in reason_set:
        return "normalize_legacy_league_match_type_or_context_fields"
    if "duplicate / superseded row" in reason_set:
        return "mark_superseded_in_source_of_truth_or_keep_latest_only"
    return "manual-review-needed"


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _safe_date(value: Any) -> str:
    date_val = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(date_val):
        return ""
    return date_val.strftime("%Y-%m-%d")


def _as_clean_str(value: Any) -> str:
    return str(value or "").strip()
