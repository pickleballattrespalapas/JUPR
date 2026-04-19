from __future__ import annotations

from collections import Counter, defaultdict
from types import SimpleNamespace
from typing import Any, Iterable

import pandas as pd

from jupr_app.data.load import load_data
from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club


KEY_FIELDS = ("player_id", "badge_id", "context_id")
DETAIL_FIELDS = [
    "id",
    "club_id",
    "player_id",
    "badge_id",
    "context_id",
    "context_type",
    "match_id",
    "value_json",
    "earned_at",
    "revoked_at",
    "awarded_by",
    "rule_version",
    "eval_run_id",
]


def build_badge_audit_report(
    supabase: Any,
    *,
    club_id: str,
    ctx: Any | None = None,
    league_id: str | None = None,
    player_id: int | None = None,
    badge_id: str | None = None,
    context_id: str | None = None,
    since: str | None = None,
    until: str | None = None,
    include_non_live: bool = False,
    include_revoked: bool = False,
    match_limit: int = 5000,
) -> dict[str, Any]:
    if ctx is None:
        ctx = _load_ctx(supabase, club_id, match_limit)

    scoped_ctx = _apply_match_filters(ctx, since=since, until=until)
    expected_rows = _build_expected_rows(
        club_id=club_id,
        ctx=scoped_ctx,
        league_id=league_id,
        player_id=player_id,
        badge_id=badge_id,
        context_id=context_id,
        include_non_live=include_non_live,
    )
    actual_rows = fetch_scoped_player_badges_rows(
        supabase,
        club_id=club_id,
        player_id=player_id,
        badge_id=badge_id,
        context_id=context_id,
    )

    expected_keys = {_row_key(row) for row in expected_rows}
    actual_active_rows = [row for row in actual_rows if row.get("revoked_at") is None]
    revoked_rows = [row for row in actual_rows if row.get("revoked_at") is not None]
    actual_active_keys = {_row_key(row) for row in actual_active_rows}
    revoked_keys = {_row_key(row) for row in revoked_rows}

    missing_keys = expected_keys - actual_active_keys
    stale_keys = actual_active_keys - expected_keys

    duplicate_groups = detect_duplicate_player_badges_rows(actual_rows)
    duplicate_rows = [row for group in duplicate_groups for row in group.get("rows", [])]

    missing_rows = [row for row in expected_rows if _row_key(row) in missing_keys]
    stale_rows = [row for row in actual_active_rows if _row_key(row) in stale_keys]

    detail_expected_rows = [project_detail_row(row) for row in expected_rows]
    detail_actual_active_rows = [project_detail_row(row) for row in actual_active_rows]
    detail_revoked_rows = [project_detail_row(row) for row in revoked_rows]

    per_badge = summarize_per_badge(
        expected_rows=expected_rows,
        actual_active_rows=actual_active_rows,
        missing_rows=missing_rows,
        stale_rows=stale_rows,
        duplicate_groups=duplicate_groups,
    )

    return {
        "scope": {
            "club_id": str(club_id),
            "league_id": league_id,
            "player_id": player_id,
            "badge_id": badge_id,
            "context_id": context_id,
            "since": since,
            "until": until,
            "include_non_live": bool(include_non_live),
            "include_revoked": bool(include_revoked),
        },
        "schema_degraded": bool(getattr(ctx, "schema_degraded", False)),
        "schema_degraded_reason": getattr(ctx, "schema_degraded_reason", None),
        "counts": {
            "expected_count": len(expected_keys),
            "actual_active_count": len(actual_active_keys),
            "missing_count": len(missing_keys),
            "stale_count": len(stale_keys),
            "duplicate_count": len(duplicate_rows),
            "duplicate_group_count": len(duplicate_groups),
            "revoked_count": len(revoked_rows),
            "revoked_key_count": len(revoked_keys),
        },
        "per_badge_summary": per_badge,
        "missing_rows": [project_detail_row(row) for row in missing_rows],
        "stale_rows": [project_detail_row(row) for row in stale_rows],
        "duplicate_rows": [project_detail_row(row) for row in duplicate_rows],
        "duplicate_groups": duplicate_groups,
        "revoked_rows": detail_revoked_rows if include_revoked else [],
        "actual_active_rows": detail_actual_active_rows,
        "expected_rows": detail_expected_rows,
        "expected_keys": sorted(expected_keys),
        "actual_active_keys": sorted(actual_active_keys),
        "revoked_keys": sorted(revoked_keys),
        "missing_keys": sorted(missing_keys),
        "stale_keys": sorted(stale_keys),
    }


def _load_ctx(supabase: Any, club_id: str, match_limit: int) -> Any:
    (
        df_players_all,
        df_players_active,
        df_leagues,
        df_matches,
        df_meta,
        df_badges,
        df_player_badges,
        name_to_id,
        id_to_name,
        schema_degraded,
        schema_degraded_reason,
    ) = load_data(supabase, club_id, match_limit=match_limit)

    return SimpleNamespace(
        supabase=supabase,
        club_id=club_id,
        df_players_all=df_players_all,
        df_players_active=df_players_active,
        df_leagues=df_leagues,
        df_meta=df_meta,
        df_badges=df_badges,
        df_matches=df_matches,
        df_player_badges=df_player_badges,
        name_to_id=name_to_id,
        id_to_name=id_to_name,
        public_mode=False,
        admin_logged_in=True,
        schema_degraded=schema_degraded,
        schema_degraded_reason=schema_degraded_reason,
    )


def _apply_match_filters(ctx: Any, *, since: str | None, until: str | None) -> Any:
    if not since and not until:
        return ctx
    df_matches = getattr(ctx, "df_matches", None)
    if df_matches is None or df_matches.empty:
        return ctx

    df = df_matches.copy()
    df["date_dt"] = pd.to_datetime(df.get("date"), utc=True, errors="coerce")
    if since:
        since_dt = pd.to_datetime(since, utc=True, errors="coerce")
        df = df[df["date_dt"] >= since_dt]
    if until:
        until_dt = pd.to_datetime(until, utc=True, errors="coerce")
        df = df[df["date_dt"] <= until_dt]

    attrs = dict(vars(ctx)) if hasattr(ctx, "__dict__") else {}
    attrs["df_matches"] = df.drop(columns=["date_dt"], errors="ignore")
    return SimpleNamespace(**attrs)


def _build_expected_rows(
    *,
    club_id: str,
    ctx: Any,
    league_id: str | None,
    player_id: int | None,
    badge_id: str | None,
    context_id: str | None,
    include_non_live: bool,
) -> list[dict[str, Any]]:
    candidates = compute_candidates_for_club(
        club_id=club_id,
        league_id=league_id,
        ctx=ctx,
        allow_non_live=include_non_live,
    )
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = {
            "player_id": int(candidate.player_id),
            "badge_id": str(candidate.badge_id),
            "context_id": str(candidate.context_id),
            "context_type": candidate.context_type,
            "match_id": candidate.match_id,
            "value_json": candidate.value_json,
            "earned_at": None,
            "revoked_at": None,
            "awarded_by": "engine_expected",
            "rule_version": None,
            "eval_run_id": None,
        }
        if player_id is not None and int(row["player_id"]) != int(player_id):
            continue
        if badge_id is not None and str(row["badge_id"]) != str(badge_id):
            continue
        if context_id is not None and str(row["context_id"]) != str(context_id):
            continue
        rows.append(row)

    deduped: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        deduped[_row_key(row)] = row
    return list(deduped.values())


def fetch_scoped_player_badges_rows(
    supabase: Any,
    *,
    club_id: str,
    player_id: int | None = None,
    badge_id: str | None = None,
    context_id: str | None = None,
) -> list[dict[str, Any]]:
    query = (
        supabase.table("player_badges")
        .select(
            "id,club_id,player_id,badge_id,context_id,context_type,match_id,value_json,"
            "earned_at,revoked_at,awarded_by,rule_version,eval_run_id"
        )
        .eq("club_id", str(club_id))
    )
    if player_id is not None:
        query = query.eq("player_id", int(player_id))
    if badge_id is not None:
        query = query.eq("badge_id", str(badge_id))
    if context_id is not None:
        query = query.eq("context_id", str(context_id))

    resp = query.execute()
    rows = []
    for row in resp.data or []:
        cleaned = dict(row)
        cleaned["player_id"] = int(cleaned.get("player_id"))
        cleaned["badge_id"] = str(cleaned.get("badge_id"))
        cleaned["context_id"] = str(cleaned.get("context_id"))
        rows.append(cleaned)
    return rows


def detect_duplicate_player_badges_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("club_id") or ""),
            int(row.get("player_id")),
            str(row.get("badge_id")),
            str(row.get("context_id")),
        )
        grouped[key].append(row)

    duplicates: list[dict[str, Any]] = []
    for key, key_rows in grouped.items():
        if len(key_rows) <= 1:
            continue
        duplicates.append(
            {
                "club_id": key[0],
                "player_id": key[1],
                "badge_id": key[2],
                "context_id": key[3],
                "row_count": len(key_rows),
                "rows": [project_detail_row(row) for row in key_rows],
            }
        )
    return duplicates


def summarize_per_badge(
    *,
    expected_rows: list[dict[str, Any]],
    actual_active_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    stale_rows: list[dict[str, Any]],
    duplicate_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    expected_counts = Counter(str(row.get("badge_id")) for row in expected_rows)
    active_counts = Counter(str(row.get("badge_id")) for row in actual_active_rows)
    missing_counts = Counter(str(row.get("badge_id")) for row in missing_rows)
    stale_counts = Counter(str(row.get("badge_id")) for row in stale_rows)
    duplicate_counts = Counter(
        str(group.get("badge_id")) for group in duplicate_groups for _ in range(int(group.get("row_count", 0)))
    )

    all_badges = sorted(
        set(expected_counts.keys())
        | set(active_counts.keys())
        | set(missing_counts.keys())
        | set(stale_counts.keys())
        | set(duplicate_counts.keys())
    )
    return [
        {
            "badge_id": badge,
            "expected_count": int(expected_counts.get(badge, 0)),
            "actual_active_count": int(active_counts.get(badge, 0)),
            "missing_count": int(missing_counts.get(badge, 0)),
            "stale_count": int(stale_counts.get(badge, 0)),
            "duplicate_count": int(duplicate_counts.get(badge, 0)),
        }
        for badge in all_badges
    ]


def project_detail_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in DETAIL_FIELDS}


def _row_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (int(row.get("player_id")), str(row.get("badge_id")), str(row.get("context_id")))
