from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import date, datetime
from typing import Any

from jupr_app.domain.bulk_match_editor import compute_recompute_scope
from jupr_app.domain.dupes import canonical_dup_key

MATCH_LOG_SELECT = (
    "id,date,league,week_tag,match_type,t1_p1,t1_p2,t2_p1,t2_p2,"
    "score_t1,score_t2,is_active,context_type,context_id,created_at,updated_at"
)
MATCH_LOG_MINIMAL_SELECT = "id,date,league,week_tag,match_type,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2"
MAX_FETCH_ROWS = 5000
MAX_RETURN_ROWS = 1000


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def is_admin_match_log_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG")


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _clean_text(value: Any, *, limit: int = 200) -> str:
    text = str(value or "").replace("<", "").replace(">", "").strip()
    return text[:limit]


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _date_sort_key(row: dict[str, Any]) -> tuple[str, int]:
    return (str(row.get("date") or row.get("created_at") or ""), int(_safe_int(row.get("id")) or 0))


def _fetch_match_rows(supabase: Any, *, club_id: str, fetch_limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select(MATCH_LOG_SELECT)
            .eq("club_id", str(club_id))
            .order("date", desc=True)
            .limit(int(fetch_limit))
            .execute()
        )
        return rows, warnings
    except Exception as exc:
        warnings.append(f"Fell back to minimal match columns: {exc.__class__.__name__}")

    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select(MATCH_LOG_MINIMAL_SELECT)
            .eq("club_id", str(club_id))
            .order("date", desc=True)
            .limit(int(fetch_limit))
            .execute()
        )
        return rows, warnings
    except Exception as exc:
        warnings.append(f"Could not load matches: {exc.__class__.__name__}")
        return [], warnings


def _player_names(supabase: Any, *, club_id: str, player_ids: set[int]) -> dict[int, str]:
    if not player_ids:
        return {}
    try:
        rows = _safe_rows(supabase.table("players").select("id,name").eq("club_id", str(club_id)).execute())
    except Exception:
        return {}
    wanted = {int(pid) for pid in player_ids if pid is not None}
    names: dict[int, str] = {}
    for row in rows:
        pid = _safe_int(row.get("id"))
        if pid is not None and int(pid) in wanted:
            names[int(pid)] = _clean_text(row.get("name") or f"Player {pid}", limit=120)
    return names


def _matches_filter(
    rows: list[dict[str, Any]],
    *,
    filter_type: str,
    match_id: int | None,
    league: str | None,
    week_tag: str | None,
    start_date: str | None,
    end_date: str | None,
) -> list[dict[str, Any]]:
    result = list(rows)
    normalized_filter = str(filter_type or "All").strip().lower()
    if normalized_filter in {"league", "leagues"}:
        result = [row for row in result if str(row.get("match_type") or "") != "PopUp"]
    elif normalized_filter in {"pop-up", "popup", "pop up"}:
        result = [row for row in result if str(row.get("match_type") or "") == "PopUp"]

    if match_id is not None:
        result = [row for row in result if _safe_int(row.get("id")) == int(match_id)]
    if league:
        result = [row for row in result if str(row.get("league") or "").strip() == str(league).strip()]
    if week_tag:
        result = [row for row in result if str(row.get("week_tag") or "").strip() == str(week_tag).strip()]
    if start_date:
        result = [row for row in result if str(row.get("date") or "")[:10] >= str(start_date)[:10]]
    if end_date:
        result = [row for row in result if str(row.get("date") or "")[:10] <= str(end_date)[:10]]
    return sorted(result, key=_date_sort_key, reverse=True)


def _format_player(pid: int | None, names: dict[int, str]) -> dict[str, Any]:
    if pid is None:
        return {"id": None, "name": "—"}
    return {"id": int(pid), "name": names.get(int(pid), f"Player {int(pid)}")}


def _match_payload(row: dict[str, Any], *, club_id: str, names: dict[int, str]) -> dict[str, Any]:
    p1 = _safe_int(row.get("t1_p1"))
    p2 = _safe_int(row.get("t1_p2"))
    p3 = _safe_int(row.get("t2_p1"))
    p4 = _safe_int(row.get("t2_p2"))
    s1 = _safe_int(row.get("score_t1")) or 0
    s2 = _safe_int(row.get("score_t2")) or 0
    dup_key = canonical_dup_key(row, str(club_id))
    return {
        "id": _safe_int(row.get("id")),
        "date": _json_safe(row.get("date")),
        "league": _clean_text(row.get("league"), limit=120),
        "week_tag": _clean_text(row.get("week_tag"), limit=80),
        "match_type": _clean_text(row.get("match_type"), limit=80),
        "score": {"team1": s1, "team2": s2, "display": f"{s1}-{s2}"},
        "team1": [_format_player(p1, names), _format_player(p2, names)],
        "team2": [_format_player(p3, names), _format_player(p4, names)],
        "is_active": bool(row.get("is_active", True)),
        "context_type": _clean_text(row.get("context_type"), limit=80),
        "context_id": row.get("context_id"),
        "created_at": _json_safe(row.get("created_at")),
        "updated_at": _json_safe(row.get("updated_at")),
        "dup_key": dup_key,
    }


def _duplicate_scan(rows: list[dict[str, Any]], *, club_id: str, names: dict[int, str]) -> dict[str, Any]:
    if not rows:
        return {"duplicate_groups": [], "duplicate_rows": [], "delete_preview": None}
    keyed = []
    for row in rows:
        keyed.append({"key": canonical_dup_key(row, str(club_id)), "row": row})
    counts = Counter(item["key"] for item in keyed)
    duplicate_keys = {key for key, count in counts.items() if count > 1}
    groups = []
    duplicate_rows = []
    delete_ids: list[int] = []
    keep_ids: list[int] = []
    affected_leagues: set[str] = set()
    affected_players: set[int] = set()

    for key in sorted(duplicate_keys):
        group_rows = [item["row"] for item in keyed if item["key"] == key]
        group_rows = sorted(group_rows, key=lambda row: int(_safe_int(row.get("id")) or 0))
        keep_id = int(_safe_int(group_rows[0].get("id")) or 0)
        group_delete_ids = [int(_safe_int(row.get("id")) or 0) for row in group_rows[1:] if _safe_int(row.get("id")) is not None]
        keep_ids.append(keep_id)
        delete_ids.extend(group_delete_ids)
        sample = _match_payload(group_rows[0], club_id=str(club_id), names=names)
        if sample.get("league"):
            affected_leagues.add(str(sample.get("league")))
        for row in group_rows:
            for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
                pid = _safe_int(row.get(col))
                if pid is not None:
                    affected_players.add(int(pid))
        groups.append(
            {
                "dup_key": key,
                "dup_count": len(group_rows),
                "keep_id": keep_id,
                "delete_ids": group_delete_ids,
                "ids": [int(_safe_int(row.get("id")) or 0) for row in group_rows],
                "league": sample.get("league"),
                "week_tag": sample.get("week_tag"),
                "match_type": sample.get("match_type"),
                "score": sample.get("score"),
                "team1": sample.get("team1"),
                "team2": sample.get("team2"),
            }
        )
        for idx, row in enumerate(group_rows, start=1):
            payload = _match_payload(row, club_id=str(club_id), names=names)
            payload["dup_rank"] = idx
            payload["dup_count"] = len(group_rows)
            payload["would_keep"] = idx == 1
            duplicate_rows.append(payload)

    delete_preview = None
    if delete_ids:
        delete_preview = {
            "mode": "planning_only",
            "keep_ids": keep_ids,
            "delete_ids": sorted(delete_ids),
            "delete_count": len(delete_ids),
            "affected_leagues": sorted(affected_leagues),
            "affected_player_ids": sorted(affected_players),
            "recompute_scope": {"standings": True, "ratings": True},
            "recommended_replay_scope": "ALL",
            "confirmation_text": "DELETE",
        }
    return {"duplicate_groups": groups, "duplicate_rows": duplicate_rows, "delete_preview": delete_preview}


def _correction_plan() -> dict[str, Any]:
    example_patch_scope = compute_recompute_scope(
        [
            {"id": 0, "league": "Example"},
            {"id": 0, "week_tag": "Week 1"},
            {"id": 0, "score_t1": 11, "score_t2": 8},
        ]
    )
    return {
        "mode": "planning_only",
        "apply_endpoint": None,
        "future_apply_endpoint": "/admin/clubs/{club_id}/match-log/edits",
        "editable_fields_planned": [
            "league",
            "date",
            "week_tag",
            "match_type",
            "is_active",
            "t1_p1",
            "t1_p2",
            "t2_p1",
            "t2_p2",
            "score_t1",
            "score_t2",
            "notes",
        ],
        "required_confirmation_text": "APPLY",
        "recompute_scope_for_sample_edit": example_patch_scope,
        "safety_rules": [
            "This slice does not apply edits or deletes.",
            "League/date changes should auto-clear week_tag unless explicitly set.",
            "Player and score changes require rating replay review before broad use.",
            "Duplicate deletion should keep the oldest row and replay history after deletion.",
            "Future writes must use FastAPI audit attribution and the Python bulk_match_editor domain service.",
        ],
    }


def build_admin_match_log(
    supabase: Any,
    *,
    club_id: str,
    filter_type: str = "All",
    match_id: int | None = None,
    league: str | None = None,
    week_tag: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    safe_limit = max(1, min(int(limit or 500), MAX_RETURN_ROWS))
    filters = {
        "filter": filter_type or "All",
        "match_id": match_id,
        "league": league or None,
        "week_tag": week_tag or None,
        "start_date": start_date or None,
        "end_date": end_date or None,
        "limit": safe_limit,
    }
    if not is_admin_match_log_enabled():
        return {
            "enabled": False,
            "status": "streamlit_fallback",
            "filters": filters,
            "summary": {"scanned_matches": 0, "returned_matches": 0, "duplicate_groups": 0, "duplicate_delete_count": 0},
            "matches": [],
            "duplicate_groups": [],
            "duplicate_rows": [],
            "duplicate_delete_preview": None,
            "correction_plan": _correction_plan(),
            "warnings": ["Next Match Log is disabled. Use Streamlit Match Log until JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG is enabled for the pilot."],
        }

    fetch_limit = min(MAX_FETCH_ROWS, max(safe_limit * 5, safe_limit))
    raw_rows, warnings = _fetch_match_rows(supabase, club_id=str(club_id), fetch_limit=fetch_limit)
    filtered_rows = _matches_filter(
        raw_rows,
        filter_type=filter_type,
        match_id=match_id,
        league=league,
        week_tag=week_tag,
        start_date=start_date,
        end_date=end_date,
    )
    visible_rows = filtered_rows[:safe_limit]
    player_ids: set[int] = set()
    for row in visible_rows:
        for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            pid = _safe_int(row.get(col))
            if pid is not None:
                player_ids.add(int(pid))
    names = _player_names(supabase, club_id=str(club_id), player_ids=player_ids)
    duplicate_payload = _duplicate_scan(visible_rows, club_id=str(club_id), names=names)
    matches = [_match_payload(row, club_id=str(club_id), names=names) for row in visible_rows]
    duplicate_delete_count = len((duplicate_payload.get("delete_preview") or {}).get("delete_ids") or [])
    return {
        "enabled": True,
        "status": "planning_only",
        "filters": filters,
        "summary": {
            "scanned_matches": len(raw_rows),
            "filtered_matches": len(filtered_rows),
            "returned_matches": len(matches),
            "duplicate_groups": len(duplicate_payload["duplicate_groups"]),
            "duplicate_delete_count": duplicate_delete_count,
        },
        "matches": matches,
        "duplicate_groups": duplicate_payload["duplicate_groups"],
        "duplicate_rows": duplicate_payload["duplicate_rows"],
        "duplicate_delete_preview": duplicate_payload["delete_preview"],
        "correction_plan": _correction_plan(),
        "warnings": warnings,
    }
