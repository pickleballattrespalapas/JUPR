from __future__ import annotations

import os
from collections import Counter
from datetime import date, datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.bulk_match_editor import compute_recompute_scope
from jupr_app.domain.dupes import canonical_dup_key
from jupr_app.services.match_edit_durability_service import apply_atomic_match_edits

MATCH_LOG_SELECT = (
    "id,date,league,week_tag,match_type,t1_p1,t1_p2,t2_p1,t2_p2,"
    "score_t1,score_t2,notes,deleted_at,context_type,context_id,updated_at"
)
MATCH_LOG_MINIMAL_SELECT = "id,date,league,week_tag,match_type,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2,deleted_at"
DUPLICATE_RESOLUTIONS_TABLE = "admin_match_log_duplicate_resolutions"
MAX_FETCH_ROWS = 5000
MAX_RETURN_ROWS = 1000
MAX_PATCHES = 100
MAX_CLEANUP_IDS = 500
MAX_RESOLUTION_IDS = 20


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def is_admin_match_log_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG")


def is_admin_match_log_apply_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY")


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


def _match_id_key(match_ids: list[int]) -> str:
    return ",".join(str(int(match_id)) for match_id in sorted({int(match_id) for match_id in match_ids}))


def _resolution_lookup_key(*, dup_key: str, match_ids: list[int]) -> tuple[str, str]:
    return (str(dup_key or "").strip(), _match_id_key(match_ids))


def _fetch_match_rows(supabase: Any, *, club_id: str, fetch_limit: int) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select(MATCH_LOG_SELECT)
            .eq("club_id", str(club_id))
            .is_("deleted_at", None)
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
            .is_("deleted_at", None)
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
        "notes": _clean_text(row.get("notes"), limit=2000),
        "score": {"team1": s1, "team2": s2, "display": f"{s1}-{s2}"},
        "team1": [_format_player(p1, names), _format_player(p2, names)],
        "team2": [_format_player(p3, names), _format_player(p4, names)],
        "is_active": row.get("deleted_at") in (None, ""),
        "context_type": _clean_text(row.get("context_type"), limit=80),
        "context_id": row.get("context_id"),
        "created_at": _json_safe(row.get("created_at")),
        "updated_at": _json_safe(row.get("updated_at")),
        "dup_key": dup_key,
    }


def _resolution_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "resolution": _clean_text(row.get("resolution") or "no_issue", limit=80),
        "reason": _clean_text(row.get("reason"), limit=500),
        "actor_email": _clean_text(row.get("actor_email"), limit=200),
        "actor_role": _clean_text(row.get("actor_role"), limit=80),
        "source_page": _clean_text(row.get("source_page"), limit=120),
        "resolved_at": _json_safe(row.get("resolved_at") or row.get("created_at")),
    }


def _fetch_duplicate_resolutions(supabase: Any, *, club_id: str) -> tuple[dict[tuple[str, str], dict[str, Any]], str | None]:
    try:
        rows = _safe_rows(
            supabase.table(DUPLICATE_RESOLUTIONS_TABLE)
            .select("dup_key,match_id_key,match_ids,resolution,reason,actor_email,actor_role,source_page,resolved_at,created_at")
            .eq("club_id", str(club_id))
            .eq("is_active", True)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001 - keep the scanner usable while migrations roll out
        return {}, f"Duplicate no-issue resolutions are unavailable: {exc.__class__.__name__}"

    resolutions: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        dup_key = str(row.get("dup_key") or "").strip()
        match_id_key = str(row.get("match_id_key") or "").strip()
        if dup_key and match_id_key:
            resolutions[(dup_key, match_id_key)] = row
    return resolutions, None


def _recent_match_edit_operations(supabase: Any, *, club_id: str) -> tuple[list[dict[str, Any]], str | None]:
    try:
        rows = _safe_rows(
            supabase.table("match_edit_operations")
            .select("id,status,recompute_scope,replay_target,replay_job_id,error_text,actor_email,source,created_at,finished_at")
            .eq("club_id", str(club_id))
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
    except Exception as exc:
        return [], f"Durable match edit operation history is unavailable: {exc.__class__.__name__}"
    return [
        {
            "id": str(row.get("id") or ""),
            "status": _clean_text(row.get("status") or "unknown", limit=40),
            "recompute_scope": dict(row.get("recompute_scope") or {}),
            "replay_target": _clean_text(row.get("replay_target"), limit=160),
            "replay_job_id": str(row.get("replay_job_id") or "") or None,
            "error_text": _clean_text(row.get("error_text"), limit=500),
            "actor_email": _clean_text(row.get("actor_email"), limit=240),
            "source": _clean_text(row.get("source"), limit=120),
            "created_at": row.get("created_at"),
            "finished_at": row.get("finished_at"),
        }
        for row in rows
    ], None


def _duplicate_scan(
    rows: list[dict[str, Any]],
    *,
    club_id: str,
    names: dict[int, str],
    resolved_lookup: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not rows:
        return {"duplicate_groups": [], "duplicate_rows": [], "delete_preview": None, "resolved_duplicate_groups": []}
    resolved_lookup = resolved_lookup or {}
    keyed = []
    for row in rows:
        keyed.append({"key": canonical_dup_key(row, str(club_id)), "row": row})
    counts = Counter(item["key"] for item in keyed)
    duplicate_keys = {key for key, count in counts.items() if count > 1}
    groups = []
    resolved_groups = []
    duplicate_rows = []
    delete_ids: list[int] = []
    keep_ids: list[int] = []
    affected_leagues: set[str] = set()
    affected_players: set[int] = set()

    for key in sorted(duplicate_keys):
        group_rows = [item["row"] for item in keyed if item["key"] == key]
        group_rows = sorted(group_rows, key=lambda row: int(_safe_int(row.get("id")) or 0))
        group_ids = [int(_safe_int(row.get("id")) or 0) for row in group_rows if _safe_int(row.get("id")) is not None]
        resolution = resolved_lookup.get(_resolution_lookup_key(dup_key=key, match_ids=group_ids))
        keep_id = int(_safe_int(group_rows[0].get("id")) or 0)
        group_delete_ids = [int(_safe_int(row.get("id")) or 0) for row in group_rows[1:] if _safe_int(row.get("id")) is not None]
        sample = _match_payload(group_rows[0], club_id=str(club_id), names=names)
        group_payload = {
            "dup_key": key,
            "dup_count": len(group_rows),
            "keep_id": keep_id,
            "delete_ids": group_delete_ids,
            "ids": group_ids,
            "league": sample.get("league"),
            "week_tag": sample.get("week_tag"),
            "match_type": sample.get("match_type"),
            "score": sample.get("score"),
            "team1": sample.get("team1"),
            "team2": sample.get("team2"),
        }

        if resolution:
            group_payload["resolution"] = _resolution_metadata(resolution)
            resolved_groups.append(group_payload)
            continue

        keep_ids.append(keep_id)
        delete_ids.extend(group_delete_ids)
        if sample.get("league"):
            affected_leagues.add(str(sample.get("league")))
        for row in group_rows:
            for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
                pid = _safe_int(row.get(col))
                if pid is not None:
                    affected_players.add(int(pid))
        groups.append(group_payload)
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
    return {
        "duplicate_groups": groups,
        "duplicate_rows": duplicate_rows,
        "delete_preview": delete_preview,
        "resolved_duplicate_groups": resolved_groups,
    }


def _correction_plan() -> dict[str, Any]:
    example_patch_scope = compute_recompute_scope(
        [
            {"id": 0, "league": "Example"},
            {"id": 0, "week_tag": "Week 1"},
            {"id": 0, "score_t1": 11, "score_t2": 8},
        ]
    )
    apply_enabled = is_admin_match_log_apply_enabled()
    return {
        "mode": "apply_enabled" if apply_enabled else "planning_only",
        "apply_endpoint": "/admin/clubs/{club_id}/match-log/edits" if apply_enabled else None,
        "duplicate_cleanup_endpoint": "/admin/clubs/{club_id}/match-log/duplicates/cleanup" if apply_enabled else None,
        "duplicate_no_issue_endpoint": "/admin/clubs/{club_id}/match-log/duplicates/resolve" if apply_enabled else None,
        "future_apply_endpoint": "/admin/clubs/{club_id}/match-log/edits",
        "editable_fields_planned": [
            "league",
            "date",
            "week_tag",
            "match_type",
            "t1_p1",
            "t1_p2",
            "t2_p1",
            "t2_p2",
            "score_t1",
            "score_t2",
            "notes",
        ],
        "required_confirmation_text": "APPLY",
        "duplicate_cleanup_confirmation_text": "DELETE",
        "duplicate_no_issue_confirmation_text": "NO ISSUE",
        "recompute_scope_for_sample_edit": example_patch_scope,
        "safety_rules": [
            "Writes require Supabase JWT auth plus manage/delete match permissions.",
            "Rated-match removal uses the guarded soft-exclude workflow, not guided field edits.",
            "League/date changes auto-clear week_tag unless explicitly set.",
            "Player and score changes require rating replay review before broad use.",
            "Duplicate cleanup keeps the oldest row and recommends replay history afterward.",
            "False-positive duplicate groups can be resolved as no issue without deleting rows.",
            "Writes use FastAPI audit attribution and Python domain services.",
            "Match edits are committed through one service-role-only transaction; rating-affecting edits create and complete a mandatory replay job before success is reported.",
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
            "apply_enabled": is_admin_match_log_apply_enabled(),
            "status": "streamlit_fallback",
            "filters": filters,
            "summary": {
                "scanned_matches": 0,
                "returned_matches": 0,
                "duplicate_groups": 0,
                "duplicate_delete_count": 0,
                "resolved_duplicate_groups": 0,
            },
            "matches": [],
            "duplicate_groups": [],
            "duplicate_rows": [],
            "duplicate_delete_preview": None,
            "resolved_duplicate_groups": [],
            "recent_edit_operations": [],
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
    resolved_lookup, resolution_warning = _fetch_duplicate_resolutions(supabase, club_id=str(club_id))
    if resolution_warning:
        warnings.append(resolution_warning)
    recent_operations, operations_warning = _recent_match_edit_operations(supabase, club_id=str(club_id))
    if operations_warning:
        warnings.append(operations_warning)
    duplicate_payload = _duplicate_scan(visible_rows, club_id=str(club_id), names=names, resolved_lookup=resolved_lookup)
    matches = [_match_payload(row, club_id=str(club_id), names=names) for row in visible_rows]
    duplicate_delete_count = len((duplicate_payload.get("delete_preview") or {}).get("delete_ids") or [])
    return {
        "enabled": True,
        "apply_enabled": is_admin_match_log_apply_enabled(),
        "status": "apply_enabled" if is_admin_match_log_apply_enabled() else "planning_only",
        "filters": filters,
        "summary": {
            "scanned_matches": len(raw_rows),
            "filtered_matches": len(filtered_rows),
            "returned_matches": len(matches),
            "duplicate_groups": len(duplicate_payload["duplicate_groups"]),
            "duplicate_delete_count": duplicate_delete_count,
            "resolved_duplicate_groups": len(duplicate_payload["resolved_duplicate_groups"]),
        },
        "matches": matches,
        "duplicate_groups": duplicate_payload["duplicate_groups"],
        "duplicate_rows": duplicate_payload["duplicate_rows"],
        "duplicate_delete_preview": duplicate_payload["delete_preview"],
        "resolved_duplicate_groups": duplicate_payload["resolved_duplicate_groups"],
        "recent_edit_operations": recent_operations,
        "correction_plan": _correction_plan(),
        "warnings": warnings,
    }


def apply_admin_match_log_edits(
    supabase: Any,
    *,
    club_id: str,
    patches: list[dict[str, Any]],
    actor_email: str,
    actor_role: str,
    correction_note: str | None = None,
    source: str = "next_match_log",
    confirmation_text: str = "",
    idempotency_key: str | None = None,
    replay_target: str = "ALL (Full System Reset)",
) -> dict[str, Any]:
    if not is_admin_match_log_apply_enabled():
        raise PermissionError("Next Match Log apply is disabled.")
    if str(confirmation_text or "").strip().upper() != "APPLY":
        raise ValueError("Type APPLY to confirm match edits.")
    clean_patches = [dict(patch) for patch in (patches or []) if isinstance(patch, dict)]
    if not clean_patches:
        raise ValueError("No patches provided.")
    if len(clean_patches) > MAX_PATCHES:
        raise ValueError(f"No more than {MAX_PATCHES} patches can be applied at once.")
    if any("is_active" in patch for patch in clean_patches):
        raise ValueError("Use the guarded rated-match exclude workflow to change match activity.")

    return apply_atomic_match_edits(
        supabase,
        club_id=str(club_id),
        patches=clean_patches,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        correction_note=correction_note,
        source=source,
        idempotency_key=str(idempotency_key or ""),
        replay_target=replay_target,
    )


def _cleanup_candidate_payload(
    supabase: Any,
    *,
    club_id: str,
    suppress_resolved: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    rows, warnings = _fetch_match_rows(supabase, club_id=str(club_id), fetch_limit=MAX_FETCH_ROWS)
    player_ids: set[int] = set()
    for row in rows:
        for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            pid = _safe_int(row.get(col))
            if pid is not None:
                player_ids.add(int(pid))
    names = _player_names(supabase, club_id=str(club_id), player_ids=player_ids)
    resolved_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    if suppress_resolved:
        resolution_lookup, resolution_warning = _fetch_duplicate_resolutions(supabase, club_id=str(club_id))
        resolved_lookup = resolution_lookup
        if resolution_warning:
            warnings.append(resolution_warning)
    return _duplicate_scan(rows, club_id=str(club_id), names=names, resolved_lookup=resolved_lookup), rows, warnings


def apply_admin_match_log_duplicate_cleanup(
    supabase: Any,
    *,
    club_id: str,
    delete_ids: list[int],
    actor_email: str,
    actor_role: str,
    source: str = "next_match_log_duplicate_cleanup",
    confirmation_text: str = "",
) -> dict[str, Any]:
    if not is_admin_match_log_apply_enabled():
        raise PermissionError("Next Match Log apply is disabled.")
    if str(confirmation_text or "").strip().upper() != "DELETE":
        raise ValueError("Type DELETE to confirm duplicate cleanup.")
    requested_ids = sorted({int(match_id) for match_id in (delete_ids or []) if _safe_int(match_id) is not None})
    if not requested_ids:
        raise ValueError("No duplicate IDs were provided.")
    if len(requested_ids) > MAX_CLEANUP_IDS:
        raise ValueError(f"No more than {MAX_CLEANUP_IDS} duplicate IDs can be cleaned up at once.")

    duplicate_payload, all_rows, scan_warnings = _cleanup_candidate_payload(supabase, club_id=str(club_id), suppress_resolved=True)
    preview = duplicate_payload.get("delete_preview") or {}
    allowed_ids = {int(match_id) for match_id in (preview.get("delete_ids") or [])}
    invalid_ids = [match_id for match_id in requested_ids if match_id not in allowed_ids]
    if invalid_ids:
        raise ValueError(f"Some requested IDs are not currently active duplicate cleanup candidates: {invalid_ids[:10]}")

    rows_by_id = {int(_safe_int(row.get("id")) or 0): dict(row) for row in all_rows if _safe_int(row.get("id")) is not None}
    rows_to_remove = [rows_by_id[match_id] for match_id in requested_ids if match_id in rows_by_id]
    if len(rows_to_remove) != len(requested_ids):
        missing = [match_id for match_id in requested_ids if match_id not in rows_by_id]
        raise ValueError(f"Some requested IDs could not be loaded for this club: {missing[:10]}")

    affected_players: set[int] = set()
    affected_leagues: set[str] = set()
    for row in rows_to_remove:
        if row.get("league"):
            affected_leagues.add(str(row.get("league")))
        for col in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            pid = _safe_int(row.get(col))
            if pid is not None:
                affected_players.add(int(pid))

    supabase.table("matches").delete().eq("club_id", str(club_id)).in_("id", requested_ids).execute()
    warnings: list[str] = list(scan_warnings)
    try:
        from jupr_app.domain.player_activity import recompute_last_game_at_for_players

        if affected_players:
            recompute_last_game_at_for_players(
                supabase=supabase,
                club_id=str(club_id),
                player_ids=affected_players,
            )
    except Exception:
        warnings.append("Unable to recompute last_game_at for affected players automatically. Run replay/history maintenance if needed.")

    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="match_duplicate_cleanup",
        entity_type="match",
        entity_id="bulk",
        before_json=rows_to_remove,
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "deleted_ids": requested_ids,
            "affected_leagues": sorted(affected_leagues),
            "affected_player_ids": sorted(affected_players),
            "recommended_replay_scope": "ALL",
        },
        note="Duplicate cleanup from Next Match Log",
        source_page=source,
        flagged_for_review=True,
    )
    audit_result = write_admin_activity_log(supabase, audit_payload)
    if audit_result.warning:
        warnings.append(audit_result.warning)
    if not audit_result.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")

    return {
        "ok": True,
        "mode": "duplicates_cleaned",
        "deleted_count": len(requested_ids),
        "deleted_ids": requested_ids,
        "affected_leagues": sorted(affected_leagues),
        "affected_player_ids": sorted(affected_players),
        "recompute_scope": {"standings": True, "ratings": True},
        "recommended_replay_scope": "ALL",
        "warnings": warnings,
    }


def resolve_admin_match_log_duplicate_false_positive(
    supabase: Any,
    *,
    club_id: str,
    match_ids: list[int],
    actor_email: str,
    actor_role: str,
    reason: str,
    dup_key: str | None = None,
    source: str = "next_match_log_duplicate_no_issue",
    confirmation_text: str = "",
) -> dict[str, Any]:
    if not is_admin_match_log_apply_enabled():
        raise PermissionError("Next Match Log apply is disabled.")
    normalized_confirmation = str(confirmation_text or "").strip().upper().replace("_", " ")
    if normalized_confirmation != "NO ISSUE":
        raise ValueError("Type NO ISSUE to confirm this duplicate group is a false positive.")

    requested_ids = sorted({int(match_id) for match_id in (match_ids or []) if _safe_int(match_id) is not None})
    if len(requested_ids) < 2:
        raise ValueError("At least two match IDs are required to resolve a duplicate false positive.")
    if len(requested_ids) > MAX_RESOLUTION_IDS:
        raise ValueError(f"No more than {MAX_RESOLUTION_IDS} match IDs can be resolved at once.")

    clean_reason = _clean_text(reason, limit=500)
    if not clean_reason:
        raise ValueError("Add a reason before marking a duplicate group as no issue.")

    duplicate_payload, _all_rows, scan_warnings = _cleanup_candidate_payload(supabase, club_id=str(club_id), suppress_resolved=False)
    matched_group = None
    requested_key = _match_id_key(requested_ids)
    for group in duplicate_payload.get("duplicate_groups") or []:
        group_ids = [int(match_id) for match_id in (group.get("ids") or []) if _safe_int(match_id) is not None]
        if _match_id_key(group_ids) == requested_key:
            matched_group = group
            break
    if matched_group is None:
        raise ValueError(f"Match IDs {requested_ids} are not a current duplicate group.")

    matched_dup_key = str(matched_group.get("dup_key") or "").strip()
    if dup_key and str(dup_key).strip() != matched_dup_key:
        raise ValueError("Duplicate key no longer matches the current scan. Refresh and try again.")

    now_iso = datetime.now(timezone.utc).isoformat()
    resolution_payload = {
        "club_id": str(club_id),
        "dup_key": matched_dup_key,
        "match_id_key": requested_key,
        "match_ids": requested_ids,
        "resolution": "no_issue",
        "reason": clean_reason,
        "actor_email": str(actor_email or "").strip().lower(),
        "actor_role": str(actor_role or "").strip(),
        "source_page": source,
        "is_active": True,
        "resolved_at": now_iso,
        "updated_at": now_iso,
    }

    try:
        existing_rows = _safe_rows(
            supabase.table(DUPLICATE_RESOLUTIONS_TABLE)
            .select("id")
            .eq("club_id", str(club_id))
            .eq("dup_key", matched_dup_key)
            .eq("match_id_key", requested_key)
            .execute()
        )
        if existing_rows:
            supabase.table(DUPLICATE_RESOLUTIONS_TABLE).update(resolution_payload).eq("club_id", str(club_id)).eq("dup_key", matched_dup_key).eq("match_id_key", requested_key).execute()
        else:
            supabase.table(DUPLICATE_RESOLUTIONS_TABLE).insert(resolution_payload).execute()
    except Exception as exc:  # noqa: BLE001 - expose migration/configuration problems clearly to operators
        raise RuntimeError(f"Could not persist duplicate no-issue resolution: {exc.__class__.__name__}") from exc

    warnings: list[str] = list(scan_warnings)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="match_duplicate_false_positive_resolved",
        entity_type="match_duplicate_group",
        entity_id=requested_key,
        before_json={"duplicate_group": matched_group},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "dup_key": matched_dup_key,
            "match_ids": requested_ids,
            "resolution": "no_issue",
            "reason": clean_reason,
        },
        note=f"Duplicate group marked no issue: {clean_reason}",
        source_page=source,
        flagged_for_review=True,
    )
    audit_result = write_admin_activity_log(supabase, audit_payload)
    if audit_result.warning:
        warnings.append(audit_result.warning)
    if not audit_result.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")

    return {
        "ok": True,
        "mode": "duplicate_no_issue",
        "resolution": "no_issue",
        "dup_key": matched_dup_key,
        "match_ids": requested_ids,
        "reason": clean_reason,
        "warnings": warnings,
    }
