from __future__ import annotations

import os
import re
from collections import Counter
from datetime import date, datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.bulk_match_editor import compute_recompute_scope
from jupr_app.domain.dupes import canonical_dup_key
from jupr_app.services.match_edit_durability_service import apply_atomic_match_edits
from jupr_app.services.match_exclusion_durability_service import (
    apply_atomic_match_exclusions,
    find_match_exclusion_operation_by_idempotency_key,
)

MATCH_LOG_SELECT = (
    "id,date,league,week_tag,match_type,t1_p1,t1_p2,t2_p1,t2_p2,"
    "score_t1,score_t2,notes,deleted_at,context_type,context_id,tournament_id,"
    "tournament_game_id,updated_at,row_version"
)
MATCH_LOG_MINIMAL_SELECT = (
    "id,date,league,week_tag,match_type,t1_p1,t1_p2,t2_p1,t2_p2,"
    "score_t1,score_t2,deleted_at,tournament_id,tournament_game_id,row_version"
)
MATCH_LOG_RECOVERY_SELECT = (
    f"{MATCH_LOG_MINIMAL_SELECT},context_type,context_id"
)
MATCH_LOG_LEGACY_MINIMAL_SELECT = (
    "id,date,league,week_tag,match_type,t1_p1,t1_p2,t2_p1,t2_p2,"
    "score_t1,score_t2,deleted_at"
)
MATCH_LOG_LEGACY_RECOVERY_SELECT = (
    f"{MATCH_LOG_LEGACY_MINIMAL_SELECT},context_type,context_id"
)
DUPLICATE_RESOLUTIONS_TABLE = "admin_match_log_duplicate_resolutions"
MAX_FETCH_ROWS = 5000
MAX_RETURN_ROWS = 1000
MAX_PATCHES = 100
MAX_CLEANUP_IDS = 100
MAX_RESOLUTION_IDS = 20
MAX_CONTEXT_IDS = 200
MAX_CONTEXT_ID_LENGTH = 200
MAX_MATCH_IDS = 100


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "y", "on"}


def is_admin_match_log_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG")


def is_admin_match_log_apply_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY")


def is_admin_match_log_destructive_enabled() -> bool:
    return is_admin_match_log_apply_enabled() and _truthy_env(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE"
    )


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


def _reject_official_tournament_match_mutations(
    supabase: Any,
    *,
    club_id: str,
    match_ids: list[int],
) -> None:
    """Keep generic Match Log writes from diverging tournament authority.

    Official tournament rows are projections of ``tournament_games``. Their
    score, participants, and active state must be corrected through Tournament
    Manager so bracket/standings state and the canonical rating match stay in
    lockstep.
    """

    clean_ids = sorted({int(value) for value in match_ids if int(value) > 0})
    if not clean_ids:
        return
    try:
        rows = _safe_rows(
            supabase.table("matches")
            .select("id,tournament_id,tournament_game_id,context_type,context_id")
            .eq("club_id", str(club_id))
            .in_("id", clean_ids)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Match Log could not verify tournament linkage before mutation."
        ) from exc

    linked_ids = sorted(
        int(match_id)
        for row in rows
        if (
            row.get("tournament_game_id") not in (None, "")
            or str(row.get("context_type") or "").strip().casefold()
            == "tournament_game"
        )
        if (match_id := _safe_int(row.get("id"))) is not None
    )
    if linked_ids:
        raise ValueError(
            "Official tournament matches cannot be edited or excluded from "
            "the generic Match Log. Use Tournament Manager correction and "
            f"recovery tools for match IDs {linked_ids[:10]}."
        )


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


def _normalize_match_ids(
    *,
    match_id: int | None = None,
    match_ids: str | list[int | str] | tuple[int | str, ...] | None = None,
) -> list[int]:
    raw_values: list[Any] = []
    if match_id not in (None, ""):
        raw_values.append(match_id)
    if isinstance(match_ids, str):
        raw_values.extend(re.split(r"[\s,;]+", match_ids))
    elif match_ids:
        for value in match_ids:
            raw_values.extend(re.split(r"[\s,;]+", str(value or "")))

    normalized: list[int] = []
    seen: set[int] = set()
    for value in raw_values:
        token = str(value or "").strip()
        if token.startswith("#"):
            token = token[1:].strip()
        if not token:
            continue
        if not re.fullmatch(r"\d+", token):
            raise ValueError("Match IDs must be positive whole numbers separated by commas or spaces.")
        parsed = int(token)
        if parsed < 1:
            raise ValueError("Match IDs must be positive whole numbers separated by commas or spaces.")
        if parsed in seen:
            continue
        seen.add(parsed)
        normalized.append(parsed)
    if len(normalized) > MAX_MATCH_IDS:
        raise ValueError(f"No more than {MAX_MATCH_IDS} match IDs may be loaded at once.")
    return normalized


def _normalize_context_ids(
    *,
    context_id: str | None = None,
    context_ids: str | list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    raw_values: list[Any] = []
    if context_id not in (None, ""):
        raw_values.append(context_id)
    if isinstance(context_ids, str):
        raw_values.extend(context_ids.split(","))
    elif context_ids:
        for value in context_ids:
            raw_values.extend(str(value or "").split(","))

    normalized: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        clean = str(value or "").strip()
        if not clean or clean in seen:
            continue
        if len(clean) > MAX_CONTEXT_ID_LENGTH:
            raise ValueError(
                f"Each recovery context ID must be {MAX_CONTEXT_ID_LENGTH} characters or fewer."
            )
        seen.add(clean)
        normalized.append(clean)
    if len(normalized) > MAX_CONTEXT_IDS:
        raise ValueError(
            f"No more than {MAX_CONTEXT_IDS} recovery context IDs may be loaded at once."
        )
    return normalized


def _fetch_match_rows(
    supabase: Any,
    *,
    club_id: str,
    fetch_limit: int,
    match_ids: list[int] | None = None,
    context_type: str | None = None,
    context_ids: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    requested_match_ids = _normalize_match_ids(match_ids=match_ids)
    requested_context_type = str(context_type or "").strip().casefold() or None
    requested_context_ids = _normalize_context_ids(context_ids=context_ids)

    def _query(columns: str):
        query = (
            supabase.table("matches")
            .select(columns)
            .eq("club_id", str(club_id))
            .is_("deleted_at", None)
        )
        if len(requested_match_ids) == 1:
            query = query.eq("id", requested_match_ids[0])
        elif requested_match_ids:
            query = query.in_("id", requested_match_ids)
        if requested_context_type:
            query = query.eq("context_type", requested_context_type)
        if len(requested_context_ids) == 1:
            query = query.eq("context_id", requested_context_ids[0])
        elif requested_context_ids:
            query = query.in_("context_id", requested_context_ids)
        return query.order("date", desc=True).limit(int(fetch_limit)).execute()

    try:
        rows = _safe_rows(_query(MATCH_LOG_SELECT))
        return rows, warnings
    except Exception as exc:
        warnings.append(f"Fell back to minimal match columns: {exc.__class__.__name__}")

    fallback_columns = (
        MATCH_LOG_RECOVERY_SELECT
        if requested_context_type or requested_context_ids
        else MATCH_LOG_MINIMAL_SELECT
    )
    try:
        rows = _safe_rows(_query(fallback_columns))
        return rows, warnings
    except Exception as exc:
        warnings.append(
            "Versioned match columns are unavailable until the atomic "
            f"exclusion migration is applied: {exc.__class__.__name__}"
        )

    legacy_columns = (
        MATCH_LOG_LEGACY_RECOVERY_SELECT
        if requested_context_type or requested_context_ids
        else MATCH_LOG_LEGACY_MINIMAL_SELECT
    )
    try:
        rows = _safe_rows(_query(legacy_columns))
        warnings.append(
            "Match Log is read-only because matches.row_version is not "
            "available yet."
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
    match_ids: list[int],
    league: str | None,
    week_tag: str | None,
    context_type: str | None,
    context_ids: list[str],
    start_date: str | None,
    end_date: str | None,
) -> list[dict[str, Any]]:
    result = list(rows)
    normalized_filter = str(filter_type or "All").strip().lower()
    if normalized_filter in {"league", "leagues"}:
        result = [row for row in result if str(row.get("match_type") or "") != "PopUp"]
    elif normalized_filter in {"pop-up", "popup", "pop up"}:
        result = [row for row in result if str(row.get("match_type") or "") == "PopUp"]

    if match_ids:
        expected_match_ids = set(match_ids)
        result = [row for row in result if _safe_int(row.get("id")) in expected_match_ids]
    if league:
        result = [row for row in result if str(row.get("league") or "").strip() == str(league).strip()]
    if week_tag:
        result = [row for row in result if str(row.get("week_tag") or "").strip() == str(week_tag).strip()]
    if context_type:
        expected_context_type = str(context_type).strip().casefold()
        result = [
            row
            for row in result
            if str(row.get("context_type") or "").strip().casefold()
            == expected_context_type
        ]
    if context_ids:
        expected_context_ids = set(context_ids)
        result = [
            row
            for row in result
            if str(row.get("context_id") or "").strip() in expected_context_ids
        ]
    if start_date:
        result = [row for row in result if str(row.get("date") or "")[:10] >= str(start_date)[:10]]
    if end_date:
        result = [row for row in result if str(row.get("date") or "")[:10] <= str(end_date)[:10]]
    return sorted(result, key=_date_sort_key, reverse=True)


def _filter_options(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Return stable, human-sorted filter values from the already-loaded match rows."""

    leagues = {
        _clean_text(row.get("league"), limit=120)
        for row in rows
        if _clean_text(row.get("league"), limit=120)
    }
    week_tags = {
        _clean_text(row.get("week_tag"), limit=80)
        for row in rows
        if _clean_text(row.get("week_tag"), limit=80)
    }
    return {
        "leagues": sorted(leagues, key=str.casefold),
        "week_tags": sorted(week_tags, key=str.casefold),
    }


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
        "row_version": _safe_int(row.get("row_version")),
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
        "tournament_id": row.get("tournament_id"),
        "tournament_game_id": row.get("tournament_game_id"),
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


def _recent_match_exclusion_operations(
    supabase: Any,
    *,
    club_id: str,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        rows = _safe_rows(
            supabase.table("match_exclusion_operations")
            .select(
                "id,mode,status,recovery_stage,replay_job_id,error_text,"
                "affected_player_ids,result_json,source,created_at,finished_at"
            )
            .eq("club_id", str(club_id))
            .order("created_at", desc=True)
            .limit(20)
            .execute()
        )
    except Exception as exc:
        return (
            [],
            "Durable match exclusion operation history is unavailable: "
            f"{exc.__class__.__name__}",
        )
    return [
        {
            "id": str(row.get("id") or ""),
            "mode": _clean_text(row.get("mode") or "exclude", limit=40),
            "status": _clean_text(
                row.get("status") or "unknown",
                limit=40,
            ),
            "recovery_stage": _clean_text(
                row.get("recovery_stage"),
                limit=40,
            )
            or None,
            "replay_job_id": str(row.get("replay_job_id") or "") or None,
            "error_text": _clean_text(row.get("error_text"), limit=500)
            or None,
            "source": _clean_text(row.get("source"), limit=120) or None,
            "affected_player_ids": [
                int(value) for value in row.get("affected_player_ids") or []
            ],
            "created_at": row.get("created_at"),
            "finished_at": row.get("finished_at"),
            "result_json": dict(row.get("result_json") or {}),
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
        group_delete_targets = [
            {
                "match_id": int(row["id"]),
                "expected_row_version": int(row["row_version"]),
            }
            for row in group_rows[1:]
            if _safe_int(row.get("id")) is not None
            and _safe_int(row.get("row_version")) is not None
        ]
        sample = _match_payload(group_rows[0], club_id=str(club_id), names=names)
        group_payload = {
            "dup_key": key,
            "dup_count": len(group_rows),
            "keep_id": keep_id,
            "delete_ids": group_delete_ids,
            "delete_targets": group_delete_targets,
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
        rows_by_id = {
            int(row["id"]): row
            for row in rows
            if _safe_int(row.get("id")) is not None
        }
        delete_targets = [
            {
                "match_id": int(match_id),
                "expected_row_version": int(rows_by_id[match_id]["row_version"]),
            }
            for match_id in sorted(delete_ids)
            if match_id in rows_by_id
            and _safe_int(rows_by_id[match_id].get("row_version")) is not None
        ]
        delete_preview = {
            "mode": "planning_only",
            "keep_ids": keep_ids,
            "delete_ids": sorted(delete_ids),
            "targets": delete_targets,
            "delete_targets": delete_targets,
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
    destructive_enabled = is_admin_match_log_destructive_enabled()
    return {
        "mode": "apply_enabled" if apply_enabled else "planning_only",
        "apply_endpoint": "/admin/clubs/{club_id}/match-log/edits" if apply_enabled else None,
        "duplicate_cleanup_endpoint": (
            "/admin/clubs/{club_id}/match-log/duplicates/cleanup"
            if destructive_enabled
            else None
        ),
        "exclude_endpoint": (
            "/admin/clubs/{club_id}/match-log/exclude"
            if destructive_enabled
            else None
        ),
        "exclusion_status_endpoint": (
            "/admin/clubs/{club_id}/match-log/exclusions/{operation_id}"
            if destructive_enabled
            else None
        ),
        "exclusion_recovery_endpoint": (
            "/admin/clubs/{club_id}/match-log/exclusions/{operation_id}/recover"
            if destructive_enabled
            else None
        ),
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
            "Duplicate cleanup keeps the oldest row and completes mandatory rating replay before success is reported.",
            "False-positive duplicate groups can be resolved as no issue without deleting rows.",
            "Writes use FastAPI audit attribution and Python domain services.",
            "Match edits are committed through one service-role-only transaction; rating-affecting edits create and complete a mandatory replay job before success is reported.",
            "Rated-match exclusion and duplicate cleanup require exact row versions, a UUID idempotency key, leased full replay, and narrow match-trigger badge reconciliation.",
        ],
    }


def build_admin_match_log(
    supabase: Any,
    *,
    club_id: str,
    filter_type: str = "All",
    match_id: int | None = None,
    match_ids: str | list[int | str] | tuple[int | str, ...] | None = None,
    league: str | None = None,
    week_tag: str | None = None,
    context_type: str | None = None,
    context_id: str | None = None,
    context_ids: str | list[str] | tuple[str, ...] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    safe_limit = max(1, min(int(limit or 500), MAX_RETURN_ROWS))
    requested_match_ids = _normalize_match_ids(match_id=match_id, match_ids=match_ids)
    requested_context_ids = _normalize_context_ids(
        context_id=context_id,
        context_ids=context_ids,
    )
    filters = {
        "filter": filter_type or "All",
        "match_id": requested_match_ids[0] if len(requested_match_ids) == 1 else None,
        "match_ids": requested_match_ids,
        "league": league or None,
        "week_tag": week_tag or None,
        "context_type": str(context_type or "").strip() or None,
        "context_id": requested_context_ids[0]
        if len(requested_context_ids) == 1
        else None,
        "context_ids": requested_context_ids,
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
            "filter_options": {"leagues": [], "week_tags": []},
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
            "recent_exclusion_operations": [],
            "correction_plan": _correction_plan(),
            "warnings": ["Next Match Log is disabled. Use Streamlit Match Log until JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG is enabled for the pilot."],
        }

    fetch_limit = min(MAX_FETCH_ROWS, max(safe_limit * 5, safe_limit))
    raw_rows, warnings = _fetch_match_rows(
        supabase,
        club_id=str(club_id),
        fetch_limit=fetch_limit,
        match_ids=requested_match_ids,
        context_type=context_type,
        context_ids=requested_context_ids,
    )
    filtered_rows = _matches_filter(
        raw_rows,
        filter_type=filter_type,
        match_ids=requested_match_ids,
        league=league,
        week_tag=week_tag,
        context_type=context_type,
        context_ids=requested_context_ids,
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
    recent_exclusion_operations, exclusion_operations_warning = (
        _recent_match_exclusion_operations(
            supabase,
            club_id=str(club_id),
        )
    )
    if exclusion_operations_warning:
        warnings.append(exclusion_operations_warning)
    duplicate_payload = _duplicate_scan(visible_rows, club_id=str(club_id), names=names, resolved_lookup=resolved_lookup)
    matches = [_match_payload(row, club_id=str(club_id), names=names) for row in visible_rows]
    duplicate_delete_count = len((duplicate_payload.get("delete_preview") or {}).get("delete_ids") or [])
    return {
        "enabled": True,
        "apply_enabled": is_admin_match_log_apply_enabled(),
        "status": "apply_enabled" if is_admin_match_log_apply_enabled() else "planning_only",
        "filters": filters,
        "filter_options": _filter_options(raw_rows),
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
        "recent_exclusion_operations": recent_exclusion_operations,
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
    patch_ids = [
        int(match_id)
        for patch in clean_patches
        if (match_id := _safe_int(patch.get("id"))) is not None
    ]
    _reject_official_tournament_match_mutations(
        supabase,
        club_id=str(club_id),
        match_ids=patch_ids,
    )

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
    targets: list[dict[str, Any]],
    actor_email: str,
    actor_role: str,
    source: str = "next_match_log_duplicate_cleanup",
    confirmation_text: str = "",
    idempotency_key: str = "",
    note: str | None = None,
) -> dict[str, Any]:
    if not is_admin_match_log_apply_enabled():
        raise PermissionError("Next Match Log apply is disabled.")
    if not is_admin_match_log_destructive_enabled():
        raise PermissionError("Next Match Log destructive actions are disabled.")
    if str(confirmation_text or "").strip().upper() != "DELETE":
        raise ValueError("Type DELETE to confirm duplicate cleanup.")
    clean_targets = [
        dict(target) for target in (targets or []) if isinstance(target, dict)
    ]
    requested_ids = sorted(
        {
            int(target["match_id"])
            for target in clean_targets
            if _safe_int(target.get("match_id")) is not None
        }
    )
    if not requested_ids:
        raise ValueError("No duplicate targets were provided.")
    if len(clean_targets) != len(requested_ids):
        raise ValueError(
            "Each duplicate cleanup target must be unique and include an exact "
            "row version."
        )
    if len(requested_ids) > MAX_CLEANUP_IDS:
        raise ValueError(f"No more than {MAX_CLEANUP_IDS} duplicate IDs can be cleaned up at once.")

    stored_operation = find_match_exclusion_operation_by_idempotency_key(
        supabase,
        club_id=str(club_id),
        idempotency_key=idempotency_key,
    )
    scan_warnings: list[str] = []
    if stored_operation is None:
        _reject_official_tournament_match_mutations(
            supabase,
            club_id=str(club_id),
            match_ids=requested_ids,
        )
        duplicate_payload, _all_rows, scan_warnings = _cleanup_candidate_payload(
            supabase,
            club_id=str(club_id),
            suppress_resolved=True,
        )
        preview = duplicate_payload.get("delete_preview") or {}
        allowed_ids = {int(match_id) for match_id in (preview.get("delete_ids") or [])}
        invalid_ids = [match_id for match_id in requested_ids if match_id not in allowed_ids]
        if invalid_ids:
            raise ValueError(f"Some requested IDs are not currently active duplicate cleanup candidates: {invalid_ids[:10]}")
        expected_targets = {
            int(target["match_id"]): int(target["expected_row_version"])
            for target in (preview.get("targets") or [])
            if _safe_int(target.get("match_id")) is not None
            and _safe_int(target.get("expected_row_version")) is not None
        }
        for target in clean_targets:
            match_id = int(target["match_id"])
            expected_version = _safe_int(target.get("expected_row_version"))
            if expected_version is None:
                raise ValueError(
                    "Each duplicate cleanup target requires expected_row_version."
                )
            if expected_targets.get(match_id) != int(expected_version):
                raise ValueError(
                    "Duplicate cleanup preview is stale. Refresh Match Log and "
                    "select the rows again."
                )

    result = apply_atomic_match_exclusions(
        supabase,
        club_id=str(club_id),
        targets=clean_targets,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        source=source,
        note=note or "Duplicate cleanup from Next Match Log",
        idempotency_key=idempotency_key,
        mode="duplicate_cleanup",
    )
    result["warnings"] = [
        *list(scan_warnings),
        *list(result.get("warnings") or []),
    ]
    return result


def apply_admin_match_log_exclusions(
    supabase: Any,
    *,
    club_id: str,
    targets: list[dict[str, Any]],
    actor_email: str,
    actor_role: str,
    source: str = "next_match_log_bulk_exclude",
    confirmation_text: str = "",
    idempotency_key: str = "",
    note: str | None = None,
) -> dict[str, Any]:
    if not is_admin_match_log_apply_enabled():
        raise PermissionError("Next Match Log apply is disabled.")
    if not is_admin_match_log_destructive_enabled():
        raise PermissionError("Next Match Log destructive actions are disabled.")
    if str(confirmation_text or "").strip().upper() != "DELETE":
        raise ValueError("Type DELETE to confirm rated match exclusion.")
    clean_targets = [
        dict(target) for target in (targets or []) if isinstance(target, dict)
    ]
    stored_operation = find_match_exclusion_operation_by_idempotency_key(
        supabase,
        club_id=str(club_id),
        idempotency_key=idempotency_key,
    )
    if stored_operation is None:
        _reject_official_tournament_match_mutations(
            supabase,
            club_id=str(club_id),
            match_ids=[
                int(match_id)
                for target in clean_targets
                if (match_id := _safe_int(target.get("match_id"))) is not None
            ],
        )
    return apply_atomic_match_exclusions(
        supabase,
        club_id=str(club_id),
        targets=clean_targets,
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        source=source,
        note=note,
        idempotency_key=idempotency_key,
        mode="exclude",
    )


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
