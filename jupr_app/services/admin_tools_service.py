from __future__ import annotations

import os
from typing import Any

from jupr_app.domain.admin.role_assignments import (
    delete_role_assignment,
    has_other_super_admin_support,
    list_role_assignments,
    normalize_email,
    upsert_role_assignment,
)
from jupr_app.domain.admin.roles import ALL_ROLES, ROLE_SUPER_ADMIN, normalize_role
from jupr_app.domain.admin_activity_log import (
    RETENTION_DAYS,
    build_activity_payload,
    list_recent_admin_activity_logs,
    retention_cutoff_iso,
    write_admin_activity_log,
)
from jupr_app.domain.gamification.badge_worker import (
    process_badge_eval_queue,
    process_badge_eval_queue_until_empty,
)
from jupr_app.domain.gamification.recompute import run_badge_recompute

TRUTHY = {"1", "true", "yes", "y", "on"}
CONFIRM_ROLE = "SAVE ROLE"
CONFIRM_REVOKE = "REVOKE ROLE"
CONFIRM_QUEUE_BATCH = "PROCESS BADGE QUEUE"
CONFIRM_QUEUE_DRAIN = "DRAIN BADGE QUEUE"
CONFIRM_BADGE_RECOMPUTE = "RUN BADGE RECOMPUTE"
SNAPSHOT_COLUMNS = {
    "t1_p1_r",
    "t1_p2_r",
    "t2_p1_r",
    "t2_p2_r",
    "t1_p1_r_end",
    "t1_p2_r_end",
    "t2_p1_r_end",
    "t2_p2_r_end",
}
QUEUE_STATUS_VALUES = ("pending", "processing", "done", "error")
BADGE_RECOMPUTE_MODES = ("dry-run", "append-only", "strict")


def is_admin_tools_enabled() -> bool:
    return os.getenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "").strip().lower() in TRUTHY


def build_admin_tools_status(*, club_id: str) -> dict[str, Any]:
    enabled = is_admin_tools_enabled()
    return {
        "ok": True,
        "enabled": enabled,
        "status": "ready" if enabled else "disabled",
        "club_id": str(club_id),
        "roles": list(ALL_ROLES),
        "retention_days": RETENTION_DAYS,
        "retention_cutoff": retention_cutoff_iso(),
        "required_permissions": {
            "overview": "view_audit_log",
            "roles": "manage_roles",
            "workers": "run_replay",
        },
        "confirmation_text": {
            "save": CONFIRM_ROLE,
            "revoke": CONFIRM_REVOKE,
            "process_queue": CONFIRM_QUEUE_BATCH,
            "drain_queue": CONFIRM_QUEUE_DRAIN,
            "badge_recompute": CONFIRM_BADGE_RECOMPUTE,
        },
    }


def _sample_match_schema(supabase: Any, *, club_id: str) -> dict[str, Any]:
    try:
        rows = supabase.table("matches").select("*").eq("club_id", str(club_id)).limit(1).execute().data or []
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {"ok": False, "message": f"Unable to inspect matches table: {exc}"}
    if not rows:
        return {"ok": True, "sample_found": False, "snapshot_columns_present": False, "missing_snapshot_columns": sorted(SNAPSHOT_COLUMNS)}
    keys = set((rows[0] or {}).keys())
    missing = sorted(SNAPSHOT_COLUMNS - keys)
    return {"ok": True, "sample_found": True, "snapshot_columns_present": not missing, "missing_snapshot_columns": missing}


def _null_snapshot_count(supabase: Any, *, club_id: str) -> dict[str, Any]:
    try:
        rows = (
            supabase.table("matches")
            .select("id,t1_p1_r,t1_p1_r_end")
            .eq("club_id", str(club_id))
            .is_("t1_p1_r", None)
            .limit(5000)
            .execute()
            .data
            or []
        )
        return {"ok": True, "sample_limit": 5000, "null_snapshot_count": len(rows), "sample_match_ids": [row.get("id") for row in rows[:25]]}
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {"ok": False, "message": f"Unable to inspect null snapshots: {exc}"}


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _table_count(supabase: Any, table: str, *, club_id: str, status: str | None = None, limit: int = 10000) -> dict[str, Any]:
    try:
        query = supabase.table(table).select("id,status").eq("club_id", str(club_id))
        if status is not None:
            query = query.eq("status", str(status))
        rows = _safe_rows(query.limit(int(limit)).execute())
        return {"ok": True, "count": len(rows), "sample_limit": int(limit)}
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        return {"ok": False, "count": None, "message": str(exc)}


def build_admin_worker_status(supabase: Any, *, club_id: str) -> dict[str, Any]:
    queue_counts = {
        status: _table_count(supabase, "badge_eval_queue", club_id=str(club_id), status=status)
        for status in QUEUE_STATUS_VALUES
    }
    eval_runs = _table_count(supabase, "badge_recompute_runs", club_id=str(club_id), limit=1000)
    return {
        "ok": True,
        "queue_counts": queue_counts,
        "badge_recompute_run_count": eval_runs,
        "queue_modes": ["batch", "drain"],
        "badge_recompute_modes": list(BADGE_RECOMPUTE_MODES),
        "confirmation_text": {
            "batch": CONFIRM_QUEUE_BATCH,
            "drain": CONFIRM_QUEUE_DRAIN,
            "badge_recompute": CONFIRM_BADGE_RECOMPUTE,
        },
    }


def build_admin_tools_overview(
    supabase: Any,
    *,
    club_id: str,
    include_flagged_only: bool = False,
    activity_limit: int = 200,
) -> dict[str, Any]:
    if not is_admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    role_rows = list_role_assignments(supabase, str(club_id))
    activity_rows, activity_warning = list_recent_admin_activity_logs(
        supabase,
        club_id=str(club_id),
        include_flagged_only=bool(include_flagged_only),
        limit=max(1, min(int(activity_limit or 200), 500)),
    )
    health = {
        "match_schema": _sample_match_schema(supabase, club_id=str(club_id)),
        "null_snapshots": _null_snapshot_count(supabase, club_id=str(club_id)),
        "workers": build_admin_worker_status(supabase, club_id=str(club_id)),
    }
    return {
        "ok": True,
        "roles": role_rows,
        "activity": activity_rows,
        "activity_warning": activity_warning,
        "health": health,
        "role_options": list(ALL_ROLES),
        "retention_days": RETENTION_DAYS,
        "retention_cutoff": retention_cutoff_iso(),
    }


def update_admin_role_assignment(
    supabase: Any,
    *,
    club_id: str,
    target_email: str,
    target_role: str,
    user_id: str | None,
    action: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_admin_tools_roles",
) -> dict[str, Any]:
    if not is_admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    normalized_email = normalize_email(target_email)
    if "@" not in normalized_email or "." not in normalized_email.split("@")[-1]:
        raise ValueError("Enter a valid email address.")
    normalized_action = str(action or "").strip().lower()
    rows = list_role_assignments(supabase, str(club_id))
    existing = next((row for row in rows if normalize_email(row.get("email")) == normalized_email), None)

    if normalized_action == "upsert":
        if str(confirmation_text or "").strip().upper() != CONFIRM_ROLE:
            raise ValueError(f"Type {CONFIRM_ROLE} to save role assignments.")
        selected_role = normalize_role(target_role)
        if existing and normalize_role(existing.get("role")) == ROLE_SUPER_ADMIN and selected_role != ROLE_SUPER_ADMIN:
            if not has_other_super_admin_support(rows=rows, target_email=normalized_email, admin_allowlist=set()):
                raise ValueError("Unsafe change blocked: this would remove the final super_admin access.")
        upsert_role_assignment(supabase, str(club_id), normalized_email, selected_role, user_id=str(user_id or "").strip() or None)
        after = {"email": normalized_email, "role": selected_role, "user_id": str(user_id or "").strip() or None}
        action_type = "role_assignment_upsert"
        note = "Admin role assignment create/update"
    elif normalized_action == "revoke":
        if str(confirmation_text or "").strip().upper() != CONFIRM_REVOKE:
            raise ValueError(f"Type {CONFIRM_REVOKE} to revoke role assignments.")
        if existing and normalize_role(existing.get("role")) == ROLE_SUPER_ADMIN:
            if not has_other_super_admin_support(rows=rows, target_email=normalized_email, admin_allowlist=set()):
                raise ValueError("Unsafe revoke blocked: this would remove the final super_admin access.")
        delete_role_assignment(supabase, str(club_id), normalized_email)
        after = None
        action_type = "role_assignment_revoke"
        note = "Admin role assignment revoked"
    else:
        raise ValueError("action must be upsert or revoke")

    log_result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type=action_type,
            entity_type="admin_role_assignment",
            entity_id=normalized_email,
            before_json={"role": str((existing or {}).get("role") or "") or None, "user_id": (existing or {}).get("user_id")},
            after_json=after,
            note=note,
            source_page=source,
            flagged_for_review=True,
        ),
    )
    return {
        "ok": True,
        "mode": "admin_role_assignment_update",
        "action": normalized_action,
        "target_email": normalized_email,
        "audit_warning": log_result.warning,
        "roles": list_role_assignments(supabase, str(club_id)),
    }


def run_admin_badge_queue_worker(
    supabase: Any,
    *,
    club_id: str,
    mode: str,
    max_jobs: int,
    time_budget_seconds: float,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_admin_tools_workers",
) -> dict[str, Any]:
    if not is_admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    normalized_mode = str(mode or "batch").strip().lower()
    if normalized_mode not in {"batch", "drain"}:
        raise ValueError("mode must be batch or drain")
    expected = CONFIRM_QUEUE_DRAIN if normalized_mode == "drain" else CONFIRM_QUEUE_BATCH
    if str(confirmation_text or "").strip().upper() != expected:
        raise ValueError(f"Type {expected} to run the badge queue worker.")

    safe_max_jobs = max(1, min(int(max_jobs or 10), 5000))
    safe_budget = max(1.0, min(float(time_budget_seconds or 5.0), 180.0))
    before = build_admin_worker_status(supabase, club_id=str(club_id))
    if normalized_mode == "drain":
        result = process_badge_eval_queue_until_empty(
            supabase,
            str(club_id),
            max_total_jobs=safe_max_jobs,
            batch_max_jobs=min(50, safe_max_jobs),
            per_batch_time_budget_seconds=min(5.0, safe_budget),
            max_wall_clock_seconds=safe_budget,
            max_errors=max(1, min(50, safe_max_jobs)),
        )
    else:
        result = process_badge_eval_queue(
            supabase,
            max_jobs=safe_max_jobs,
            time_budget_seconds=int(safe_budget),
        )
    after = build_admin_worker_status(supabase, club_id=str(club_id))
    log_result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="admin_badge_queue_worker_run",
            entity_type="badge_eval_queue",
            entity_id=normalized_mode,
            before_json={"worker_status": before},
            after_json={"mode": normalized_mode, "result": result, "worker_status": after},
            note="Admin Tools badge eval queue worker run",
            source_page=source,
            flagged_for_review=True,
        ),
    )
    return {"ok": True, "mode": normalized_mode, "result": result, "worker_status": after, "audit_warning": log_result.warning}


def run_admin_badge_recompute_job(
    supabase: Any,
    *,
    club_id: str,
    mode: str,
    player_id: int | None,
    badge_id: str | None,
    league_id: str | None,
    context_id: str | None,
    since: str | None,
    until: str | None,
    include_non_live: bool,
    allow_strict_global: bool,
    match_limit: int,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_admin_tools_badge_recompute",
) -> dict[str, Any]:
    if not is_admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    normalized_mode = str(mode or "dry-run").strip().lower()
    if normalized_mode not in BADGE_RECOMPUTE_MODES:
        raise ValueError("mode must be dry-run, append-only, or strict")
    if normalized_mode != "dry-run" and str(confirmation_text or "").strip().upper() != CONFIRM_BADGE_RECOMPUTE:
        raise ValueError(f"Type {CONFIRM_BADGE_RECOMPUTE} to run an applying badge recompute.")
    summary = run_badge_recompute(
        supabase,
        club_id=str(club_id),
        mode=normalized_mode,
        league_id=str(league_id).strip() or None if league_id is not None else None,
        context_id=str(context_id).strip() or None if context_id is not None else None,
        player_id=int(player_id) if player_id is not None else None,
        badge_id=str(badge_id).strip() or None if badge_id is not None else None,
        since=str(since).strip() or None if since is not None else None,
        until=str(until).strip() or None if until is not None else None,
        created_by=actor_email,
        allow_strict_global=bool(allow_strict_global),
        match_limit=max(100, min(int(match_limit or 5000), 50000)),
        include_non_live=bool(include_non_live),
    )
    log_result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="admin_badge_recompute_run",
            entity_type="player_badges",
            entity_id=f"badge_recompute:{normalized_mode}",
            after_json={"mode": normalized_mode, "summary": summary},
            note="Admin Tools badge recompute job",
            source_page=source,
            flagged_for_review=normalized_mode != "dry-run",
        ),
    )
    return {"ok": True, "mode": normalized_mode, "summary": summary, "audit_warning": log_result.warning}
