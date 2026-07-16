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

TRUTHY = {"1", "true", "yes", "y", "on"}
CONFIRM_ROLE = "SAVE ROLE"
CONFIRM_REVOKE = "REVOKE ROLE"
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
        "required_permissions": {"overview": "view_audit_log", "roles": "manage_roles"},
        "confirmation_text": {"save": CONFIRM_ROLE, "revoke": CONFIRM_REVOKE},
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
