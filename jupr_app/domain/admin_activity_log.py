from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from postgrest.exceptions import APIError

from jupr_app.domain.admin.roles import ROLE_CLUB_OWNER, ROLE_SUPER_ADMIN, normalize_role

RETENTION_DAYS = 365


@dataclass(frozen=True)
class ActivityLogWriteResult:
    ok: bool
    warning: str | None = None


def can_view_admin_activity(role: str) -> bool:
    normalized = normalize_role(role)
    return normalized in {ROLE_SUPER_ADMIN, ROLE_CLUB_OWNER}


def retention_cutoff_iso(*, now: datetime | None = None) -> str:
    anchor = now or datetime.now(timezone.utc)
    return (anchor - timedelta(days=RETENTION_DAYS)).isoformat()


def build_activity_payload(
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_type: str,
    entity_id: str,
    before_json: Any = None,
    after_json: Any = None,
    note: str | None = None,
    source_page: str | None = None,
    flagged_for_review: bool = False,
    effective_role: str | None = None,
    role_source: str | None = None,
) -> dict[str, Any]:
    normalized_after_json = after_json if isinstance(after_json, dict) else ({"value": after_json} if after_json is not None else None)
    if effective_role or role_source:
        audit_context: dict[str, str] = {}
        if effective_role:
            audit_context["effective_role"] = normalize_role(effective_role)
        if role_source:
            audit_context["role_source"] = str(role_source or "").strip()
        if normalized_after_json is None:
            normalized_after_json = {"_audit_context": audit_context}
        else:
            normalized_after_json = {
                **normalized_after_json,
                "_audit_context": audit_context,
            }

    return {
        "club_id": str(club_id),
        "actor_email": str(actor_email or "").strip().lower() or "unknown",
        "actor_role": normalize_role(actor_role),
        "action_type": str(action_type or "").strip(),
        "entity_type": str(entity_type or "").strip(),
        "entity_id": str(entity_id or "").strip(),
        "before_json": before_json,
        "after_json": normalized_after_json,
        "note": str(note or "").strip() or None,
        "source_page": str(source_page or "").strip() or None,
        "flagged_for_review": bool(flagged_for_review),
    }


def write_admin_activity_log(supabase, payload: dict[str, Any]) -> ActivityLogWriteResult:
    try:
        supabase.table("admin_activity_log").insert(payload).execute()
        return ActivityLogWriteResult(ok=True)
    except Exception as exc:  # noqa: BLE001 - intentionally degrade gracefully
        warning = "Admin activity log write failed."
        if isinstance(exc, APIError):
            code = getattr(exc, "code", None) or ((exc.args[0].get("code")) if exc.args and isinstance(exc.args[0], dict) else None)
            if code in {"42P01", "PGRST205"}:
                warning = "Admin activity log table is not available yet. Apply migrations to enable audit visibility."
        return ActivityLogWriteResult(ok=False, warning=warning)


def list_recent_admin_activity_logs(
    supabase,
    *,
    club_id: str,
    include_flagged_only: bool = False,
    limit: int = 200,
) -> tuple[list[dict[str, Any]], str | None]:
    try:
        query = (
            supabase.table("admin_activity_log")
            .select(
                "id,club_id,actor_email,actor_role,action_type,entity_type,entity_id,before_json,after_json,note,source_page,flagged_for_review,created_at"
            )
            .eq("club_id", str(club_id))
            .order("created_at", desc=True)
            .limit(int(limit))
        )
        if include_flagged_only:
            query = query.eq("flagged_for_review", True)
        result = query.execute()
        return list(result.data or []), None
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, APIError):
            code = getattr(exc, "code", None) or ((exc.args[0].get("code")) if exc.args and isinstance(exc.args[0], dict) else None)
            if code in {"42P01", "PGRST205"}:
                return [], "Admin activity migration is not applied yet."
        return [], "Unable to load admin activity logs right now."
