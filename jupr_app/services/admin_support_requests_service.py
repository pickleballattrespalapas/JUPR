from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import os

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
REQUEST_STATUSES = {"new", "in_review", "resolved", "dismissed"}
REQUEST_TYPES = {"data_correction", "profile_privacy", "general_support"}
IDENTITY_STATUSES = {"not_required", "pending", "verified", "rejected"}
FULFILLMENT_STATUSES = {"not_required", "pending", "in_progress", "completed", "declined"}
RESOLUTION_ACTIONS = {"none", "alias", "hide", "anonymize", "contact_update", "correction", "other"}
CONFIRM_STATUS = "SAVE REQUEST STATUS"


class SupportRequestConflictError(RuntimeError):
    """Raised when a reviewer attempts to save a stale queue item."""


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def is_admin_support_requests_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _clean_text(value: Any, *, limit: int = 1000) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _request_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "club_id": str(row.get("club_id") or ""),
        "club_slug": _clean_text(row.get("club_slug"), limit=120),
        "request_type": _clean_text(row.get("request_type"), limit=60),
        "status": _clean_text(row.get("status"), limit=40),
        "requester_name": _clean_text(row.get("requester_name"), limit=160),
        "requester_email": _clean_text(row.get("requester_email"), limit=240),
        "player_name": _clean_text(row.get("player_name"), limit=160),
        "player_id": row.get("player_id"),
        "match_id": _clean_text(row.get("match_id"), limit=120),
        "tournament_id": _clean_text(row.get("tournament_id"), limit=120),
        "subject": _clean_text(row.get("subject"), limit=240),
        "description": _clean_text(row.get("description"), limit=2400),
        "requested_action": _clean_text(row.get("requested_action"), limit=1200),
        "evidence_url": _clean_text(row.get("evidence_url"), limit=600),
        "identity_status": _clean_text(row.get("identity_status") or "not_required", limit=40),
        "fulfillment_status": _clean_text(row.get("fulfillment_status") or "not_required", limit=40),
        "resolution_action": _clean_text(row.get("resolution_action") or "none", limit=60),
        "resolution_evidence": _clean_text(row.get("resolution_evidence"), limit=1200),
        "source": _clean_text(row.get("source"), limit=120),
        "admin_note": _clean_text(row.get("admin_note"), limit=1200),
        "reviewed_by": _clean_text(row.get("reviewed_by"), limit=240),
        "reviewed_at": row.get("reviewed_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def build_admin_support_requests_status(supabase: Any | None, *, club_id: str) -> dict[str, Any]:
    if not is_admin_support_requests_enabled():
        return {
            "enabled": False,
            "status": "guarded_off",
            "requests_endpoint": None,
            "update_endpoint": None,
            "request_count": None,
            "warnings": ["Next Support Requests is disabled. Enable JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS on FastAPI for staff review."],
        }
    request_count = None
    if supabase is not None:
        try:
            rows = _safe_rows(supabase.table("public_support_requests").select("id").eq("club_id", str(club_id)).limit(1000).execute())
            request_count = len(rows)
        except Exception:
            request_count = None
    return {
        "enabled": True,
        "status": "ready_for_support_request_review",
        "requests_endpoint": "/admin/clubs/{club_id}/support-requests",
        "update_endpoint": "/admin/clubs/{club_id}/support-requests/{request_id}",
        "request_count": request_count,
        "warnings": [],
    }


def list_admin_support_requests(
    supabase: Any,
    *,
    club_id: str,
    status: str | None = None,
    request_type: str | None = None,
    limit: int = 200,
) -> dict[str, Any]:
    if not is_admin_support_requests_enabled():
        raise PermissionError("Next Support Requests is disabled.")
    query = supabase.table("public_support_requests").select("*").eq("club_id", str(club_id))
    clean_status = _clean_text(status, limit=40).lower()
    if clean_status:
        if clean_status not in REQUEST_STATUSES:
            raise ValueError("Unsupported request status filter.")
        query = query.eq("status", clean_status)
    clean_type = _clean_text(request_type, limit=60).lower()
    if clean_type:
        if clean_type not in REQUEST_TYPES:
            raise ValueError("Unsupported request type filter.")
        query = query.eq("request_type", clean_type)
    rows = _safe_rows(query.order("created_at", desc=True).limit(max(1, min(int(limit or 200), 500))).execute())
    summary = {"total": len(rows), "by_status": {}, "by_type": {}}
    for row in rows:
        row_status = _clean_text(row.get("status") or "new", limit=40).lower()
        row_type = _clean_text(row.get("request_type") or "general_support", limit=60).lower()
        summary["by_status"][row_status] = int(summary["by_status"].get(row_status, 0)) + 1
        summary["by_type"][row_type] = int(summary["by_type"].get(row_type, 0)) + 1
    return {
        "ok": True,
        "mode": "admin_support_requests_list",
        "requests": [_request_payload(row) for row in rows],
        "summary": summary,
        "warnings": [],
    }


def update_admin_support_request(
    supabase: Any,
    *,
    club_id: str,
    request_id: str,
    status: str,
    admin_note: str | None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_updated_at: str | None = None,
    identity_status: str | None = None,
    fulfillment_status: str | None = None,
    resolution_action: str | None = None,
    resolution_evidence: str | None = None,
    source: str = "next_admin_support_requests",
) -> dict[str, Any]:
    if not is_admin_support_requests_enabled():
        raise PermissionError("Next Support Requests is disabled.")
    if str(confirmation_text or "").strip().upper() != CONFIRM_STATUS:
        raise ValueError(f"Type {CONFIRM_STATUS} to update this request.")
    clean_request_id = _clean_text(request_id, limit=120)
    clean_status = _clean_text(status, limit=40).lower()
    if clean_status not in REQUEST_STATUSES:
        raise ValueError("Unsupported request status.")
    before = _safe_first(
        supabase.table("public_support_requests")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("id", clean_request_id)
        .limit(1)
        .execute()
    )
    if not before:
        raise ValueError("Support request not found.")
    current_updated_at = _clean_text(before.get("updated_at"), limit=80)
    clean_expected_updated_at = _clean_text(expected_updated_at, limit=80)
    if clean_expected_updated_at and clean_expected_updated_at != current_updated_at:
        raise SupportRequestConflictError("This request changed after it was loaded. Refresh before saving.")

    clean_note = _clean_text(admin_note, limit=1200)
    if clean_status in {"resolved", "dismissed"} and not clean_note:
        raise ValueError("An admin note is required before resolving or dismissing a request.")

    current_identity = _clean_text(before.get("identity_status") or "not_required", limit=40).lower()
    current_fulfillment = _clean_text(before.get("fulfillment_status") or "not_required", limit=40).lower()
    current_action = _clean_text(before.get("resolution_action") or "none", limit=60).lower()
    clean_identity = _clean_text(identity_status if identity_status is not None else current_identity, limit=40).lower()
    clean_fulfillment = _clean_text(
        fulfillment_status if fulfillment_status is not None else current_fulfillment,
        limit=40,
    ).lower()
    clean_action = _clean_text(resolution_action if resolution_action is not None else current_action, limit=60).lower()
    clean_resolution_evidence = _clean_text(
        resolution_evidence if resolution_evidence is not None else before.get("resolution_evidence"),
        limit=1200,
    )
    if clean_identity not in IDENTITY_STATUSES:
        raise ValueError("Unsupported identity-verification status.")
    if clean_fulfillment not in FULFILLMENT_STATUSES:
        raise ValueError("Unsupported privacy-fulfillment status.")
    if clean_action not in RESOLUTION_ACTIONS:
        raise ValueError("Unsupported resolution action.")
    if _clean_text(before.get("request_type"), limit=60).lower() == "profile_privacy" and clean_status == "resolved":
        if clean_identity != "verified":
            raise ValueError("Verify the requester identity before resolving a profile privacy request.")
        if clean_fulfillment != "completed":
            raise ValueError("Complete and verify the privacy fulfillment checklist before resolving the request.")
        if clean_action == "none":
            raise ValueError("Record the approved privacy resolution action before resolving the request.")
        if not clean_resolution_evidence:
            raise ValueError("Record privacy fulfillment evidence before resolving the request.")

    updated_at = _now_iso()
    payload = {
        "status": clean_status,
        "admin_note": clean_note or None,
        "identity_status": clean_identity,
        "fulfillment_status": clean_fulfillment,
        "resolution_action": clean_action,
        "resolution_evidence": clean_resolution_evidence or None,
        "reviewed_by": str(actor_email or "").strip() or None,
        "reviewed_at": updated_at,
        "updated_at": updated_at,
    }
    update_query = (
        supabase.table("public_support_requests")
        .update(payload)
        .eq("club_id", str(club_id))
        .eq("id", clean_request_id)
    )
    if current_updated_at:
        update_query = update_query.eq("updated_at", current_updated_at)
    updated = _safe_first(update_query.execute())
    if not updated:
        raise SupportRequestConflictError("This request changed after it was loaded. Refresh before saving.")
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_public_support_request_admin",
        entity_type="public_support_request",
        entity_id=clean_request_id,
        before_json=_request_payload(before),
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "request": _request_payload(updated)},
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and _truthy_env("JUPR_REQUIRE_API_AUDIT_LOG"):
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "admin_support_request_update",
        "request": _request_payload(updated),
        "warnings": warnings,
    }
