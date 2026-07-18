from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log

TRUTHY_ENV_VALUES = {"1", "true", "yes", "y", "on"}
SOCIAL_SUBMISSION_STATUSES = ("pending", "saved", "rejected")
SOCIAL_SUBMISSION_ACTIONS = ("approve", "reject")
CONFIRM_APPROVE_SOCIAL_SUBMISSION = "APPROVE SOCIAL SUBMISSION"
CONFIRM_REJECT_SOCIAL_SUBMISSION = "REJECT SOCIAL SUBMISSION"
MAX_SOCIAL_SUBMISSION_ROWS = 100
_BASE_SELECT = (
    "id,club_id,name,event_type,event_date,status,result_mode,submission_mode,"
    "summary_json,raw_event_json,created_at,updated_at,rejection_reason,moderated_at,moderated_by"
)


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY_ENV_VALUES


def _admin_tools_enabled() -> bool:
    return _truthy_env("JUPR_ENABLE_NEXT_ADMIN_TOOLS")


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _safe_first(response: Any) -> dict[str, Any] | None:
    rows = _safe_rows(response)
    return rows[0] if rows else None


def _clean_text(value: Any, *, limit: int = 1000) -> str:
    return str(value or "").replace("<", "").replace(">", "").strip()[:limit]


def _json_object(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _submitted_by_name(row: dict[str, Any]) -> str:
    return _clean_text(row.get("submitted_by_name") or row.get("submitted_by"), limit=160) or "unknown"


def _submission_payload(row: dict[str, Any], *, include_raw: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": _clean_text(row.get("id"), limit=120),
        "club_id": _clean_text(row.get("club_id"), limit=120),
        "name": _clean_text(row.get("name"), limit=240) or "Untitled",
        "event_type": _clean_text(row.get("event_type"), limit=80),
        "event_date": row.get("event_date"),
        "status": _clean_text(row.get("status"), limit=40).lower(),
        "submission_mode": _clean_text(row.get("submission_mode"), limit=40),
        "submitted_by_name": _submitted_by_name(row),
        "summary_json": _json_object(row.get("summary_json")),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "rejection_reason": _clean_text(row.get("rejection_reason"), limit=1200) or None,
        "moderated_at": row.get("moderated_at"),
        "moderated_by": _clean_text(row.get("moderated_by"), limit=240) or None,
    }
    if include_raw:
        payload["raw_event_json"] = _json_object(row.get("raw_event_json"))
    return payload


def _missing_submitted_by_name(exc: Exception) -> bool:
    text = " ".join(str(part) for part in getattr(exc, "args", ()) if part).lower()
    return "submitted_by_name" in text and ("schema" in text or "column" in text or "42703" in text or "pgrst204" in text)


def _select_submission_query(supabase: Any, *, club_id: str, submitted_by_column: str):
    return (
        supabase.table("live_events")
        .select(f"{_BASE_SELECT},{submitted_by_column}")
        .eq("club_id", str(club_id))
        .eq("result_mode", "social_unrated")
    )


def _list_submission_rows(
    supabase: Any,
    *,
    club_id: str,
    status: str,
    limit: int,
) -> list[dict[str, Any]]:
    try:
        response = (
            _select_submission_query(supabase, club_id=club_id, submitted_by_column="submitted_by_name")
            .eq("status", status)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        if not _missing_submitted_by_name(exc):
            raise
        response = (
            _select_submission_query(supabase, club_id=club_id, submitted_by_column="submitted_by")
            .eq("status", status)
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
    return _safe_rows(response)


def _get_submission_row(supabase: Any, *, club_id: str, event_id: str) -> dict[str, Any] | None:
    try:
        response = (
            _select_submission_query(supabase, club_id=club_id, submitted_by_column="submitted_by_name")
            .eq("id", event_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        if not _missing_submitted_by_name(exc):
            raise
        response = (
            _select_submission_query(supabase, club_id=club_id, submitted_by_column="submitted_by")
            .eq("id", event_id)
            .limit(1)
            .execute()
        )
    return _safe_first(response)


def list_admin_social_submissions(
    supabase: Any,
    *,
    club_id: str,
    status: str = "pending",
    limit: int = MAX_SOCIAL_SUBMISSION_ROWS,
) -> dict[str, Any]:
    if not _admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    normalized_status = _clean_text(status, limit=40).lower()
    if normalized_status not in SOCIAL_SUBMISSION_STATUSES:
        raise ValueError("Unsupported Club Social submission status filter.")
    safe_limit = max(1, min(int(limit or MAX_SOCIAL_SUBMISSION_ROWS), MAX_SOCIAL_SUBMISSION_ROWS))
    rows = _list_submission_rows(
        supabase,
        club_id=str(club_id),
        status=normalized_status,
        limit=safe_limit,
    )
    return {
        "ok": True,
        "mode": "admin_social_submission_review",
        "read_only": True,
        "status": normalized_status,
        "statuses": list(SOCIAL_SUBMISSION_STATUSES),
        "confirmation_text": {
            "approve": CONFIRM_APPROVE_SOCIAL_SUBMISSION,
            "reject": CONFIRM_REJECT_SOCIAL_SUBMISSION,
        },
        "summary": {"returned_count": len(rows), "limit": safe_limit},
        "submissions": [_submission_payload(row) for row in rows],
        "warnings": [],
    }


def moderate_admin_social_submission(
    supabase: Any,
    *,
    club_id: str,
    event_id: str,
    action: str,
    expected_status: str,
    rejection_reason: str = "",
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    source: str = "next_admin_tools_social_review",
) -> dict[str, Any]:
    if not _admin_tools_enabled():
        raise PermissionError("Next Admin Tools are disabled.")
    clean_event_id = _clean_text(event_id, limit=120)
    if not clean_event_id:
        raise ValueError("Club Social submission ID is required.")
    normalized_action = _clean_text(action, limit=40).lower()
    if normalized_action not in SOCIAL_SUBMISSION_ACTIONS:
        raise ValueError("action must be 'approve' or 'reject'.")
    normalized_expected_status = _clean_text(expected_status, limit=40).lower()
    if normalized_expected_status not in SOCIAL_SUBMISSION_STATUSES:
        raise ValueError("A valid expected_status is required.")
    expected_confirmation = (
        CONFIRM_APPROVE_SOCIAL_SUBMISSION
        if normalized_action == "approve"
        else CONFIRM_REJECT_SOCIAL_SUBMISSION
    )
    if str(confirmation_text or "").strip().upper() != expected_confirmation:
        raise ValueError(f"Type {expected_confirmation} to moderate this submission.")
    clean_rejection_reason = _clean_text(rejection_reason, limit=1200)
    if normalized_action == "reject" and not clean_rejection_reason:
        raise ValueError("A rejection reason is required.")

    before = _get_submission_row(supabase, club_id=str(club_id), event_id=clean_event_id)
    if not before:
        raise ValueError("Club Social submission not found for this club.")
    current_status = _clean_text(before.get("status"), limit=40).lower()
    if current_status != normalized_expected_status:
        raise ValueError(
            f"Submission status changed from {normalized_expected_status} to {current_status or 'unknown'}. Reload the queue before moderating."
        )
    target_status = "saved" if normalized_action == "approve" else "rejected"
    if current_status == target_status:
        raise ValueError(f"Submission is already {target_status}.")

    update_payload = {
        "status": target_status,
        "moderated_at": datetime.now(timezone.utc).isoformat(),
        "moderated_by": _clean_text(actor_email, limit=240) or None,
        "rejection_reason": clean_rejection_reason if normalized_action == "reject" else None,
    }
    updated = _safe_first(
        supabase.table("live_events")
        .update(update_payload)
        .eq("id", clean_event_id)
        .eq("club_id", str(club_id))
        .eq("result_mode", "social_unrated")
        .eq("status", normalized_expected_status)
        .execute()
    )
    if not updated:
        raise ValueError("Submission changed while it was being moderated. Reload the queue and review it again.")

    normalized_updated = {**before, **updated}
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type=f"{normalized_action}_club_social_submission",
        entity_type="live_event",
        entity_id=clean_event_id,
        before_json=_submission_payload(before, include_raw=False),
        after_json={
            "source_client": "fastapi/nextjs",
            "submission": _submission_payload(normalized_updated, include_raw=False),
        },
        note=clean_rejection_reason if normalized_action == "reject" else None,
        source_page=_clean_text(source, limit=120) or "next_admin_tools_social_review",
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
        "mode": "admin_social_submission_moderation",
        "action": normalized_action,
        "submission": _submission_payload(normalized_updated),
        "warnings": warnings,
    }
