from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log


TRUTHY = {"1", "true", "yes", "y", "on"}
_OPERATION_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,159}$")


class GuardedWriteRecoveryRequired(RuntimeError):
    """The mutation may have completed and must be reconciled before retrying."""

    def __init__(self, operation_key: str, message: str):
        self.operation_key = str(operation_key or "")
        super().__init__(message)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def clean_operation_key(value: Any) -> str:
    key = str(value or "").strip()
    if not _OPERATION_KEY_RE.fullmatch(key):
        raise ValueError(
            "operation_key must be 8-160 characters and use only letters, numbers, dot, underscore, colon, or hyphen."
        )
    return key


def require_staging_service_role_write(
    supabase: Any,
    *,
    workflow: str,
    required_tables: tuple[str, ...] = (),
) -> None:
    """Fail closed before a privileged write when staging/server schema is not ready."""
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    if environment != "staging":
        raise PermissionError(
            f"{workflow} writes are staging-only. Keep using the Streamlit fallback outside the isolated staging API."
        )
    if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
        raise RuntimeError(
            f"{workflow} is not write-ready: SUPABASE_SERVICE_ROLE_KEY must be configured on FastAPI only. No data was changed."
        )

    tables = ("admin_activity_log", "admin_guarded_operations", *required_tables)
    for table in dict.fromkeys(tables):
        try:
            supabase.table(table).select("*").limit(1).execute()
        except Exception as exc:
            raise RuntimeError(
                f"{workflow} is not write-ready: required staging table {table} is unavailable. Apply the migration and reload the API schema; no data was changed."
            ) from exc


def required_audit_event(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_type: str,
    entity_id: str,
    source: str,
    before: Any = None,
    after: Any = None,
    note: str | None = None,
    intent: bool,
) -> None:
    result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            action_type=str(action_type),
            entity_type=str(entity_type),
            entity_id=str(entity_id),
            before_json=before,
            after_json=after,
            note=note,
            source_page=str(source or "")[:120],
            flagged_for_review=True,
        ),
    )
    if not result.ok:
        if intent:
            raise RuntimeError("Required audit intent could not be persisted; no data was changed or sent.")
        raise RuntimeError(
            "Required completion audit could not be persisted; the operation may have completed. Stop and reconcile before retrying."
        )


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    data = getattr(response, "data", None)
    return [dict(row) for row in (data or []) if isinstance(row, dict)]


def get_guarded_operation(
    supabase: Any,
    *,
    club_id: str,
    workflow: str,
    operation_key: str,
) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("admin_guarded_operations")
        .select("*")
        .eq("club_id", str(club_id))
        .eq("workflow", str(workflow))
        .eq("operation_key", str(operation_key))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def begin_guarded_operation(
    supabase: Any,
    *,
    club_id: str,
    workflow: str,
    action: str,
    operation_key: str,
    request_payload: Any,
    actor_email: str,
    actor_role: str,
    source: str,
    before_json: Any = None,
) -> tuple[dict[str, Any], bool]:
    """Persist audit intent first, then an idempotent durable operation row."""
    key = clean_operation_key(operation_key)
    request_fingerprint = canonical_fingerprint(request_payload)
    existing = get_guarded_operation(
        supabase,
        club_id=str(club_id),
        workflow=str(workflow),
        operation_key=key,
    )
    if existing is not None:
        if str(existing.get("request_fingerprint") or "") != request_fingerprint:
            raise ValueError("operation_key was already used for a different request.")
        status = str(existing.get("status") or "")
        if status == "completed":
            return existing, True
        raise GuardedWriteRecoveryRequired(
            key,
            f"Operation {key} is {status or 'incomplete'}. Reconcile or recover it before retrying; the write will not run again.",
        )

    required_audit_event(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type=f"{action}_intent",
        entity_type="admin_guarded_operation",
        entity_id=key,
        source=source,
        before=before_json,
        after={
            "workflow": str(workflow),
            "action": str(action),
            "request_fingerprint": request_fingerprint,
            "source_client": "fastapi/nextjs",
        },
        intent=True,
    )
    payload = {
        "club_id": str(club_id),
        "workflow": str(workflow),
        "action": str(action),
        "operation_key": key,
        "request_fingerprint": request_fingerprint,
        "status": "intent_recorded",
        "before_json": before_json,
        "actor_email": str(actor_email or "").strip().lower(),
        "actor_role": str(actor_role or "").strip().lower(),
        "source": str(source or "")[:120],
    }
    try:
        rows = _safe_rows(supabase.table("admin_guarded_operations").insert(payload).execute())
    except Exception as exc:
        # A racing request may have won the unique key. Re-read and only replay an
        # exactly matching completed operation; never run a second mutation.
        raced = get_guarded_operation(
            supabase,
            club_id=str(club_id),
            workflow=str(workflow),
            operation_key=key,
        )
        if raced is not None and str(raced.get("request_fingerprint") or "") == request_fingerprint:
            if str(raced.get("status") or "") == "completed":
                return raced, True
            raise GuardedWriteRecoveryRequired(
                key,
                f"Operation {key} is already in progress or requires recovery. The write was not started twice.",
            ) from exc
        raise RuntimeError("Unable to persist the guarded operation record; no domain data was changed.") from exc
    if len(rows) != 1:
        raise RuntimeError("Unable to persist the guarded operation record; no domain data was changed.")
    return rows[0], False


def update_guarded_operation(
    supabase: Any,
    *,
    operation_id: Any,
    operation_key: str = "",
    status: str,
    after_json: Any = None,
    result_json: Any = None,
    error_text: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": str(status),
        "updated_at": utc_now_iso(),
        "error_text": str(error_text or "")[:2000] or None,
    }
    if after_json is not None:
        payload["after_json"] = after_json
    if result_json is not None:
        payload["result_json"] = result_json
    if status in {"completed", "compensated", "failed"}:
        payload["finished_at"] = utc_now_iso()
    try:
        rows = _safe_rows(
            supabase.table("admin_guarded_operations")
            .update(payload)
            .eq("id", operation_id)
            .execute()
        )
    except Exception as exc:
        raise GuardedWriteRecoveryRequired(
            str(operation_key or operation_id or "unknown-operation"),
            "Guarded operation state could not be persisted. The domain outcome may be complete while the ledger is stale; stop and reconcile before retrying.",
        ) from exc
    if len(rows) != 1:
        raise GuardedWriteRecoveryRequired(
            str(operation_key or operation_id or "unknown-operation"),
            "Guarded operation state could not be verified. The domain outcome may be complete while the ledger is stale; stop and reconcile before retrying.",
        )
    return rows[0]


def operation_result(existing: dict[str, Any]) -> dict[str, Any]:
    result = existing.get("result_json")
    if not isinstance(result, dict):
        raise GuardedWriteRecoveryRequired(
            str(existing.get("operation_key") or ""),
            "The completed operation has no readable result. Reconcile it before retrying.",
        )
    return {**result, "idempotent": True}
