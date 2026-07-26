from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlencode
from uuid import NAMESPACE_URL, uuid5

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log


TRUTHY = {"1", "true", "yes", "y", "on"}
OPERATION_KEY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,160}$")
OPERATION_TABLE = "live_ladder_admin_operations"


class LiveLadderConflictError(RuntimeError):
    """Raised when a mutation request is stale or reuses an operation key."""


class LiveLadderPersistenceError(RuntimeError):
    """Raised when the durable intent/completion contract cannot be persisted."""


class LiveLadderUncertainError(RuntimeError):
    """Raised when a write may have happened and blind retry is unsafe."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _first(response: Any) -> dict[str, Any] | None:
    rows = _safe_rows(response)
    return rows[0] if rows else None


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str))


def stable_request_fingerprint(payload: Any) -> str:
    canonical = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_idempotency_key(value: Any) -> str:
    clean = str(value or "").strip()
    if not OPERATION_KEY_PATTERN.fullmatch(clean):
        raise ValueError(
            "idempotency_key must be 8-160 characters using letters, numbers, dot, underscore, colon, or hyphen."
        )
    return clean


def deterministic_operation_key(
    *,
    club_id: str,
    surface: str,
    operation_type: str,
    entity_id: str,
    idempotency_key: str,
) -> str:
    key = normalize_idempotency_key(idempotency_key)
    scope = "\n".join(
        (
            str(club_id or "").strip(),
            str(surface or "").strip(),
            str(operation_type or "").strip(),
            str(entity_id or "").strip(),
            key,
        )
    )
    return hashlib.sha256(scope.encode("utf-8")).hexdigest()


def deterministic_match_context_id(*, operation_key: str, slot: str | int) -> str:
    """Return a UUIDv5 accepted by the canonical matches.context_id column."""
    clean_operation_key = str(operation_key or "").strip()
    if not re.fullmatch(r"[a-f0-9]{64}", clean_operation_key):
        raise ValueError("operation_key must be a lowercase SHA-256 value")
    clean_slot = str(slot or "").strip()
    if not clean_slot:
        raise ValueError("match context slot is required")
    return str(uuid5(NAMESPACE_URL, f"jupr:live-ladder:{clean_operation_key}:slot:{clean_slot}"))


def is_staging_write_gate_enabled(flag_name: str) -> bool:
    return (
        os.getenv("JUPR_ENV", "").strip().lower() == "staging"
        and os.getenv(str(flag_name), "").strip().lower() in TRUTHY
    )


def require_staging_write_gate(*, surface_label: str, flag_name: str) -> None:
    if is_staging_write_gate_enabled(flag_name):
        return
    raise PermissionError(
        f"{surface_label} writes are staging-only and disabled. Set JUPR_ENV=staging and {flag_name}=1 on FastAPI, "
        "or use the Streamlit fallback."
    )


def build_match_log_recovery_url(
    *,
    context_type: str,
    context_ids: list[str] | tuple[str, ...] | None = None,
    fallback_context_id: str = "",
) -> str:
    clean_type = str(context_type or "").strip()
    contexts: list[str] = []
    seen: set[str] = set()
    for value in context_ids or ():
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            contexts.append(clean)

    query: dict[str, str] = {}
    if clean_type:
        query["context_type"] = clean_type
    if len(contexts) == 1:
        query["context_id"] = contexts[0]
    elif contexts:
        query["context_ids"] = ",".join(contexts)
    else:
        fallback = str(fallback_context_id or "").strip()
        if fallback:
            query["context_id"] = fallback
    return f"/admin/match-log?{urlencode(query)}" if query else "/admin/match-log"


def operation_recovery_handoff(
    *,
    surface: str,
    entity_id: str,
    match_context_ids: list[str] | None = None,
) -> dict[str, Any]:
    context_type = {
        "challenge_ladder": "challenge_ladder",
        "moneyball": "moneyball",
        "jupr_live_admin": "jupr_live",
    }.get(str(surface), str(surface))
    contexts: list[str] = []
    seen: set[str] = set()
    for value in match_context_ids or []:
        clean = str(value or "").strip()
        if clean and clean not in seen:
            seen.add(clean)
            contexts.append(clean)
    return {
        "outcome": "verify_before_retry",
        "match_context_ids": contexts,
        "match_log_url": build_match_log_recovery_url(
            context_type=context_type,
            context_ids=contexts,
            fallback_context_id=str(entity_id or ""),
        ),
        "replay_history_url": "/admin/replay-history",
        "instructions": (
            "Reload the durable operation first. If its outcome is still uncertain, stop: inspect the listed "
            "contexts in Match Log, make any correction there, then run/verify Replay History before retrying."
        ),
    }


def _required_audit(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_type: str,
    entity_id: str,
    source: str,
    before_json: dict[str, Any] | None = None,
    after_json: dict[str, Any] | None = None,
    after_domain_mutation: bool = False,
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
            before_json=_json_safe(before_json or {}),
            after_json={"source_client": "fastapi/nextjs", **_json_safe(after_json or {})},
            source_page=str(source),
            flagged_for_review=True,
        ),
    )
    if result.ok:
        return
    if after_domain_mutation:
        raise LiveLadderUncertainError(
            "The domain write may have completed, but its required audit did not persist. Do not retry blindly; "
            "reload the durable operation and use Match Log/Replay History if it remains uncertain."
        )
    raise LiveLadderPersistenceError(
        "Required durable audit intent could not be persisted. No domain mutation was attempted."
    )


def ensure_live_ladder_operation_schema_ready(supabase: Any) -> None:
    try:
        supabase.table(OPERATION_TABLE).select(
            "operation_key,club_id,surface,operation_type,idempotency_key,request_fingerprint,expected_version,status,result_json,recovery_json"
        ).limit(1).execute()
    except Exception as exc:
        raise LiveLadderPersistenceError(
            "The order-24 live-ladder operation ledger is unavailable. Apply the canonical migration to staging "
            "before enabling any Next writes."
        ) from exc


def _get_operation(supabase: Any, *, club_id: str, operation_key: str) -> dict[str, Any] | None:
    try:
        return _first(
            supabase.table(OPERATION_TABLE)
            .select("*")
            .eq("club_id", str(club_id))
            .eq("operation_key", str(operation_key))
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise LiveLadderPersistenceError("Unable to read the durable live-ladder operation ledger.") from exc


def _get_active_version_lease(
    supabase: Any,
    *,
    club_id: str,
    surface: str,
    expected_version: str,
) -> dict[str, Any] | None:
    """Return the operation currently owning one authoritative surface snapshot."""
    try:
        return _first(
            supabase.table(OPERATION_TABLE)
            .select("*")
            .eq("club_id", str(club_id))
            .eq("surface", str(surface))
            .eq("expected_version", str(expected_version))
            .in_("status", ["intent", "running", "mutated", "recovery_required"])
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise LiveLadderPersistenceError("Unable to inspect the durable version lease.") from exc


def _update_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_key: str,
    patch: dict[str, Any],
) -> dict[str, Any]:
    payload = {**_json_safe(patch), "updated_at": _now_iso()}
    try:
        updated = _first(
            supabase.table(OPERATION_TABLE)
            .update(payload)
            .eq("club_id", str(club_id))
            .eq("operation_key", str(operation_key))
            .execute()
        )
    except Exception as exc:
        raise LiveLadderUncertainError(
            "The operation ledger could not be updated. The write outcome is uncertain; do not retry blindly."
        ) from exc
    if updated is None:
        raise LiveLadderUncertainError(
            "The operation ledger update returned no row. The write outcome is uncertain; do not retry blindly."
        )
    return updated


def _public_operation(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "operation_key": str(row.get("operation_key") or ""),
        "surface": str(row.get("surface") or ""),
        "operation_type": str(row.get("operation_type") or ""),
        "entity_id": str(row.get("entity_id") or ""),
        "request_fingerprint": str(row.get("request_fingerprint") or ""),
        "expected_version": str(row.get("expected_version") or ""),
        "status": str(row.get("status") or "intent"),
        "attempt_count": int(row.get("attempt_count") or 0),
        "error_text": row.get("error_text"),
        "recovery": _json_safe(row.get("recovery_json") or {}),
        "completed_at": row.get("completed_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _operation_response(row: dict[str, Any], *, idempotent_replay: bool) -> dict[str, Any]:
    result = _json_safe(row.get("result_json") or {})
    if not isinstance(result, dict):
        result = {"result": result}
    return {
        **result,
        "ok": bool(result.get("ok", True)),
        "idempotent_replay": bool(idempotent_replay),
        "operation": _public_operation(row),
        "operation_key": str(row.get("operation_key") or ""),
        "request_fingerprint": str(row.get("request_fingerprint") or ""),
        "recovery": _json_safe(row.get("recovery_json") or {}),
    }


def _requires_surface_receipt_recovery(row: dict[str, Any]) -> bool:
    result = row.get("result_json")
    return isinstance(result, dict) and result.get("core_committed") is True


def replay_durable_admin_operation_if_present(
    supabase: Any,
    *,
    club_id: str,
    surface: str,
    operation_type: str,
    entity_id: str,
    idempotency_key: str,
    request_payload: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
    recover_incomplete: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> dict[str, Any] | None:
    """Recover an exact prior operation before rerunning mutable previews."""
    ensure_live_ladder_operation_schema_ready(supabase)
    operation_key = deterministic_operation_key(
        club_id=str(club_id),
        surface=str(surface),
        operation_type=str(operation_type),
        entity_id=str(entity_id),
        idempotency_key=idempotency_key,
    )
    fingerprint = stable_request_fingerprint(request_payload)
    existing = _get_operation(supabase, club_id=str(club_id), operation_key=operation_key)
    if existing is None:
        return None
    if str(existing.get("request_fingerprint") or "") != fingerprint:
        raise LiveLadderConflictError(
            "This idempotency key was already used with a different request. Reload and use a new key."
        )
    status = str(existing.get("status") or "")
    if status == "completed" and not _requires_surface_receipt_recovery(existing):
        return _operation_response(existing, idempotent_replay=True)
    if recover_incomplete is not None:
        recovered_result = recover_incomplete(existing)
        if recovered_result is not None:
            completed = _complete_recovered_domain_result(
                supabase,
                operation=existing,
                result=recovered_result,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
            )
            return _operation_response(completed, idempotent_replay=True)
    if status == "completed":
        raise LiveLadderUncertainError(
            "The operation was marked completed with only a database core receipt; "
            "verified surface recovery is still required."
        )
    if status in {"mutated", "recovery_required"} and existing.get("result_json"):
        if _requires_surface_receipt_recovery(existing):
            raise LiveLadderUncertainError(
                "The database core committed, but this surface receipt still requires "
                "verified recovery before the operation can be completed."
            )
        completed = _complete_operation_audit(
            supabase,
            operation=existing,
            actor_email=actor_email,
            actor_role=actor_role,
            source=source,
            recovered=True,
        )
        return _operation_response(completed, idempotent_replay=True)
    if status == "failed":
        raise LiveLadderConflictError(
            "This operation was rejected before domain mutation because its authoritative version became stale. "
            "Reload the Python state and use a new idempotency key."
        )
    raise LiveLadderUncertainError(
        "A durable operation with this key is incomplete and its outcome is uncertain. Do not retry the "
        "domain write; inspect/reconcile this operation and use Match Log/Replay History if needed."
    )


def _complete_operation_audit(
    supabase: Any,
    *,
    operation: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
    recovered: bool,
) -> dict[str, Any]:
    operation_key = str(operation.get("operation_key") or "")
    _required_audit(
        supabase,
        club_id=str(operation.get("club_id") or ""),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type=(
            "reconcile_live_ladder_operation_complete_admin"
            if recovered
            else "complete_live_ladder_operation_admin"
        ),
        entity_type="live_ladder_admin_operation",
        entity_id=operation_key,
        source=source,
        before_json={"status": operation.get("status")},
        after_json={
            "surface": operation.get("surface"),
            "operation_type": operation.get("operation_type"),
            "request_fingerprint": operation.get("request_fingerprint"),
            "recovered_response_loss": bool(recovered),
        },
        after_domain_mutation=True,
    )
    return _update_operation(
        supabase,
        club_id=str(operation.get("club_id") or ""),
        operation_key=operation_key,
        patch={
            "status": "completed",
            "error_text": None,
            "completed_at": operation.get("completed_at") or _now_iso(),
            "completion_audited_at": _now_iso(),
            "updated_by": str(actor_email or ""),
        },
    )


def _complete_recovered_domain_result(
    supabase: Any,
    *,
    operation: dict[str, Any],
    result: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    """Persist a surface-verified recovery result before completing its audit."""

    mutated = _update_operation(
        supabase,
        club_id=str(operation.get("club_id") or ""),
        operation_key=str(operation.get("operation_key") or ""),
        patch={
            "status": "mutated",
            "result_json": _json_safe(result),
            "error_text": None,
            "completed_at": operation.get("completed_at") or _now_iso(),
            "updated_by": str(actor_email or ""),
        },
    )
    return _complete_operation_audit(
        supabase,
        operation=mutated,
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
        recovered=True,
    )


def run_durable_admin_operation(
    supabase: Any,
    *,
    club_id: str,
    surface: str,
    operation_type: str,
    entity_id: str,
    idempotency_key: str,
    expected_version: str,
    current_version: str,
    request_payload: dict[str, Any],
    stored_request_json: dict[str, Any] | None = None,
    recovery: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
    mutate: Callable[[], dict[str, Any]],
    current_version_resolver: Callable[[], str] | None = None,
    recover_incomplete: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Run one FastAPI-authoritative write behind a durable, replayable intent."""

    ensure_live_ladder_operation_schema_ready(supabase)
    clean_expected = str(expected_version or "").strip()
    clean_current = str(current_version or "").strip()
    if not clean_expected:
        raise ValueError("expected_version is required; reload the authoritative Python preview/state.")
    key = normalize_idempotency_key(idempotency_key)
    operation_key = deterministic_operation_key(
        club_id=str(club_id),
        surface=str(surface),
        operation_type=str(operation_type),
        entity_id=str(entity_id),
        idempotency_key=key,
    )
    fingerprint = stable_request_fingerprint(request_payload)
    existing = _get_operation(supabase, club_id=str(club_id), operation_key=operation_key)
    if existing is not None:
        if str(existing.get("request_fingerprint") or "") != fingerprint:
            raise LiveLadderConflictError(
                "This idempotency key was already used with a different request. Reload and use a new key."
            )
        status = str(existing.get("status") or "")
        if status == "completed" and not _requires_surface_receipt_recovery(existing):
            return _operation_response(existing, idempotent_replay=True)
        if status == "failed":
            raise LiveLadderConflictError(
                "This operation was rejected before domain mutation because its authoritative version became stale. "
                "Reload the Python state and use a new idempotency key."
            )
        if recover_incomplete is not None:
            recovered_result = recover_incomplete(existing)
            if recovered_result is not None:
                completed = _complete_recovered_domain_result(
                    supabase,
                    operation=existing,
                    result=recovered_result,
                    actor_email=actor_email,
                    actor_role=actor_role,
                    source=source,
                )
                return _operation_response(completed, idempotent_replay=True)
        if status == "completed":
            raise LiveLadderUncertainError(
                "The operation was marked completed with only a database core receipt; "
                "verified surface recovery is still required."
            )
        if status in {"mutated", "recovery_required"} and existing.get("result_json"):
            if _requires_surface_receipt_recovery(existing):
                raise LiveLadderUncertainError(
                    "The database core committed, but this surface receipt still requires "
                    "verified recovery before the operation can be completed."
                )
            completed = _complete_operation_audit(
                supabase,
                operation=existing,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
                recovered=True,
            )
            return _operation_response(completed, idempotent_replay=True)
        raise LiveLadderUncertainError(
            "A durable operation with this key is incomplete and its outcome is uncertain. Do not retry the "
            "domain write; inspect/reconcile this operation and use Match Log/Replay History if needed."
        )
    if clean_expected != clean_current:
        raise LiveLadderConflictError(
            "The authoritative state changed after preview. Reload, review the Python result again, and use a new idempotency key."
        )

    now = _now_iso()
    recovery_payload = _json_safe(recovery)
    _required_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="intent_live_ladder_operation_admin",
        entity_type="live_ladder_admin_operation",
        entity_id=operation_key,
        source=source,
        before_json={"authoritative_version": clean_current},
        after_json={
            "surface": surface,
            "operation_type": operation_type,
            "entity_id": entity_id,
            "request_fingerprint": fingerprint,
            "expected_version": clean_expected,
            "recovery": recovery_payload,
        },
    )
    operation_payload = {
        "operation_key": operation_key,
        "club_id": str(club_id),
        "surface": str(surface),
        "operation_type": str(operation_type),
        "entity_id": str(entity_id),
        "idempotency_key": key,
        "request_fingerprint": fingerprint,
        "expected_version": clean_expected,
        "status": "intent",
        "request_json": _json_safe(
            stored_request_json
            if stored_request_json is not None
            else request_payload
        ),
        "result_json": {},
        "recovery_json": recovery_payload,
        "error_text": None,
        "attempt_count": 0,
        "created_by": str(actor_email or ""),
        "updated_by": str(actor_email or ""),
        "created_at": now,
        "updated_at": now,
    }
    try:
        operation = _first(supabase.table(OPERATION_TABLE).insert(operation_payload).execute())
    except Exception as exc:
        raced = _get_operation(supabase, club_id=str(club_id), operation_key=operation_key)
        if raced and str(raced.get("request_fingerprint") or "") == fingerprint:
            if (
                str(raced.get("status") or "") == "completed"
                and not _requires_surface_receipt_recovery(raced)
            ):
                return _operation_response(raced, idempotent_replay=True)
            if recover_incomplete is not None:
                recovered_result = recover_incomplete(raced)
                if recovered_result is not None:
                    completed = _complete_recovered_domain_result(
                        supabase,
                        operation=raced,
                        result=recovered_result,
                        actor_email=actor_email,
                        actor_role=actor_role,
                        source=source,
                    )
                    return _operation_response(completed, idempotent_replay=True)
            if str(raced.get("status") or "") == "completed":
                raise LiveLadderUncertainError(
                    "The concurrent operation was marked completed with only a database "
                    "core receipt; verified surface recovery is still required."
                )
            raise LiveLadderUncertainError(
                "A concurrent request owns this operation. Reload its durable status instead of retrying."
            ) from exc
        lease = _get_active_version_lease(
            supabase,
            club_id=str(club_id),
            surface=str(surface),
            expected_version=clean_expected,
        )
        if lease is not None:
            raise LiveLadderConflictError(
                "Another operation already owns this authoritative version. Reload its outcome and refresh the "
                "Python state before attempting a different write."
            ) from exc
        raise LiveLadderPersistenceError(
            "The durable operation intent could not be persisted. No domain mutation was attempted."
        ) from exc
    if operation is None:
        raise LiveLadderPersistenceError(
            "The durable operation intent returned no row. No domain mutation was attempted."
        )
    operation = _update_operation(
        supabase,
        club_id=str(club_id),
        operation_key=operation_key,
        patch={"status": "running", "attempt_count": 1, "updated_by": str(actor_email or "")},
    )
    if current_version_resolver is not None:
        try:
            observed_version = str(current_version_resolver() or "").strip()
        except Exception as exc:
            _update_operation(
                supabase,
                club_id=str(club_id),
                operation_key=operation_key,
                patch={
                    "status": "failed",
                    "error_text": "authoritative version recheck failed before domain mutation",
                    "updated_by": str(actor_email or ""),
                },
            )
            _required_audit(
                supabase,
                club_id=str(club_id),
                actor_email=actor_email,
                actor_role=actor_role,
                action_type="fail_live_ladder_operation_admin",
                entity_type="live_ladder_admin_operation",
                entity_id=operation_key,
                source=source,
                before_json={"status": "running"},
                after_json={
                    "status": "failed",
                    "domain_mutation_attempted": False,
                    "reason": "authoritative_version_recheck_failed",
                },
            )
            raise LiveLadderPersistenceError(
                "The authoritative version could not be rechecked. No domain mutation was attempted."
            ) from exc
        if observed_version != clean_expected:
            operation = _update_operation(
                supabase,
                club_id=str(club_id),
                operation_key=operation_key,
                patch={
                    "status": "failed",
                    "error_text": "authoritative version changed before domain mutation",
                    "updated_by": str(actor_email or ""),
                },
            )
            _required_audit(
                supabase,
                club_id=str(club_id),
                actor_email=actor_email,
                actor_role=actor_role,
                action_type="fail_live_ladder_operation_admin",
                entity_type="live_ladder_admin_operation",
                entity_id=operation_key,
                source=source,
                before_json={"status": "running", "expected_version": clean_expected},
                after_json={
                    "status": "failed",
                    "domain_mutation_attempted": False,
                    "observed_version": observed_version,
                    "reason": "stale_after_intent",
                },
            )
            raise LiveLadderConflictError(
                "The authoritative state changed while acquiring the write lease. No domain mutation was "
                "attempted; reload the Python state and use a new idempotency key."
            )
    try:
        result = _json_safe(mutate())
    except Exception as exc:
        try:
            operation = _update_operation(
                supabase,
                club_id=str(club_id),
                operation_key=operation_key,
                patch={
                    "status": "recovery_required",
                    "error_text": str(exc)[:1000],
                    "updated_by": str(actor_email or ""),
                },
            )
            _required_audit(
                supabase,
                club_id=str(club_id),
                actor_email=actor_email,
                actor_role=actor_role,
                action_type="fail_live_ladder_operation_admin",
                entity_type="live_ladder_admin_operation",
                entity_id=operation_key,
                source=source,
                before_json={"status": "running"},
                after_json={
                    "status": "recovery_required",
                    "surface": surface,
                    "operation_type": operation_type,
                    "request_fingerprint": fingerprint,
                    "error": str(exc)[:500],
                    "recovery": recovery_payload,
                },
                after_domain_mutation=True,
            )
        except Exception as audit_exc:
            raise LiveLadderUncertainError(
                "The operation failed or is uncertain, and its required failure audit also failed. Do not "
                "retry blindly; inspect Match Log/Replay History and the operation ledger."
            ) from audit_exc
        raise LiveLadderUncertainError(
            "The operation did not return a verified completion and may have changed domain state. Do not "
            "retry blindly; reconcile the durable operation and inspect Match Log/Replay History."
        ) from exc

    operation = _update_operation(
        supabase,
        club_id=str(club_id),
        operation_key=operation_key,
        patch={
            "status": "mutated",
            "result_json": result,
            "error_text": None,
            "completed_at": _now_iso(),
            "updated_by": str(actor_email or ""),
        },
    )
    try:
        operation = _complete_operation_audit(
            supabase,
            operation=operation,
            actor_email=actor_email,
            actor_role=actor_role,
            source=source,
            recovered=False,
        )
    except Exception:
        try:
            _update_operation(
                supabase,
                club_id=str(club_id),
                operation_key=operation_key,
                patch={"status": "recovery_required", "updated_by": str(actor_email or "")},
            )
        except Exception:
            pass
        raise
    return _operation_response(operation, idempotent_replay=False)


def get_durable_admin_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_key: str,
    surface: str,
) -> dict[str, Any]:
    ensure_live_ladder_operation_schema_ready(supabase)
    row = _get_operation(supabase, club_id=str(club_id), operation_key=str(operation_key))
    if row is None or str(row.get("surface") or "") != str(surface):
        raise ValueError("durable operation not found")
    payload = {
        "ok": True,
        "mode": "live_ladder_operation_status",
        "operation": _public_operation(row),
        "recovery": _json_safe(row.get("recovery_json") or {}),
    }
    if str(row.get("status") or "") == "completed":
        payload["completed_result"] = _json_safe(row.get("result_json") or {})
    else:
        payload["uncertainty"] = (
            "This operation is not durably complete. Do not infer success or retry the write from this status alone."
        )
    return payload


def reconcile_durable_admin_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_key: str,
    surface: str,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    expected_confirmation: str,
    source: str,
    recover_incomplete: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    if str(confirmation_text or "").strip().upper() != str(expected_confirmation).upper():
        raise ValueError(f"Type {expected_confirmation} to reconcile this interrupted operation.")
    ensure_live_ladder_operation_schema_ready(supabase)
    row = _get_operation(supabase, club_id=str(club_id), operation_key=str(operation_key))
    if row is None or str(row.get("surface") or "") != str(surface):
        raise ValueError("durable operation not found")
    if (
        str(row.get("status") or "") == "completed"
        and not _requires_surface_receipt_recovery(row)
    ):
        return _operation_response(row, idempotent_replay=True)
    if recover_incomplete is not None:
        recovered_result = recover_incomplete(row)
        if recovered_result is not None:
            completed = _complete_recovered_domain_result(
                supabase,
                operation=row,
                result=recovered_result,
                actor_email=actor_email,
                actor_role=actor_role,
                source=source,
            )
            return _operation_response(completed, idempotent_replay=True)
    if str(row.get("status") or "") == "completed":
        raise LiveLadderUncertainError(
            "The operation was marked completed with only a database core receipt; "
            "verified surface recovery is still required."
        )
    if row.get("result_json"):
        if _requires_surface_receipt_recovery(row):
            raise LiveLadderUncertainError(
                "The database core committed, but this surface receipt still requires "
                "verified recovery before the operation can be completed."
            )
        completed = _complete_operation_audit(
            supabase,
            operation=row,
            actor_email=actor_email,
            actor_role=actor_role,
            source=source,
            recovered=True,
        )
        return _operation_response(completed, idempotent_replay=True)
    _required_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="inspect_uncertain_live_ladder_operation_admin",
        entity_type="live_ladder_admin_operation",
        entity_id=str(operation_key),
        source=source,
        before_json={"status": row.get("status")},
        after_json={
            "surface": surface,
            "request_fingerprint": row.get("request_fingerprint"),
            "recovery": row.get("recovery_json") or {},
            "outcome": "still_uncertain_no_result_snapshot",
        },
    )
    return {
        "ok": False,
        "mode": "live_ladder_operation_recovery_required",
        "operation": _public_operation(row),
        "outcome": "uncertain",
        "uncertainty": (
            "No durable completion snapshot exists, so FastAPI cannot truthfully claim success or safe retry. "
            "Use the concrete Match Log/Replay History handoff before any new operation."
        ),
        "recovery": _json_safe(row.get("recovery_json") or {}),
    }
