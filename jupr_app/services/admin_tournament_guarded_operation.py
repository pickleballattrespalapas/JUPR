from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any, Callable

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_admin_operations import build_tournament_admin_operation_request


TOURNAMENT_ADMIN_OPERATION_TABLE = "tournament_admin_operations"
SURFACE_MUTATION_FLAGS = {
    "tournament": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS",
    "setup": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS",
    "registration": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS",
    "import_handoff": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF",
    "operations": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
    "tournament_live": "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES",
}
TRUTHY = {"1", "true", "yes", "y", "on"}


class StaleTournamentAdminStateError(ValueError):
    """The reviewed Tournament Admin state is no longer current."""


class TournamentAdminRecoveryRequiredError(RuntimeError):
    """A mutation may have completed and must be reconciled, not repeated."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in TRUTHY


def tournament_admin_guarded_runtime_enabled(surface: str) -> bool:
    """Return whether the durable staging mutation path is active.

    Local/test callers retain the legacy in-memory/fake-friendly path. Every
    deployed staging mutation must opt into its own surface flag; production is
    deliberately refused by ``require_tournament_admin_mutation_runtime``.
    """

    environment = os.getenv("JUPR_ENV", "").strip().lower()
    flag = SURFACE_MUTATION_FLAGS.get(str(surface), "")
    return environment == "staging" and bool(flag) and _truthy_env(flag)


def require_tournament_admin_mutation_runtime(surface: str) -> None:
    environment = os.getenv("JUPR_ENV", "").strip().lower()
    flag = SURFACE_MUTATION_FLAGS.get(str(surface))
    if flag is None:
        raise PermissionError("Unknown Tournament Admin mutation surface.")
    if environment == "production":
        raise PermissionError(
            "Tournament Admin mutations are staging-only until manual parity acceptance is complete."
        )
    if environment == "staging":
        if not _truthy_env(flag):
            raise PermissionError(f"Tournament Admin mutation is disabled. Enable {flag} only for the approved staging exercise.")
        if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
            raise RuntimeError(
                "Tournament Admin staging mutations require SUPABASE_SERVICE_ROLE_KEY on FastAPI. Never expose this secret to Next or the browser."
            )


def tournament_admin_mutation_status() -> dict[str, Any]:
    environment = os.getenv("JUPR_ENV", "").strip().lower() or "local"
    return {
        "environment": environment,
        "staging_only": True,
        "service_role_ready": bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()),
        "surface_flags": {
            surface: {"name": flag, "enabled": environment == "staging" and _truthy_env(flag)}
            for surface, flag in SURFACE_MUTATION_FLAGS.items()
        },
    }


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _get_operation(supabase: Any, *, club_id: str, operation_key: str) -> dict[str, Any] | None:
    try:
        rows = _safe_rows(
            supabase.table(TOURNAMENT_ADMIN_OPERATION_TABLE)
            .select("*")
            .eq("club_id", str(club_id))
            .eq("operation_key", str(operation_key))
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Tournament Admin durable operation storage is unavailable. Apply the order-26 migration before enabling staging mutations."
        ) from exc
    return rows[0] if rows else None


def _get_operation_by_idempotency_key(
    supabase: Any,
    *,
    club_id: str,
    surface: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    try:
        rows = _safe_rows(
            supabase.table(TOURNAMENT_ADMIN_OPERATION_TABLE)
            .select("*")
            .eq("club_id", str(club_id))
            .eq("surface", str(surface))
            .eq("client_idempotency_key", str(idempotency_key))
            .limit(1)
            .execute()
        )
    except Exception as exc:
        raise RuntimeError(
            "Tournament Admin idempotency storage is unavailable. Apply the surface migration before enabling staging writes."
        ) from exc
    return rows[0] if rows else None


def _insert_operation(supabase: Any, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        rows = _safe_rows(supabase.table(TOURNAMENT_ADMIN_OPERATION_TABLE).insert(payload).execute())
    except Exception as exc:
        detail = str(exc).lower()
        if "idx_tournament_admin_operations_active_lock" in detail or "duplicate key" in detail or "23505" in detail:
            raise StaleTournamentAdminStateError(
                "Another Tournament Admin operation already owns this tournament lock. Wait for it to complete or reconcile before reloading."
            ) from exc
        raise RuntimeError(
            "Tournament Admin could not persist mutation intent; no domain mutation was attempted."
        ) from exc
    if not rows:
        raise RuntimeError("Tournament Admin could not persist mutation intent; no domain mutation was attempted.")
    return rows[0]


def _update_operation(
    supabase: Any,
    *,
    club_id: str,
    operation_key: str,
    patch: dict[str, Any],
) -> dict[str, Any]:
    try:
        rows = _safe_rows(
            supabase.table(TOURNAMENT_ADMIN_OPERATION_TABLE)
            .update({**patch, "updated_at": _now_iso()})
            .eq("club_id", str(club_id))
            .eq("operation_key", str(operation_key))
            .execute()
        )
    except Exception as exc:
        raise TournamentAdminRecoveryRequiredError(
            "Tournament Admin may already have changed data, but recovery state did not persist. Reload the authoritative detail before any retry."
        ) from exc
    if not rows:
        raise TournamentAdminRecoveryRequiredError(
            "Tournament Admin may already have changed data, but recovery state did not persist. Reload the authoritative detail before any retry."
        )
    return rows[0]


def _write_required_audit(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_type: str,
    entity_id: str,
    operation: dict[str, Any],
    source: str,
    before_json: Any = None,
    after_json: Any = None,
) -> None:
    marker = {
        "operation_key": operation["operation_key"],
        "request_fingerprint": operation["request_fingerprint"],
        "operation_status": action_type.rsplit("_", 1)[-1],
    }
    result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            action_type=action_type,
            entity_type=str(entity_type),
            entity_id=str(entity_id),
            before_json=before_json,
            after_json={"audit_marker": marker, "value": after_json},
            source_page=source,
            flagged_for_review=True,
        ),
    )
    if not result.ok:
        raise RuntimeError("Required Tournament Admin audit record could not be persisted.")


def _attempt_failure_audit(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action: str,
    entity_type: str,
    entity_id: str,
    operation: dict[str, Any],
    source: str,
    error_text: str,
) -> Exception | None:
    try:
        _write_required_audit(
            supabase,
            club_id=club_id,
            actor_email=actor_email,
            actor_role=actor_role,
            action_type=f"{action}_failure",
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation,
            source=source,
            after_json={"error": error_text},
        )
    except Exception as exc:  # the caller must keep the outcome recovery-required
        return exc
    return None


def _public_result(operation: dict[str, Any], result: dict[str, Any], *, replay: bool, reconciled: bool = False) -> dict[str, Any]:
    return {
        **dict(result or {}),
        "operation_key": str(operation.get("operation_key") or ""),
        "request_fingerprint": str(operation.get("request_fingerprint") or ""),
        "client_idempotency_key": str(
            operation.get("client_idempotency_key")
            or (operation.get("request_json") or {}).get("idempotency_key")
            or ""
        ),
        "idempotent_replay": bool(replay),
        "reconciled": bool(reconciled),
    }


def get_tournament_admin_operation_record(
    supabase: Any,
    *,
    club_id: str,
    operation_key: str,
) -> dict[str, Any] | None:
    """Return one server-private operation for an authorized service caller."""

    return _get_operation(
        supabase,
        club_id=str(club_id),
        operation_key=str(operation_key),
    )


def get_tournament_admin_operation_record_by_idempotency_key(
    supabase: Any,
    *,
    club_id: str,
    surface: str,
    idempotency_key: str,
) -> dict[str, Any] | None:
    """Return one private operation for an exact client retry.

    Surface services use this before rebuilding server-derived request evidence.
    If the first attempt changed domain state, the original evidence must be
    reused so the guarded runner can reconcile the same request instead of
    deriving a different request from post-mutation state.
    """

    return _get_operation_by_idempotency_key(
        supabase,
        club_id=str(club_id),
        surface=str(surface),
        idempotency_key=str(idempotency_key),
    )


def reconcile_tournament_admin_guarded_operation(
    supabase: Any,
    *,
    club_id: str,
    surface: str,
    operation_key: str,
    entity_type: str,
    entity_id: str,
    actor_email: str,
    actor_role: str,
    source: str,
    verify_outcome: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Resolve an interrupted mutation from authoritative domain evidence.

    The verifier may report ``completed``, ``not_applied``, or ``uncertain``.
    Only the first two release the durable lock, and both require a dedicated
    audit record before the operation status is changed.
    """

    require_tournament_admin_mutation_runtime(surface)
    operation = _get_operation(
        supabase,
        club_id=str(club_id),
        operation_key=str(operation_key),
    )
    if not operation:
        raise ValueError("Tournament Admin operation not found for this club.")
    if str(operation.get("surface") or "") != str(surface):
        raise ValueError("Tournament Admin operation belongs to another surface.")
    if str(operation.get("entity_type") or "") != str(entity_type) or str(operation.get("entity_id") or "") != str(entity_id):
        raise ValueError("Tournament Admin operation does not belong to this draw.")

    status = str(operation.get("status") or "")
    stored_result = operation.get("result_json")
    if status == "completed" and isinstance(stored_result, dict):
        return {
            **_public_result(operation, stored_result, replay=True, reconciled=True),
            "recovery_disposition": "already_completed",
            "recovery_evidence": {},
        }
    if status == "failed":
        raise ValueError("This Tournament Admin operation is already closed as not applied or failed.")

    if status in {"mutated", "recovery_required"} and isinstance(stored_result, dict) and stored_result:
        verification = {
            "status": "completed",
            "result": stored_result,
            "evidence": {"source": "stored_result"},
        }
    else:
        verification = dict(verify_outcome(dict(operation)) or {})
    disposition = str(verification.get("status") or "uncertain").strip().lower()
    evidence = verification.get("evidence") if isinstance(verification.get("evidence"), dict) else {}
    if disposition not in {"completed", "not_applied"}:
        raise TournamentAdminRecoveryRequiredError(
            "Authoritative Tournament Admin evidence cannot prove that this operation fully completed or never started. Keep its scope locked, use the documented fallback, and inspect audit/replay evidence before any new write."
        )

    action = str(operation.get("action") or "tournament_admin")
    audit_action = f"{action}_reconciliation" if disposition == "completed" else f"{action}_recovery_not_applied"
    _write_required_audit(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type=audit_action,
        entity_type=entity_type,
        entity_id=entity_id,
        operation=operation,
        source=source,
        before_json={"status": status, "error": operation.get("error_text")},
        after_json={"disposition": disposition, "evidence": evidence},
    )

    attempt_count = max(1, int(operation.get("attempt_count") or 1)) + 1
    if disposition == "completed":
        verified_result = verification.get("result") if isinstance(verification.get("result"), dict) else {}
        completed = _update_operation(
            supabase,
            club_id=str(club_id),
            operation_key=str(operation_key),
            patch={
                "status": "completed",
                "result_json": verified_result,
                "error_text": None,
                "attempt_count": attempt_count,
                "updated_by": str(actor_email or ""),
                "completion_audited_at": _now_iso(),
            },
        )
        return {
            **_public_result(completed, verified_result, replay=True, reconciled=True),
            "recovery_disposition": "completed",
            "recovery_evidence": evidence,
        }

    closed = _update_operation(
        supabase,
        club_id=str(club_id),
        operation_key=str(operation_key),
        patch={
            "status": "failed",
            "result_json": {},
            "error_text": "authoritative recovery verified no domain effect",
            "attempt_count": attempt_count,
            "updated_by": str(actor_email or ""),
        },
    )
    return {
        "ok": True,
        "operation_key": str(closed.get("operation_key") or operation_key),
        "request_fingerprint": str(closed.get("request_fingerprint") or ""),
        "client_idempotency_key": str(closed.get("client_idempotency_key") or ""),
        "idempotent_replay": True,
        "reconciled": True,
        "recovery_disposition": "not_applied",
        "recovery_evidence": evidence,
    }


def run_tournament_admin_guarded_operation(
    supabase: Any,
    *,
    club_id: str,
    surface: str,
    action: str,
    entity_type: str,
    entity_id: str,
    lock_scope: str | None = None,
    expected_state: str,
    current_state: Callable[[], str],
    payload: dict[str, Any],
    actor_email: str,
    actor_role: str,
    source: str,
    preflight: Callable[[], Any] | None = None,
    reconcile: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
    mutate: Callable[[], dict[str, Any]],
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Run one tournament mutation with intent, replay, and recovery state.

    This intentionally remains tournament-specific. Order 23 can later adapt
    the persistence/audit hooks to its shared guarded-operation abstraction
    without changing this module's request or response contract.
    """

    require_tournament_admin_mutation_runtime(surface)
    reviewed_state = str(expected_state or "").strip()
    if not reviewed_state:
        raise StaleTournamentAdminStateError("A reviewed state version is required. Reload before submitting this mutation.")
    operation_request = build_tournament_admin_operation_request(
        club_id=str(club_id),
        surface=str(surface),
        action=str(action),
        entity_type=str(entity_type),
        entity_id=str(entity_id),
        lock_scope=str(lock_scope or entity_id),
        expected_state=reviewed_state,
        payload=payload,
        idempotency_key=idempotency_key,
    )
    existing = _get_operation(
        supabase,
        club_id=str(club_id),
        operation_key=operation_request["operation_key"],
    )
    clean_idempotency_key = str(idempotency_key or "").strip()
    if clean_idempotency_key:
        idempotent_existing = _get_operation_by_idempotency_key(
            supabase,
            club_id=str(club_id),
            surface=str(surface),
            idempotency_key=clean_idempotency_key,
        )
        if idempotent_existing:
            if str(idempotent_existing.get("request_fingerprint") or "") != operation_request["request_fingerprint"]:
                raise ValueError(
                    "This idempotency key was already used for a different Tournament Admin request. Reload and create a new command."
                )
            existing = idempotent_existing
    if existing:
        if str(existing.get("request_fingerprint") or "") != operation_request["request_fingerprint"]:
            raise ValueError("Tournament Admin operation key conflicts with a different request.")
        stored_result = existing.get("result_json")
        if str(existing.get("status") or "") == "completed" and isinstance(stored_result, dict):
            return _public_result(existing, stored_result, replay=True)
        status = str(existing.get("status") or "")
        has_reconcilable_result = isinstance(stored_result, dict) and bool(stored_result)
        if status == "recovery_required" and not has_reconcilable_result and reconcile is not None:
            # A durable intent exists but the response/result marker was lost.  The
            # callback is deliberately read-only: it may reconstruct a result only
            # from exact authoritative evidence and must never repeat the mutation.
            reconciled_result = reconcile(existing)
            if isinstance(reconciled_result, dict) and reconciled_result:
                existing = _update_operation(
                    supabase,
                    club_id=str(club_id),
                    operation_key=operation_request["operation_key"],
                    patch={
                        "status": "mutated",
                        "result_json": reconciled_result,
                        "error_text": None,
                    },
                )
                stored_result = reconciled_result
                status = "mutated"
                has_reconcilable_result = True
        if (status == "mutated" and isinstance(stored_result, dict)) or (status == "recovery_required" and has_reconcilable_result):
            try:
                _write_required_audit(
                    supabase,
                    club_id=str(club_id),
                    actor_email=actor_email,
                    actor_role=actor_role,
                    action_type=f"{action}_completion",
                    entity_type=entity_type,
                    entity_id=entity_id,
                    operation=operation_request,
                    source=source,
                    after_json={"result": stored_result, "reconciled": True},
                )
            except Exception as exc:
                failure_audit_error = _attempt_failure_audit(
                    supabase,
                    club_id=str(club_id),
                    actor_email=actor_email,
                    actor_role=actor_role,
                    action=action,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    operation=operation_request,
                    source=source,
                    error_text=str(exc)[:1000],
                )
                suffix = " The failure audit also did not persist." if failure_audit_error is not None else " A failure audit was recorded."
                raise TournamentAdminRecoveryRequiredError(
                    "Tournament Admin could not persist the reconciliation completion audit. The stored result remains recovery-required; do not repeat the domain mutation."
                    + suffix
                ) from exc
            try:
                completed = _update_operation(
                    supabase,
                    club_id=str(club_id),
                    operation_key=operation_request["operation_key"],
                    patch={"status": "completed", "completion_audited_at": _now_iso(), "error_text": None},
                )
            except Exception as exc:
                failure_audit_error = _attempt_failure_audit(
                    supabase,
                    club_id=str(club_id),
                    actor_email=actor_email,
                    actor_role=actor_role,
                    action=action,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    operation=operation_request,
                    source=source,
                    error_text=str(exc)[:1000],
                )
                suffix = " The failure audit also did not persist." if failure_audit_error is not None else " A failure audit was recorded."
                raise TournamentAdminRecoveryRequiredError(
                    "Tournament Admin reconciliation was audited, but the durable completed marker did not persist. Reload and retry the identical request only to reconcile."
                    + suffix
                ) from exc
            return _public_result(completed, stored_result, replay=True, reconciled=True)
        raise TournamentAdminRecoveryRequiredError(
            "A prior attempt with this operation key has unresolved recovery state. "
            f"Operation key: {operation_request['operation_key']}. Reload authoritative detail; do not repeat the mutation."
        )

    authoritative_state = str(current_state() or "").strip()
    if authoritative_state != reviewed_state:
        raise StaleTournamentAdminStateError(
            "Tournament Admin data changed after it was loaded. Reload the authoritative detail, review the impact, and submit again."
        )
    if preflight is not None:
        preflight()

    now = _now_iso()
    operation_payload = {
        "operation_key": operation_request["operation_key"],
        "request_fingerprint": operation_request["request_fingerprint"],
        "club_id": str(club_id),
        "surface": str(surface),
        "action": str(action),
        "entity_type": str(entity_type),
        "entity_id": str(entity_id),
        "lock_scope": str(lock_scope or entity_id),
        "expected_state": reviewed_state,
        "status": "intent",
        "request_json": operation_request,
        "result_json": {},
        "attempt_count": 1,
        "created_by": str(actor_email or ""),
        "updated_by": str(actor_email or ""),
        "created_at": now,
        "updated_at": now,
    }
    if clean_idempotency_key:
        operation_payload["client_idempotency_key"] = clean_idempotency_key
    operation = _insert_operation(
        supabase,
        operation_payload,
    )
    locked_state = str(current_state() or "").strip()
    if locked_state != reviewed_state:
        _update_operation(
            supabase,
            club_id=str(club_id),
            operation_key=operation_request["operation_key"],
            patch={"status": "failed", "error_text": "state changed before durable audit intent"},
        )
        raise StaleTournamentAdminStateError(
            "Tournament Admin data changed while the operation lock was being acquired. Reload before submitting a new mutation."
        )
    try:
        _write_required_audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type=f"{action}_intent",
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation_request,
            source=source,
            before_json={"expected_state": reviewed_state},
            after_json={"payload": payload},
        )
    except Exception:
        _update_operation(
            supabase,
            club_id=str(club_id),
            operation_key=operation_request["operation_key"],
            patch={"status": "failed", "error_text": "required audit intent unavailable"},
        )
        raise RuntimeError("Required Tournament Admin audit intent could not be persisted; no domain mutation was attempted.")

    try:
        result = dict(mutate() or {})
    except StaleTournamentAdminStateError as exc:
        error_text = str(exc)[:1000]
        try:
            _update_operation(
                supabase,
                club_id=str(club_id),
                operation_key=operation_request["operation_key"],
                patch={"status": "failed", "error_text": error_text},
            )
        except Exception as state_exc:
            raise TournamentAdminRecoveryRequiredError(
                "A SQL compare-and-swap rejected stale Tournament Admin state without a domain write, but the failed marker did not persist. Reconcile the operation before continuing."
            ) from state_exc
        audit_error = _attempt_failure_audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation_request,
            source=source,
            error_text=error_text,
        )
        if audit_error is not None:
            raise TournamentAdminRecoveryRequiredError(
                "A SQL compare-and-swap rejected stale Tournament Admin state without a domain write, but its required failure audit did not persist."
            ) from audit_error
        raise
    except Exception as exc:
        error_text = str(exc)[:1000]
        recovery_state_error: Exception | None = None
        try:
            _update_operation(
                supabase,
                club_id=str(club_id),
                operation_key=operation_request["operation_key"],
                patch={"status": "recovery_required", "error_text": error_text},
            )
        except Exception as state_exc:
            recovery_state_error = state_exc
        audit_error = _attempt_failure_audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation_request,
            source=source,
            error_text=error_text,
        )
        suffix_parts = []
        if recovery_state_error is not None:
            suffix_parts.append("The recovery state could not be persisted.")
        if audit_error is not None:
            suffix_parts.append("The failure audit also did not persist.")
        if not suffix_parts:
            suffix_parts.append("A failure audit and recovery state were recorded.")
        raise TournamentAdminRecoveryRequiredError(
            "Tournament Admin raised after durable intent, so the write outcome may be partial or response-lost. Reload authoritative state and reconcile; do not submit a new operation."
            + " "
            + " ".join(suffix_parts)
            + f" Operation key: {operation_request['operation_key']}."
        ) from exc

    try:
        mutated = _update_operation(
            supabase,
            club_id=str(club_id),
            operation_key=operation_request["operation_key"],
            patch={"status": "mutated", "result_json": result, "error_text": None},
        )
    except Exception as exc:
        audit_error = _attempt_failure_audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation_request,
            source=source,
            error_text=str(exc)[:1000],
        )
        suffix = " The failure audit also did not persist." if audit_error is not None else " A failure audit was recorded."
        raise TournamentAdminRecoveryRequiredError(
            "Tournament Admin returned from the domain mutation, but its durable result did not persist. Reload authoritative state and reconcile; do not retry blindly."
            + suffix
        ) from exc
    try:
        _write_required_audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type=f"{action}_completion",
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation_request,
            source=source,
            after_json={"result": result},
        )
    except Exception as exc:
        recovery_state_error: Exception | None = None
        try:
            _update_operation(
                supabase,
                club_id=str(club_id),
                operation_key=operation_request["operation_key"],
                patch={"status": "recovery_required", "error_text": str(exc)[:1000]},
            )
        except Exception as state_exc:
            recovery_state_error = state_exc
        failure_audit_error = _attempt_failure_audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation_request,
            source=source,
            error_text=str(exc)[:1000],
        )
        suffix = ""
        if recovery_state_error is not None:
            suffix += " Recovery state could not be updated."
        if failure_audit_error is not None:
            suffix += " The failure audit also did not persist."
        raise TournamentAdminRecoveryRequiredError(
            "Tournament Admin data may already have changed, but completion audit failed. Reload and retry the identical request to reconcile; do not submit a new mutation."
            + suffix
        ) from exc
    try:
        completed = _update_operation(
            supabase,
            club_id=str(club_id),
            operation_key=operation_request["operation_key"],
            patch={"status": "completed", "completion_audited_at": _now_iso(), "error_text": None},
        )
    except Exception as exc:
        failure_audit_error = _attempt_failure_audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            operation=operation_request,
            source=source,
            error_text=str(exc)[:1000],
        )
        suffix = " The failure audit also did not persist." if failure_audit_error is not None else " A failure audit was recorded."
        raise TournamentAdminRecoveryRequiredError(
            "Tournament Admin completion was audited, but the durable completed marker did not persist. Reload and retry the identical request only to reconcile."
            + suffix
        ) from exc
    return _public_result(completed, result, replay=False)
