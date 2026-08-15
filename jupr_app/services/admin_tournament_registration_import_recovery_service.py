from __future__ import annotations

from typing import Any

from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
)
from jupr_app.services.admin_tournament_guarded_operation import (
    TOURNAMENT_ADMIN_INTENT_AUDIT_UNAVAILABLE_ERROR,
    TOURNAMENT_ADMIN_RECOVERY_TOMBSTONE_ERROR,
    TOURNAMENT_ADMIN_STALE_CAS_NO_WRITE_PREFIX,
    TOURNAMENT_ADMIN_STATE_CHANGED_BEFORE_INTENT_ERROR,
    TournamentAdminRecoveryRequiredError,
    reconcile_tournament_admin_guarded_operation,
    require_tournament_admin_mutation_runtime,
    reserve_tournament_admin_recovery_tombstone,
)
REGISTRATION_IMPORT_ACTION = "ops_registration_import"
REGISTRATION_IMPORT_RECONCILE_CONFIRMATION = "RECONCILE REGISTRATION IMPORT"
REGISTRATION_IMPORT_SURFACE = "operations"
_RUNNER_PROVEN_NO_WRITE_ERRORS = {
    TOURNAMENT_ADMIN_STATE_CHANGED_BEFORE_INTENT_ERROR,
    TOURNAMENT_ADMIN_INTENT_AUDIT_UNAVAILABLE_ERROR,
}


def _validate_operation_scope(
    operation: dict[str, Any],
    *,
    tournament_id: str,
    draw_id: str,
) -> None:
    if str(operation.get("surface") or "") != REGISTRATION_IMPORT_SURFACE:
        raise ValueError("Operation is not a Tournament Ops registration import.")
    if str(operation.get("action") or "") != REGISTRATION_IMPORT_ACTION:
        raise ValueError("Operation is not a Tournament Ops registration import.")
    if (
        str(operation.get("entity_type") or "") != "tournament_event_draw"
        or str(operation.get("entity_id") or "") != str(draw_id)
    ):
        raise ValueError("Registration import operation does not belong to this draw.")
    if str(operation.get("lock_scope") or "") != str(tournament_id):
        raise ValueError("Registration import operation does not belong to this tournament.")


def _not_applied_result(operation: dict[str, Any]) -> dict[str, Any]:
    """Return an idempotent result after a prior proof closed the operation."""

    return {
        "ok": True,
        "operation_key": str(operation.get("operation_key") or ""),
        "request_fingerprint": str(operation.get("request_fingerprint") or ""),
        "client_idempotency_key": str(operation.get("client_idempotency_key") or ""),
        "idempotent_replay": True,
        "reconciled": True,
        "recovery_disposition": "not_applied",
        "recovery_evidence": {"source": "stored_not_applied_reconciliation"},
    }


def reconcile_admin_tournament_registration_import_operation(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    draw_id: str,
    operation_reference: str,
    retained_request: dict[str, Any],
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str = "next_tournament_ops_registration_import_reconcile",
) -> dict[str, Any]:
    """Prove a registration import completed or never changed domain state.

    This endpoint never reruns the import.  When no operation is visible, it
    first reserves the exact retained request as a non-runnable tombstone.
    A normal interrupted operation closes only from a stored result. An empty
    recovery-required row remains uncertain: a momentarily unchanged readback
    cannot fence an original PostgREST request that may still commit later.
    """

    require_tournament_admin_mutation_runtime(REGISTRATION_IMPORT_SURFACE)
    if str(confirmation_text or "").strip() != REGISTRATION_IMPORT_RECONCILE_CONFIRMATION:
        raise ValueError(
            f"Type {REGISTRATION_IMPORT_RECONCILE_CONFIRMATION} exactly to reconcile this registration import."
        )
    retained_idempotency_key = str(retained_request.get("idempotency_key") or "").strip()
    retained_expected_state = str(
        retained_request.get("expected_state_fingerprint") or ""
    ).strip()
    if not retained_idempotency_key:
        raise ValueError("The retained registration import idempotency key is required.")
    retained_operation_payload = {
        "draw_id": str(draw_id),
        "import_mode": str(retained_request.get("import_mode") or "REPLACE"),
        "expected_draw_updated_at": retained_request.get("expected_draw_updated_at"),
    }
    canonical_request = build_tournament_admin_operation_request(
        club_id=str(club_id),
        surface=REGISTRATION_IMPORT_SURFACE,
        action=REGISTRATION_IMPORT_ACTION,
        entity_type="tournament_event_draw",
        entity_id=str(draw_id),
        lock_scope=str(tournament_id),
        expected_state=retained_expected_state,
        payload=retained_operation_payload,
        idempotency_key=retained_idempotency_key,
    )
    if str(operation_reference) not in {
        retained_idempotency_key,
        str(canonical_request["operation_key"]),
    }:
        # Reject a route/body mismatch before reserving any ledger row.  The
        # caller must not be able to write a tombstone for one request while
        # presenting another request's recovery reference.
        raise ValueError(
            "Registration import operation reference does not match the retained request."
        )
    operation, tombstone_created = reserve_tournament_admin_recovery_tombstone(
        supabase,
        club_id=str(club_id),
        surface=REGISTRATION_IMPORT_SURFACE,
        action=REGISTRATION_IMPORT_ACTION,
        entity_type="tournament_event_draw",
        entity_id=str(draw_id),
        lock_scope=str(tournament_id),
        expected_state=retained_expected_state,
        payload=retained_operation_payload,
        idempotency_key=retained_idempotency_key,
        actor_email=str(actor_email),
    )
    _validate_operation_scope(
        operation,
        tournament_id=str(tournament_id),
        draw_id=str(draw_id),
    )
    valid_references = {
        str(operation.get("operation_key") or ""),
        str(operation.get("client_idempotency_key") or ""),
    }
    if str(operation_reference) not in valid_references:
        raise ValueError("Registration import operation reference does not match the retained request.")

    status = str(operation.get("status") or "")
    error_text = str(operation.get("error_text") or "")
    if (
        status == "failed"
        and error_text == "authoritative recovery verified no domain effect"
    ):
        result = _not_applied_result(operation)
    elif status == "failed" and (
        error_text in _RUNNER_PROVEN_NO_WRITE_ERRORS
        or error_text.startswith(TOURNAMENT_ADMIN_STALE_CAS_NO_WRITE_PREFIX)
    ):
        # A duplicate request can lose the insert race while the exact winner
        # is still at intent, then observe that winner settle as a no-write
        # failure. Only runner-defined failure markers are conclusive here;
        # arbitrary/legacy failed rows remain closed and are never guessed.
        result = reconcile_tournament_admin_guarded_operation(
            supabase,
            club_id=str(club_id),
            surface=REGISTRATION_IMPORT_SURFACE,
            operation_key=str(operation.get("operation_key") or ""),
            entity_type="tournament_event_draw",
            entity_id=str(draw_id),
            actor_email=str(actor_email),
            actor_role=str(actor_role),
            source=str(source),
            verify_outcome=lambda _row: {"status": "uncertain"},
            settled_failed_is_not_applied=True,
        )
    elif error_text == TOURNAMENT_ADMIN_RECOVERY_TOMBSTONE_ERROR:
        result = reconcile_tournament_admin_guarded_operation(
            supabase,
            club_id=str(club_id),
            surface=REGISTRATION_IMPORT_SURFACE,
            operation_key=str(operation.get("operation_key") or ""),
            entity_type="tournament_event_draw",
            entity_id=str(draw_id),
            actor_email=str(actor_email),
            actor_role=str(actor_role),
            source=str(source),
            verify_outcome=lambda _row: {
                "status": "not_applied",
                "result": {},
                "evidence": {
                    "authority": "tournament_admin_operations",
                    "recovery_tombstone_reserved": True,
                    "tombstone_created_by_this_request": bool(tombstone_created),
                },
            },
        )
    elif status == "intent":
        # The original request won the unique-key race and may still be
        # advancing from durable intent to mutation.  State equality at this
        # instant cannot safely close it as not applied.
        raise TournamentAdminRecoveryRequiredError(
            "The original registration import has durable intent and may still be in flight. "
            "Keep the retained request blocked and check authoritative state again."
        )
    else:
        result = reconcile_tournament_admin_guarded_operation(
            supabase,
            club_id=str(club_id),
            surface=REGISTRATION_IMPORT_SURFACE,
            operation_key=str(operation.get("operation_key") or ""),
            entity_type="tournament_event_draw",
            entity_id=str(draw_id),
            actor_email=str(actor_email),
            actor_role=str(actor_role),
            source=str(source),
            verify_outcome=lambda _row: {
                "status": "uncertain",
                "result": {},
                "evidence": {
                    "authority": "tournament_admin_operations",
                    "reason": "empty recovery has no commit fence",
                },
            },
        )
    return {
        **result,
        "authority": "python_fastapi",
        "tournament_id": str(tournament_id),
        "draw_id": str(draw_id),
    }
__all__ = [
    "REGISTRATION_IMPORT_ACTION",
    "REGISTRATION_IMPORT_RECONCILE_CONFIRMATION",
    "REGISTRATION_IMPORT_SURFACE",
    "reconcile_admin_tournament_registration_import_operation",
]
