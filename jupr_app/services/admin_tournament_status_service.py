from __future__ import annotations

from datetime import datetime, timezone
import os
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.tournament_admin_operations import stable_tournament_admin_fingerprint
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
    tournament_admin_guarded_runtime_enabled,
)
from jupr_app.services.admin_tournament_lifecycle_service import (
    require_admin_tournament_completion_readiness,
)
from jupr_app.services.admin_tournament_service import (
    TOURNAMENT_SELECT,
    _clean_text,
    _first_row,
    _tournament_payload,
    is_admin_tournament_admin_enabled,
    is_api_audit_log_required,
)


CONFIRM_COMPLETE = "COMPLETE"
CONFIRM_ARCHIVE = "ARCHIVE"
CONFIRM_UNARCHIVE = "UNARCHIVE"
TERMINAL_ACTIONS = {
    "complete": (CONFIRM_COMPLETE, "COMPLETED"),
    "archive": (CONFIRM_ARCHIVE, "ARCHIVED"),
    "unarchive": (CONFIRM_UNARCHIVE, "COMPLETED"),
}
TERMINAL_RECEIPT_TABLE = "tournament_lifecycle_receipts"


def _deployed_environment() -> bool:
    return os.getenv("JUPR_ENV", "").strip().lower() in {"staging", "production"}


def _rpc_object(response: Any, *, key: str) -> dict[str, Any]:
    data = getattr(response, "data", None)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = data[0]
    if not isinstance(data, dict) or not isinstance(data.get(key), dict):
        raise RuntimeError(
            "Atomic tournament terminal transition returned no authoritative result."
        )
    return dict(data)


def _completion_snapshot_fingerprint(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    ignore_operation_key: str | None,
) -> str:
    """Read one database-owned completion snapshot fingerprint."""

    try:
        response = supabase.rpc(
            "admin_tournament_completion_snapshot",
            {
                "p_club_id": str(club_id),
                "p_tournament_id": str(tournament_id),
                "p_ignore_operation_key": str(ignore_operation_key or "") or None,
            },
        ).execute()
        payload = _rpc_object(response, key="snapshot")
        snapshot = dict(payload["snapshot"])
    except Exception as exc:
        if isinstance(exc, RuntimeError) and "authoritative" in str(exc):
            raise
        raise RuntimeError(
            "Atomic tournament closeout snapshot is unavailable; no status was changed."
        ) from exc
    fingerprint = str(snapshot.get("snapshot_fingerprint") or "")
    if not fingerprint:
        raise RuntimeError(
            "Atomic tournament closeout snapshot returned no fingerprint; no status was changed."
        )
    return fingerprint


def _transition_error(exc: Exception) -> Exception:
    detail = str(exc)
    if "JUPR_TOURNAMENT_STALE" in detail or "JUPR_TOURNAMENT_CLOSEOUT_SNAPSHOT_STALE" in detail:
        return StaleTournamentAdminStateError(
            "Tournament closeout evidence changed while completion was being committed. Reload Tournament Closeout."
        )
    if "JUPR_TOURNAMENT_CLOSEOUT_NOT_READY" in detail:
        return ValueError(
            "Tournament completion is blocked because the atomic closeout evidence is not ready. Reload Tournament Closeout."
        )
    if "JUPR_TOURNAMENT_NOT_COMPLETED" in detail:
        return ValueError("Only a completed tournament can be moved to the archive.")
    if "JUPR_TOURNAMENT_NOT_ARCHIVED" in detail:
        return ValueError("Only an archived tournament can be restored.")
    if "JUPR_TOURNAMENT_OPERATION_INVALID" in detail:
        return RuntimeError(
            "The terminal transition has no matching durable operation intent; no status was changed."
        )
    return RuntimeError(
        "Atomic tournament terminal transition failed; tournament status was not changed."
    )


def _local_terminal_transition(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    expected_updated_at: str | None,
    action: str,
    next_status: str,
    operation_key: str | None,
    request_fingerprint: str | None,
    evidence_fingerprint: str,
    evidence_json: dict[str, Any],
    actor_email: str,
) -> dict[str, Any]:
    """Small fake/local adapter; deployed routes always use the atomic RPC."""

    updated_at = datetime.now(timezone.utc).isoformat()
    query = (
        supabase.table("tournaments")
        .update({"status": next_status, "updated_at": updated_at})
        .eq("club_id", str(club_id))
        .eq("id", str(tournament_id))
    )
    if expected_updated_at:
        query = query.eq("updated_at", str(expected_updated_at))
    rows = list(getattr(query.execute(), "data", None) or [])
    if expected_updated_at and not rows:
        raise StaleTournamentAdminStateError(
            "Tournament changed after it was loaded. Reload before changing status."
        )
    tournament = dict(rows[0]) if rows else {
        "id": str(tournament_id),
        "club_id": str(club_id),
        "status": next_status,
        "updated_at": updated_at,
    }
    receipt = {
        "club_id": str(club_id),
        "tournament_id": str(tournament_id),
        "action": str(action),
        "to_status": str(next_status),
        "operation_key": str(operation_key or f"local:{action}:{tournament_id}"),
        "request_fingerprint": str(request_fingerprint or evidence_fingerprint),
        "evidence_fingerprint": str(evidence_fingerprint),
        "evidence_json": dict(evidence_json),
        "created_by": str(actor_email or "local"),
        "created_at": updated_at,
    }
    try:
        inserted = supabase.table(TERMINAL_RECEIPT_TABLE).insert(receipt).execute()
        receipt_rows = list(getattr(inserted, "data", None) or [])
        if receipt_rows:
            receipt = dict(receipt_rows[0])
    except Exception:
        # This adapter exists only for unit fakes and local development before a
        # migration is installed. Deployed calls never reach it.
        pass
    return {"tournament": tournament, "receipt": receipt}


def apply_admin_tournament_status_action(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    action: str,
    expected_updated_at: str | None = None,
    actor_email: str,
    actor_role: str,
    confirmation_text: str,
    guarded_operation_key: str | None = None,
    request_fingerprint: str | None = None,
    source: str = "next_tournament_admin_status_action",
    dry_run: bool = False,
    atomic: bool | None = None,
) -> dict[str, Any]:
    """Commit terminal status together with its immutable lifecycle receipt."""

    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    clean_tournament_id = _clean_text(tournament_id, limit=120)
    normalized_action = _clean_text(action, limit=40).lower()
    if normalized_action not in TERMINAL_ACTIONS:
        raise ValueError("action must be complete, archive, or unarchive")
    expected_confirmation, next_status = TERMINAL_ACTIONS[normalized_action]
    if str(confirmation_text or "").strip().upper() != expected_confirmation:
        raise ValueError(
            f"Type {expected_confirmation} to confirm tournament status change."
        )

    before = _first_row(
        supabase,
        "tournaments",
        TOURNAMENT_SELECT,
        key="id",
        value=clean_tournament_id,
    )
    if not before or str(before.get("club_id") or "") != str(club_id):
        raise ValueError("tournament not found")
    before_payload = _tournament_payload(before)
    current_status = str(before.get("status") or "").upper()
    if normalized_action == "complete" and current_status != "ACTIVE":
        if current_status in {"COMPLETED", "ARCHIVED"}:
            raise ValueError("Tournament closeout is already complete.")
        raise ValueError("Only an active tournament can be completed.")
    if normalized_action == "archive" and current_status != "COMPLETED":
        raise ValueError("Only a completed tournament can be moved to the archive.")
    if normalized_action == "unarchive" and current_status != "ARCHIVED":
        raise ValueError("Only an archived tournament can be restored.")
    if expected_updated_at and str(before.get("updated_at") or "") != str(expected_updated_at):
        raise StaleTournamentAdminStateError(
            "Tournament changed after it was loaded. Reload before changing status."
        )

    use_atomic = (
        tournament_admin_guarded_runtime_enabled("tournament")
        if atomic is None
        else bool(atomic)
    )
    if not dry_run and _deployed_environment() and not use_atomic:
        raise PermissionError(
            "Tournament terminal transitions require the atomic lifecycle receipt RPC in deployed environments."
        )
    if not dry_run and use_atomic and (not guarded_operation_key or not request_fingerprint):
        raise RuntimeError(
            "Atomic tournament terminal transition requires its durable operation identity."
        )

    snapshot_fingerprint = ""
    pre_read_snapshot_fingerprint = ""
    if normalized_action == "complete" and use_atomic and not dry_run:
        # Readiness is assembled through several bounded Data API reads. Bracket
        # those reads with the same database-owned snapshot used by the terminal
        # RPC so stale "ready" evidence can never be paired with a newer state.
        pre_read_snapshot_fingerprint = _completion_snapshot_fingerprint(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            ignore_operation_key=str(guarded_operation_key or "") or None,
        )

    lifecycle: dict[str, Any] | None = None
    if normalized_action == "complete":
        lifecycle = require_admin_tournament_completion_readiness(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            ignore_operation_key=str(guarded_operation_key or "") or None,
        )
        evidence_json: dict[str, Any] = {
            "contract": str(lifecycle.get("contract") or ""),
            "authority": str(lifecycle.get("authority") or ""),
            "club_id": str(club_id),
            "tournament_id": clean_tournament_id,
            "phase": str(lifecycle.get("phase") or ""),
            "counts": dict(lifecycle.get("counts") or {}),
            "domain_readiness": {
                "completion": dict(
                    (lifecycle.get("domain_readiness") or {}).get("completion") or {}
                )
            },
            "draws": list(lifecycle.get("draws") or []),
            "warnings": list(lifecycle.get("warnings") or []),
        }
    else:
        evidence_json = {
            "contract": "jupr:tournament-terminal-visibility:v1",
            "club_id": str(club_id),
            "tournament_id": clean_tournament_id,
            "action": normalized_action,
            "from_status": current_status,
            "to_status": next_status,
        }
    evidence_fingerprint = stable_tournament_admin_fingerprint(evidence_json)

    if normalized_action == "complete" and use_atomic and not dry_run:
        snapshot_fingerprint = _completion_snapshot_fingerprint(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            ignore_operation_key=str(guarded_operation_key or "") or None,
        )
        if snapshot_fingerprint != pre_read_snapshot_fingerprint:
            raise StaleTournamentAdminStateError(
                "Tournament closeout evidence changed while readiness was being reviewed. Reload Tournament Closeout."
            )

    if dry_run:
        return {
            "ok": True,
            "mode": "tournament_status_action_preflight",
            "dry_run": True,
            "write_count": 0,
            "action": normalized_action,
            "to_status": next_status,
            "evidence_fingerprint": evidence_fingerprint,
        }

    if use_atomic:
        try:
            response = supabase.rpc(
                "admin_transition_tournament_terminal_status_cas",
                {
                    "p_club_id": str(club_id),
                    "p_tournament_id": clean_tournament_id,
                    "p_action": normalized_action,
                    "p_expected_updated_at": str(expected_updated_at or before.get("updated_at") or ""),
                    "p_operation_key": str(guarded_operation_key),
                    "p_request_fingerprint": str(request_fingerprint),
                    "p_snapshot_fingerprint": snapshot_fingerprint,
                    "p_evidence_fingerprint": evidence_fingerprint,
                    "p_evidence_json": evidence_json,
                    "p_actor": str(actor_email or ""),
                },
            ).execute()
            committed = _rpc_object(response, key="tournament")
        except Exception as exc:
            raise _transition_error(exc) from exc
    else:
        committed = _local_terminal_transition(
            supabase,
            club_id=str(club_id),
            tournament_id=clean_tournament_id,
            expected_updated_at=expected_updated_at,
            action=normalized_action,
            next_status=next_status,
            operation_key=guarded_operation_key,
            request_fingerprint=request_fingerprint,
            evidence_fingerprint=evidence_fingerprint,
            evidence_json=evidence_json,
            actor_email=actor_email,
        )

    tournament = _tournament_payload(dict(committed["tournament"]))
    receipt = dict(committed.get("receipt") or {})
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type=f"{normalized_action}_tournament_admin",
        entity_type="tournament",
        entity_id=clean_tournament_id,
        before_json={"tournament": before_payload},
        after_json={
            "source_client": "fastapi/nextjs",
            "source_page": source,
            "action": normalized_action,
            "tournament": tournament,
            "lifecycle_receipt": receipt,
        },
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        raise RuntimeError("audit log write required but unavailable")
    return {
        "ok": True,
        "mode": "tournament_status_action",
        "action": normalized_action,
        "tournament": tournament,
        "lifecycle_receipt": receipt,
        "evidence_fingerprint": evidence_fingerprint,
        "warnings": warnings,
    }


def reconcile_admin_tournament_status_action(
    supabase: Any,
    *,
    club_id: str,
    tournament_id: str,
    action: str,
    operation_key: str,
) -> dict[str, Any] | None:
    """Prove a response-loss terminal transition from its atomic receipt."""

    normalized_action = _clean_text(action, limit=40).lower()
    if normalized_action not in TERMINAL_ACTIONS or not str(operation_key or ""):
        return None
    try:
        rows = list(
            getattr(
                supabase.table(TERMINAL_RECEIPT_TABLE)
                .select("*")
                .eq("club_id", str(club_id))
                .eq("tournament_id", str(tournament_id))
                .eq("operation_key", str(operation_key))
                .eq("action", normalized_action)
                .limit(1)
                .execute(),
                "data",
                None,
            )
            or []
        )
    except Exception:
        return None
    if not rows:
        return None
    receipt = dict(rows[0])
    tournament = _first_row(
        supabase,
        "tournaments",
        TOURNAMENT_SELECT,
        key="id",
        value=str(tournament_id),
    )
    expected_status = TERMINAL_ACTIONS[normalized_action][1]
    if (
        not tournament
        or str(tournament.get("club_id") or "") != str(club_id)
        or str(tournament.get("status") or "").upper() != expected_status
        or str(receipt.get("to_status") or "").upper() != expected_status
    ):
        return None
    return {
        "ok": True,
        "mode": "tournament_status_action",
        "action": normalized_action,
        "tournament": _tournament_payload(tournament),
        "lifecycle_receipt": receipt,
        "evidence_fingerprint": str(receipt.get("evidence_fingerprint") or ""),
        "warnings": [],
        "reconciled": True,
    }


__all__ = [
    "CONFIRM_ARCHIVE",
    "CONFIRM_COMPLETE",
    "CONFIRM_UNARCHIVE",
    "apply_admin_tournament_status_action",
    "reconcile_admin_tournament_status_action",
]
