from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.services.admin_player_editor_service import (
    PlayerEditorConflictError,
    _fetch_league_ratings,
    _jupr_to_elo,
    _league_rating_payload,
    _mark_recovery_required,
    _safe_int,
    _safe_rows,
    is_admin_player_editor_enabled,
    is_api_audit_log_required,
)
from jupr_app.services.admin_guarded_write_service import (
    GuardedWriteRecoveryRequired,
    begin_guarded_operation,
    canonical_fingerprint,
    get_guarded_operation,
    operation_result,
    update_guarded_operation,
)

REQUIRED_CONFIRMATION = "SAVE LEAGUE RATING"


def _fetch_league_rating_row(supabase: Any, *, club_id: str, player_id: int, league_rating_id: int) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("league_ratings")
        .select("id,club_id,player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active,inactive_at")
        .eq("club_id", str(club_id))
        .eq("player_id", int(player_id))
        .eq("id", int(league_rating_id))
        .limit(1)
        .execute()
    )
    return rows[0] if rows else None


def update_admin_player_editor_league_rating(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    league_rating_id: int,
    patch: dict[str, Any],
    actor_email: str,
    actor_role: str,
    expected_state_fingerprint: str,
    idempotency_key: str,
    confirmation_text: str = "",
    source: str = "next_player_editor_league_rating",
) -> dict[str, Any]:
    if not is_admin_player_editor_enabled():
        raise PermissionError("Next Player Editor is disabled.")
    normalized_confirmation = str(confirmation_text or "").strip().upper()
    if normalized_confirmation != REQUIRED_CONFIRMATION:
        raise ValueError(f"Type {REQUIRED_CONFIRMATION} to confirm league-rating edits.")
    update_payload: dict[str, Any] = {}
    request_patch: dict[str, Any] = {}
    if "rating_jupr" in patch:
        update_payload["rating"] = _jupr_to_elo(patch.get("rating_jupr"), field_name="League JUPR")
        request_patch["rating"] = update_payload["rating"]
    if "starting_jupr" in patch:
        update_payload["starting_rating"] = _jupr_to_elo(patch.get("starting_jupr"), field_name="League starting JUPR")
        request_patch["starting_rating"] = update_payload["starting_rating"]
    if "is_active" in patch:
        next_active = bool(patch.get("is_active"))
        update_payload["is_active"] = next_active
        request_patch["is_active"] = next_active
    if not update_payload:
        raise ValueError("No supported league-rating fields were provided.")
    request_payload = {
        "player_id": int(player_id),
        "league_rating_id": int(league_rating_id),
        "expected_state_fingerprint": str(expected_state_fingerprint or "").strip().lower(),
        "patch": request_patch,
        "confirmation_text": REQUIRED_CONFIRMATION,
    }
    existing_operation = get_guarded_operation(
        supabase,
        club_id=str(club_id),
        workflow="player_editor_league_rating_update",
        operation_key=idempotency_key,
    )
    if existing_operation is not None:
        if str(existing_operation.get("request_fingerprint") or "") != canonical_fingerprint(request_payload):
            raise ValueError("operation_key was already used for a different request.")
        if str(existing_operation.get("status") or "") == "completed":
            return operation_result(existing_operation)
        raise GuardedWriteRecoveryRequired(
            idempotency_key,
            "The league-rating operation is incomplete. Inspect its recovery status before retrying.",
        )
    before = _fetch_league_rating_row(
        supabase,
        club_id=str(club_id),
        player_id=int(player_id),
        league_rating_id=int(league_rating_id),
    )
    if before is None:
        raise ValueError("league rating not found")
    before_public = _league_rating_payload(before)
    if str(expected_state_fingerprint or "").strip().lower() != str(before_public.get("state_fingerprint") or ""):
        raise PlayerEditorConflictError(
            "League rating changed after it was loaded. Reload Player Editor and review the edit.",
            operation_key=idempotency_key,
        )
    if "is_active" in request_patch:
        update_payload["inactive_at"] = (
            None
            if bool(request_patch["is_active"])
            else (before.get("inactive_at") or datetime.now(timezone.utc).isoformat())
        )
    operation, idempotent = begin_guarded_operation(
        supabase,
        club_id=str(club_id),
        workflow="player_editor_league_rating_update",
        action="update_player_editor_league_rating",
        operation_key=idempotency_key,
        request_payload=request_payload,
        actor_email=actor_email,
        actor_role=actor_role,
        source=source,
        before_json={"league_rating": before_public},
    )
    if idempotent:
        return operation_result(operation)
    planned_after = _league_rating_payload({**before, **update_payload})
    update_guarded_operation(
        supabase,
        operation_id=operation.get("id"),
        operation_key=idempotency_key,
        status="intent_recorded",
        result_json={"phase": "preflight", "planned": {"league_rating": planned_after}},
    )
    update_query = (
        supabase.table("league_ratings")
        .update(update_payload)
        .eq("club_id", str(club_id))
        .eq("player_id", int(player_id))
        .eq("id", int(league_rating_id))
        .eq("league_name", before.get("league_name"))
        .eq("rating", before.get("rating"))
        .eq("starting_rating", before.get("starting_rating"))
        .eq("wins", before.get("wins"))
        .eq("losses", before.get("losses"))
        .eq("matches_played", before.get("matches_played"))
        .eq("is_active", before.get("is_active"))
    )
    if before.get("inactive_at") is None:
        update_query = update_query.is_("inactive_at", None)
    else:
        update_query = update_query.eq("inactive_at", before.get("inactive_at"))
    try:
        updated_rows = _safe_rows(update_query.execute())
    except Exception as exc:
        readback: dict[str, Any] | None = None
        try:
            readback = _fetch_league_rating_row(
                supabase,
                club_id=str(club_id),
                player_id=int(player_id),
                league_rating_id=int(league_rating_id),
            )
        except Exception:
            readback = None
        _mark_recovery_required(
            supabase,
            operation=operation,
            operation_key=idempotency_key,
            result_json={
                "planned": {"league_rating": planned_after},
                "readback_verified": readback is not None,
                "league_rating": _league_rating_payload(readback) if readback else None,
            },
            error_text="League-rating compare-and-swap update returned an ambiguous transport result.",
        )
        raise GuardedWriteRecoveryRequired(
            idempotency_key,
            "The league-rating edit may have committed. Inspect this exact operation before retrying; do not use a new key.",
        ) from exc
    if not updated_rows:
        update_guarded_operation(
            supabase,
            operation_id=operation.get("id"),
            operation_key=idempotency_key,
            status="failed",
            error_text="League-rating state changed before compare-and-swap update.",
        )
        raise PlayerEditorConflictError(
            "League rating changed after it was loaded. Reload Player Editor and review the edit.",
            operation_key=idempotency_key,
        )
    after = _league_rating_payload(updated_rows[0]) if updated_rows else _league_rating_payload(before | update_payload)
    audit_payload = build_activity_payload(
        club_id=str(club_id),
        actor_email=str(actor_email or ""),
        actor_role=str(actor_role or ""),
        action_type="update_player_editor_league_rating",
        entity_type="league_rating",
        entity_id=str(int(league_rating_id)),
        before_json={"league_rating": _league_rating_payload(before)},
        after_json={"source_client": "fastapi/nextjs", "source_page": source, "patch": update_payload, "league_rating": after},
        source_page=source,
        flagged_for_review=True,
    )
    audit_write = write_admin_activity_log(supabase, audit_payload)
    warnings: list[str] = []
    if audit_write.warning:
        warnings.append(audit_write.warning)
    if not audit_write.ok and is_api_audit_log_required():
        _mark_recovery_required(
            supabase,
            operation=operation,
            operation_key=idempotency_key,
            result_json={"planned": {"league_rating": planned_after}, "league_rating": after},
            error_text="Required completion audit did not persist.",
        )
        raise GuardedWriteRecoveryRequired(
            idempotency_key,
            "The league-rating edit may be committed, but its required audit is unavailable. Inspect Player Editor before retrying.",
        )
    result = {
        "ok": True,
        "mode": "player_editor_league_rating_update",
        "league_rating": after,
        "league_ratings": _fetch_league_ratings(supabase, club_id=str(club_id), player_id=int(player_id)),
        "operation_key": idempotency_key,
        "idempotent_replay": False,
        "recovery": {
            "operation_status": f"/admin/clubs/{{club_id}}/players/editor/operations/{idempotency_key}",
            "operator_rule": "Retry the exact unchanged request with the same idempotency key after an interrupted response.",
        },
        "warnings": warnings,
    }
    update_guarded_operation(
        supabase,
        operation_id=operation.get("id"),
        operation_key=idempotency_key,
        status="completed",
        after_json={"league_rating": after},
        result_json=result,
    )
    return result
