from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any
from uuid import UUID

from jupr_app.config import get_email_mode, get_next_web_base_url
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.notifications.player_profile_update_repo import (
    REQUEST_STATUS_ACTIVE,
    SEND_STATUS_SENDING,
    StaleCommunicationsStateError,
    claim_communications_admin_operation,
    complete_communications_admin_operation,
    delete_pending_outbox_rows_guarded,
    get_communications_admin_operation,
    get_subscription,
    list_digests_for_range,
    list_outbox_rows,
    list_subscriptions_by_status,
    mark_unsubscribed_guarded,
    normalize_email,
    replace_verified_subscriber_atomic,
    retry_outbox_rows_guarded,
    validate_email_address,
    validate_communications_admin_operation,
)
from jupr_app.domain.notifications.player_update_sender import (
    generate_and_queue_digest_for_player,
    generate_and_queue_digests_for_active_subscriptions,
)
from jupr_app.domain.recaps.player_weekly_digest import compute_player_weekly_digest
from jupr_app.services.admin_player_updates_service import (
    _build_ctx,
    _coerce_date,
    is_admin_player_updates_enabled,
    is_api_audit_log_required,
    send_pending_player_update_emails_for_range,
)
from jupr_app.services.staging_write_guard import (
    require_staging_communications_mutations,
)

CONFIRM_QUEUE = "QUEUE PLAYER UPDATES"
CONFIRM_SEND = "SEND PLAYER UPDATES"
CONFIRM_RETRY = "RETRY PLAYER UPDATES"
CONFIRM_RETRY_UNCERTAIN = "RETRY UNCERTAIN EMAILS"
CONFIRM_DELETE = "DELETE QUEUED UPDATES"
CONFIRM_REPLACE = "REPLACE VERIFIED SUBSCRIBER"
CONFIRM_DEACTIVATE = "UNSUBSCRIBE VERIFIED SUBSCRIBER"
COMMUNICATION_STATUSES = ("pending_admin_review", "active", "rejected", "unsubscribed")


def _confirm(value: str, expected: str) -> None:
    if str(value or "").strip().upper() != expected:
        raise ValueError(f"Type {expected} to confirm this operation.")


def _operation_key(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("operation_key is required")
    try:
        return str(UUID(raw))
    except Exception as exc:
        raise ValueError("operation_key must be a UUID") from exc


def _date_window(start_date: Any, end_date: Any, *, max_days: int = 45) -> tuple[date, date]:
    start = _coerce_date(start_date)
    end = _coerce_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    if (end - start).days > max_days:
        raise ValueError(f"Player update ranges are capped at {max_days} days")
    return start, end


def _safe_rows(response: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (response.data or [])]
    except Exception:
        return []


def _mask_email(value: Any) -> str:
    email = str(value or "").strip()
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    return f"{local[:1]}***@{domain}"


def _audit(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_type: str,
    entity_id: str,
    after_json: dict[str, Any],
    source: str,
    before_json: dict[str, Any] | None = None,
    post_mutation: bool = True,
) -> list[str]:
    result = write_admin_activity_log(
        supabase,
        build_activity_payload(
            club_id=str(club_id),
            actor_email=str(actor_email or ""),
            actor_role=str(actor_role or ""),
            action_type=action_type,
            entity_type=entity_type,
            entity_id=entity_id,
            before_json=before_json or {},
            after_json={"source_client": "fastapi/nextjs", **after_json},
            source_page=source,
            flagged_for_review=True,
        ),
    )
    if not result.ok and is_api_audit_log_required():
        if not post_mutation:
            raise RuntimeError("Required audit intent could not be persisted; nothing was changed or sent.")
        raise RuntimeError(
            "The operation may have completed, but its required completion audit could not be persisted. Reload state before retrying."
        )
    return [result.warning] if result.warning else []


def _required_audit_intent(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_type: str,
    entity_id: str,
    reviewed_scope: dict[str, Any],
    source: str,
) -> None:
    """Fail closed before a communications mutation when strict audit is on."""

    if not is_api_audit_log_required():
        return
    _audit(
        supabase,
        club_id=club_id,
        actor_email=actor_email,
        actor_role=actor_role,
        action_type=f"{action_type}_intent",
        entity_type=entity_type,
        entity_id=entity_id,
        after_json={"phase": "intent", "reviewed_scope": reviewed_scope},
        source=source,
        post_mutation=False,
    )


def _audit_failure(
    supabase: Any,
    *,
    club_id: str,
    actor_email: str,
    actor_role: str,
    action_type: str,
    entity_type: str,
    entity_id: str,
    reviewed_scope: dict[str, Any],
    source: str,
    error: Exception,
) -> None:
    """Persist a distinct failure outcome after a recorded intent."""

    try:
        _audit(
            supabase,
            club_id=club_id,
            actor_email=actor_email,
            actor_role=actor_role,
            action_type=f"{action_type}_failed",
            entity_type=entity_type,
            entity_id=entity_id,
            after_json={
                "phase": "failed",
                "reviewed_scope": reviewed_scope,
                "error_type": type(error).__name__,
                "error": str(error or "")[:500],
            },
            source=source,
        )
    except Exception:
        raise RuntimeError(
            "The communications operation failed or is uncertain, and its required failure audit also failed. Reload state before retrying."
        ) from error


def _player_names(supabase: Any, club_id: str) -> dict[int, str]:
    rows = _safe_rows(supabase.table("players").select("id,name").eq("club_id", str(club_id)).execute())
    result: dict[int, str] = {}
    for row in rows:
        try:
            result[int(row.get("id"))] = str(row.get("name") or "").strip()
        except Exception:
            continue
    return result


def build_communications_workspace(
    supabase: Any,
    *,
    club_id: str,
    start_date: Any,
    end_date: Any,
    outbox_status: str | None = None,
    limit: int = 500,
) -> dict[str, Any]:
    if not is_admin_player_updates_enabled():
        raise PermissionError("Next Player Updates Admin is disabled.")
    start, end = _date_window(start_date, end_date, max_days=366)
    names = _player_names(supabase, str(club_id))
    subscriptions = list_subscriptions_by_status(
        supabase,
        str(club_id),
        statuses=list(COMMUNICATION_STATUSES),
        limit=min(max(1, int(limit)), 1000),
    )
    digests = list_digests_for_range(
        supabase,
        str(club_id),
        week_start_from=start,
        week_start_to=end,
    )
    outbox = list_outbox_rows(
        supabase,
        str(club_id),
        status=outbox_status,
        limit=min(max(1, int(limit)), 1000),
        week_start=start,
        week_end=end,
    )
    for row in subscriptions:
        row["player_name"] = names.get(int(row.get("player_id") or 0), "")
    for row in digests:
        row["player_name"] = names.get(int(row.get("player_id") or 0), "")
    for row in outbox:
        row["player_name"] = names.get(int(row.get("player_id") or 0), "")

    subscription_counts: dict[str, int] = defaultdict(int)
    for row in subscriptions:
        subscription_counts[str(row.get("request_status") or "unknown")] += 1
    outbox_counts: dict[str, int] = defaultdict(int)
    for row in outbox:
        outbox_counts[str(row.get("send_status") or "unknown")] += 1
    return {
        "ok": True,
        "mode": "communications_workspace",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "subscriptions": subscriptions,
        "digests": digests,
        "outbox": outbox,
        "subscription_counts": dict(subscription_counts),
        "outbox_counts": dict(outbox_counts),
        "email_mode": get_email_mode(),
    }


def preview_player_digest(
    supabase: Any,
    *,
    club_id: str,
    player_id: int,
    start_date: Any,
    end_date: Any,
) -> dict[str, Any]:
    if not is_admin_player_updates_enabled():
        raise PermissionError("Next Player Updates Admin is disabled.")
    start, end = _date_window(start_date, end_date)
    ctx = _build_ctx(supabase, club_id=str(club_id))
    digest = compute_player_weekly_digest(ctx, player_id=int(player_id), start_date=start, end_date=end)
    return {
        "ok": True,
        "mode": "player_digest_preview",
        "player_id": int(player_id),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "digest": digest,
        "persisted": False,
        "queued": False,
    }


def queue_player_digests(
    supabase: Any,
    *,
    club_id: str,
    start_date: Any,
    end_date: Any,
    player_id: int | None,
    only_players_with_matches: bool,
    confirmation_text: str,
    operation_key: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    if not is_admin_player_updates_enabled():
        raise PermissionError("Next Player Updates Admin is disabled.")
    require_staging_communications_mutations()
    _confirm(confirmation_text, CONFIRM_QUEUE)
    op_key = _operation_key(operation_key)
    start, end = _date_window(start_date, end_date)
    operation_request = {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "player_id": int(player_id) if player_id is not None else None,
        "only_players_with_matches": bool(only_players_with_matches),
    }
    operation = get_communications_admin_operation(supabase, operation_key=op_key)
    if operation is not None:
        operation = validate_communications_admin_operation(
            operation,
            club_id=str(club_id),
            operation_type="queue_player_update_digests",
            request_json=operation_request,
        )
        if str(operation.get("status") or "") == "completed" and isinstance(operation.get("result_json"), dict):
            return {**dict(operation["result_json"]), "idempotent_replay": True}
    _required_audit_intent(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="queue_player_update_digests_admin",
        entity_type="player_update_outbox",
        entity_id=op_key,
        reviewed_scope=operation_request,
        source=source,
    )
    try:
        if operation is None:
            operation = claim_communications_admin_operation(
                supabase,
                club_id=str(club_id),
                operation_key=op_key,
                operation_type="queue_player_update_digests",
                request_json=operation_request,
            )
        ctx = _build_ctx(supabase, club_id=str(club_id))
        if player_id is None:
            result = generate_and_queue_digests_for_active_subscriptions(
                ctx,
                start_date=start,
                end_date=end,
                only_players_with_matches=bool(only_players_with_matches),
                operation_key=op_key,
            )
        else:
            result = generate_and_queue_digest_for_player(
                ctx,
                player_id=int(player_id),
                start_date=start,
                end_date=end,
                operation_key=op_key,
            )
        warnings = _audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="queue_player_update_digests_admin",
            entity_type="player_update_outbox",
            entity_id=op_key,
            after_json={
                "operation_key": op_key,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "player_id": player_id,
                "only_players_with_matches": bool(only_players_with_matches),
                "result": result,
            },
            source=source,
        )
        if int(result.get("failed") or 0) > 0:
            raise RuntimeError(
                f"Queue partially completed with {int(result.get('failed') or 0)} failed row(s). "
                "Retry the unchanged request with the same operation key."
            )
        response = {
            "ok": True,
            "mode": "player_update_queue",
            "operation_key": op_key,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "result": result,
            "warnings": warnings,
        }
        complete_communications_admin_operation(
            supabase,
            club_id=str(club_id),
            operation_key=op_key,
            result_json=response,
        )
    except Exception as exc:
        _audit_failure(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="queue_player_update_digests_admin",
            entity_type="player_update_outbox",
            entity_id=op_key,
            reviewed_scope=operation_request,
            source=source,
            error=exc,
        )
        raise
    return response


def send_selected_outbox_rows(
    supabase: Any,
    *,
    club_id: str,
    items: list[dict[str, Any]],
    confirmation_text: str,
    operation_key: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    if not is_admin_player_updates_enabled():
        raise PermissionError("Next Player Updates Admin is disabled.")
    require_staging_communications_mutations()
    _confirm(confirmation_text, CONFIRM_SEND)
    op_key = _operation_key(operation_key)
    if not items:
        raise ValueError("Select at least one pending outbox row")
    current_rows = {str(row.get("id") or ""): row for row in list_outbox_rows(supabase, str(club_id), limit=1000)}
    grouped: dict[tuple[date, date], list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        outbox_id = str((item or {}).get("id") or "").strip()
        row = current_rows.get(outbox_id)
        if row is None:
            raise StaleCommunicationsStateError("An outbox row is missing. Reload the queue.")
        grouped[(_coerce_date(row.get("week_start")), _coerce_date(row.get("week_end")))].append(dict(item))
    reviewed_scope = {"operation_key": op_key, "items": items, "email_mode": get_email_mode()}
    _required_audit_intent(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="send_selected_player_updates_admin",
        entity_type="player_update_outbox",
        entity_id=op_key,
        reviewed_scope=reviewed_scope,
        source=source,
    )
    try:
        totals = {"attempted": 0, "sent": 0, "skipped": 0, "errors": 0, "stale": 0, "uncertain": 0}
        windows: list[dict[str, Any]] = []
        ctx = _build_ctx(supabase, club_id=str(club_id))
        for (start, end), window_items in grouped.items():
            result = send_pending_player_update_emails_for_range(
                ctx,
                start_date=start,
                end_date=end,
                limit=2000,
                public_base_url=get_next_web_base_url(),
                outbox_items=window_items,
                actor_email=actor_email,
            )
            windows.append({"start_date": start.isoformat(), "end_date": end.isoformat(), **result})
            for key in totals:
                totals[key] += int(result.get(key, 0) or 0)
        warnings = _audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="send_selected_player_updates_admin",
            entity_type="player_update_outbox",
            entity_id=op_key,
            after_json={"operation_key": op_key, "selected_count": len(items), "totals": totals, "email_mode": get_email_mode()},
            source=source,
        )
    except Exception as exc:
        _audit_failure(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="send_selected_player_updates_admin",
            entity_type="player_update_outbox",
            entity_id=op_key,
            reviewed_scope=reviewed_scope,
            source=source,
            error=exc,
        )
        raise
    return {"ok": True, "mode": "player_update_send_selected", "operation_key": op_key, "windows": windows, **totals, "email_mode": get_email_mode(), "warnings": warnings}


def retry_outbox_rows(
    supabase: Any,
    *,
    club_id: str,
    items: list[dict[str, Any]],
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    if not is_admin_player_updates_enabled():
        raise PermissionError("Next Player Updates Admin is disabled.")
    require_staging_communications_mutations()
    current = {str(row.get("id") or ""): row for row in list_outbox_rows(supabase, str(club_id), limit=1000)}
    includes_uncertain = any(str(current.get(str((item or {}).get("id") or ""), {}).get("send_status") or "") == SEND_STATUS_SENDING for item in items)
    _confirm(confirmation_text, CONFIRM_RETRY_UNCERTAIN if includes_uncertain else CONFIRM_RETRY)
    reviewed_scope = {"items": items, "included_uncertain_sending": includes_uncertain}
    _required_audit_intent(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="retry_player_update_outbox_admin",
        entity_type="player_update_outbox",
        entity_id="bulk_retry",
        reviewed_scope=reviewed_scope,
        source=source,
    )
    try:
        result = retry_outbox_rows_guarded(
            supabase,
            club_id=str(club_id),
            items=items,
            allow_uncertain=includes_uncertain,
        )
        warnings = _audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="retry_player_update_outbox_admin",
            entity_type="player_update_outbox",
            entity_id="bulk_retry",
            after_json={"result": result, "included_uncertain_sending": includes_uncertain},
            source=source,
        )
    except Exception as exc:
        _audit_failure(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="retry_player_update_outbox_admin",
            entity_type="player_update_outbox",
            entity_id="bulk_retry",
            reviewed_scope=reviewed_scope,
            source=source,
            error=exc,
        )
        raise
    return {"ok": True, "mode": "player_update_outbox_retry", **result, "warnings": warnings}


def delete_outbox_rows(
    supabase: Any,
    *,
    club_id: str,
    items: list[dict[str, Any]],
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    if not is_admin_player_updates_enabled():
        raise PermissionError("Next Player Updates Admin is disabled.")
    require_staging_communications_mutations()
    _confirm(confirmation_text, CONFIRM_DELETE)
    reviewed_scope = {"items": items}
    _required_audit_intent(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="delete_player_update_outbox_admin",
        entity_type="player_update_outbox",
        entity_id="bulk_delete",
        reviewed_scope=reviewed_scope,
        source=source,
    )
    try:
        result = delete_pending_outbox_rows_guarded(supabase, club_id=str(club_id), items=items)
        audit_result = {key: value for key, value in result.items() if key != "deleted_rows"}
        warnings = _audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="delete_player_update_outbox_admin",
            entity_type="player_update_outbox",
            entity_id="bulk_delete",
            after_json={"result": audit_result},
            source=source,
        )
    except Exception as exc:
        _audit_failure(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="delete_player_update_outbox_admin",
            entity_type="player_update_outbox",
            entity_id="bulk_delete",
            reviewed_scope=reviewed_scope,
            source=source,
            error=exc,
        )
        raise
    return {"ok": True, "mode": "player_update_outbox_delete", **audit_result, "warnings": warnings}


def replace_active_subscription(
    supabase: Any,
    *,
    club_id: str,
    subscription_id: str,
    expected_row_version: int,
    new_email: str,
    request_note: str | None,
    admin_note: str | None,
    confirmation_text: str,
    operation_key: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    if not is_admin_player_updates_enabled():
        raise PermissionError("Next Player Updates Admin is disabled.")
    require_staging_communications_mutations()
    _confirm(confirmation_text, CONFIRM_REPLACE)
    op_key = _operation_key(operation_key)
    validated_email = validate_email_address(new_email, field_name="new_email")
    operation_request = {
        "old_subscription_id": str(subscription_id),
        "new_email_normalized": normalize_email(validated_email),
        "request_note": str(request_note or "").strip(),
        "admin_note": str(admin_note or "").strip(),
        "actor_email": str(actor_email or "").strip().lower(),
    }
    operation = get_communications_admin_operation(supabase, operation_key=op_key)
    if operation is not None:
        operation = validate_communications_admin_operation(
            operation,
            club_id=str(club_id),
            operation_type="replace_verified_subscriber",
            request_json=operation_request,
        )
        if str(operation.get("status") or "") == "completed" and isinstance(operation.get("result_json"), dict):
            return {**dict(operation["result_json"]), "idempotent_replay": True}
    before = get_subscription(supabase, club_id=str(club_id), subscription_id=str(subscription_id))
    if before is None:
        raise ValueError("Subscription not found")
    reviewed_scope = {
        "operation_key": op_key,
        "subscription_id": str(subscription_id),
        "expected_row_version": int(expected_row_version),
        "new_email_masked": _mask_email(new_email),
    }
    _required_audit_intent(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="replace_verified_subscriber_admin",
        entity_type="subscription",
        entity_id=str(subscription_id),
        reviewed_scope=reviewed_scope,
        source=source,
    )
    try:
        if operation is None:
            operation = claim_communications_admin_operation(
                supabase,
                club_id=str(club_id),
                operation_key=op_key,
                operation_type="replace_verified_subscriber",
                request_json=operation_request,
            )
        row = replace_verified_subscriber_atomic(
            supabase,
            club_id=str(club_id),
            operation_key=op_key,
            old_subscription_id=str(subscription_id),
            new_email=new_email,
            new_request_note=request_note,
            verified_by=actor_email,
            admin_note=admin_note,
            expected_row_version=expected_row_version,
        )
        warnings = _audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="replace_verified_subscriber_admin",
            entity_type="subscription",
            entity_id=str(subscription_id),
            before_json={"id": before.get("id"), "player_id": before.get("player_id"), "email_masked": _mask_email(before.get("email")), "row_version": before.get("row_version")},
            after_json={"replacement_id": row.get("id"), "player_id": row.get("player_id"), "email_masked": _mask_email(row.get("email")), "operation_key": op_key},
            source=source,
        )
        response = {"ok": True, "mode": "replace_verified_subscriber", "subscription": row, "operation_key": op_key, "warnings": warnings}
        complete_communications_admin_operation(
            supabase,
            club_id=str(club_id),
            operation_key=op_key,
            result_json=response,
        )
    except Exception as exc:
        _audit_failure(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="replace_verified_subscriber_admin",
            entity_type="subscription",
            entity_id=str(subscription_id),
            reviewed_scope=reviewed_scope,
            source=source,
            error=exc,
        )
        raise
    return response


def deactivate_active_subscription(
    supabase: Any,
    *,
    club_id: str,
    subscription_id: str,
    expected_row_version: int,
    confirmation_text: str,
    actor_email: str,
    actor_role: str,
    source: str,
) -> dict[str, Any]:
    if not is_admin_player_updates_enabled():
        raise PermissionError("Next Player Updates Admin is disabled.")
    require_staging_communications_mutations()
    _confirm(confirmation_text, CONFIRM_DEACTIVATE)
    before = get_subscription(supabase, club_id=str(club_id), subscription_id=str(subscription_id))
    if before is None:
        raise ValueError("Subscription not found")
    if int(before.get("row_version") or 1) != int(expected_row_version):
        raise StaleCommunicationsStateError("Subscription changed. Reload before deactivating it.")
    if str(before.get("request_status") or "") != REQUEST_STATUS_ACTIVE:
        raise StaleCommunicationsStateError("Subscription is no longer active. Reload before deactivating it.")
    reviewed_scope = {"subscription_id": str(subscription_id), "expected_row_version": int(expected_row_version)}
    _required_audit_intent(
        supabase,
        club_id=str(club_id),
        actor_email=actor_email,
        actor_role=actor_role,
        action_type="deactivate_verified_subscriber_admin",
        entity_type="subscription",
        entity_id=str(subscription_id),
        reviewed_scope=reviewed_scope,
        source=source,
    )
    try:
        row = mark_unsubscribed_guarded(
            supabase,
            club_id=str(club_id),
            subscription_id=str(subscription_id),
            expected_row_version=expected_row_version,
        )
        warnings = _audit(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="deactivate_verified_subscriber_admin",
            entity_type="subscription",
            entity_id=str(subscription_id),
            before_json={"id": before.get("id"), "player_id": before.get("player_id"), "email_masked": _mask_email(before.get("email")), "row_version": before.get("row_version")},
            after_json={"request_status": row.get("request_status"), "row_version": row.get("row_version")},
            source=source,
        )
    except Exception as exc:
        _audit_failure(
            supabase,
            club_id=str(club_id),
            actor_email=actor_email,
            actor_role=actor_role,
            action_type="deactivate_verified_subscriber_admin",
            entity_type="subscription",
            entity_id=str(subscription_id),
            reviewed_scope=reviewed_scope,
            source=source,
            error=exc,
        )
        raise
    return {"ok": True, "mode": "deactivate_verified_subscriber", "subscription": row, "warnings": warnings}
