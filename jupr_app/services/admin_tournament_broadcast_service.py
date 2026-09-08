"""Reviewed tournament emails with durable, at-most-once SMTP attempts.

A broadcast records the reviewed audience. Each HTTP request attempts only one
recipient, so large selections do not require a long-running web request. A
unique recipient claim is committed before SMTP; uncertain attempts are never
automatically retried. The existing private communications ledger stores both.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid5

from jupr_app.config import EMAIL_MODE_DRY_RUN
from jupr_app.services.tournament_broadcast_edit_link_service import build_tournament_broadcast_edit_links
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.notifications.player_profile_update_repo import (
    claim_communications_admin_operation,
    complete_communications_admin_operation,
    get_communications_admin_operation,
    validate_communications_admin_operation,
    validate_email_address,
)
from jupr_app.domain.notifications.tournament_registrant_broadcast_email import send_tournament_registrant_broadcast_email
from jupr_app.services.admin_tournament_registration_reporting_service import (
    _require_tournament,
    broadcast_delivery_settings,
    build_admin_tournament_broadcast_preview,
)
from jupr_app.services.admin_tournament_service import is_admin_tournament_admin_enabled
from jupr_app.services.staging_write_guard import require_staging_communications_mutations

TABLE = "communications_admin_operations"
CONFIRM_SEND = "SEND TO SELECTED PARTICIPANTS"


def _root_type(tournament_id: str) -> str:
    return f"tournament_broadcast:{tournament_id}"


def _recipient_type(operation_key: str) -> str:
    return f"tournament_broadcast_recipient:{operation_key}"


def _key(value: str) -> str:
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("A valid email operation ID is required.") from exc


def _authorize(supabase: Any, club_id: str, tournament_id: str) -> None:
    if not is_admin_tournament_admin_enabled():
        raise PermissionError("Next Tournament Admin is disabled.")
    _require_tournament(supabase, club_id=club_id, tournament_id=tournament_id)


def _operation(supabase: Any, club_id: str, tournament_id: str, operation_key: str) -> dict:
    _authorize(supabase, club_id, tournament_id)
    row = get_communications_admin_operation(supabase, operation_key=_key(operation_key))
    if not row or row.get("club_id") != club_id or row.get("operation_type") != _root_type(tournament_id):
        raise ValueError("Email not found in this tournament.")
    return row


def _audit(supabase: Any, *, club_id: str, actor_email: str, actor_role: str,
           operation_key: str, action: str, details: dict) -> None:
    result = write_admin_activity_log(supabase, build_activity_payload(
        club_id=club_id, actor_email=actor_email, actor_role=actor_role,
        action_type=action, entity_type="tournament_broadcast", entity_id=operation_key,
        after_json={"source_client": "fastapi/nextjs", **details},
        source_page="next_tournament_communications", flagged_for_review=True,
    ))
    if not result.ok:
        raise RuntimeError("The email audit could not be saved. Check the email results before continuing.")


def create_tournament_broadcast(supabase: Any, *, club_id: str, tournament_id: str,
        operation_key: str, preview_fingerprint: str, registration_ids: list[str],
        subject: str, message: str, include_cancelled: bool,
        confirmation_text: str, actor_email: str, actor_role: str,
        include_registration_events: bool = False, include_registration_edit_links: bool = False) -> dict:
    _authorize(supabase, club_id, tournament_id)
    require_staging_communications_mutations()
    if confirmation_text != CONFIRM_SEND:
        raise ValueError("Confirm the selected participants before sending.")
    operation_key = _key(operation_key)
    request = dict(tournament_id=tournament_id, registration_ids=sorted(set(registration_ids)),
        subject=subject.strip(), message=message.strip(), include_cancelled=include_cancelled,
        preview_fingerprint=preview_fingerprint, actor_email=actor_email.lower())
    if include_registration_events:
        request["include_registration_events"] = True
    if include_registration_edit_links:
        request["include_registration_edit_links"] = True
    if not request["registration_ids"] or not request["subject"] or not request["message"]:
        raise ValueError("Select participants and enter a subject and message.")
    if "\x00" in message or len(subject) > 200 or len(message) > 10000:
        raise ValueError("Use a subject up to 200 characters and a message up to 10,000 characters without null characters.")
    existing = get_communications_admin_operation(supabase, operation_key=operation_key)
    if existing:
        validate_communications_admin_operation(existing, club_id=club_id,
            operation_type=_root_type(tournament_id), request_json={**request,
                "review": existing["request_json"].get("review")})
        return get_tournament_broadcast(supabase, club_id=club_id,
            tournament_id=tournament_id, operation_key=operation_key)
    preview = build_admin_tournament_broadcast_preview(supabase, club_id=club_id,
        tournament_id=tournament_id, registration_ids=registration_ids,
        subject=subject, message=message, include_cancelled=include_cancelled,
        include_registration_events=include_registration_events,
        include_registration_edit_links=include_registration_edit_links)
    if not preview.get("send_available"):
        raise ValueError(preview.get("send_unavailable_reason") or "Email sending is unavailable.")
    if preview.get("preview_fingerprint") != preview_fingerprint:
        raise ValueError("The recipients, message, or sender changed. Preview the email again before sending.")
    for recipient in preview["recipients"]:
        validate_email_address(recipient["email"], field_name="Participant email")
    review = {key: preview[key] for key in ("recipients", "preview", "delivery_mode", "sender")}
    review["email_sponsors"] = preview.get("email_sponsors") or []
    if include_registration_edit_links:
        review["edit_link_context"] = preview["edit_link_context"]
    request["review"] = review
    _audit(supabase, club_id=club_id, actor_email=actor_email, actor_role=actor_role,
        operation_key=operation_key, action="tournament_broadcast_confirmed",
        details={"tournament_id": tournament_id, "recipient_count": len(review["recipients"]),
                 "preview_fingerprint": preview_fingerprint, "delivery_mode": review["delivery_mode"]})
    claim_communications_admin_operation(supabase, club_id=club_id,
        operation_key=operation_key, operation_type=_root_type(tournament_id), request_json=request)
    return get_tournament_broadcast(supabase, club_id=club_id,
        tournament_id=tournament_id, operation_key=operation_key)


def list_tournament_broadcasts(supabase: Any, *, club_id: str, tournament_id: str) -> dict:
    _authorize(supabase, club_id, tournament_id)
    rows = supabase.table(TABLE).select("operation_key,request_json,created_at").eq("club_id", club_id).eq(
        "operation_type", _root_type(tournament_id)).order("created_at", desc=True).limit(10).execute().data or []
    return {"broadcasts": [{"operation_key": row["operation_key"], "created_at": row.get("created_at"),
        "subject": row["request_json"]["review"]["preview"]["subject"],
        "recipient_count": len(row["request_json"]["review"]["recipients"])} for row in rows]}


def get_tournament_broadcast(supabase: Any, *, club_id: str, tournament_id: str, operation_key: str) -> dict:
    operation_key = _key(operation_key)
    row = _operation(supabase, club_id, tournament_id, operation_key)
    request = row["request_json"]
    recipients = request["review"]["recipients"]
    # Chunk the IN query to avoid URL and PostgREST row-limit truncation.
    keys = [str(uuid5(UUID(operation_key), recipient["email"])) for recipient in recipients]
    outcomes = {}
    for start in range(0, len(keys), 100):
        attempts = supabase.table(TABLE).select("operation_key,status,result_json").eq("club_id", club_id).eq(
            "operation_type", _recipient_type(operation_key)).in_("operation_key", keys[start:start+100]).execute().data or []
        outcomes.update({attempt["operation_key"]: attempt for attempt in attempts})
    results = []
    for index, (recipient, key) in enumerate(zip(recipients, keys)):
        attempt = outcomes.get(key)
        result = (attempt or {}).get("result_json") or {}
        results.append({"index": index, "name": recipient["name"], "email": recipient["email"],
            "status": result.get("status", "uncertain" if attempt else "pending"),
            "detail": result.get("detail", "Check delivery before sending another email." if attempt else "")})
    return {"ok": True, "operation_key": operation_key, "subject": request["review"]["preview"]["subject"],
        "message": request["message"], "created_at": row.get("created_at"),
        "include_registration_events": request.get("include_registration_events", False),
        "include_registration_edit_links": request.get("include_registration_edit_links", False),
        "delivery_mode": request["review"]["delivery_mode"], "sender": request["review"]["sender"],
        "recipients": results, "recipient_count": len(results),
        "pending_count": sum(result["status"] == "pending" for result in results)}


def send_tournament_broadcast_recipient(supabase: Any, *, club_id: str, tournament_id: str,
        operation_key: str, recipient_index: int, confirmation_text: str,
        actor_email: str, actor_role: str) -> dict:
    operation_key = _key(operation_key)
    row = _operation(supabase, club_id, tournament_id, operation_key)
    require_staging_communications_mutations()
    if confirmation_text != CONFIRM_SEND:
        raise ValueError("Confirm the selected participants before sending.")
    request = row["request_json"]
    review = request["review"]
    if not 0 <= recipient_index < len(review["recipients"]):
        raise ValueError("Recipient not found in this email.")
    recipient = review["recipients"][recipient_index]
    child_key = str(uuid5(UUID(operation_key), recipient["email"]))
    child_request = {"broadcast_id": operation_key, "recipient_index": recipient_index,
        "preview_fingerprint": request["preview_fingerprint"]}
    child_type = _recipient_type(operation_key)
    existing = get_communications_admin_operation(supabase, operation_key=child_key)
    if existing:
        validate_communications_admin_operation(existing, club_id=club_id,
            operation_type=child_type, request_json=child_request)
        return {"index": recipient_index, **(existing.get("result_json") or {
            "status": "uncertain", "detail": "This email is sending or its delivery is uncertain. It will not be sent again."})}

    current = broadcast_delivery_settings()
    if not current["enabled"] or current["delivery_mode"] != review["delivery_mode"] or current["sender"] != review["sender"]:
        raise ValueError("Email delivery settings changed. Review the results before preparing a new email.")
    # Revalidate this recipient against the current registration, but never
    # substitute a new email address into the already-confirmed audience.
    preview = build_admin_tournament_broadcast_preview(supabase, club_id=club_id,
        tournament_id=tournament_id, registration_ids=request["registration_ids"],
        subject=request["subject"], message=request["message"], include_cancelled=request["include_cancelled"],
        include_sponsor_logos=False, reviewed_email_sponsors=review.get("email_sponsors"),
        include_registration_events=request.get("include_registration_events", False),
        include_registration_edit_links=request.get("include_registration_edit_links", False))
    if preview["preview_fingerprint"] != request["preview_fingerprint"]:
        raise ValueError("Participant details changed, registration events changed, sponsor details changed, or edit links changed. Review the results and preview a new email for the remaining participants.")
    _audit(supabase, club_id=club_id, actor_email=actor_email, actor_role=actor_role,
        operation_key=operation_key, action="tournament_broadcast_recipient_intent",
        details={"recipient_index": recipient_index, "attempt_id": child_key})
    try:
        # INSERT, never upsert: only the request that owns the unique claim may
        # enter SMTP. The general claim helper intentionally permits replays.
        claimed = supabase.table(TABLE).insert({"operation_key": child_key,
            "club_id": club_id, "operation_type": child_type, "request_json": child_request,
            "status": "started"}).execute().data
    except Exception as exc:
        existing = get_communications_admin_operation(supabase, operation_key=child_key)
        if existing:
            validate_communications_admin_operation(existing, club_id=club_id,
                operation_type=child_type, request_json=child_request)
            return {"index": recipient_index, **(existing.get("result_json") or {
                "status": "uncertain", "detail": "This email is already being handled. Check its result shortly."})}
        raise RuntimeError("The email attempt could not be recorded. No email was sent.") from exc
    if not claimed:
        raise RuntimeError("The email claim was not confirmed. Check the email results before continuing.")
    try:
        edit_links = None
        if request.get("include_registration_edit_links"):
            # A dry run never issues working bearer links. Live links start their
            # normal 48-hour lifetime here, even when a batch is resumed later.
            edit_links = recipient["registration_edit_links"] if review["delivery_mode"] == EMAIL_MODE_DRY_RUN else build_tournament_broadcast_edit_links(
                tournament_id=tournament_id, recipient=recipient, context=review["edit_link_context"])
        delivery = send_tournament_registrant_broadcast_email(
            tournament_name="", recipient_email=recipient["email"], recipient_name=recipient["name"],
            subject=review["preview"]["subject"], message=request["message"],
            personalize_greeting=False, message_id=child_key, email_sponsors=review.get("email_sponsors"),
            registration_events=recipient.get("registration_events"), registration_edit_links=edit_links)
        result = {"status": delivery["status"], "provider_message_id": delivery.get("provider_message_id"),
            "detail": "Accepted by the mail server." if delivery["status"] == "sent" else "Test only; no participant email was sent."}
    except Exception:
        # SMTP can disconnect after accepting DATA. Treat all ambiguous failures
        # conservatively and never expose credentials or provider diagnostics.
        result = {"status": "uncertain", "detail": "Delivery could not be confirmed. Check the recipient's inbox before sending another email."}
    try:
        complete_communications_admin_operation(supabase, club_id=club_id,
            operation_key=child_key, result_json={**result, "finished_at": datetime.now(timezone.utc).isoformat()})
    except Exception:
        return {"index": recipient_index, "status": "uncertain",
            "detail": "The delivery result could not be saved. This email will not be sent again automatically."}
    return {"index": recipient_index, **result}
