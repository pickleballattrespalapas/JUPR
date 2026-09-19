"""Reviewed, club-scoped interclub emails with durable at-most-once delivery.

An email operation freezes its audience and message. Each recipient is claimed
before preparing a capability or touching SMTP; ambiguous attempts are not
retried automatically. Staging exercises the same path in dry-run mode.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
from html import escape
import json
from typing import Any
from uuid import UUID, uuid5
from zoneinfo import ZoneInfo

from jupr_app.config import EMAIL_MODE_DRY_RUN, EMAIL_MODE_LIVE, EMAIL_MODE_STAGING_REDIRECT, get_email_mode, get_env_or_default
from jupr_app.domain.admin_activity_log import build_activity_payload, write_admin_activity_log
from jupr_app.domain.notifications.player_profile_update_repo import (
    claim_communications_admin_operation, complete_communications_admin_operation,
    get_communications_admin_operation, validate_communications_admin_operation,
    validate_email_address,
)
from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status, send_email_with_inline_chart
from jupr_app.services.staging_write_guard import require_staging_communications_mutations, staging_communications_mutations_enabled

TABLE = "communications_admin_operations"
MAX_RECIPIENTS = 200


class EmailNotFoundError(ValueError):
    """A batch is absent in the authenticated club and season."""


def _text(value: Any) -> str:
    return str(value or "").strip()


def _uuid(value: str) -> str:
    try:
        return str(UUID(str(value)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("A valid email operation ID is required.") from exc


def _all(query_factory) -> list[dict]:
    result = []
    for offset in range(0, 10000, 500):
        rows = query_factory().range(offset, offset + 499).execute().data or []
        result.extend(rows)
        if len(rows) < 500:
            return result
    raise ValueError("Too many contacts to review at once. Use the club's shared season signup link.")


def _one(db, table, **keys) -> dict | None:
    query = db.table(table).select("*")
    for key, value in keys.items():
        query = query.eq(key, value)
    rows = query.limit(1).execute().data or []
    return rows[0] if rows else None


def _authorized_context(db, club_id: str, season_id: str) -> tuple[dict, dict]:
    participation = _one(db, "pcs_interclub_participations", club_id=club_id, season_id=season_id)
    if not participation or participation.get("status") != "accepted":
        raise PermissionError("Your club must accept this season before inviting players.")
    season = _one(db, "pcs_interclub_seasons", id=season_id)
    club = _one(db, "clubs", id=club_id)
    if not season or not club:
        raise ValueError("Club or season unavailable.")
    return club, season


def delivery_settings() -> dict:
    try:
        mode = get_email_mode()
        smtp = get_smtp_config_status()
        return {"enabled": staging_communications_mutations_enabled() and (mode == EMAIL_MODE_DRY_RUN or smtp["ok"]),
            "delivery_mode": mode, "sender": {key: smtp[key] for key in ("from_email", "from_name", "reply_to")}}
    except (ValueError, RuntimeError):
        return {"enabled": False, "delivery_mode": "unavailable", "sender": {}}


def _email(value: Any) -> str | None:
    value = _text(value).lower()
    if len(value) > 254 or any(c in value for c in "\r\n\x00"):
        return None
    try:
        return validate_email_address(value)
    except ValueError:
        return None


def _date(value, tz="UTC") -> str:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(ZoneInfo(tz)).strftime("%B %-d, %Y at %-I:%M %p")


def build_audience(db, *, club_id: str, season_id: str, kind: str, meet_id: str | None = None) -> dict:
    club, season = _authorized_context(db, club_id, season_id)
    if kind not in {"season", "meet"} or (kind == "season" and meet_id) or (kind == "meet" and not meet_id):
        raise ValueError("Choose a season signup email or a specific meet invitation.")
    settings = _one(db, "pcs_interclub_pool_settings", club_id=club_id, season_id=season_id) or {}
    candidates = []
    state = {"club": {"id": club_id, "name": club["name"]}, "season": season["details"], "settings": settings}
    if kind == "season":
        contacts = _all(lambda: db.table("player_profile_update_subscriptions")
            .select("id,player_id,email,email_normalized,request_status,preferences_json,verified_at")
            .eq("club_id", club_id).eq("request_status", "active").order("id"))
        by_player: dict[str, list[dict]] = {}
        for contact in contacts:
            preferences = contact.get("preferences_json") or {}
            if preferences.get("optional_emails_enabled") is False or preferences.get("unsubscribe_scope") == "global":
                continue
            address = _email(contact.get("email_normalized") or contact.get("email"))
            if address:
                by_player.setdefault(str(contact["player_id"]), []).append({"id": contact["id"], "email": address})
        players = _all(lambda: db.table("players").select("id,name,active,inactive_at").eq("club_id", club_id).order("id"))
        for player in players:
            if player.get("active") is False or player.get("inactive_at"):
                continue
            matches = by_player.get(str(player["id"]), [])
            addresses = {r["email"] for r in matches}
            address = next(iter(addresses)) if len(addresses) == 1 else None
            candidates.append({"id": str(player["id"]), "name": player["name"], "email": address or "",
                "available": bool(address), "unavailable_reason": "" if address else "No single verified club email. Share the season signup link with this player.",
                "contact_ids": sorted(r["id"] for r in matches)})
        subject = f"Join {club['name']} for {season['details']['name']}"
        message = f"Would you like to play interclub pickleball for {club['name']} this season? Join our player pool using the link below. We will invite you separately for each meet, so you can choose the dates that work for you."
        ends_at = datetime.fromisoformat(season["details"]["end_date"]).replace(
            tzinfo=ZoneInfo(season["details"].get("timezone") or "America/Mazatlan")) + timedelta(days=1)
        season_open = ends_at > datetime.now(timezone.utc)
        available = bool(season_open and settings.get("open") and settings.get("share_id"))
        reason = ("Open season player signup before preparing invitation emails." if season_open
            else "This season has ended. Player signup invitations are closed.")
    else:
        meet = _one(db, "pcs_interclub_meets", id=meet_id, season_id=season_id)
        if not meet or club_id not in (meet.get("club_ids") or []):
            raise ValueError("This meet is not available for your club.")
        availability = _one(db, "pcs_interclub_availability_settings", club_id=club_id, season_id=season_id, meet_id=meet_id) or {}
        members = _all(lambda: db.table("pcs_interclub_pool_members").select("id,name,email,status,revision,player_id")
            .eq("club_id", club_id).eq("season_id", season_id).eq("status", "active").order("id"))
        for member in members:
            address = _email(member.get("email"))
            candidates.append({"id": str(member["id"]), "name": member["name"], "email": address or "",
                "available": bool(address), "unavailable_reason": "" if address else "This player needs a valid season signup email.",
                "revision": member["revision"]})
        host = _one(db, "clubs", id=meet["host_club_id"]) or {}
        timezone_name = season["details"].get("timezone") or "America/Mazatlan"
        when = _date(meet["starts_at"], timezone_name)
        deadline = _date(availability["deadline"], timezone_name) if availability.get("deadline") else None
        subject = f"Are you available? {season['details']['name']} — {when}"
        message = f"Our next interclub meet is at {host.get('name') or 'the host club'} on {when} ({timezone_name}). Let us know whether you are available, maybe, or unavailable using your personal link below. Your club administrator will confirm the final lineup."
        if deadline:
            message += f" Please respond by {deadline} ({timezone_name})."
        available = bool(availability.get("open") and availability.get("deadline")
            and datetime.fromisoformat(availability["deadline"].replace("Z", "+00:00")) > datetime.now(timezone.utc)
            and datetime.fromisoformat(meet["starts_at"].replace("Z", "+00:00")) > datetime.now(timezone.utc))
        reason = "Open availability signup with a future response deadline before inviting players."
        state.update(meet=meet, availability=availability)
    candidates.sort(key=lambda row: (row["name"].casefold(), row["id"]))
    delivery = delivery_settings()
    return {"kind": kind, "meet_id": meet_id, "candidates": candidates, "defaults": {"subject": subject, "message": message},
        "delivery_mode": delivery["delivery_mode"], "send_available": bool(available and delivery["enabled"]),
        "send_unavailable_reason": "" if available and delivery["enabled"] else (reason if not available else "Email preparation is unavailable in this environment."),
        "_state": state, "_delivery": delivery}


def _bodies(subject: str, message: str, name: str, links: list[dict], kind: str) -> dict:
    label = "Join the season player pool" if kind == "season" else "Respond for this meet"
    text = f"Hi {name},\n\n{message}\n\n" + "\n".join(f"{row.get('name') or label}: {row['url']}" for row in links)
    paragraphs = "".join(f"<p>{escape(p).replace(chr(10), '<br>')}</p>" for p in message.split("\n\n"))
    actions = "".join(f'<p><a href="{escape(row["url"], quote=True)}" style="display:inline-block;padding:14px 20px;background:#1d4ed8;color:#fff;border-radius:8px;text-decoration:none;font-weight:bold">{escape((row.get("name") + ": ") if row.get("name") else "")}{label}</a></p>' for row in links)
    html = f'<!doctype html><html><body style="font-family:Arial,sans-serif;color:#1f2937"><h1>{escape(subject)}</h1><p>Hi {escape(name)},</p>{paragraphs}{actions}<p>No player account is needed. Joining the pool or reporting availability does not assign you to a team.</p></body></html>'
    return {"subject": subject, "text": text, "html": html}


def preview_email(db, *, club_id: str, season_id: str, kind: str, recipient_ids: list[str], subject: str, message: str, meet_id: str | None = None) -> dict:
    subject, message = _text(subject), _text(message)
    if not subject or len(subject) > 200 or any(c in subject for c in "\r\n\x00") or not message or len(message) > 10000 or "\x00" in message:
        raise ValueError("Enter a subject up to 200 characters and a message up to 10,000 characters.")
    ids = sorted(set(str(value) for value in recipient_ids))
    if not ids or len(ids) > MAX_RECIPIENTS:
        raise ValueError(f"Choose between 1 and {MAX_RECIPIENTS} players.")
    audience = build_audience(db, club_id=club_id, season_id=season_id, kind=kind, meet_id=meet_id)
    available = {r["id"]: r for r in audience["candidates"] if r["available"]}
    if any(i not in available for i in ids):
        raise ValueError("A selected player or email is unavailable in this club. Refresh the players and select again.")
    grouped: dict[str, dict] = {}
    for member_id in ids:
        row = available[member_id]
        group = grouped.setdefault(row["email"], {"id": member_id, "name": row["name"], "email": row["email"], "members": []})
        group["members"].append(row)
    recipients = sorted(grouped.values(), key=lambda row: (row["name"].casefold(), row["email"]))
    for row in recipients:
        row["name"] = ", ".join(member["name"] for member in row["members"])
    from services.api.interclub_player_pool_routes import pool_signup_url
    preview_url = pool_signup_url(audience["_state"]["settings"].get("share_id")) if kind == "season" else "#personal-meet-response-link"
    first = recipients[0]
    links = [{"name": m["name"] if kind == "meet" else "", "url": preview_url} for m in first["members"]] if kind == "meet" else [{"url": preview_url}]
    preview = _bodies(subject, message, first["name"], links, kind)
    fingerprint = sha256(json.dumps({"club_id": club_id, "season_id": season_id, "kind": kind, "meet_id": meet_id,
        "recipient_ids": ids, "recipients": recipients, "subject": subject, "message": message,
        "state": audience["_state"], "delivery": audience["_delivery"]}, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    return {"kind": kind, "meet_id": meet_id, "recipient_count": len(recipients), "player_count": len(ids), "recipients": recipients,
        "preview": preview, "preview_fingerprint": fingerprint, "delivery_mode": audience["delivery_mode"],
        "sender": audience["_delivery"]["sender"], "send_available": audience["send_available"], "send_unavailable_reason": audience["send_unavailable_reason"]}


def _root_type(season_id):
    return f"interclub_player_email:{season_id}"


def _recipient_type(operation_key):
    return f"interclub_player_email_recipient:{operation_key}"


def _operation(db, club_id, season_id, operation_key):
    _authorized_context(db, club_id, season_id)
    row = get_communications_admin_operation(db, operation_key=_uuid(operation_key))
    if not row or row.get("club_id") != club_id or row.get("operation_type") != _root_type(season_id):
        raise EmailNotFoundError("Email not found in this club and season.")
    return row


def _audit(db, *, club_id, user, operation_key, action, details):
    result = write_admin_activity_log(db, build_activity_payload(club_id=club_id, actor_email=user.email,
        actor_role="administrator", action_type=action, entity_type="interclub_player_email", entity_id=operation_key,
        after_json=details, source_page="next_interclub_player_pool", flagged_for_review=True))
    if not result.ok:
        raise RuntimeError("The email audit could not be saved. Check the email results before continuing.")


def create_email(db, *, club_id, season_id, user, operation_key, preview_fingerprint, **payload):
    _authorized_context(db, club_id, season_id)
    require_staging_communications_mutations()
    operation_key = _uuid(operation_key)
    request = {**payload, "recipient_ids": sorted(set(payload["recipient_ids"])), "subject": payload["subject"].strip(),
        "message": payload["message"].strip(), "actor_id": user.user_id, "preview_fingerprint": preview_fingerprint}
    existing = get_communications_admin_operation(db, operation_key=operation_key)
    if existing:
        validate_communications_admin_operation(existing, club_id=club_id, operation_type=_root_type(season_id),
            request_json={**request, "review": existing["request_json"].get("review")})
        return get_email(db, club_id=club_id, season_id=season_id, operation_key=operation_key)
    preview = preview_email(db, club_id=club_id, season_id=season_id, **payload)
    if not preview["send_available"]:
        raise ValueError(preview["send_unavailable_reason"])
    if preview_fingerprint != preview["preview_fingerprint"]:
        raise ValueError("The selected players, email, or meet changed. Preview the email again.")
    request["review"] = {key: preview[key] for key in ("recipients", "delivery_mode", "sender")}
    _audit(db, club_id=club_id, user=user, operation_key=operation_key, action="interclub_player_email_confirmed",
        details={"season_id": season_id, "meet_id": payload.get("meet_id"), "recipient_count": preview["recipient_count"], "delivery_mode": preview["delivery_mode"]})
    claim_communications_admin_operation(db, club_id=club_id, operation_key=operation_key, operation_type=_root_type(season_id), request_json=request)
    return get_email(db, club_id=club_id, season_id=season_id, operation_key=operation_key)


def get_email(db, *, club_id, season_id, operation_key):
    operation = _operation(db, club_id, season_id, operation_key)
    request = operation["request_json"]
    recipients = request["review"]["recipients"]
    keys = [str(uuid5(UUID(operation_key), recipient["email"])) for recipient in recipients]
    attempts = {}
    for offset in range(0, len(keys), 100):
        rows = db.table(TABLE).select("operation_key,result_json").eq("club_id", club_id).eq(
            "operation_type", _recipient_type(operation_key)).in_("operation_key", keys[offset:offset+100]).execute().data or []
        attempts.update({row["operation_key"]: row for row in rows})
    results = []
    for index, (recipient, key) in enumerate(zip(recipients, keys)):
        attempt = attempts.get(key)
        result = (attempt or {}).get("result_json") or {}
        results.append({"index": index, "name": recipient["name"], "email": recipient["email"],
            "status": result.get("status", "uncertain" if attempt else "pending"),
            "detail": result.get("detail", "Check delivery before preparing another email." if attempt else ""),
            "links": result.get("links", [])})
    return {"operation_key": operation_key, "kind": request["kind"], "meet_id": request.get("meet_id"),
        "subject": request["subject"], "recipients": results, "recipient_count": len(results),
        "pending_count": sum(row["status"] == "pending" for row in results), "delivery_mode": request["review"]["delivery_mode"]}


def _links(db, user, club_id, season_id, request, recipient):
    from services.api.interclub_player_pool_routes import prepare_meet_invitations, pool_response_url, pool_signup_url
    if request["kind"] == "season":
        settings = _one(db, "pcs_interclub_pool_settings", club_id=club_id, season_id=season_id)
        return [{"name": "Season signup", "url": pool_signup_url(settings["share_id"])}]
    prepared = prepare_meet_invitations(db, user, club_id, season_id, request["meet_id"], [m["id"] for m in recipient["members"]])
    season = _one(db, "pcs_interclub_seasons", id=season_id)
    by_member = {r["member_id"]: r for r in prepared["responses"]}
    return [{"name": m["name"], "url": pool_response_url(by_member[m["id"]], season, prepared["meet"])} for m in recipient["members"]]


def _deliver(*, recipient, request, links, message_id):
    mode = get_email_mode()
    if mode != request["review"]["delivery_mode"]:
        raise ValueError("Email delivery settings changed.")
    if mode == EMAIL_MODE_DRY_RUN:
        return {"status": "dry_run", "detail": "Test invitation prepared; no email was sent."}
    target = recipient["email"]
    subject = request["subject"]
    if mode == EMAIL_MODE_STAGING_REDIRECT:
        target = get_env_or_default("JUPR_STAGING_EMAIL_REDIRECT_TO")
        if not _email(target):
            raise ValueError("Staging email redirect is unavailable.")
        subject = f"[STAGING] {subject}"
    bodies = _bodies(subject, request["message"], recipient["name"], links, request["kind"])
    provider = send_email_with_inline_chart(to_email=target, subject=subject, html_body=bodies["html"], text_body=bodies["text"], message_id=message_id)
    return {"status": "sent" if mode == EMAIL_MODE_LIVE else "staging_redirect", "provider_message_id": provider, "detail": "Accepted by the mail server."}


def send_recipient(db, *, club_id, season_id, user, operation_key, recipient_index):
    operation_key = _uuid(operation_key)
    operation = _operation(db, club_id, season_id, operation_key)
    require_staging_communications_mutations()
    request = operation["request_json"]
    if not 0 <= recipient_index < len(request["review"]["recipients"]):
        raise ValueError("Recipient not found in this email.")
    recipient = request["review"]["recipients"][recipient_index]
    child_key = str(uuid5(UUID(operation_key), recipient["email"]))
    child_type = _recipient_type(operation_key)
    child_request = {"operation_key": operation_key, "recipient_index": recipient_index, "preview_fingerprint": request["preview_fingerprint"]}
    def replay(existing):
        validate_communications_admin_operation(existing, club_id=club_id, operation_type=child_type, request_json=child_request)
        return {"index": recipient_index, **(existing.get("result_json") or {"status": "uncertain", "detail": "This email is already being handled. It will not be sent again automatically."})}
    existing = get_communications_admin_operation(db, operation_key=child_key)
    if existing:
        return replay(existing)
    current = preview_email(db, club_id=club_id, season_id=season_id, **{k: request.get(k) for k in ("kind", "meet_id", "recipient_ids", "subject", "message")})
    if not current["send_available"] or current["preview_fingerprint"] != request["preview_fingerprint"]:
        raise ValueError("Players, signup settings, or the meet changed. Preview a new email before sending the remaining invitations.")
    _audit(db, club_id=club_id, user=user, operation_key=operation_key, action="interclub_player_email_recipient_intent", details={"recipient_index": recipient_index, "attempt_id": child_key})
    try:
        claimed = db.table(TABLE).insert({"operation_key": child_key, "club_id": club_id,
            "operation_type": child_type, "request_json": child_request, "status": "started"}).execute().data
    except Exception as exc:
        existing = get_communications_admin_operation(db, operation_key=child_key)
        if existing:
            return replay(existing)
        raise RuntimeError("The email attempt could not be recorded. No email was sent.") from exc
    if not claimed:
        raise RuntimeError("The email claim was not confirmed. Check its results before continuing.")
    try:
        links = _links(db, user, club_id, season_id, request, recipient)
        result = {**_deliver(recipient=recipient, request=request, links=links, message_id=child_key), "links": links}
    except Exception:
        result = {"status": "uncertain", "detail": "Delivery could not be confirmed. Check the invitation and email results before preparing another email."}
    try:
        complete_communications_admin_operation(db, club_id=club_id, operation_key=child_key, result_json=result)
    except Exception:
        result = {"status": "uncertain", "detail": "The result could not be saved. This invitation will not be sent again automatically."}
    return {"index": recipient_index, **result}
