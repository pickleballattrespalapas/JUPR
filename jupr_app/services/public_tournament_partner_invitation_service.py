"""Public partner messages and scoped email acceptance, including guest senders."""
from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

from jupr_app.domain.tournament_partner_invitation_tokens import build_partner_invitation_token, verify_partner_invitation_token
from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token, registration_edit_email_hash
from jupr_app.domain.tournament_registration_repo import get_public_tournament_bundle, registration_has_imported_draw_selection
from jupr_app.domain.notifications.tournament_partner_invitation_email import send_partner_invitation_email
from jupr_app.services.public_tournament_partner_request_service import (
    _display_name, _email_is_contact_allowed, _is_public_partner_board_target,
    _registration_is_active, _selection_by_public_entry_key,
)
from jupr_app.services.public_tournament_registration_edit_service import _public_web_base_url, _verified_bundle
from jupr_app.services.public_tournament_registration_service import (
    _clean_email, _EMAIL_RE, build_tournament_registration_player_profile,
    validate_and_clean_tournament_selection,
)
from jupr_app.services.tournament_email_sponsor_service import load_tournament_email_sponsors

TABLE = "tournament_partner_invitations"
TERMINAL = {"COMPLETED", "DECLINED", "CANCELLED", "EXPIRED"}


class InvitationConflictError(ValueError):
    pass


class InvitationRateLimitError(ValueError):
    pass


def _rows(response: Any) -> list[dict]:
    return list(response.data or [])


def _one(db: Any, table: str, row_id: str) -> dict:
    rows = _rows(db.table(table).select("*").eq("id", row_id).limit(1).execute())
    return rows[0] if rows else {}


def _rpc(db: Any, name: str, args: dict) -> Any:
    try:
        return db.rpc(name, args).execute().data
    except Exception as exc:
        message = str(getattr(exc, "message", ""))
        if message.startswith("JUPR_INVITATION_RATE: "):
            raise InvitationRateLimitError(message.split(": ", 1)[1]) from exc
        if message.startswith("JUPR_INVITATION: "):
            raise InvitationConflictError(message.split(": ", 1)[1]) from exc
        raise RuntimeError("We couldn’t update this partner request. Please try again.") from exc


def _expired(row: dict) -> bool:
    return datetime.fromisoformat(str(row["expires_at"]).replace("Z", "+00:00")) <= datetime.now(timezone.utc)


def _context(db: Any, row: dict) -> dict:
    tournament, settings, _, events = get_public_tournament_bundle(db,
        club_id=row["club_id"], tournament_id=str(row["tournament_id"]))
    if not tournament or not settings:
        raise ValueError("This tournament is no longer available.")
    target = _one(db, "tournament_registration_selections", row["target_selection_id"])
    registration = _one(db, "tournament_registrations", str(target.get("registration_id") or ""))
    event = next((e for e in events if str(e.get("id")) == row["event_option_id"]), {})
    if not target or not registration or not event or str(target.get("tournament_id")) != str(row["tournament_id"]):
        raise ValueError("This partner request is no longer available.")
    return {"tournament": tournament, "settings": settings, "event": event,
            "target": target, "target_registration": registration}


def _available(ctx: dict) -> bool:
    return bool(ctx["settings"].get("partner_board_enabled")
        and _registration_is_active(ctx["target_registration"])
        and ctx["target_registration"].get("wants_partner_board_contact")
        and _is_public_partner_board_target(ctx["target"], ctx["event"]))


def _token(row: dict, ctx: dict, role: str) -> str:
    email = row["requester_email"] if role == "requester" else ctx["target_registration"]["email"]
    return build_partner_invitation_token(invitation_id=row["id"], club_id=row["club_id"],
        tournament_id=str(row["tournament_id"]), role=role, email=email,
        expires_at=int(datetime.fromisoformat(str(row["expires_at"]).replace("Z", "+00:00")).timestamp()))


def _action_url(club_slug: str, token: str) -> str:
    # Fragments are never sent in HTTP requests or Referer headers.
    return f"{_public_web_base_url()}/clubs/{club_slug}/tournament-partner-request#token={token}"


def _verified(db: Any, club_id: str, token: str) -> tuple[dict, dict, str]:
    claims = verify_partner_invitation_token(token, club_id=str(club_id))
    row = _one(db, TABLE, claims["id"])
    if not row or row["club_id"] != str(club_id) or str(row["tournament_id"]) != claims["tournament"]:
        raise ValueError("This partner request link is no longer available.")
    ctx = _context(db, row)
    email = row["requester_email"] if claims["role"] == "requester" else ctx["target_registration"]["email"]
    if claims["email_hash"] != registration_edit_email_hash(email):
        raise ValueError("This partner request link is no longer available.")
    return row, ctx, claims["role"]


def _requester_registration(db: Any, row: dict, registration_id: str | None = None, *, email_verified: bool = False) -> dict | None:
    rows = [_one(db, "tournament_registrations", registration_id)] if registration_id else _rows(
        db.table("tournament_registrations").select("*").eq("tournament_id", str(row["tournament_id"])).execute())
    normalize = lambda name: re.sub(r"\s+", " ", str(name or "")).strip().casefold()
    email_matches = [r for r in rows if str(r.get("tournament_id")) == str(row["tournament_id"])
        and _clean_email(r.get("email")) == row["requester_email"] and _registration_is_active(r)]
    if len(email_matches) == 1 and (row.get("verified_at") or email_verified):
        return email_matches[0]
    matches = [r for r in email_matches if normalize(_display_name(r)) == normalize(row["requester_name"])]
    if len(email_matches) == 1 and not matches:
        raise ValueError("The name on this request does not match the sender’s registration. Ask them to resend using their full registered name.")
    if len(matches) > 1 or email_matches and not matches:
        raise ValueError("More than one registration matches your details. Open your registration edit link to send this request.")
    return matches[0] if matches else None


def _pairing_candidate(db: Any, row: dict, ctx: dict, registration_id: str | None = None) -> tuple[str | None, dict]:
    registration = _requester_registration(db, row, registration_id)
    if not registration:
        return None, {}
    if registration["id"] == ctx["target_registration"]["id"]:
        raise ValueError("You cannot request yourself as a partner.")
    selections = _rows(db.table("tournament_registration_selections").select("*")
        .eq("registration_id", registration["id"]).eq("event_option_id", row["event_option_id"]).execute())
    if not selections:
        return None, {}
    if len(selections) != 1 or selections[0].get("partner_mode") != "NEEDS_PARTNER":
        raise InvitationConflictError("The sender already has a partner or needs to update their entry for this division.")
    for reg in (registration, ctx["target_registration"]):
        if registration_has_imported_draw_selection(db, tournament_id=str(row["tournament_id"]), registration_id=reg["id"]):
            raise InvitationConflictError("Tournament play has already been prepared for this registration. Please contact the organizer to change partners.")
    profiles = [build_tournament_registration_player_profile(db, club_id=row["club_id"], registration=r)
        for r in (registration, ctx["target_registration"])]
    # Use the same age, gender and skill rules as entering a named partner in registration.
    for primary, partner, reg, partner_reg in (
        (profiles[0], profiles[1], registration, ctx["target_registration"]),
        (profiles[1], profiles[0], ctx["target_registration"], registration),
    ):
        validate_and_clean_tournament_selection(db, club_id=row["club_id"], tournament_id=str(row["tournament_id"]),
            event=ctx["event"], settings=ctx["settings"], primary_registration_id=reg["id"], player_profile=primary,
            raw_selection={"event_option_id": row["event_option_id"], "partner_mode": "HAS_PARTNER",
                "partner_name": _display_name(partner_reg), "partner_email": partner_reg["email"],
                "partner_skill": partner.get("doubles_skill"), "partner_age": partner.get("age"), "partner_gender": partner.get("gender")})
    return selections[0]["id"], {"event": ctx["event"].get("updated_at"), "requester_registration": registration.get("updated_at"),
        "target_registration": ctx["target_registration"].get("updated_at"),
        "requester_selection": selections[0].get("updated_at"), "target_selection": ctx["target"].get("updated_at")}


def _deliver(db: Any, row: dict, ctx: dict, club_slug: str, kind: str) -> str:
    requester = _requester_registration(db, row) if row.get("verified_at") else None
    requester_name = _display_name(requester) if requester else row["requester_name"]
    target_notice = kind.endswith("target")
    role = "target" if target_notice else "requester"
    address = ctx["target_registration"]["email"] if target_notice else row["requester_email"]
    if not _email_is_contact_allowed(address):
        return "failed"
    attempt = uuid4().hex
    claimed = _rpc(db, "claim_partner_invitation_delivery", {"p_invitation_id": row["id"], "p_kind": kind, "p_attempt_id": attempt})
    if not claimed:
        rows = _rows(db.table("tournament_partner_invitation_deliveries").select("status")
            .eq("invitation_id", row["id"]).eq("kind", kind).limit(1).execute())
        return str(rows[0]["status"]) if rows else "sending"
    copies = {
        "verify_requester": ("Confirm your partner request", "Confirm your email to send your message and partner request.", "Send my partner request"),
        "request_target": (f"{requester_name} would like to partner with you", "Read their message below. Accepting will pair your registrations automatically. If they still need to register, your partnership will be reserved until they finish.", "Accept partnership"),
        "reserved_requester": ("Your partner request was accepted", "Your partnership is reserved. Complete registration for this division and PCS will pair you automatically. Your reservation lasts until " + str(row['expires_at'])[:10] + ".", "Complete registration"),
        "reserved_target": ("Your partnership is reserved", "We’ve asked your partner to complete registration. PCS will finish pairing you automatically when they do. Your reservation lasts until " + str(row['expires_at'])[:10] + ".", "View partnership"),
        "completed_requester": ("You’re partnered up!", "Both registrations are now paired for this division and your team is on the roster.", "View partnership"),
        "completed_target": ("You’re partnered up!", "Both registrations are now paired for this division and your team is on the roster.", "View partnership"),
        "declined_requester": ("Partner request declined", "This player declined your request. You can find another player on the Partner Board.", "View request"),
        "cancelled_requester": ("Partner request cancelled", "This partner request has been cancelled. You can find another player on the Partner Board.", "View request"),
        "cancelled_target": ("Partner request cancelled", "This partner request has been cancelled. You can find another player on the Partner Board.", "View request"),
    }
    title, description, label = copies[kind]
    try:
        status = send_partner_invitation_email(to_email=address, title=title, description=description,
            tournament_name=ctx["tournament"]["name"], division_name=ctx["event"].get("label") or ctx["event"].get("division_name") or "Division",
            requester_name=requester_name, target_name=_display_name(ctx["target_registration"]),
            message=row["message"] if kind in {"verify_requester", "request_target"} else "",
            requester_email=row["requester_email"] if kind == "request_target" else "",
            action_url=_action_url(club_slug, _token(row, ctx, role)), action_label=label,
            sponsors=load_tournament_email_sponsors(db, club_id=row["club_id"], tournament_id=str(row["tournament_id"])))
    except Exception:
        status = "failed"
    db.table("tournament_partner_invitation_deliveries").update({"status": status, "updated_at": datetime.now(timezone.utc).isoformat()})\
        .eq("invitation_id", row["id"]).eq("kind", kind).eq("attempt_id", attempt).execute()
    return status


def _notify(db: Any, row: dict, ctx: dict, club_slug: str) -> dict[str, str]:
    kinds = {
        "UNVERIFIED": ["verify_requester"], "PENDING": ["request_target"],
        "RESERVED": ["reserved_requester", "reserved_target"],
        "COMPLETED": ["completed_requester", "completed_target"],
        "DECLINED": ["declined_requester"], "CANCELLED": ["cancelled_requester", "cancelled_target"],
    }.get(row["status"], [])
    result = {}
    for kind in kinds:
        try:
            result[kind] = _deliver(db, row, ctx, club_slug, kind)
        except Exception:
            result[kind] = "failed"
    return result


def create_invitation(db: Any, *, club_id: str, club_slug: str, payload: dict) -> dict:
    if payload.get("website"):
        return {"ok": True, "status": "PENDING", "notification_status": {}}
    tournament, settings, _, events = get_public_tournament_bundle(db, club_id=str(club_id),
        tournament_id=payload.get("tournament_id"), registration_slug=payload.get("registration_slug"))
    if not tournament or not settings or not settings.get("partner_board_enabled"):
        raise ValueError("The Partner Board is not available for this tournament.")
    tid = str(tournament["id"])
    target = _selection_by_public_entry_key(db, tournament_id=tid, board_entry_key=payload["board_entry_key"])
    if not target:
        raise ValueError("This player is no longer listed on the Partner Board.")
    name = str(payload.get("name") or "").strip()
    email = _clean_email(payload.get("email"))
    verified = False
    if payload.get("edit_token"):
        _, bundle = _verified_bundle(db, club_id=str(club_id), tournament_id=tid, edit_token=payload["edit_token"])
        registration = bundle["registration"]
        if not _registration_is_active(registration):
            raise ValueError("Your registration is no longer active.")
        name, email, verified = _display_name(registration), _clean_email(registration["email"]), True
        if registration["id"] == target["registration_id"]:
            raise ValueError("You cannot request yourself as a partner.")
    message = str(payload.get("message") or "").strip()
    if not name or len(name) > 160 or any(c in name for c in "\r\n") or not _EMAIL_RE.fullmatch(email):
        raise ValueError("Enter your full name and a valid email address.")
    if not message or len(message) > 2000:
        raise ValueError("Enter a message of up to 2,000 characters.")
    if not _email_is_contact_allowed(email):
        raise ValueError("We couldn’t send to this email address. Check the address or contact the organizer.")
    row = _rpc(db, "create_tournament_partner_invitation", {"p_invitation": {
        "id": "pinv_" + uuid4().hex, "club_id": str(club_id), "tournament_id": tid,
        "target_selection_id": target["id"], "requester_name": name, "requester_email": email,
        "message": message, "request_key": payload["request_key"], "verified": verified,
        "send_directly": True}})
    # Anonymous responses always have the same shape; no tokens, contact details,
    # registration lookup results, or target email leave this boundary.
    notices = _notify(db, row, _context(db, row), club_slug)
    return {"ok": True, "status": "PENDING", "notification_status": notices}


def review_invitation(db: Any, *, club_id: str, club_slug: str, token: str) -> dict:
    row, ctx, role = _verified(db, club_id, token)
    # A requester capability is sent only to their mailbox, never returned by
    # the public form. Possession still proves email ownership for edit links.
    requester = _requester_registration(db, row, email_verified=role == "requester") if role == "requester" or row.get("verified_at") else None
    status = "EXPIRED" if _expired(row) and row["status"] not in TERMINAL else row["status"]
    if status in {"UNVERIFIED", "PENDING", "RESERVED"} and not _available(ctx):
        status = "CANCELLED"
    actions = []
    if role == "requester":
        if status == "UNVERIFIED": actions = ["verify", "cancel"]
        elif status in {"PENDING", "RESERVED"}: actions = ["cancel"]
    elif status == "PENDING": actions = ["accept", "decline"]
    elif status == "RESERVED": actions = ["cancel"]
    query = urlencode({"tournament": ctx["settings"].get("registration_slug")}) if ctx["settings"].get("registration_slug") else urlencode({"tournament_id": str(row["tournament_id"])})
    result = {"ok": True, "role": role, "status": status, "actions": actions,
        "requester_name": _display_name(requester) if requester else row["requester_name"], "target_name": _display_name(ctx["target_registration"]),
        "tournament_name": ctx["tournament"]["name"], "division_name": ctx["event"].get("label") or ctx["event"].get("division_name"),
        "message": row["message"], "expires_at": row["expires_at"],
        "board_url": f"/clubs/{club_slug}/tournament-partner-board?{query}",
        "roster_url": f"/clubs/{club_slug}/tournament-roster?{query}"}
    if role == "requester" and status in {"PENDING", "RESERVED"}:
        registration = requester
        base = f"/clubs/{club_slug}/tournament-registration"
        if registration:
            edit = build_registration_edit_token(tournament_id=str(row["tournament_id"]), registration_id=registration["id"], email=registration["email"])
            base += "/edit"
            query += "&" + urlencode({"edit_token": edit})
        result["registration_url"] = f"{base}?{query}#partner_invitation={token}"
        # These are the verified sender's own details, only on their private link.
        result["registration_prefill"] = {"name": result["requester_name"], "email": row["requester_email"], "event_option_id": row["event_option_id"]}
    return result


def act_on_invitation(db: Any, *, club_id: str, club_slug: str, token: str, action: str) -> dict:
    row, ctx, role = _verified(db, club_id, token)
    allowed = {"requester": {"verify", "cancel", "complete", "retry_email"}, "target": {"accept", "decline", "cancel", "retry_email"}}[role]
    if action not in allowed or action == "cancel" and role == "target" and row["status"] != "RESERVED":
        raise ValueError("This link does not allow that action.")
    if action != "retry_email":
        selection_id, versions = (None, {})
        if action in {"accept", "complete"} and row["status"] not in TERMINAL:
            selection_id, versions = _pairing_candidate(db, row, ctx)
        row = _rpc(db, "transition_tournament_partner_invitation", {"p_invitation_id": row["id"], "p_action": action,
            "p_requester_selection_id": selection_id, "p_versions": versions})
        if row.get("stale"):
            raise InvitationConflictError("This request is no longer available. One player may already have a partner, or the request has expired.")
    notices = _notify(db, row, ctx, club_slug)
    return {**review_invitation(db, club_id=club_id, club_slug=club_slug, token=token), "notification_status": notices}


def validate_invitation_registration(db: Any, *, club_id: str, tournament_id: str, token: str, payload: dict) -> None:
    row, _, role = _verified(db, club_id, token)
    if role != "requester" or str(row["tournament_id"]) != str(tournament_id):
        raise ValueError("Use the registration link from your partner request email.")
    if _clean_email(payload.get("email")) != row["requester_email"]:
        raise ValueError("Use the same email address as your partner request.")
    if row["status"] not in {"PENDING", "RESERVED", "COMPLETED"}:
        raise InvitationConflictError("This partnership is no longer available. Return to the Partner Board.")
    selections = [s for s in payload.get("selections", []) if s.get("event_option_id") == row["event_option_id"]]
    if not selections or (row["status"] != "COMPLETED" and selections[0].get("partner_mode") != "NEEDS_PARTNER"):
        raise ValueError("Keep the requested division selected. PCS will add your partner automatically after registration.")


def finish_invitation_registration(db: Any, *, club_id: str, club_slug: str, token: str, registration_id: str) -> dict:
    try:
        row, ctx, role = _verified(db, club_id, token)
        if role != "requester": raise ValueError("Use your own partner request link.")
        if row["status"] == "RESERVED":
            selection_id, versions = _pairing_candidate(db, row, ctx, registration_id)
            row = _rpc(db, "transition_tournament_partner_invitation", {"p_invitation_id": row["id"], "p_action": "complete",
                "p_requester_selection_id": selection_id, "p_versions": versions})
            if row.get("stale"): raise InvitationConflictError("The partnership is no longer available.")
            _notify(db, row, ctx, club_slug)
        return {"status": row["status"]}
    except Exception:
        # Registration already saved. Never report that save as failed or invite
        # a duplicate submission; the secure page provides completion recovery.
        return {"status": "RETRY_REQUIRED", "message": "Your registration is saved. Open your partner request to finish pairing."}
