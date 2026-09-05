from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlencode
from uuid import uuid4

from jupr_app.domain.notifications.tournament_pairing_interest_email import (
    send_pairing_interest_emails,
    send_pairing_status_emails,
)
from jupr_app.domain.tournament_partner_service import (
    ACCEPTED_STATUS,
    CANCELLED_STATUS,
    DECLINED_STATUS,
    PENDING_STATUS,
    create_partner_request_atomic,
    transition_partner_request_atomic,
)
from jupr_app.domain.tournament_public_references import public_tournament_reference_matches
from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token
from jupr_app.services.public_tournament_registration_edit_service import _public_web_base_url, _verified_bundle
from jupr_app.services.public_tournament_registration_service import _clean_email, _clean_text, _safe_bool


class PartnerRequestStaleError(ValueError):
    """A valid actor attempted a transition against superseded state."""


_PRIVATE_CONTACT_FIELDS = {
    "email",
    "phone",
    "partner_email",
    "partner_phone",
    "dupr_id",
    "notes",
    "wants_partner_board_contact",
}


def _safe_rows(resp: Any) -> list[dict[str, Any]]:
    try:
        return [dict(row) for row in (resp.data or [])]
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_rows(resp)
    return rows[0] if rows else None


def _selection_by_id(supabase: Any, selection_id: str) -> dict[str, Any] | None:
    return _safe_first(
        supabase.table("tournament_registration_selections")
        .select("*")
        .eq("id", str(selection_id))
        .limit(1)
        .execute()
    )


def _selection_by_public_entry_key(
    supabase: Any,
    *,
    tournament_id: str,
    board_entry_key: str,
) -> dict[str, Any] | None:
    rows = _safe_rows(
        supabase.table("tournament_registration_selections")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .execute()
    )
    for row in rows:
        selection_id = str(row.get("id") or "").strip()
        if public_tournament_reference_matches(
            board_entry_key,
            tournament_id=str(tournament_id),
            namespace="partner-board-selection",
            source_id=selection_id,
        ):
            return row
    return None


def _event_by_id(supabase: Any, event_option_id: str) -> dict[str, Any] | None:
    return _safe_first(
        supabase.table("tournament_event_options")
        .select("*")
        .eq("id", str(event_option_id))
        .limit(1)
        .execute()
    )


def _tournament_by_id(supabase: Any, tournament_id: str) -> dict[str, Any] | None:
    return _safe_first(
        supabase.table("tournaments")
        .select("id,name")
        .eq("id", str(tournament_id))
        .limit(1)
        .execute()
    )


def _registration_by_id(supabase: Any, registration_id: str) -> dict[str, Any] | None:
    return _safe_first(
        supabase.table("tournament_registrations")
        .select("id,tournament_id,display_name,first_name,last_name,player_id,email,wants_partner_board_contact,status")
        .eq("id", str(registration_id))
        .limit(1)
        .execute()
    )


def _partner_request_by_id(supabase: Any, request_id: str) -> dict[str, Any] | None:
    return _safe_first(
        supabase.table("tournament_registration_partner_requests")
        .select("*")
        .eq("id", str(request_id))
        .limit(1)
        .execute()
    )


def _partner_requests_for_tournament(supabase: Any, tournament_id: str) -> list[dict[str, Any]]:
    return _safe_rows(
        supabase.table("tournament_registration_partner_requests")
        .select("*")
        .eq("tournament_id", str(tournament_id))
        .execute()
    )


def _display_name(registration: dict[str, Any] | None) -> str:
    row = registration or {}
    display = _clean_text(row.get("display_name"), limit=160)
    if display:
        return display
    return " ".join(part for part in [_clean_text(row.get("first_name"), limit=80), _clean_text(row.get("last_name"), limit=80)] if part).strip()


def _registration_is_active(registration: dict[str, Any] | None) -> bool:
    status = _clean_text((registration or {}).get("status") or "confirmed", limit=40).upper()
    return status not in {"CANCELLED", "WITHDRAWN"}


def _contact_denylist() -> set[str]:
    raw = os.getenv("JUPR_TOURNAMENT_PARTNER_CONTACT_DENYLIST", "")
    return {_clean_email(value) for value in raw.split(",") if _clean_email(value)}


def _email_is_contact_allowed(email: str | None) -> bool:
    clean = _clean_email(email)
    if not clean:
        return False
    denylist = _contact_denylist()
    domain = f"@{clean.rsplit('@', 1)[-1]}" if "@" in clean else ""
    return clean not in denylist and (not domain or domain not in denylist)


def _notification_failure_status(keys: list[str], exc: Exception) -> dict[str, dict[str, str]]:
    reason = _clean_text(exc.__class__.__name__, limit=80) or "notification_error"
    return {
        key: {"status": "failed", "provider_message_id": reason, "to_email": ""}
        for key in keys
    }


def _notification_statuses(rows: dict[str, dict[str, str]]) -> dict[str, str]:
    return {key: _clean_text(value.get("status") or "unknown", limit=40) for key, value in rows.items()}


def _is_public_partner_board_target(selection: dict[str, Any], event: dict[str, Any] | None) -> bool:
    if _clean_text(selection.get("partner_mode"), limit=40).upper() != "NEEDS_PARTNER":
        return False
    if not _safe_bool(selection.get("show_on_partner_board")):
        return False
    if not event:
        return False
    if not _safe_bool(event.get("partner_board_enabled", event.get("public_partner_board"))):
        return False
    if _clean_text(event.get("status") or "draft", limit=40).lower() not in {
        "open",
        "tentative",
        "confirmed",
        "published",
        "active",
    }:
        return False
    if not _safe_bool(event.get("enabled", True)):
        return False
    return True


def _board_url(
    *,
    club_slug: str,
    tournament_id: str,
    registration_slug: str | None = None,
    edit_token: str | None = None,
    partner_request_id: str | None = None,
) -> str:
    query: dict[str, str] = {}
    if _clean_text(registration_slug, limit=120):
        query["tournament"] = _clean_text(registration_slug, limit=120)
    else:
        query["tournament_id"] = str(tournament_id)
    if _clean_text(edit_token, limit=4000):
        query["edit_token"] = _clean_text(edit_token, limit=4000)
    if _clean_text(partner_request_id, limit=160):
        query["partner_request_id"] = _clean_text(partner_request_id, limit=160)
    suffix = f"?{urlencode(query)}" if query else ""
    return f"{_public_web_base_url()}/clubs/{_clean_text(club_slug, limit=120) or 'tres-palapas'}/tournament-partner-board{suffix}"


def _generic_honeypot_response() -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "public_partner_request",
        "status": "accepted",
        "message": "Partner request submitted.",
    }


def _selection_ids(bundle: dict[str, Any]) -> set[str]:
    return {str(row.get("id") or "").strip() for row in (bundle.get("selections") or []) if str(row.get("id") or "").strip()}


def _selection_payload_for_request(supabase: Any, selection_id: str) -> dict[str, Any]:
    selection = _selection_by_id(supabase, selection_id) or {}
    registration = _registration_by_id(supabase, str(selection.get("registration_id") or "")) if selection.get("registration_id") else None
    event = _event_by_id(supabase, str(selection.get("event_option_id") or "")) if selection.get("event_option_id") else None
    return {
        "display_name": _display_name(registration) or "Player",
        "division_name": _clean_text((event or {}).get("label") or (event or {}).get("division_name") or "Division"),
    }


def _partner_request_payload(supabase: Any, request: dict[str, Any], *, perspective_selection_ids: set[str]) -> dict[str, Any]:
    requester = _selection_payload_for_request(supabase, str(request.get("requester_selection_id") or ""))
    target = _selection_payload_for_request(supabase, str(request.get("target_selection_id") or ""))
    direction = "incoming" if str(request.get("target_selection_id") or "") in perspective_selection_ids else "outgoing"
    status = _clean_text(request.get("status") or PENDING_STATUS, limit=40).upper()
    available_actions: list[str] = []
    if status == PENDING_STATUS and direction == "incoming":
        available_actions = ["accept", "decline"]
    elif status == PENDING_STATUS and direction == "outgoing":
        available_actions = ["cancel"]
    payload = {
        "id": str(request.get("id") or ""),
        "status": status,
        "requester": requester,
        "target": target,
        "created_at": request.get("created_at"),
        "updated_at": request.get("updated_at"),
        "responded_at": request.get("responded_at"),
        "direction": direction,
        "available_actions": available_actions,
    }
    if _PRIVATE_CONTACT_FIELDS.intersection(payload):
        raise RuntimeError("Private contact field escaped the partner request projection.")
    return payload


def create_public_tournament_partner_request(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    requester_selection_id: str,
    target_selection_id: str | None = None,
    target_public_entry_key: str | None = None,
    tournament_id: str | None = None,
    website: str | None = None,
    club_slug: str = "tres-palapas",
) -> dict[str, Any]:
    """Create a pending partner request from a token-verified requester.

    The edit token authenticates the requesting registration. The target must be a
    public partner-board entry in the same tournament/division. Acceptance is handled
    by the target registration's edit token and automatically creates the confirmed
    team link.
    """

    if _clean_text(website, limit=200):
        return _generic_honeypot_response()

    verified, bundle = _verified_bundle(
        supabase,
        club_id=str(club_id),
        edit_token=edit_token,
        tournament_id=tournament_id,
    )
    tid = str(verified.get("tournament_id") or "").strip()
    registration = bundle.get("registration") or {}
    settings = bundle.get("settings") or {}
    selections = bundle.get("selections") or []
    if not _safe_bool(settings.get("partner_board_enabled")):
        raise ValueError("The Players Needing Partners page is not available.")
    requester_selection_id = _clean_text(requester_selection_id, limit=160)
    public_target_key = _clean_text(target_public_entry_key, limit=80)
    if public_target_key:
        public_target = _selection_by_public_entry_key(
            supabase,
            tournament_id=tid,
            board_entry_key=public_target_key,
        )
        if not public_target:
            raise ValueError("That Players Needing Partners entry was not found.")
        target_selection_id = str(public_target.get("id") or "")
    target_selection_id = _clean_text(target_selection_id, limit=160)
    if not requester_selection_id or not target_selection_id:
        raise ValueError("Requester and target selections are required.")
    if requester_selection_id == target_selection_id:
        raise ValueError("A player cannot request themselves as a partner.")

    requester_selection = next((row for row in selections if str(row.get("id") or "") == requester_selection_id), None)
    if not requester_selection:
        raise ValueError("Requester selection is not part of this verified registration.")
    if str(requester_selection.get("registration_id") or "") != str(registration.get("id") or ""):
        raise ValueError("Requester selection is not part of this verified registration.")
    if not _registration_is_active(registration):
        raise ValueError("Requester registration is no longer active.")

    target_selection = _selection_by_id(supabase, target_selection_id)
    if not target_selection:
        raise ValueError("That Players Needing Partners entry was not found.")
    if str(target_selection.get("tournament_id") or "") != tid:
        raise ValueError("Partner selections must be in the same tournament.")
    if str(requester_selection.get("event_option_id") or "") != str(target_selection.get("event_option_id") or ""):
        raise ValueError("Partner selections must be in the same division.")

    event = _event_by_id(supabase, str(target_selection.get("event_option_id") or ""))
    if not _is_public_partner_board_target(target_selection, event):
        raise ValueError("That Players Needing Partners entry is no longer available.")

    target_registration = _registration_by_id(supabase, str(target_selection.get("registration_id") or ""))
    if not target_registration or not _registration_is_active(target_registration):
        raise ValueError("That Players Needing Partners entry is no longer available.")
    if not _safe_bool(target_registration.get("wants_partner_board_contact")):
        raise ValueError("That Players Needing Partners entry is no longer available.")
    target_name = _display_name(target_registration)
    requester_name = _display_name(registration)
    created = create_partner_request_atomic(
        supabase,
        request_id=f"preq_{uuid4().hex[:20]}",
        tournament_id=tid,
        event_option_id=str(target_selection.get("event_option_id") or ""),
        requester_selection_id=requester_selection_id,
        target_selection_id=target_selection_id,
        target_display_name_snapshot=target_name,
        source="PUBLIC_PARTNER_BOARD",
    )
    tournament = _tournament_by_id(supabase, tid) or {}
    registration_slug_value = _clean_text(settings.get("registration_slug"), limit=120) or None
    board_url = _board_url(club_slug=club_slug, tournament_id=tid, registration_slug=registration_slug_value)
    target_email = _clean_email((target_registration or {}).get("email"))
    target_email_for_delivery = target_email if _email_is_contact_allowed(target_email) else ""
    target_edit_token = ""
    accept_url = board_url
    if target_email_for_delivery and target_registration:
        target_edit_token = build_registration_edit_token(
            tournament_id=tid,
            registration_id=str(target_registration.get("id") or ""),
            email=target_email_for_delivery,
        )
        accept_url = _board_url(
            club_slug=club_slug,
            tournament_id=tid,
            registration_slug=registration_slug_value,
            edit_token=target_edit_token,
            partner_request_id=str(created.get("id") or ""),
        )
    idempotent = bool(created.get("idempotent"))
    if idempotent:
        notifications = {
            "player": {"status": "not_repeated", "provider_message_id": "idempotent", "to_email": ""},
            "organizer": {"status": "not_repeated", "provider_message_id": "idempotent", "to_email": ""},
        }
    else:
        try:
            notifications = send_pairing_interest_emails(
                tournament_name=_clean_text(tournament.get("name") or "Tournament"),
                division_name=_clean_text((event or {}).get("label") or (event or {}).get("division_name") or "Division"),
                requester_name=requester_name,
                target_name=target_name,
                target_email=target_email_for_delivery,
                board_url=board_url,
                accept_url=accept_url,
            )
        except Exception as exc:
            notifications = _notification_failure_status(["player", "organizer"], exc)
    return {
        "ok": True,
        "mode": "public_partner_request",
        "status": str(created.get("status") or PENDING_STATUS),
        "partner_request_id": str(created.get("id") or ""),
        "idempotent": idempotent,
        "notification_status": _notification_statuses(notifications),
        "message": (
            "This partner request is already pending; no duplicate notification was sent."
            if idempotent
            else "Partner request submitted. If accepted, both registrations will automatically be paired."
        ),
    }


def list_public_tournament_partner_requests(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    tournament_id: str | None = None,
) -> dict[str, Any]:
    verified, bundle = _verified_bundle(
        supabase,
        club_id=str(club_id),
        edit_token=edit_token,
        tournament_id=tournament_id,
    )
    tid = str(verified.get("tournament_id") or "").strip()
    ids = _selection_ids(bundle)
    requests = _partner_requests_for_tournament(supabase, tid)
    related = [
        request
        for request in requests
        if str(request.get("target_selection_id") or "") in ids
        or str(request.get("requester_selection_id") or "") in ids
    ]
    related.sort(key=lambda row: (str(row.get("created_at") or ""), str(row.get("id") or "")), reverse=True)
    incoming = [request for request in related if str(request.get("target_selection_id") or "") in ids]
    outgoing = [request for request in related if str(request.get("requester_selection_id") or "") in ids]
    return {
        "ok": True,
        "mode": "public_partner_requests",
        "incoming": [_partner_request_payload(supabase, request, perspective_selection_ids=ids) for request in incoming],
        "outgoing": [_partner_request_payload(supabase, request, perspective_selection_ids=ids) for request in outgoing],
        "summary": {
            "incoming": len(incoming),
            "outgoing": len(outgoing),
            "pending_incoming": len([r for r in incoming if _clean_text(r.get("status"), limit=40).upper() == "PENDING"]),
            "pending_outgoing": len([r for r in outgoing if _clean_text(r.get("status"), limit=40).upper() == "PENDING"]),
        },
    }


def _recipient_board_url(
    *,
    club_slug: str,
    tournament_id: str,
    registration_slug: str | None,
    registration: dict[str, Any] | None,
    partner_request_id: str,
) -> tuple[str, str]:
    registration = registration or {}
    email = _clean_email(registration.get("email"))
    if not email or not _email_is_contact_allowed(email):
        return "", _board_url(
            club_slug=club_slug,
            tournament_id=tournament_id,
            registration_slug=registration_slug,
        )
    token = build_registration_edit_token(
        tournament_id=tournament_id,
        registration_id=str(registration.get("id") or ""),
        email=email,
    )
    return email, _board_url(
        club_slug=club_slug,
        tournament_id=tournament_id,
        registration_slug=registration_slug,
        edit_token=token,
        partner_request_id=partner_request_id,
    )


def _transition_public_tournament_partner_request(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    partner_request_id: str,
    action: str,
    tournament_id: str | None = None,
    website: str | None = None,
    club_slug: str = "tres-palapas",
) -> dict[str, Any]:
    clean_action = _clean_text(action, limit=20).lower()
    desired_status = {
        "accept": ACCEPTED_STATUS,
        "decline": DECLINED_STATUS,
        "cancel": CANCELLED_STATUS,
    }.get(clean_action)
    if not desired_status:
        raise ValueError("Unsupported partner request action.")
    if _clean_text(website, limit=200):
        return {
            "ok": True,
            "mode": f"public_partner_request_{clean_action}",
            "status": desired_status,
            "message": "Partner request updated.",
        }
    verified, bundle = _verified_bundle(
        supabase,
        club_id=str(club_id),
        edit_token=edit_token,
        tournament_id=tournament_id,
    )
    ids = _selection_ids(bundle)
    request = _partner_request_by_id(supabase, partner_request_id)
    if not request:
        raise ValueError("Partner request was not found.")
    if str(request.get("tournament_id") or "") != str(verified.get("tournament_id") or ""):
        raise ValueError("Partner request is for a different tournament.")
    request_id = str(request.get("id") or partner_request_id)
    requester_selection_id = str(request.get("requester_selection_id") or "").strip()
    target_selection_id = str(request.get("target_selection_id") or "").strip()
    if clean_action in {"accept", "decline"}:
        if target_selection_id not in ids:
            raise ValueError("Only the requested partner can respond to this request.")
        actor_selection_id = target_selection_id
    else:
        if requester_selection_id not in ids:
            raise ValueError("Only the requester can cancel this request.")
        actor_selection_id = requester_selection_id

    transition = transition_partner_request_atomic(
        supabase,
        request_id=request_id,
        actor_selection_id=actor_selection_id,
        action=clean_action,
    )
    if _clean_text(transition.get("outcome"), limit=40).lower() == "stale":
        current = _clean_text(transition.get("status") or request.get("status"), limit=40).upper()
        raise PartnerRequestStaleError(
            f"This partner request is no longer pending (current status: {current or 'UNKNOWN'}). Refresh before trying another action."
        )

    idempotent = bool(transition.get("idempotent"))
    requester_selection = _selection_by_id(supabase, requester_selection_id) or {}
    target_selection = _selection_by_id(supabase, target_selection_id) or {}
    requester_registration = _registration_by_id(supabase, str(requester_selection.get("registration_id") or ""))
    target_registration = _registration_by_id(supabase, str(target_selection.get("registration_id") or ""))
    event = _event_by_id(supabase, str(request.get("event_option_id") or "")) or {}
    tournament = _tournament_by_id(supabase, str(request.get("tournament_id") or "")) or {}
    settings = bundle.get("settings") or {}
    registration_slug = _clean_text(settings.get("registration_slug"), limit=120) or None

    if idempotent:
        notifications: dict[str, dict[str, str]] = {
            "lifecycle": {"status": "not_repeated", "provider_message_id": "idempotent", "to_email": ""}
        }
    else:
        try:
            requester_email, requester_url = _recipient_board_url(
                club_slug=club_slug,
                tournament_id=str(request.get("tournament_id") or ""),
                registration_slug=registration_slug,
                registration=requester_registration,
                partner_request_id=request_id,
            )
            target_email, target_url = _recipient_board_url(
                club_slug=club_slug,
                tournament_id=str(request.get("tournament_id") or ""),
                registration_slug=registration_slug,
                registration=target_registration,
                partner_request_id=request_id,
            )
            notifications = send_pairing_status_emails(
                action=desired_status.lower(),
                tournament_name=_clean_text(tournament.get("name") or "Tournament"),
                division_name=_clean_text(event.get("label") or event.get("division_name") or "Division"),
                requester_name=_display_name(requester_registration) or "Requester",
                target_name=_display_name(target_registration) or "Requested player",
                requester_email=requester_email,
                target_email=target_email,
                requester_url=requester_url,
                target_url=target_url,
            )
        except Exception as exc:
            expected_keys = ["requester", "target"] if clean_action == "accept" else (["requester"] if clean_action == "decline" else ["target"])
            notifications = _notification_failure_status(expected_keys, exc)

    messages = {
        "accept": "Partner request accepted. Both registrations are now automatically paired.",
        "decline": "Partner request declined. No team was created.",
        "cancel": "Partner request cancelled. No team was created.",
    }
    if idempotent:
        messages[clean_action] = f"Partner request was already {desired_status.lower()}; no duplicate notification was sent."
    return {
        "ok": True,
        "mode": f"public_partner_request_{clean_action}",
        "status": desired_status,
        "partner_request_id": request_id,
        "team_link_id": str(transition.get("team_link_id") or ""),
        "event_option_id": str(request.get("event_option_id") or ""),
        "idempotent": idempotent,
        "cancelled_request_ids": [str(value) for value in (transition.get("cancelled_request_ids") or [])],
        "notification_status": _notification_statuses(notifications),
        "message": messages[clean_action],
    }


def accept_public_tournament_partner_request(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    partner_request_id: str,
    tournament_id: str | None = None,
    website: str | None = None,
    club_slug: str = "tres-palapas",
) -> dict[str, Any]:
    return _transition_public_tournament_partner_request(
        supabase,
        club_id=club_id,
        edit_token=edit_token,
        partner_request_id=partner_request_id,
        action="accept",
        tournament_id=tournament_id,
        website=website,
        club_slug=club_slug,
    )


def decline_public_tournament_partner_request(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    partner_request_id: str,
    tournament_id: str | None = None,
    website: str | None = None,
    club_slug: str = "tres-palapas",
) -> dict[str, Any]:
    return _transition_public_tournament_partner_request(
        supabase,
        club_id=club_id,
        edit_token=edit_token,
        partner_request_id=partner_request_id,
        action="decline",
        tournament_id=tournament_id,
        website=website,
        club_slug=club_slug,
    )


def cancel_public_tournament_partner_request(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    partner_request_id: str,
    tournament_id: str | None = None,
    website: str | None = None,
    club_slug: str = "tres-palapas",
) -> dict[str, Any]:
    return _transition_public_tournament_partner_request(
        supabase,
        club_id=club_id,
        edit_token=edit_token,
        partner_request_id=partner_request_id,
        action="cancel",
        tournament_id=tournament_id,
        website=website,
        club_slug=club_slug,
    )
