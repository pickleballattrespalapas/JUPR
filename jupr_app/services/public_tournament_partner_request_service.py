from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from jupr_app.domain.notifications.tournament_pairing_interest_email import send_pairing_interest_emails
from jupr_app.domain.tournament_partner_service import create_partner_request
from jupr_app.services.public_tournament_registration_edit_service import _public_web_base_url, _verified_bundle
from jupr_app.services.public_tournament_registration_service import _clean_text, _safe_bool


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
        .select("id,display_name,first_name,last_name,player_id,email")
        .eq("id", str(registration_id))
        .limit(1)
        .execute()
    )


def _display_name(registration: dict[str, Any] | None) -> str:
    row = registration or {}
    display = _clean_text(row.get("display_name"), limit=160)
    if display:
        return display
    return " ".join(part for part in [_clean_text(row.get("first_name"), limit=80), _clean_text(row.get("last_name"), limit=80)] if part).strip()


def _is_public_partner_board_target(selection: dict[str, Any], event: dict[str, Any] | None) -> bool:
    if _clean_text(selection.get("partner_mode"), limit=40).upper() != "NEEDS_PARTNER":
        return False
    if not _safe_bool(selection.get("show_on_partner_board")):
        return False
    if not event:
        return False
    if not _safe_bool(event.get("partner_board_enabled", event.get("public_partner_board"))):
        return False
    if _clean_text(event.get("status") or "draft", limit=40).lower() not in {"open", "published", "active"}:
        return False
    if not _safe_bool(event.get("enabled", True)):
        return False
    return True


def _board_url(*, tournament_id: str, registration_slug: str | None = None) -> str:
    query: dict[str, str] = {}
    if _clean_text(registration_slug, limit=120):
        query["tournament"] = _clean_text(registration_slug, limit=120)
    else:
        query["tournament_id"] = str(tournament_id)
    suffix = f"?{urlencode(query)}" if query else ""
    return f"{_public_web_base_url()}/clubs/tres-palapas/tournament-partner-board{suffix}"


def _generic_honeypot_response() -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "public_partner_request",
        "status": "accepted",
        "message": "Partner request submitted.",
    }


def create_public_tournament_partner_request(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    requester_selection_id: str,
    target_selection_id: str,
    tournament_id: str | None = None,
    website: str | None = None,
) -> dict[str, Any]:
    """Create a pending partner request from a token-verified requester.

    The edit token authenticates the requesting registration. The target must be a
    public partner-board entry in the same tournament/division.
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
    requester_selection_id = _clean_text(requester_selection_id, limit=160)
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

    target_selection = _selection_by_id(supabase, target_selection_id)
    if not target_selection:
        raise ValueError("Target partner-board entry was not found.")
    if str(target_selection.get("tournament_id") or "") != tid:
        raise ValueError("Partner selections must be in the same tournament.")
    if str(requester_selection.get("event_option_id") or "") != str(target_selection.get("event_option_id") or ""):
        raise ValueError("Partner selections must be in the same division.")

    event = _event_by_id(supabase, str(target_selection.get("event_option_id") or ""))
    if not _is_public_partner_board_target(target_selection, event):
        raise ValueError("Target partner-board entry is no longer available.")

    target_registration = _registration_by_id(supabase, str(target_selection.get("registration_id") or ""))
    target_name = _display_name(target_registration)
    requester_name = _display_name(registration)
    created = create_partner_request(
        supabase,
        tournament_id=tid,
        event_option_id=str(target_selection.get("event_option_id") or ""),
        requester_selection_id=requester_selection_id,
        target_selection_id=target_selection_id,
        target_player_id=target_selection.get("player_id"),
        target_display_name_snapshot=target_name,
        source="PUBLIC_PARTNER_BOARD",
    )
    tournament = _tournament_by_id(supabase, tid) or {}
    notifications = send_pairing_interest_emails(
        tournament_name=_clean_text(tournament.get("name") or "Tournament"),
        division_name=_clean_text((event or {}).get("label") or (event or {}).get("division_name") or "Division"),
        requester_name=requester_name,
        target_name=target_name,
        target_email=_clean_text((target_registration or {}).get("email")),
        board_url=_board_url(tournament_id=tid, registration_slug=_clean_text(settings.get("registration_slug"), limit=120) or None),
    )
    return {
        "ok": True,
        "mode": "public_partner_request",
        "status": str(created.get("status") or "PENDING"),
        "partner_request_id": str(created.get("id") or ""),
        "event_option_id": str(created.get("event_option_id") or ""),
        "requester_selection_id": str(created.get("requester_selection_id") or ""),
        "target_selection_id": str(created.get("target_selection_id") or ""),
        "notification_status": {key: value.get("status") for key, value in notifications.items()},
        "message": "Partner request submitted.",
    }
