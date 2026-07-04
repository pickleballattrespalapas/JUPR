from __future__ import annotations

from typing import Any

from jupr_app.domain.tournament_registration_edit_tokens import verify_registration_edit_token
from jupr_app.domain.tournament_registration_repo import get_registration_confirmation_bundle, save_registration
from jupr_app.services.public_tournament_registration_service import (
    _clean_email,
    _clean_selection,
    _clean_text,
    _safe_bool,
    _safe_float,
    _safe_int,
    build_public_tournament_registration_page,
)


def _registration_public_payload(registration: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(registration.get("id") or ""),
        "first_name": _clean_text(registration.get("first_name"), limit=80),
        "last_name": _clean_text(registration.get("last_name"), limit=80),
        "display_name": _clean_text(registration.get("display_name"), limit=160),
        "email": _clean_email(registration.get("email")),
        "phone": _clean_text(registration.get("phone"), limit=60),
        "dupr_id": _clean_text(registration.get("dupr_id"), limit=80),
        "doubles_skill": _safe_float(registration.get("doubles_skill")),
        "singles_skill": _safe_float(registration.get("singles_skill")),
        "age": _safe_int(registration.get("age")),
        "gender": _clean_text(registration.get("gender"), limit=40),
        "notes": _clean_text(registration.get("notes"), limit=800),
        "wants_partner_board_contact": _safe_bool(registration.get("wants_partner_board_contact")),
        "status": _clean_text(registration.get("status"), limit=40),
        "payment_status": _clean_text(registration.get("payment_status"), limit=40),
        "submitted_at": registration.get("submitted_at"),
    }


def _selection_public_payload(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(selection.get("id") or ""),
        "event_option_id": str(selection.get("event_option_id") or ""),
        "registration_day_id": str(selection.get("registration_day_id") or ""),
        "partner_mode": _clean_text(selection.get("partner_mode") or "NONE", limit=40).upper(),
        "partner_name": _clean_text(selection.get("partner_name"), limit=160),
        "partner_email": _clean_email(selection.get("partner_email")),
        "partner_phone": _clean_text(selection.get("partner_phone"), limit=60),
        "partner_dupr_id": _clean_text(selection.get("partner_dupr_id"), limit=80),
        "partner_skill": _safe_float(selection.get("partner_skill")),
        "partner_age": _safe_int(selection.get("partner_age")),
        "partner_note": _clean_text(selection.get("partner_note"), limit=500),
        "show_on_partner_board": _safe_bool(selection.get("show_on_partner_board")),
    }


def _verified_bundle(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    tournament_id: str | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    expected_tournament_id = _clean_text(tournament_id, limit=120) or None
    verified = verify_registration_edit_token(edit_token, expected_tournament_id=expected_tournament_id)
    tid = str(verified.get("tournament_id") or "").strip()
    registration_id = str(verified.get("registration_id") or "").strip()
    if not tid or not registration_id:
        raise ValueError("Invalid registration edit link.")
    bundle = get_registration_confirmation_bundle(supabase, tid, registration_id)
    registration = bundle.get("registration") or None
    tournament = bundle.get("tournament") or {}
    if not registration:
        raise ValueError("Registration was not found.")
    if str(tournament.get("club_id") or club_id) != str(club_id):
        raise ValueError("Registration edit link is for a different club.")
    verify_registration_edit_token(
        edit_token,
        expected_tournament_id=tid,
        expected_registration_id=registration_id,
        expected_email=_clean_email(registration.get("email")),
    )
    return verified, bundle


def build_public_tournament_registration_edit_page(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    tournament_id: str | None = None,
    registration_slug: str | None = None,
) -> dict[str, Any]:
    verified, bundle = _verified_bundle(
        supabase,
        club_id=str(club_id),
        edit_token=edit_token,
        tournament_id=tournament_id,
    )
    tid = str(verified.get("tournament_id") or "")
    page = build_public_tournament_registration_page(
        supabase,
        club_id=str(club_id),
        tournament_id=tid,
        registration_slug=registration_slug,
    )
    if not page.get("available"):
        raise ValueError(str(page.get("setup_error") or "Tournament registration is not configured."))
    if not page.get("tournament"):
        raise ValueError("Tournament registration was not found.")
    return {
        **page,
        "edit_mode": True,
        "edit_token_valid": True,
        "edit_token_expires_at": verified.get("exp"),
        "registration": _registration_public_payload(bundle.get("registration") or {}),
        "selections": [_selection_public_payload(selection) for selection in (bundle.get("selections") or [])],
        "total_price_usd": float(bundle.get("total_price_usd") or 0),
    }


def submit_public_tournament_registration_edit(
    supabase: Any,
    *,
    club_id: str,
    edit_token: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    verified, bundle = _verified_bundle(
        supabase,
        club_id=str(club_id),
        edit_token=edit_token,
        tournament_id=_clean_text(payload.get("tournament_id"), limit=120) or None,
    )
    tournament = bundle.get("tournament") or {}
    settings = bundle.get("settings") or {}
    registration = bundle.get("registration") or {}
    tournament_id = str(verified.get("tournament_id") or "").strip()
    registration_id = str(verified.get("registration_id") or "").strip()
    page = build_public_tournament_registration_page(
        supabase,
        club_id=str(club_id),
        tournament_id=tournament_id,
        registration_slug=_clean_text(payload.get("registration_slug"), limit=120) or _clean_text(settings.get("registration_slug"), limit=120) or None,
    )
    if not page.get("registration_open"):
        raise ValueError(str(page.get("registration_closed_reason") or "Registration is not open."))
    selectable = {str(event.get("id")): event for event in (page.get("events") or []) if event.get("selectable")}
    selections: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_selection in payload.get("selections") or []:
        if not isinstance(raw_selection, dict):
            continue
        clean_selection = _clean_selection(raw_selection)
        event_option_id = str(clean_selection.get("event_option_id") or "").strip()
        if not event_option_id or event_option_id in seen:
            continue
        event = selectable.get(event_option_id)
        if not event:
            raise ValueError("One or more selected events is no longer open for registration.")
        clean_selection["registration_day_id"] = str(event.get("registration_day_id") or clean_selection.get("registration_day_id") or "")
        selections.append(clean_selection)
        seen.add(event_option_id)
    if not selections:
        raise ValueError("Select at least one open event.")

    save_payload = {
        "first_name": _clean_text(payload.get("first_name"), limit=80),
        "last_name": _clean_text(payload.get("last_name"), limit=80),
        "display_name": _clean_text(payload.get("display_name"), limit=160),
        "email": _clean_email(registration.get("email")),
        "phone": _clean_text(payload.get("phone"), limit=60),
        "player_id": registration.get("player_id"),
        "dupr_id": _clean_text(payload.get("dupr_id"), limit=80),
        "doubles_skill": _safe_float(payload.get("doubles_skill")),
        "singles_skill": _safe_float(payload.get("singles_skill")),
        "age": _safe_int(payload.get("age")),
        "gender": _clean_text(payload.get("gender"), limit=40),
        "notes": _clean_text(payload.get("notes"), limit=800),
        "wants_partner_board_contact": _safe_bool(payload.get("wants_partner_board_contact")),
        "payment_status": registration.get("payment_status") or "unpaid",
        "status": registration.get("status") or "confirmed",
        "selections": selections,
    }
    result = save_registration(
        supabase,
        tournament_id=tournament_id,
        payload=save_payload,
        expected_registration_id=registration_id,
    )
    return {
        "ok": True,
        "mode": "registration_edit",
        "tournament": {
            "id": str(tournament.get("id") or tournament_id),
            "name": _clean_text(tournament.get("name") or "Tournament"),
            "status": _clean_text(tournament.get("status")),
            "start_date": tournament.get("start_date"),
            "end_date": tournament.get("end_date"),
        },
        "settings": {
            "registration_slug": _clean_text(settings.get("registration_slug"), limit=120),
            "registration_status": _clean_text(settings.get("registration_status") or "draft", limit=40).lower(),
        },
        "registration_id": result.get("registration_id"),
        "submitted_at": result.get("submitted_at"),
        "selection_count": result.get("selection_count"),
    }
