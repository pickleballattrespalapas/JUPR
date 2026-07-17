from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from jupr_app.config import get_env_or_default, get_public_base_url
from jupr_app.domain.notifications.tournament_registration_edit_email import send_tournament_registration_edit_email
from jupr_app.domain.tournament_registration_edit_tokens import build_registration_edit_token, verify_registration_edit_token
from jupr_app.domain.tournament_registration_repo import get_registration_by_email, get_registration_confirmation_bundle, save_registration
from jupr_app.services.public_tournament_registration_service import (
    _clean_email,
    _clean_text,
    _get_club_player,
    _public_day,
    _public_event,
    _public_registration_player,
    _safe_bool,
    _safe_float,
    _safe_int,
    build_validated_public_registration_save_payload,
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
        "player_id": registration.get("player_id"),
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


def _public_web_base_url(public_base_url: str | None = None) -> str:
    for candidate in (
        public_base_url,
        get_env_or_default("JUPR_WEB_BASE_URL"),
        get_env_or_default("STAGING_WEB_BASE_URL"),
        get_env_or_default("NEXT_PUBLIC_JUPR_WEB_BASE_URL"),
        get_env_or_default("JUPR_PUBLIC_BASE_URL"),
        get_public_base_url(),
    ):
        value = str(candidate or "").strip().rstrip("/")
        if value:
            return value
    return ""


def _edit_url(*, club_slug: str, edit_token: str, tournament_id: str, registration_slug: str | None, public_base_url: str | None = None) -> str:
    query: dict[str, str] = {"edit_token": str(edit_token)}
    if _clean_text(registration_slug, limit=120):
        query["tournament"] = _clean_text(registration_slug, limit=120)
    else:
        query["tournament_id"] = str(tournament_id)
    return f"{_public_web_base_url(public_base_url)}/clubs/{club_slug}/tournament-registration/edit?{urlencode(query)}"


def _generic_edit_link_response() -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "registration_edit_link_request",
        "accepted": True,
        "message": "If a matching registration exists, an edit link will be sent to that email address.",
    }


def _require_token_bound_registration_slug(bundle: dict[str, Any], registration_slug: Any) -> None:
    """Reject routing hints that do not belong to the token-bound tournament."""

    supplied_slug = _clean_text(registration_slug, limit=120)
    expected_slug = _clean_text((bundle.get("settings") or {}).get("registration_slug"), limit=120)
    if supplied_slug and supplied_slug != expected_slug:
        raise ValueError("Registration edit link is for a different tournament.")


def request_public_tournament_registration_edit_link(
    supabase: Any,
    *,
    club_id: str,
    club_slug: str,
    email: str,
    tournament_id: str | None = None,
    registration_slug: str | None = None,
    website: str | None = None,
    public_base_url: str | None = None,
) -> dict[str, Any]:
    if _clean_text(website, limit=200):
        return _generic_edit_link_response()
    clean_email = _clean_email(email)
    if not clean_email or "@" not in clean_email:
        raise ValueError("A valid email is required.")
    page = build_public_tournament_registration_page(
        supabase,
        club_id=str(club_id),
        tournament_id=_clean_text(tournament_id, limit=120) or None,
        registration_slug=_clean_text(registration_slug, limit=120) or None,
    )
    if not page.get("available"):
        raise ValueError(str(page.get("setup_error") or "Tournament registration is not configured."))
    tournament = page.get("tournament") or {}
    settings = page.get("settings") or {}
    tid = str(tournament.get("id") or "").strip()
    if not tid:
        raise ValueError("Tournament registration was not found.")
    registration = get_registration_by_email(supabase, tid, clean_email)
    if not registration:
        return _generic_edit_link_response()
    token = build_registration_edit_token(
        tournament_id=tid,
        registration_id=str(registration.get("id") or ""),
        email=clean_email,
    )
    send_tournament_registration_edit_email(
        tournament_name=_clean_text(tournament.get("name") or "Tournament"),
        registered_email=clean_email,
        edit_url=_edit_url(
            club_slug=str(club_slug),
            edit_token=token,
            tournament_id=tid,
            registration_slug=_clean_text(settings.get("registration_slug"), limit=120) or _clean_text(registration_slug, limit=120) or None,
            public_base_url=public_base_url,
        ),
    )
    return _generic_edit_link_response()


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
    _require_token_bound_registration_slug(bundle, registration_slug)
    page = build_public_tournament_registration_page(
        supabase,
        club_id=str(club_id),
        tournament_id=tid,
        registration_slug=None,
    )
    if not page.get("available"):
        raise ValueError(str(page.get("setup_error") or "Tournament registration is not configured."))
    if not page.get("tournament"):
        raise ValueError("Tournament registration was not found.")

    registration = bundle.get("registration") or {}
    linked_player_id = registration.get("player_id")
    linked_player = (
        _get_club_player(
            supabase,
            club_id=str(club_id),
            player_id=linked_player_id,
            require_active=False,
        )
        if linked_player_id not in (None, "")
        else None
    )
    page["players"] = [_public_registration_player(linked_player)] if linked_player else []

    # Existing selections remain visible and may be preserved even when an
    # organizer safely closes or disables that division after registration.
    # They are not made selectable for anyone else.
    selected_event_ids = {
        str(selection.get("event_option_id") or "")
        for selection in (bundle.get("selections") or [])
        if str(selection.get("event_option_id") or "")
    }
    page_event_ids = {str(event.get("id") or "") for event in (page.get("events") or [])}
    missing_events = [
        event
        for event in (bundle.get("event_options") or [])
        if str(event.get("id") or "") in selected_event_ids and str(event.get("id") or "") not in page_event_ids
    ]
    if missing_events:
        page["events"] = [
            *(page.get("events") or []),
            *[_public_event(event, registration_open=bool(page.get("registration_open"))) for event in missing_events],
        ]
    selected_day_ids = {
        str(event.get("registration_day_id") or "")
        for event in (bundle.get("event_options") or [])
        if str(event.get("id") or "") in selected_event_ids
    }
    page_day_ids = {str(day.get("id") or "") for day in (page.get("days") or [])}
    missing_days = [
        day
        for day in (bundle.get("days") or [])
        if str(day.get("id") or "") in selected_day_ids and str(day.get("id") or "") not in page_day_ids
    ]
    if missing_days:
        page["days"] = [*(page.get("days") or []), *[_public_day(day) for day in missing_days]]
    return {
        **page,
        "edit_mode": True,
        "edit_token_valid": True,
        "edit_token_expires_at": verified.get("exp"),
        "registration": _registration_public_payload(registration),
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
    _require_token_bound_registration_slug(bundle, payload.get("registration_slug"))
    page = build_public_tournament_registration_page(
        supabase,
        club_id=str(club_id),
        tournament_id=tournament_id,
        registration_slug=None,
    )
    if not page.get("registration_open"):
        raise ValueError(str(page.get("registration_closed_reason") or "Registration is not open."))
    existing_selected_ids = {
        str(selection.get("event_option_id") or "")
        for selection in (bundle.get("selections") or [])
        if str(selection.get("event_option_id") or "")
    }
    existing_event_options = {
        str(event.get("id") or ""): _public_event(event, registration_open=bool(page.get("registration_open")))
        for event in (bundle.get("event_options") or [])
        if str(event.get("id") or "") in existing_selected_ids
    }
    save_payload = build_validated_public_registration_save_payload(
        supabase,
        club_id=str(club_id),
        tournament_id=tournament_id,
        page=page,
        payload=payload,
        locked_registration=registration,
        existing_event_options=existing_event_options,
    )
    result = save_registration(
        supabase,
        tournament_id=tournament_id,
        payload=save_payload,
        expected_registration_id=registration_id,
        allow_existing_unselectable_event_ids=existing_selected_ids,
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
