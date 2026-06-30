from __future__ import annotations

from typing import Any

import streamlit as st

from jupr_app.domain.notifications.tournament_partner_request_email import send_tournament_partner_request_email
from jupr_app.domain.tournament_registration_repo import get_public_tournament_bundle, registration_feature_available
from jupr_app.ui.layout import page_shell
from jupr_app.ui.public_links import navigate_same_tab


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_data(resp: Any) -> list[dict[str, Any]]:
    try:
        return list(resp.data or [])
    except Exception:
        return []


def _safe_first(resp: Any) -> dict[str, Any] | None:
    rows = _safe_data(resp)
    return rows[0] if rows else None


def _normalize_email(value: Any) -> str:
    return _safe_text(value).lower()


def _valid_email(value: Any) -> bool:
    text = _normalize_email(value)
    if not text:
        return False
    return "@" in text and "." in text.rsplit("@", 1)[-1]


def _display_name(registration: dict[str, Any]) -> str:
    display = _safe_text(registration.get("display_name"))
    if display:
        return display
    name = " ".join(part for part in [_safe_text(registration.get("first_name")), _safe_text(registration.get("last_name"))] if part)
    return name or "Player"


def _load_target_selection(supabase, *, tournament_id: str, target_selection_id: str) -> dict[str, Any] | None:
    if not target_selection_id:
        return None
    return _safe_first(
        supabase.table("tournament_registration_selections")
        .select("*")
        .eq("tournament_id", tournament_id)
        .eq("id", target_selection_id)
        .limit(1)
        .execute()
    )


def _load_registration(supabase, *, tournament_id: str, registration_id: str) -> dict[str, Any] | None:
    if not registration_id:
        return None
    return _safe_first(
        supabase.table("tournament_registrations")
        .select("*")
        .eq("tournament_id", tournament_id)
        .eq("id", registration_id)
        .limit(1)
        .execute()
    )


def _event_lookup(event_options: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in event_options}


def _day_lookup(days: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in days}


def _roster_params(*, tournament_id: str, registration_slug: str) -> dict[str, str]:
    params = {"tournament_id": tournament_id}
    if registration_slug:
        params["tournament"] = registration_slug
    return params


def _back_to_roster_button(*, tournament_id: str, registration_slug: str, label: str = "Back to tournament roster", primary: bool = False) -> None:
    if st.button(label, type="primary" if primary else "secondary"):
        navigate_same_tab(page="tournament_roster", params=_roster_params(tournament_id=tournament_id, registration_slug=registration_slug), public_mode=True)


def _resolve_tournament(ctx, supabase, *, tournament_id: str, slug: str):
    if not tournament_id and not slug:
        return None, None, [], []
    return get_public_tournament_bundle(
        supabase,
        club_id=_safe_text(getattr(ctx, "club_id", "")),
        tournament_id=tournament_id or None,
        registration_slug=slug or None,
    )


def render(ctx) -> None:
    page_shell("🤝 Request Partner", "Send a private partner request without exposing the requested player's email.", mode_label="Public")
    supabase = getattr(ctx, "supabase", None)
    if supabase is None:
        st.error("Missing database context.")
        st.stop()

    available, reason = registration_feature_available(supabase)
    if not available:
        st.error(reason or "Tournament registration is not available.")
        st.stop()

    tournament_id = _safe_text(st.query_params.get("tournament_id"))
    slug = _safe_text(st.query_params.get("tournament"))
    target_selection_id = _safe_text(st.query_params.get("target_selection_id"))
    tournament, settings, days, event_options = _resolve_tournament(ctx, supabase, tournament_id=tournament_id, slug=slug)
    if not tournament:
        st.error("This partner request link is missing or references an unavailable tournament.")
        st.stop()
    tournament_id = _safe_text(tournament.get("id"))
    registration_slug = _safe_text((settings or {}).get("registration_slug")) or slug

    if not target_selection_id:
        st.error("This partner request link is missing the requested player.")
        _back_to_roster_button(tournament_id=tournament_id, registration_slug=registration_slug)
        st.stop()

    selection = _load_target_selection(supabase, tournament_id=tournament_id, target_selection_id=target_selection_id)
    if not selection:
        st.error("The requested player entry could not be found.")
        _back_to_roster_button(tournament_id=tournament_id, registration_slug=registration_slug)
        st.stop()
    if _safe_text(selection.get("partner_mode")).upper() != "NEEDS_PARTNER":
        st.warning("This player is no longer marked as looking for a partner in this division.")
        _back_to_roster_button(tournament_id=tournament_id, registration_slug=registration_slug)
        st.stop()

    registration = _load_registration(supabase, tournament_id=tournament_id, registration_id=_safe_text(selection.get("registration_id"))) or {}
    target_email = _normalize_email(registration.get("email"))
    target_name = _display_name(registration)
    event = _event_lookup(event_options).get(_safe_text(selection.get("event_option_id"))) or {}
    day = _day_lookup(days).get(_safe_text(selection.get("registration_day_id"))) or {}
    event_label = _safe_text(event.get("event_family_label") or event.get("label") or "Event")
    division_label = _safe_text(event.get("division_name") or event.get("label") or "Division")
    day_label = _safe_text(day.get("label") or "Day")

    st.subheader(f"Request {target_name} as a partner")
    st.write(f"Division: **{day_label} / {event_label} / {division_label}**")
    st.caption("The requested player's email address will not be shown or shared with you. Your contact information will be included in the email so they can reply directly.")

    if not target_email:
        st.error("This player does not have an email address on their registration, so the request cannot be sent automatically.")
        _back_to_roster_button(tournament_id=tournament_id, registration_slug=registration_slug)
        st.stop()

    sent_key = f"partner_request_sent_{tournament_id}_{target_selection_id}"
    if st.session_state.get(sent_key):
        st.success("Your partner request was sent. The requested player can contact you using the information you provided.")
        _back_to_roster_button(tournament_id=tournament_id, registration_slug=registration_slug, primary=True)
        return

    with st.form(f"partner_request_form_{tournament_id}_{target_selection_id}"):
        requester_name = st.text_input("Your name *")
        requester_email = st.text_input("Your email")
        requester_phone = st.text_input("Your phone / WhatsApp")
        message = st.text_area("Message to the requested player", value="Hi, would you like to partner for this division?", height=120)
        submitted = st.form_submit_button("Send partner request", type="primary", use_container_width=True)

    if submitted:
        errors: list[str] = []
        if not _safe_text(requester_name):
            errors.append("Enter your name.")
        if not _safe_text(requester_email) and not _safe_text(requester_phone):
            errors.append("Enter your email or phone number so the requested player can contact you.")
        if _safe_text(requester_email) and not _valid_email(requester_email):
            errors.append("Enter a valid email address or leave the email field blank and provide a phone number.")
        if errors:
            for error in errors:
                st.error(error)
            st.stop()

        try:
            result = send_tournament_partner_request_email(
                tournament_name=_safe_text(tournament.get("name") or "Tournament"),
                target_name=target_name,
                target_email=target_email,
                requester_name=_safe_text(requester_name),
                requester_email=_normalize_email(requester_email),
                requester_phone=_safe_text(requester_phone),
                event_label=event_label,
                division_label=division_label,
                day_label=day_label,
                message=_safe_text(message),
            )
            st.session_state[sent_key] = True
            status = _safe_text(result.get("status"))
            if status == "dry_run":
                st.success("Partner request prepared. Email sending is currently in dry-run mode.")
            else:
                st.success("Partner request sent. The requested player can contact you using the information you provided.")
            _back_to_roster_button(tournament_id=tournament_id, registration_slug=registration_slug, primary=True)
            return
        except Exception as exc:
            st.error(f"Could not send partner request: {exc}")

    _back_to_roster_button(tournament_id=tournament_id, registration_slug=registration_slug, label="Cancel / Back to roster")
