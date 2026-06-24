from __future__ import annotations

from typing import Any

import streamlit as st

from jupr_app.domain.tournament_registration_edit_tokens import verify_registration_edit_token
from jupr_app.domain.tournament_registration_repo import get_public_tournament_bundle, get_registration_confirmation_bundle, registration_feature_available
from jupr_app.ui.layout import page_shell
from jupr_app.ui.pages.tournament_registration import _advance_step1_registration_wizard, _hydrate_registration_wizard_from_bundle, _safe_text
from jupr_app.ui.tournament_registration_session import submission_state_key, wizard_state_key
from jupr_app.ui.public_links import navigate_same_tab

_GENDER_OPTIONS = ["", "Female", "Male", "Other", "Prefer not to say"]


def _gender_index(value: Any) -> int:
    gender = _safe_text(value)
    return _GENDER_OPTIONS.index(gender) if gender in _GENDER_OPTIONS else 0


def _registration_nav_params(*, tournament_id: str, registration_slug: str) -> dict[str, str]:
    params = {"tournament_id": tournament_id, "edit": "1"}
    if registration_slug:
        params["tournament"] = registration_slug
    return params


def _render_verified_contact_gate(*, wizard: dict[str, Any], tournament_id: str, registration_slug: str) -> None:
    step1 = wizard.get("step1") or {}
    registration_id = _safe_text(wizard.get("edit_registration_id"))
    widget_suffix = f"{tournament_id}_{registration_id or 'registration'}"

    st.info("Editing existing registration. Contact info → Events → Partner information → Confirmation.")
    st.markdown("### 1. Name and contact")
    c1, c2 = st.columns(2)
    with c1:
        first_name = st.text_input(
            "First name *",
            value=_safe_text(step1.get("first_name")),
            key=f"edit_step1_first_name_{widget_suffix}",
        )
        locked_email = _safe_text(step1.get("email"))
        st.text_input(
            "Email *",
            value=locked_email,
            disabled=True,
            key=f"edit_step1_email_{widget_suffix}",
        )
        gender = st.selectbox(
            "Gender *",
            _GENDER_OPTIONS,
            index=_gender_index(step1.get("gender")),
            key=f"edit_step1_gender_{widget_suffix}",
        )
    with c2:
        last_name = st.text_input(
            "Last name *",
            value=_safe_text(step1.get("last_name")),
            key=f"edit_step1_last_name_{widget_suffix}",
        )
        phone = st.text_input(
            "Phone / WhatsApp",
            value=_safe_text(step1.get("phone")),
            key=f"edit_step1_phone_{widget_suffix}",
        )
        age = st.text_input(
            "Age *",
            value=_safe_text(step1.get("age")),
            key=f"edit_step1_age_{widget_suffix}",
        )
    notes = st.text_area(
        "Notes for tournament staff",
        value=_safe_text(step1.get("notes")),
        height=90,
        key=f"edit_step1_notes_{widget_suffix}",
    )

    _, next_col = st.columns([4, 1])
    with next_col:
        if st.button("Next ➜", type="primary", key=f"edit_step1_next_{widget_suffix}"):
            advanced, error = _advance_step1_registration_wizard(
                wizard,
                tournament_id=tournament_id,
                first_name=first_name,
                last_name=last_name,
                email_for_submit=locked_email,
                phone=phone,
                gender=gender,
                age=age,
                notes=notes,
                find_existing_registration=None,
            )
            if not advanced:
                st.error(error)
                st.stop()
            st.session_state[wizard_state_key(tournament_id)] = wizard
            navigate_same_tab(
                page="tournament_registration",
                params=_registration_nav_params(tournament_id=tournament_id, registration_slug=registration_slug),
                public_mode=True,
                source="tournament_registration_edit:contact_next",
            )


def render(ctx) -> None:
    page_shell("✏️ Edit Registration", "Securely edit your tournament registration.", mode_label="Public")
    supabase = ctx.supabase
    available, reason = registration_feature_available(supabase)
    if not available:
        st.error(reason or "Tournament registration is not available.")
        return

    token = _safe_text(st.query_params.get("edit_token"))
    tournament_id = _safe_text(st.query_params.get("tournament_id"))
    slug = _safe_text(st.query_params.get("tournament"))
    if not tournament_id and slug:
        tournament, _settings, _days, _events = get_public_tournament_bundle(supabase, club_id=_safe_text(getattr(ctx, "club_id", "")), registration_slug=slug)
        tournament_id = _safe_text((tournament or {}).get("id"))
    if not token or not tournament_id:
        st.error("That edit link is missing required information. Please return to tournament registration and request a new link.")
        if st.button("Back to tournament registration"):
            navigate_same_tab(page="tournament_registration", params={"tournament_id": tournament_id, "tournament": slug}, public_mode=True)
        return

    try:
        verified = verify_registration_edit_token(token, expected_tournament_id=tournament_id)
        registration_id = _safe_text(verified.get("registration_id"))
        bundle = get_registration_confirmation_bundle(supabase, tournament_id, registration_id)
        registration = bundle.get("registration") or {}
        if not registration:
            raise ValueError("Registration was not found.")
        verify_registration_edit_token(token, expected_tournament_id=tournament_id, expected_registration_id=registration_id, expected_email=_safe_text(registration.get("email")))
    except Exception as exc:
        error_text = str(exc).lower()
        if "configuration" in error_text or "jupr_registration_edit_secret" in error_text:
            st.error("Registration edit links are not configured yet. Please contact tournament staff.")
        else:
            st.error("This secure edit link is invalid or expired. Please request a new edit link from tournament registration.")
        if st.button("Back to tournament registration"):
            navigate_same_tab(page="tournament_registration", params={"tournament_id": tournament_id, "tournament": slug}, public_mode=True)
        return

    st.session_state.pop(submission_state_key(tournament_id), None)
    key = wizard_state_key(tournament_id)
    wizard: dict[str, Any] = st.session_state.setdefault(key, {})
    same_registration = bool(wizard.get("edit_mode")) and _safe_text(wizard.get("edit_registration_id")) == registration_id
    _hydrate_registration_wizard_from_bundle(
        wizard,
        bundle,
        preserve_existing_progress=same_registration,
    )
    wizard["edit_link_verified"] = True
    wizard["edit_link_registration_id"] = registration_id
    st.session_state[key] = wizard

    registration_slug = _safe_text((bundle.get("settings") or {}).get("registration_slug")) or slug
    current_step = int(wizard.get("current_step") or 1)
    if current_step <= 1:
        _render_verified_contact_gate(
            wizard=wizard,
            tournament_id=tournament_id,
            registration_slug=registration_slug,
        )
        return

    navigate_same_tab(
        page="tournament_registration",
        params=_registration_nav_params(tournament_id=tournament_id, registration_slug=registration_slug),
        public_mode=True,
        source="tournament_registration_edit:resume_wizard",
    )
