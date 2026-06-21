from __future__ import annotations

from typing import Any

import streamlit as st

from jupr_app.domain.tournament_registration_edit_tokens import verify_registration_edit_token
from jupr_app.domain.tournament_registration_repo import get_public_tournament_bundle, get_registration_confirmation_bundle, registration_feature_available
from jupr_app.ui.layout import page_shell
from jupr_app.ui.pages.tournament_registration import _hydrate_registration_wizard_from_bundle, _safe_text, _wizard_key
from jupr_app.ui.public_links import navigate_same_tab


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

    key = _wizard_key(tournament_id)
    wizard: dict[str, Any] = st.session_state.setdefault(key, {})
    _hydrate_registration_wizard_from_bundle(wizard, bundle)
    st.session_state[key] = wizard
    params = {"tournament_id": tournament_id, "edit": "1"}
    registration_slug = _safe_text((bundle.get("settings") or {}).get("registration_slug")) or slug
    if registration_slug:
        params["tournament"] = registration_slug
    navigate_same_tab(page="tournament_registration", params=params, public_mode=True)
