from __future__ import annotations

from typing import Any

import streamlit as st

from jupr_app.domain.notifications.smtp_mailer import get_smtp_config_status
from jupr_app.domain.tournament_registration_repo import (
    get_public_tournament_bundle,
    get_registration_confirmation_bundle,
    registration_feature_available,
)
from jupr_app.ui.layout import page_shell
from jupr_app.ui.public_links import navigate_same_tab
from jupr_app.ui.tournament_registration_confirmation_view import render_registration_confirmation_summary
from jupr_app.ui.tournament_registration_session import get_submission_result


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _registration_id_from_query_or_session(tournament_id: str, query_registration_id: str) -> tuple[str, dict[str, Any]]:
    query_registration_id = _safe_text(query_registration_id)
    if query_registration_id:
        return query_registration_id, {}
    submission_result = get_submission_result(tournament_id)
    return _safe_text(submission_result.get("registration_id")), submission_result


def render(ctx) -> None:
    page_shell(
        "✅ Registration Confirmation",
        "Your tournament registration details and payment information.",
        mode_label="Public",
    )
    supabase = ctx.supabase
    available, reason = registration_feature_available(supabase)
    if not available:
        st.error(reason or "Tournament registration is not available.")
        return

    tournament_id = _safe_text(st.query_params.get("tournament_id"))
    slug = _safe_text(st.query_params.get("tournament"))
    query_registration_id = _safe_text(st.query_params.get("registration_id"))
    email_status = _safe_text(st.query_params.get("email_status"))
    if not tournament_id and slug:
        tournament, _settings, _days, _events = get_public_tournament_bundle(
            supabase, club_id=_safe_text(getattr(ctx, "club_id", "")), registration_slug=slug
        )
        tournament_id = _safe_text((tournament or {}).get("id"))
    registration_id, submission_result = _registration_id_from_query_or_session(tournament_id, query_registration_id)
    if not email_status:
        email_status = _safe_text(submission_result.get("email_status"))
    if not tournament_id or not registration_id:
        st.error("We could not find that registration confirmation link. If you just submitted, return to registration and your saved summary should still be available.")
        nav_params = {"tournament_id": tournament_id} if tournament_id else {}
        if slug:
            nav_params["tournament"] = slug
        if st.button("Back to tournament registration", key="confirmation_missing_back"):
            navigate_same_tab(page="tournament_registration", params=nav_params, public_mode=True)
        return

    try:
        bundle = get_registration_confirmation_bundle(supabase, tournament_id, registration_id)
    except Exception:
        st.error("We could not load that registration confirmation right now. Please contact tournament staff.")
        return
    if not (bundle.get("registration") or {}):
        st.error("We could not find that registration.")
        return

    sender_status = get_smtp_config_status()
    render_registration_confirmation_summary(
        bundle=bundle,
        email_status=email_status,
        sender_status=sender_status,
        show_title=True,
    )

    registration_slug = _safe_text((bundle.get("settings") or {}).get("registration_slug")) or slug
    nav_params = {"tournament_id": tournament_id}
    if registration_slug:
        nav_params["tournament"] = registration_slug
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Tournament Roster", key="confirmation_view_roster"):
            navigate_same_tab(page="tournament_roster", params=nav_params, public_mode=True)
    with col2:
        if st.button("Return to Tournament Registration", key="confirmation_return_registration"):
            navigate_same_tab(page="tournament_registration", params=nav_params, public_mode=True)
