from __future__ import annotations

import streamlit as st

from jupr_app.domain.notifications.player_profile_update_repo import (
    REQUEST_STATUS_UNSUBSCRIBED,
    unsubscribe_via_public_link,
)
from jupr_app.ui.helpers import qp_get
from jupr_app.ui.layout import page_shell


def parse_unsubscribe_identifiers(*, token_q: str, ut_q: str, sid_q: str, subscription_id_q: str) -> tuple[str, str]:
    token = str(token_q or ut_q or "").strip()
    subscription_id = str(sid_q or subscription_id_q or "").strip()
    return token, subscription_id


def render(ctx) -> None:
    page_shell("Email Preferences", "Manage verified player update email subscriptions.", mode_label="Public")

    token, subscription_id = parse_unsubscribe_identifiers(
        token_q=qp_get("token", ""),
        ut_q=qp_get("ut", ""),
        sid_q=qp_get("sid", ""),
        subscription_id_q=qp_get("subscription_id", ""),
    )

    if not token and not subscription_id:
        st.info("No unsubscribe link parameters were provided. Use the unsubscribe link from one of your player update emails.")
        st.caption("Need help? Contact joe@juprleagues.com")
        return

    action_key = f"email_unsubscribed:{token}:{subscription_id}"
    if action_key not in st.session_state:
        try:
            row = unsubscribe_via_public_link(
                ctx.supabase,
                unsubscribe_token=token or None,
                subscription_id=subscription_id or None,
            )
            st.session_state[action_key] = row
        except Exception as exc:
            st.error(f"We could not process this unsubscribe link: {exc}")
            st.caption("If this keeps happening, email joe@juprleagues.com and we can unsubscribe you manually.")
            return

    row = st.session_state.get(action_key) or {}
    if str(row.get("request_status") or "").strip().lower() == REQUEST_STATUS_UNSUBSCRIBED:
        st.success("You’re unsubscribed from these verified player update emails.")
    else:
        st.success("Your email preference has been updated.")

    st.caption("You can contact joe@juprleagues.com any time to resubscribe or update preferences.")
