from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from jupr_app.ui.layout import page_shell


def render(ctx) -> None:
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell(
        "📬 Player Updates Admin",
        "Review verified player update requests and monitor delivery pipeline.",
        mode_label=mode_label,
    )

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    pending_tab, active_tab, digests_tab, queue_tab = st.tabs(
        [
            "Pending Requests",
            "Active Profiles",
            "Weekly Digests",
            "Send Queue",
        ]
    )

    with pending_tab:
        st.subheader("Pending Requests")
        with st.form("player_updates_pending_filters"):
            st.text_input("Search email", key="player_updates_pending_email")
            st.number_input("Limit", min_value=1, max_value=500, value=100, step=1, key="player_updates_pending_limit")
            st.form_submit_button("Apply Filters")
        st.info("Placeholder: pending request moderation tools will be added in a follow-up.")

    with active_tab:
        st.subheader("Active Profiles")
        with st.form("player_updates_active_filters"):
            st.text_input("Search player id", key="player_updates_active_player")
            st.number_input("Limit", min_value=1, max_value=500, value=100, step=1, key="player_updates_active_limit")
            st.form_submit_button("Apply Filters")
        st.info("Placeholder: active subscription management tools will be added in a follow-up.")

    with digests_tab:
        st.subheader("Weekly Digests")
        today = date.today()
        with st.form("player_updates_digests_filters"):
            st.date_input("Week start (from)", value=today - timedelta(days=28), key="player_updates_digest_start")
            st.date_input("Week start (to)", value=today, key="player_updates_digest_end")
            st.form_submit_button("Apply Filters")
        st.info("Placeholder: digest review UI will be added after generation logic is implemented.")

    with queue_tab:
        st.subheader("Send Queue")
        with st.form("player_updates_queue_filters"):
            st.selectbox("Status", ["pending", "sent", "skipped", "error"], key="player_updates_queue_status")
            st.number_input("Limit", min_value=1, max_value=500, value=100, step=1, key="player_updates_queue_limit")
            st.form_submit_button("Apply Filters")
        st.info("Placeholder: queue operations and retries will be added in a follow-up.")
