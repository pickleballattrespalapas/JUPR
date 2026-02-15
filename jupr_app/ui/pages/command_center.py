from __future__ import annotations

import streamlit as st

from jupr_app.ui.components.theme_toggle import render_theme_toggle


def render(ctx) -> None:
    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    col_l, col_r = st.columns([0.8, 0.2])
    with col_l:
        st.title("JUPR Club Command Center")
        st.caption("Admin operations hub for match entry and league workflows.")
    with col_r:
        render_theme_toggle(key="cc_theme_toggle", label="Dark theme")

    st.info("Use the quick links below to move between common admin workflows.")

    st.subheader("Quick actions")
    row1_col1, row1_col2, row1_col3 = st.columns(3)
    with row1_col1:
        st.link_button("Record Match", "/?page=record_match", use_container_width=True)
    with row1_col2:
        st.link_button("League Manager", "/?page=league_manager", use_container_width=True)
    with row1_col3:
        st.link_button("Match Log", "/?page=match_log", use_container_width=True)

    row2_col1, row2_col2, row2_col3 = st.columns(3)
    with row2_col1:
        st.link_button("Player Editor", "/?page=player_editor", use_container_width=True)
    with row2_col2:
        st.link_button("Admin Tools", "/?page=admin_tools", use_container_width=True)
    with row2_col3:
        st.link_button("Weekly Recap Admin", "/?page=weekly_recap_admin", use_container_width=True)

    st.subheader("Alerts")
    st.info("Coming soon")

    st.subheader("Active competitions")
    st.info("Coming soon")

    st.subheader("Leaderboards snapshot")
    st.info("Coming soon")

    st.subheader("Public navigation")
    st.info("Coming soon")
