from __future__ import annotations

import streamlit as st

from jupr_app.ui.layout import page_shell


def _nav_button(label: str, target_label: str):
    if st.button(label, use_container_width=True):
        st.session_state["main_nav"] = target_label
        st.rerun()


def render(ctx):
    if not bool(ctx.admin_logged_in):
        st.error("Admin login required.")
        st.stop()

    page_shell(
        "🧭 Command Center",
        "Operational dashboard for administrators.",
        mode_label="Admin",
    )

    st.subheader("Core Operations")

    col1, col2, col3 = st.columns(3)

    with col1:
        _nav_button("🏟️ League Manager", "🏟️ League Manager")
        _nav_button("📝 Match Uploader", "📝 Match Uploader")

    with col2:
        _nav_button("📝 Match Log", "📝 Match Log")
        _nav_button("👥 Player Editor", "👥 Player Editor")

    with col3:
        _nav_button("⚙️ Admin Tools", "⚙️ Admin Tools")
        _nav_button("🗞️ Weekly Recap Admin", "🗞️ Weekly Recap Admin")

    st.divider()

    st.subheader("Advanced")

    col4, col5 = st.columns(2)

    with col4:
        _nav_button("🛠️ Challenge Ladder Admin", "🛠️ Challenge Ladder Admin")
        _nav_button("🏆 Tournament Manager", "🏆 Tournament Manager")

    with col5:
        _nav_button("💰 Moneyball", "💰 Moneyball")
        _nav_button("🎨 Theme QA", "🎨 Theme QA")
