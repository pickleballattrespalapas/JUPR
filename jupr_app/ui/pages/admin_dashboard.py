from __future__ import annotations

import html

import streamlit as st

from jupr_app.domain.job_monitor import fetch_recent_jobs
from jupr_app.domain.system_health import get_system_health
from jupr_app.ui.layout import page_shell


def _nav_card(title: str, desc: str, target_label: str):
    clicked = st.button(
        label="",
        key=f"card_{target_label}",
        help=desc,
        use_container_width=True,
    )
    st.markdown(
        f"""
        <div class="jupr-admin-card">
            <div class="jupr-admin-card__title">{html.escape(title)}</div>
            <div class="jupr-admin-card__desc">{html.escape(desc)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if clicked:
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
    st.markdown('<div class="jupr-grid">', unsafe_allow_html=True)

    _nav_card("League Manager", "Manage standings & divisions", "🏟️ League Manager")
    _nav_card("Match Uploader", "Fast score entry", "📝 Match Uploader")
    _nav_card("Match Log", "Bulk edit + replay", "📝 Match Log")
    _nav_card("Player Editor", "Merge & manage players", "👥 Player Editor")
    _nav_card("Admin Tools", "Recompute & system ops", "⚙️ Admin Tools")
    _nav_card("Weekly Recap", "Generate & publish recap", "🗞️ Weekly Recap Admin")

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("System Health")

    health = get_system_health(ctx.supabase, ctx.club_id)

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Matches", health.get("match_count"))
    c2.metric("Pending Badge Jobs", health.get("pending_badge_jobs"))
    c3.metric("Last Check (UTC)", health.get("timestamp")[:19])

    st.divider()
    st.subheader("Background Jobs")

    df_jobs = fetch_recent_jobs(ctx.supabase, ctx.club_id)

    if df_jobs.empty:
        st.info("No recent jobs.")
    else:
        st.dataframe(
            df_jobs.reindex(columns=["job_type", "status", "created_at", "completed_at"]),
            use_container_width=True,
            hide_index=True,
        )
