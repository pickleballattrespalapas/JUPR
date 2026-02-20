from __future__ import annotations

import streamlit as st

from jupr_app.domain.job_monitor import fetch_recent_jobs
from jupr_app.domain.system_health import get_system_health
from jupr_app.ui.layout import page_shell


def _nav_card(
    title: str,
    desc: str,
    target_label: str,
    metric: str | None = None,
    status: str = "green",
):
    wrapper_key = f"card_{target_label}"

    clicked = st.button(
        f"{title}\n\n{desc}",
        key=wrapper_key,
        use_container_width=True,
    )

    status_class = {
        "green": "jupr-status-green",
        "yellow": "jupr-status-yellow",
        "red": "jupr-status-red",
    }.get(status, "jupr-status-green")

    st.markdown(
        f"""
        <div class="jupr-admin-card-wrapper">
            <div class="jupr-admin-title-row">
                <div></div>
                <div class="jupr-status-dot {status_class}"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if clicked:
        st.session_state["_admin_selected_page"] = label
        # Do NOT call st.rerun()


def render(ctx):
    if not bool(ctx.admin_logged_in):
        st.error("Admin login required.")
        st.stop()

    page_shell(
        "🧭 Command Center",
        "Operational dashboard for administrators.",
        mode_label="Admin",
    )

    health = get_system_health(ctx.supabase, ctx.club_id)

    st.subheader("Core Operations")
    st.markdown('<div class="jupr-admin-grid">', unsafe_allow_html=True)

    _nav_card("League Manager", "Manage standings & divisions", "🏟️ League Manager")
    _nav_card("Match Uploader", "Fast score entry", "📝 Match Uploader")
    _nav_card("Match Log", "Bulk edit + replay", "📝 Match Log")
    _nav_card("Player Editor", "Merge & manage players", "👥 Player Editor")
    pending = health.get("pending_badge_jobs", 0)
    status = "green"
    if isinstance(pending, int) and pending > 20:
        status = "red"
    elif isinstance(pending, int) and pending > 5:
        status = "yellow"

    _nav_card(
        "Admin Tools",
        "Recompute & system ops",
        "⚙️ Admin Tools",
        metric=pending,
        status=status,
    )
    _nav_card("Weekly Recap", "Generate & publish recap", "🗞️ Weekly Recap Admin")

    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.subheader("System Health")

    st.metric("Total Matches", health.get("match_count"))
    st.metric("Pending Badge Jobs", health.get("pending_badge_jobs"))
    st.metric("Last Check (UTC)", health.get("timestamp")[:19])

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
