from __future__ import annotations

import streamlit as st

from jupr_app.ui.components.weekly_recap_layout import render_weekly_recap
from jupr_app.ui.layout import page_shell


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("🗞️ Tres Palapas Weekly Recap", "Club-wide weekly recap.", mode_label=mode_label)

    supabase = ctx.supabase
    club_id = str(ctx.club_id)

    response = (
        supabase.table("weekly_recaps")
        .select("week_start,week_end,status,final_json")
        .eq("club_id", club_id)
        .eq("status", "published")
        .order("week_start", desc=True)
        .execute()
    )
    published = response.data or []
    if not published:
        st.info("No published recaps yet.")
        return

    week_options = [row["week_start"] for row in published]
    selected_week = st.selectbox("Select week", options=week_options, format_func=str)
    selected_row = next((row for row in published if row["week_start"] == selected_week), published[0])

    recap = selected_row.get("final_json") or {}
    render_weekly_recap(recap, print_view=False)

    st.caption("Tip: use your browser print dialog for a bulletin-board-ready PDF.")
    base_url = st.session_state.get("base_url", "")
    if base_url:
        print_url = f"{base_url}/?page=weekly_recap&public=1"
        st.link_button("Open Print-Friendly View", print_url)

    if not bool(ctx.public_mode):
        draft_check = (
            supabase.table("weekly_recaps")
            .select("week_start")
            .eq("club_id", club_id)
            .eq("status", "draft")
            .eq("week_start", selected_week)
            .execute()
        )
        if draft_check.data:
            st.info("Draft exists for this week; public view shows published only.")
