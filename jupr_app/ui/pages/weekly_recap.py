from __future__ import annotations

import streamlit as st
from postgrest.exceptions import APIError

from jupr_app.ui.components.weekly_recap_layout import render_weekly_recap
from jupr_app.ui.layout import page_shell
from jupr_app.ui.url import qp_get


def _get_api_error_code(exc: APIError) -> str | None:
    code = getattr(exc, "code", None)
    if code:
        return code
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("code")
    return None


def _handle_missing_table(exc: APIError) -> bool:
    code = _get_api_error_code(exc)
    if code in {"PGRST205", "42P01"}:
        st.error("Weekly recaps table not found. Apply migration supabase/migrations/20260207_weekly_recaps.sql in Supabase.")
        return True
    return False


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("🗞️ Tres Palapas Weekly Recap", "Club-wide weekly recap.", mode_label=mode_label)

    print_mode = qp_get("print", "0").lower() in ("1", "true", "yes", "y")

    if print_mode:
        st.markdown("<style>header{visibility:hidden;} footer{visibility:hidden;} </style>", unsafe_allow_html=True)

    supabase = ctx.supabase
    club_id = str(ctx.club_id)

    try:
        response = (
            supabase.table("weekly_recaps")
            .select("week_start,week_end,status,final_json")
            .eq("club_id", club_id)
            .eq("status", "published")
            .order("week_start", desc=True)
            .execute()
        )
    except APIError as exc:
        if _handle_missing_table(exc):
            return
        raise
    published = response.data or []
    if not published:
        st.info("No published recaps yet.")
        return

    selected_row = published[0]
    if not print_mode and len(published) > 1:
        week_options = [row["week_start"] for row in published]
        selected_week = st.selectbox("Select week", options=week_options, format_func=str)
        selected_row = next((row for row in published if row["week_start"] == selected_week), published[0])

    recap = selected_row.get("final_json") or {}
    render_weekly_recap(recap, print_view=print_mode)

    if not print_mode:
        st.caption("Tip: use your browser print dialog for a bulletin-board-ready PDF.")
        base_url = st.session_state.get("base_url", "")
        if base_url:
            print_url = f"{base_url}/?page=weekly_recap&public=1&print=1"
            st.link_button("Open Print-Friendly View", print_url)

    if (not print_mode) and (not bool(ctx.public_mode)):
        try:
            draft_check = (
                supabase.table("weekly_recaps")
                .select("week_start")
                .eq("club_id", club_id)
                .eq("status", "draft")
                .eq("week_start", selected_row["week_start"])
                .execute()
            )
        except APIError as exc:
            if _handle_missing_table(exc):
                return
            raise
        if draft_check.data:
            st.info("Draft exists for this week; public view shows published only.")
