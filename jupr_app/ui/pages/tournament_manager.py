from __future__ import annotations

import streamlit as st

from jupr_app.ui.layout import page_shell


def render(ctx):
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell("🏆 Tournament Manager", "Admin-only tournament operations.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    st.info("Tournament Manager tools are under construction.")
