# jupr_app/ui/public_nav.py
from __future__ import annotations

import streamlit as st
from jupr_app.ui.url import qp_get


PUBLIC_NAV = [
    ("leaderboards", "🏆 Leaderboards"),
    ("match_explorer", "🎯 Match Explorer"),
    ("players", "🔎 Player Search"),
    ("challenge_ladder", "🪜 Challenge Ladder"),
    ("faqs", "❓ FAQs"),
]

def render_public_top_nav(*, default_page: str = "leaderboards") -> str:
    """
    Renders a horizontal, top-of-page navigation for PUBLIC_MODE.
    Returns the selected page key.
    """
    # Read current page from query params (supports deep links)
    current = (qp_get("page", "") or "").strip() or default_page

    keys = [k for k, _ in PUBLIC_NAV]
    labels = [label for _, label in PUBLIC_NAV]

    # Pick selected index (safe fallback)
    try:
        idx = keys.index(current)
    except ValueError:
        idx = keys.index(default_page) if default_page in keys else 0

    st.markdown("**Go to:**")
    selected_label = st.radio(
        label="public_nav",
        options=labels,
        index=idx,
        horizontal=True,
        label_visibility="collapsed",
        key="public_top_nav_radio",
    )

    selected_key = keys[labels.index(selected_label)]

    # Keep URL in sync (so links copy/paste nicely)
    # Note: Streamlit reruns automatically; no need to force a rerun.
    try:
        st.query_params["page"] = selected_key
    except Exception:
        pass

    return selected_key
