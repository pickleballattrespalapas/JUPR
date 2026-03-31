from __future__ import annotations

import streamlit as st

from jupr_app.ui.page_registry import PAGE_KEY_TO_LABEL, PUBLIC_NAV_KEYS
from jupr_app.ui.url import qp_get


def render_public_top_nav(
    *,
    labels_in_order: list[str] | None = None,
    current_label: str | None = None,
    default_page: str = "leaderboards",
) -> str:
    """
    Render the horizontal public navigation and return the selected label.

    If callers do not provide a label list, the shared public-page registry drives the nav.
    """
    if labels_in_order is None:
        labels_in_order = [
            PAGE_KEY_TO_LABEL[key]
            for key in PUBLIC_NAV_KEYS
            if key in PAGE_KEY_TO_LABEL
        ]

    if not labels_in_order:
        return PAGE_KEY_TO_LABEL.get(default_page, default_page)

    if current_label is None:
        current_page = (qp_get("page", "") or "").strip() or default_page
        current_label = PAGE_KEY_TO_LABEL.get(current_page, labels_in_order[0])

    st.markdown("**Go to:**")

    try:
        idx = labels_in_order.index(current_label)
    except ValueError:
        idx = 0

    selected_label = st.radio(
        label="public_nav",
        options=labels_in_order,
        index=idx,
        horizontal=True,
        label_visibility="collapsed",
        key="public_top_nav_radio",
    )

    return selected_label
