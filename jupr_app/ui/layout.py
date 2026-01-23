from __future__ import annotations

import html

import streamlit as st

from jupr_app.ui.theme_clean import topbar, divider, callout


def page_shell(
    title: str,
    subtitle: str = "",
    *,
    mode_label: str = "",
    right_html: str = "",
) -> None:
    """
    Render a consistent page chrome wrapper (topbar + spacing).
    """
    resolved_right_html = right_html
    if not resolved_right_html and mode_label:
        resolved_right_html = f'<span class="jupr-pill neutral">{html.escape(mode_label)}</span>'

    topbar(title, subtitle, right_html=resolved_right_html)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


def theme_sanity_block() -> None:
    """
    Temporary helper to confirm theme primitives are active.
    TODO: Remove after visual verification.
    """
    callout("info", "New", "This page uses the updated clean theme primitives.")
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="jupr-card jupr-card--hover">
          <div class="jupr-section-title">Theme primitives active</div>
          <div class="jupr-topbar__subtitle">
            Cards, callouts, and dividers are now available for reuse across pages.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    divider()
