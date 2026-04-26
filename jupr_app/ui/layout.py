from __future__ import annotations

import html
import streamlit as st

from jupr_app.ui.theme_clean import page_header, topbar


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
    if not resolved_right_html:
        chips: list[str] = []
        if mode_label:
            chips.append(f'<span class="jupr-pill neutral">{html.escape(mode_label)}</span>')

        resolved_right_html = "".join(chips)

    in_public_shell = bool(st.session_state.get("jupr_public_mode", False))
    if in_public_shell:
        page_header(title, subtitle, right_html=resolved_right_html)
    else:
        topbar(title, subtitle, right_html=resolved_right_html)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
