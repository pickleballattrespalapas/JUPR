from __future__ import annotations

import html
from urllib.parse import urlencode

import streamlit as st

from jupr_app.ui.theme_clean import topbar


def build_admin_entry_url() -> str:
    """
    Build a deterministic URL that explicitly enters the admin shell.
    """
    preserved_page = str(st.query_params.get("page", "") or "").strip()
    params: dict[str, str] = {"admin": "1"}
    if preserved_page:
        params["page"] = preserved_page
    return f"/?{urlencode(params)}"


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

        in_public_shell = bool(st.session_state.get("jupr_public_mode", False))
        in_admin_entry = bool(st.session_state.get("jupr_admin_entry_active", False))
        if in_public_shell and not in_admin_entry:
            admin_href = html.escape(build_admin_entry_url(), quote=True)
            chips.append(
                f'<a class="jupr-topbar-action" href="{admin_href}" '
                'aria-label="Open admin login">Admin Login</a>'
            )

        resolved_right_html = "".join(chips)

    topbar(title, subtitle, right_html=resolved_right_html)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
