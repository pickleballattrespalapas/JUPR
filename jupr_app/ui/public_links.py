# jupr_app/ui/public_links.py
from __future__ import annotations

import streamlit as st
import urllib.parse


def get_base_url() -> str:
    return (st.session_state.get("base_url", "") or "").rstrip("/")


def build_public_url(*, page: str, params: dict[str, str] | None = None) -> str:
    """
    Builds a canonical public URL using the configured Streamlit Cloud base.
    Always includes public=1.
    """
    base = get_base_url()
    q = {"page": page, "public": "1"}

    if params:
        for k, v in params.items():
            if v is None:
                continue
            q[str(k)] = str(v)

    return f"{base}/?{urllib.parse.urlencode(q)}"


def public_link_button(label: str, url: str):
    """
    A link-button with safe fallback for older Streamlit versions.
    """
    try:
        st.link_button(label, url)
    except Exception:
        st.markdown(
            f'<a href="{url}" target="_blank" rel="noopener noreferrer">'
            f'<button style="padding:0.5rem 1rem; font-size:1rem; border-radius:0.5rem; cursor:pointer;">'
            f'{label}</button></a>',
            unsafe_allow_html=True,
        )
