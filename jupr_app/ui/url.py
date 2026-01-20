# jupr_app/ui/url.py
from __future__ import annotations

import streamlit as st


def qp_get(key: str, default: str = "") -> str:
    """Streamlit query params can be str or list depending on version."""
    try:
        v = st.query_params.get(key, default)
    except Exception:
        return default
    if isinstance(v, list):
        return v[0] if v else default
    return str(v) if v is not None else default
