from __future__ import annotations

import streamlit as st


def qp_get(key: str, default=None):
    """
    Safe getter for query params.
    Returns string or default.
    """
    try:
        return st.query_params.get(key, default)
    except Exception:
        return default


def qp_set(**kwargs):
    """
    Set one or more query params.
    """
    for k, v in kwargs.items():
        if v is None:
            continue
        st.query_params[k] = str(v)


def qp_clear():
    """
    Clear all query params.
    """
    try:
        st.query_params.clear()
    except Exception:
        pass
