from __future__ import annotations

import streamlit as st

from jupr_app.ui.layout import page_shell


def render(ctx) -> None:  # noqa: ARG001
    page_shell("Contact Support", "Need help with ratings, pages, or subscriptions?", mode_label="Public")
    st.markdown(
        """
For support, email **joe@juprleagues.com**.

Please include:
- your name,
- player name or profile link (if relevant),
- and a short description of the issue.

Operator: **Joe Baumann**  
Location: **Tres Palapas, Los Barriles, BCS, Mexico**
"""
    )
