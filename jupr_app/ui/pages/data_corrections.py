from __future__ import annotations

import streamlit as st

from jupr_app.ui.layout import page_shell


def render(ctx) -> None:  # noqa: ARG001
    page_shell("Data Corrections", "How to request score or profile corrections.", mode_label="Public")
    st.markdown(
        """
If a score, player assignment, or profile detail looks wrong, request a correction at **joe@juprleagues.com**.

Please include:
- match date,
- players involved,
- what is currently shown,
- and what should be corrected.

We review requests and apply approved corrections through admin workflows.
"""
    )
