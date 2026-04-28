from __future__ import annotations

import streamlit as st

from jupr_app.ui.layout import page_shell


def render(ctx) -> None:  # noqa: ARG001
    page_shell("Terms of Use", "Basic rules for using JUPR pages and services.", mode_label="Public")
    st.markdown(
        """
### Service scope
JUPR provides player ratings, match history, standings, and related club tools for Tres Palapas.

### Acceptable use
Please use JUPR lawfully and respectfully. Do not attempt to access restricted admin functions without authorization.

### Data accuracy and corrections
We try to keep data accurate, but errors can happen. If you see an issue, request a correction at **joe@juprleagues.com**.

### Availability
JUPR may change features, page content, or availability as operations evolve.

### Contact
For support or policy questions, email **joe@juprleagues.com**.
"""
    )
