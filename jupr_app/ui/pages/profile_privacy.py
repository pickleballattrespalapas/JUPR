from __future__ import annotations

import streamlit as st

from jupr_app.ui.layout import page_shell


def render(ctx) -> None:  # noqa: ARG001
    page_shell("Profile Privacy", "How public profiles and aliases are handled.", mode_label="Public")
    st.markdown(
        """
JUPR supports public player profiles so leagues and ratings are understandable to participants.

### Public vs admin visibility
- Public pages may show a player alias.
- Admin tools may continue to show the true name for operations and result integrity.

### Removal or anonymization requests
Players may request profile removal or anonymization by emailing **joe@juprleagues.com**.
"""
    )
