from __future__ import annotations

import streamlit as st

from jupr_app.ui.layout import page_shell


def render(ctx) -> None:  # noqa: ARG001
    page_shell("Privacy Policy", "How JUPR handles player and subscriber data.", mode_label="Public")
    st.markdown(
        """
### Who operates JUPR
JUPR is operated by **Joe Baumann** at **Tres Palapas, Los Barriles, BCS, Mexico**.

### What data we use
We keep match and profile data needed to run ratings, leagues, tournaments, and player pages. For verified update emails, we store the email address and subscription history.

### How we use it
- Run ratings, standings, and profile pages.
- Send verified player update emails when requested.
- Support corrections, operations, and system integrity.

### Public profile behavior
Players may use a public alias on public pages. Admin tools may still show a player's true name for operations, results integrity, and correction workflows.

### Your choices
- You can unsubscribe from update emails using the unsubscribe link in any update email.
- You may request profile removal or anonymization.
- You may request data corrections.

### Contact for privacy and data requests
Email **joe@juprleagues.com** for privacy questions, correction requests, or profile removal/anonymization requests.
"""
    )
