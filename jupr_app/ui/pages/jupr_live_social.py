from __future__ import annotations

import streamlit as st

from jupr_app.ui.pages import jupr_live


def render(ctx):
    st.info("JUPR Live Social has moved inside JUPR Live under the Club Social mode.")
    st.session_state[jupr_live.PREFILL_MODE_KEY] = "Club Social"
    jupr_live.render(ctx)
