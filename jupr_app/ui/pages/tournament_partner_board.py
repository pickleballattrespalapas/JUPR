from __future__ import annotations

import streamlit as st

from jupr_app.ui.pages import tournament_partner_request, tournament_roster


def render(ctx):
    if str(st.query_params.get("target_selection_id") or "").strip():
        tournament_partner_request.render(ctx)
        return
    tournament_roster.render(ctx, focus_partners=True, legacy_partner_board=True)
