from __future__ import annotations

import streamlit as st

from jupr_app.domain.live_social import save_social_round_robin
from jupr_app.ui.layout import page_shell
from jupr_app.ui.live.shared import LivePageConfig, render_live_page


SOCIAL_CONFIG = LivePageConfig(
    state_key="jupr_live_social_state",
    intro_markdown=(
        "Create and save social Round Robin results for community recognition and recaps. "
        "These saves are persistent but unrated and do not affect ratings, leaderboards, "
        "match history, or replay logic."
    ),
    event_types=("Round Robin",),
    mode_pill_label="Social",
    allow_official=False,
    allow_tournament=False,
    show_official_context=False,
    persistent_save_label="Save social results",
)


def _save_rr_social(ctx, state: dict, event: dict) -> None:
    result = save_social_round_robin(ctx, event, saved_by="admin")
    state["last_saved_rounds"] = ["rr"]
    event["saved_rounds"] = ["rr"]
    st.session_state["force_data_refresh"] = True
    st.success(
        "Social results saved "
        f"({result['match_count']} matches, {result['participant_count']} participants)."
    )


def render(ctx):
    page_shell(
        "🟢 JUPR Live Social",
        "Admin-only social lane: saved, unrated Round Robin events for community recognition.",
        mode_label="Admin",
    )
    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.warning("Admin login required.")
        return
    render_live_page(
        ctx,
        SOCIAL_CONFIG,
        on_save_rr=_save_rr_social,
    )
