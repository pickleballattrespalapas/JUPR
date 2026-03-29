from __future__ import annotations

import streamlit as st

from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.live_beta_engine import clear_expired_substitutions
from jupr_app.ui.layout import page_shell
from jupr_app.ui.live.shared import (
    LivePageConfig,
    build_league_round_official_payloads,
    build_rr_official_payloads,
    build_tournament_official_payloads,
    mark_tournament_payloads_saved,
    render_live_page,
)


ADMIN_CONFIG = LivePageConfig(
    state_key="jupr_live_admin_state",
    intro_markdown=(
        "Use the official JUPR Live workflow for roster-resolved Round Robin and League / Ladder events. "
        "League/session context and official finalize/save actions only live here."
    ),
    event_types=("Round Robin", "League / Ladder", "Tournament"),
    mode_pill_label="Official",
    allow_official=True,
    allow_tournament=True,
    show_official_context=True,
)


def _process_payloads(ctx, payloads: list[dict]) -> dict:
    return process_matches(
        payloads,
        supabase=ctx.supabase,
        club_id=str(ctx.club_id),
        name_to_id=ctx.name_to_id,
        df_players_all=ctx.df_players_all,
        df_leagues=ctx.df_leagues,
        df_meta=ctx.df_meta,
    )


def _save_rr_official(ctx, state: dict, event: dict) -> None:
    if "rr" in set(state.get("last_saved_rounds") or []):
        st.info("Official round robin results were already saved in this session.")
        return
    payloads = build_rr_official_payloads(state, event)
    if not payloads:
        st.warning("Enter at least one scored match before saving officially.")
        return
    res = _process_payloads(ctx, payloads)
    state["last_saved_rounds"] = ["rr"]
    event["saved_rounds"] = ["rr"]
    clear_expired_substitutions(event)
    st.session_state["force_data_refresh"] = True
    st.success(f"Official results saved ({res['inserted']} matches).")


def _save_league_round_official(ctx, state: dict, event: dict) -> bool:
    current_round_number = int(event.get("currentRoundNumber") or 1)
    saved_rounds = set(state.get("last_saved_rounds") or [])
    if current_round_number in saved_rounds:
        return True
    payloads = build_league_round_official_payloads(state, event)
    if not payloads:
        st.warning("Enter at least one scored match before finalizing this round.")
        return False
    res = _process_payloads(ctx, payloads)
    saved_rounds.add(current_round_number)
    state["last_saved_rounds"] = sorted(saved_rounds)
    event["saved_rounds"] = sorted(saved_rounds)
    clear_expired_substitutions(event)
    st.session_state["force_data_refresh"] = True
    st.success(
        f"Official round {current_round_number} saved ({res['inserted']} matches)."
    )
    return True


def _save_tournament_official(ctx, state: dict, event: dict) -> None:
    payloads = build_tournament_official_payloads(event)
    if not payloads:
        st.info("No newly completed tournament matches to save yet.")
        return
    res = _process_payloads(ctx, payloads)
    mark_tournament_payloads_saved(event, payloads)
    st.session_state["force_data_refresh"] = True
    st.success(f"Official tournament results saved ({res['inserted']} matches).")


def render(ctx):
    page_shell(
        "🔴 JUPR Live Admin",
        "Run official JUPR Live events with save controls and league/session context.",
        mode_label="Admin",
    )
    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.warning("Admin login required.")
        return
    render_live_page(
        ctx,
        ADMIN_CONFIG,
        on_save_rr=_save_rr_official,
        on_save_league=_save_league_round_official,
        on_save_tournament=_save_tournament_official,
    )
