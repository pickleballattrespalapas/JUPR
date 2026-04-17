from __future__ import annotations

import streamlit as st

from jupr_app.domain.live_social import (
    SOCIAL_SKILL_LEVEL_OPTIONS,
    SOCIAL_TABLES_INSTALL_MESSAGE,
    SocialTablesNotInstalledError,
    normalize_skill_levels,
    save_social_live_event,
)
from jupr_app.ui.layout import page_shell
from jupr_app.ui.live import LivePageConfig, render_live_page


QUICK_SESSION_CONFIG = LivePageConfig(
    state_key="jupr_live_public_state",
    intro_markdown=(
        "Run a lightweight JUPR Live quick session with session-only scoring. "
        "No persistence, ratings effects, or official saves are used in this mode."
    ),
    event_types=("Round Robin", "League / Ladder"),
    mode_pill_label="Quick Session",
    allow_official=False,
    allow_tournament=False,
    show_official_context=False,
)

CLUB_SOCIAL_CONFIG = LivePageConfig(
    state_key="jupr_live_club_social_state",
    intro_markdown=(
        "Submit unrated club social results to recap moderation. "
        "Club Social saves are persistent but do not affect ratings, leaderboards, "
        "match history, or replay logic."
    ),
    event_types=("Round Robin", "League / Ladder"),
    mode_pill_label="Club Social",
    allow_official=False,
    allow_tournament=False,
    show_official_context=False,
    persistent_save_label="Submit club social results",
    requires_roster_resolution=True,
)

MODE_OPTIONS = ("Quick Session", "Club Social")
MODE_KEY = "jupr_live_mode"
PREFILL_MODE_KEY = "jupr_live_prefill_mode"
SOCIAL_SKILL_LEVELS_KEY = "jupr_live_social_skill_levels"


def _mode_selector() -> str:
    prefill_mode = st.session_state.pop(PREFILL_MODE_KEY, None)
    if prefill_mode in MODE_OPTIONS:
        st.session_state[MODE_KEY] = prefill_mode
    if st.session_state.get(MODE_KEY) not in MODE_OPTIONS:
        st.session_state[MODE_KEY] = MODE_OPTIONS[0]
    return st.radio(
        "Live mode",
        MODE_OPTIONS,
        horizontal=True,
        key=MODE_KEY,
    )


def _club_context(ctx) -> tuple[str | None, str | None]:
    club_id = str(getattr(ctx, "club_id", "") or "").strip()
    club_name = str(getattr(ctx, "club_name", "") or "").strip()
    return (club_id or None, club_name or None)


def _default_host_name(ctx) -> str:
    if bool(getattr(ctx, "admin_logged_in", False)):
        return "admin"
    return ""


def _save_social(ctx, state: dict, event: dict) -> bool:
    club_id, _ = _club_context(ctx)
    admin_logged_in = bool(getattr(ctx, "admin_logged_in", False))
    if not club_id:
        st.error("Club context is required to submit Club Social results.")
        return False

    host_name = normalize_host_name(
        st.session_state.get("jupr_live_social_host_name", _default_host_name(ctx))
    )
    if not admin_logged_in and not host_name:
        st.error("Host / Submitter Name is required for Club Social submissions.")
        return False
    if admin_logged_in and not host_name:
        host_name = "admin"

    skill_levels = normalize_skill_levels(st.session_state.get(SOCIAL_SKILL_LEVELS_KEY))
    submission_mode = "admin" if admin_logged_in else "public"
    try:
        result = save_social_live_event(
            ctx,
            event,
            target_club_id=club_id,
            submission_mode=submission_mode,
            host_name=host_name,
            skill_levels=skill_levels,
        )
    except SocialTablesNotInstalledError:
        st.error(SOCIAL_TABLES_INSTALL_MESSAGE)
        return False
    state["last_saved_rounds"] = list(result.get("saved_rounds") or [])
    event["saved_rounds"] = list(result.get("saved_rounds") or [])
    st.session_state["force_data_refresh"] = True
    try:
        from jupr_app.ui.pages.players import (
            fetch_player_social_event_history,
            fetch_player_social_participation,
        )

        fetch_player_social_event_history.clear()
        fetch_player_social_participation.clear()
    except Exception:
        pass
    status = str(result.get("status") or "")
    if status == "saved":
        title = "Club Social results saved"
    else:
        title = "Club Social results submitted and awaiting approval"
    st.success(
        f"{title} ({result['participant_count']} participants, {result['match_count']} matches, final status: {status})."
    )
    return True


def normalize_host_name(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _render_club_social_mode(ctx) -> None:
    club_id, club_name = _club_context(ctx)
    admin_logged_in = bool(getattr(ctx, "admin_logged_in", False))
    if "jupr_live_social_host_name" not in st.session_state:
        st.session_state["jupr_live_social_host_name"] = _default_host_name(ctx)
    if SOCIAL_SKILL_LEVELS_KEY not in st.session_state:
        st.session_state[SOCIAL_SKILL_LEVELS_KEY] = ["All"]

    if club_id:
        label = club_name or club_id
        st.caption(f"Submitting to club: **{label}**")
        st.caption(
            "Club Social submissions are unrated and may require admin approval before they appear in recap."
        )
    else:
        st.warning(
            "No club context found on this page. You can still score the event, "
            "but Club Social submit is disabled until a club context is available."
        )

    st.text_input(
        "Host / Submitter Name",
        key="jupr_live_social_host_name",
        disabled=admin_logged_in,
        help=(
            "Required for public/guest Club Social submissions. "
            "Admins default to 'admin'."
        ),
    )
    selected_skill_levels = st.multiselect(
        "Skill level tags",
        options=SOCIAL_SKILL_LEVEL_OPTIONS,
        key=SOCIAL_SKILL_LEVELS_KEY,
        help="Used for Weekly Recap social grouping (for example, Social 3.5, Social 4.0, or Social All).",
    )
    normalized_skill_levels = normalize_skill_levels(selected_skill_levels)
    if normalized_skill_levels != selected_skill_levels:
        st.session_state[SOCIAL_SKILL_LEVELS_KEY] = normalized_skill_levels
        st.rerun()
    st.caption(f"Weekly Recap grouping tags: {', '.join(normalized_skill_levels)}")

    if not club_id:
        render_live_page(ctx, CLUB_SOCIAL_CONFIG)
        return
    render_live_page(
        ctx,
        CLUB_SOCIAL_CONFIG,
        on_save_rr=_save_social,
        on_save_league=_save_social,
    )


def render(ctx):
    mode_label = (
        "Public"
        if bool(ctx.public_mode)
        else ("Admin" if bool(ctx.admin_logged_in) else "Guest")
    )
    page_shell(
        "🔴 JUPR Live",
        "Run Quick Session or Club Social events. Official rated workflows remain on JUPR Live Admin.",
        mode_label=mode_label,
    )
    selected_mode = _mode_selector()
    if selected_mode == "Club Social":
        _render_club_social_mode(ctx)
        return
    render_live_page(ctx, QUICK_SESSION_CONFIG)
