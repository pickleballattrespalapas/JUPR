from __future__ import annotations

import streamlit as st

from jupr_app.ui.layout import page_shell
from jupr_app.ui.theme_tokens import get_theme_tokens


WIZARD_STEP_KEY = "record_match_wizard_step"
SELECTED_TYPE_KEY = "record_match_competition_type"

COMPETITION_TYPES: list[dict[str, str]] = [
    {
        "id": "ladder_league",
        "title": "Ladder League",
        "icon": "🪜",
        "description": "Structured weekly ladder play with tracked standings.",
    },
    {
        "id": "challenge_ladder",
        "title": "Challenge Ladder",
        "icon": "⚔️",
        "description": "Open challenges where players climb by results.",
    },
    {
        "id": "tournament",
        "title": "Tournament",
        "icon": "🏆",
        "description": "Bracketed or pool-based tournament results.",
    },
    {
        "id": "round_robin",
        "title": "Round Robin",
        "icon": "🔄",
        "description": "Everyone plays everyone within a group.",
    },
    {
        "id": "moneyball",
        "title": "Moneyball",
        "icon": "💰",
        "description": "Moneyball format sessions and side-game outcomes.",
    },
    {
        "id": "bulk_match_entry",
        "title": "Bulk Match Entry",
        "icon": "📥",
        "description": "Fast multi-match input for admins and captains.",
    },
]


def _ensure_state() -> None:
    st.session_state.setdefault(WIZARD_STEP_KEY, 1)
    st.session_state.setdefault(SELECTED_TYPE_KEY, None)


def _step_1_competition_type(tokens: dict[str, str]) -> None:
    st.markdown("### Step 1 · Choose competition type")
    st.caption("Select the format first. The wizard will tailor next steps to this selection.")

    st.markdown(
        f"""
        <style>
        .record-match-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin: 0.5rem 0 1rem 0;
        }}
        .record-match-card {{
            border: 1px solid {tokens['border_subtle']};
            border-radius: 12px;
            background: {tokens['card_bg']};
            padding: 12px;
            color: {tokens['text_primary']};
            min-height: 112px;
        }}
        .record-match-card h4 {{
            margin: 0 0 6px 0;
            color: {tokens['text_primary']};
        }}
        .record-match-card p {{
            margin: 0;
            color: {tokens['text_secondary']};
            font-size: 0.9rem;
            line-height: 1.35;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    card_html = "".join(
        [
            (
                "<div class='record-match-card'>"
                f"<h4>{option['icon']} {option['title']}</h4>"
                f"<p>{option['description']}</p>"
                "</div>"
            )
            for option in COMPETITION_TYPES
        ]
    )
    st.markdown(f"<div class='record-match-grid'>{card_html}</div>", unsafe_allow_html=True)

    selector_options = ["Select one…"] + [option["title"] for option in COMPETITION_TYPES]
    selected_title = st.selectbox(
        "Competition type",
        options=selector_options,
        key="record_match_competition_selector",
    )

    if selected_title != "Select one…":
        selected_option = next(opt for opt in COMPETITION_TYPES if opt["title"] == selected_title)
        st.session_state[SELECTED_TYPE_KEY] = selected_option

    controls = st.columns([1, 1, 3])
    with controls[0]:
        next_disabled = st.session_state.get(SELECTED_TYPE_KEY) is None
        if st.button("Next →", type="primary", disabled=next_disabled):
            st.session_state[WIZARD_STEP_KEY] = 2
            st.rerun()


def _step_2_placeholder(tokens: dict[str, str]) -> None:
    selected = st.session_state.get(SELECTED_TYPE_KEY)
    selected_label = selected["title"] if isinstance(selected, dict) else "Not selected"

    st.markdown("### Step 2 · Match details")
    st.caption("Scaffold only: match inputs and submission wiring will be added in follow-up tasks.")

    st.markdown(
        f"""
        <div style="
            border: 1px dashed {tokens['border_subtle']};
            border-radius: 12px;
            background: {tokens['card_bg']};
            padding: 14px;
            color: {tokens['text_primary']};
            margin: 0.5rem 0 1rem 0;
        ">
            <strong>Selected competition type:</strong> {selected_label}<br/>
            <span style="color: {tokens['text_secondary']};">
                This step intentionally omits submission logic for now.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    controls = st.columns([1, 1, 3])
    with controls[0]:
        if st.button("← Back"):
            st.session_state[WIZARD_STEP_KEY] = 1
            st.rerun()
    with controls[1]:
        st.button("Submit (coming soon)", type="primary", disabled=True)


def render(ctx) -> None:
    mode_label = "Public" if bool(getattr(ctx, "public_mode", False)) else "Admin"
    page_shell("🧾 Record Match", "Unified wizard for recording results across competition types.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin only.")
        return

    _ensure_state()
    tokens = get_theme_tokens()

    step = int(st.session_state.get(WIZARD_STEP_KEY, 1))
    if step <= 1:
        _step_1_competition_type(tokens)
    else:
        _step_2_placeholder(tokens)
