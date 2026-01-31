from __future__ import annotations

import logging

import streamlit as st

from jupr_app.ui.helpers import display_requirement_text
from jupr_app.ui.pages.players import badge_icon

logger = logging.getLogger(__name__)

EARNS_PAGE_SIZE = 25

SECTION_ORDER = ["Common", "Uncommon", "Rare", "Legendary", "Unclaimed", "Unranked", "Other"]


def _badge_earner_counts(df_player_badges) -> dict[str, int]:
    if df_player_badges is None or df_player_badges.empty:
        return {}
    if "badge_id" not in df_player_badges.columns or "player_id" not in df_player_badges.columns:
        return {}
    unique = df_player_badges.drop_duplicates(subset=["badge_id", "player_id"])
    counts = unique["badge_id"].astype(str).value_counts()
    return {str(k): int(v) for k, v in counts.to_dict().items()}


def get_all_badges(df_badges, df_player_badges) -> list[dict]:
    if df_badges is None or df_badges.empty:
        return []
    earners_map = _badge_earner_counts(df_player_badges)
    has_player_badges = df_player_badges is not None
    badges = []
    for row in df_badges.itertuples(index=False):
        badge_id = str(getattr(row, "badge_id", "") or "")
        name = str(getattr(row, "name", "") or "Badge")
        earners_count = earners_map.get(badge_id)
        if earners_count is None and has_player_badges:
            earners_count = 0
        badges.append(
            {
                "badge_id": badge_id,
                "name": name,
                "category": getattr(row, "category", None),
                "prestige": getattr(row, "prestige", 0),
                "requirements": getattr(row, "requirements", None),
                "earners_count": earners_count,
            }
        )
    return sorted(badges, key=lambda item: item["name"].lower())


def _summarize_requirement(requirements) -> str:
    text = display_requirement_text(requirements)
    if not text:
        return "Req: -"
    normalized = " ".join(str(text).split())
    if len(normalized) > 40:
        normalized = f"{normalized[:37].rstrip()}..."
    return f"Req: {normalized}"


def _group_badges(badges: list[dict]) -> list[tuple[str, list[dict]]]:
    has_category = any(badge.get("category") for badge in badges)
    sections: dict[str, list[dict]] = {}

    for badge in badges:
        if has_category:
            section = badge.get("category") or "Other"
        else:
            earners_count = badge.get("earners_count")
            if earners_count is None:
                section = "Unranked"
            elif earners_count == 0:
                section = "Unclaimed"
            elif earners_count >= 100:
                section = "Common"
            elif earners_count >= 25:
                section = "Uncommon"
            elif earners_count >= 5:
                section = "Rare"
            else:
                section = "Legendary"
        sections.setdefault(section, []).append(badge)

    if has_category:
        ordered_sections = sorted(sections.items(), key=lambda item: str(item[0]).lower())
    else:
        order_index = {name: idx for idx, name in enumerate(SECTION_ORDER)}
        ordered_sections = sorted(
            sections.items(),
            key=lambda item: (order_index.get(item[0], len(order_index)), str(item[0]).lower()),
        )

    return [(section, sorted(items, key=lambda item: item["name"].lower())) for section, items in ordered_sections]


def _render_badge_card(badge: dict, column, open_key: str) -> None:
    badge_id = badge.get("badge_id", "")
    name = badge.get("name", "Badge")
    icon = badge_icon(badge_id, badge.get("category"))
    requirements_summary = _summarize_requirement(badge.get("requirements"))
    earners_count = badge.get("earners_count")

    with column:
        st.markdown(
            f"""
            <div class="badge-card">
                <div class="badge-card__icon">{icon}</div>
                <div class="badge-card__name" title="{name}">{name}</div>
                <div class="badge-card__req">{requirements_summary}</div>
                <div class="badge-card__meta">{'' if earners_count is None else f'{earners_count} earners'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("View details", key=open_key, use_container_width=True):
            st.session_state["badge_details_open"] = True
            st.session_state["badge_details_badge_id"] = badge_id
            st.rerun()


def get_badge_earners_page(
    df_player_badges,
    df_players,
    badge_id: str,
    offset: int,
    limit: int,
) -> tuple[list[dict], int]:
    if df_player_badges is None or df_player_badges.empty:
        return [], 0
    if "badge_id" not in df_player_badges.columns or "player_id" not in df_player_badges.columns:
        raise ValueError("Player badge data is missing required columns.")
    badge_id = str(badge_id)
    df_filtered = df_player_badges[df_player_badges["badge_id"].astype(str) == badge_id]
    if df_filtered.empty:
        return [], 0
    if "earned_at" in df_filtered.columns:
        df_filtered = df_filtered.sort_values("earned_at", ascending=False)
    df_filtered = df_filtered.drop_duplicates(subset=["player_id"])
    total = len(df_filtered)
    df_page = df_filtered.iloc[int(offset) : int(offset) + int(limit)]

    player_names = {}
    if df_players is not None and not df_players.empty and "id" in df_players.columns:
        names = df_players["name"] if "name" in df_players.columns else None
        if names is not None:
            player_names = dict(zip(df_players["id"], names.astype(str)))

    earners = []
    for row in df_page.itertuples(index=False):
        player_id = getattr(row, "player_id", None)
        name = player_names.get(player_id, f"Player {player_id}")
        earners.append({"player_id": player_id, "name": name})

    return earners, total


def _get_badge_state(badge_id: str, total_known: int | None) -> dict:
    key = f"badge_earners::{badge_id}"
    if key not in st.session_state:
        st.session_state[key] = {
            "earners": [],
            "offset": 0,
            "total": total_known,
            "has_more": total_known is None or total_known > 0,
            "loading": False,
            "error": None,
            "loaded_once": False,
        }
    return st.session_state[key]


def _load_more_earners(state: dict, badge_id: str, df_player_badges, df_players) -> None:
    if state["loading"] or state.get("has_more") is False:
        return
    state["loading"] = True
    try:
        new_earners, total = get_badge_earners_page(
            df_player_badges,
            df_players,
            badge_id,
            state["offset"],
            EARNS_PAGE_SIZE,
        )
        state["earners"].extend(new_earners)
        state["offset"] += len(new_earners)
        state["total"] = total
        state["has_more"] = state["offset"] < total
        state["error"] = None
    except Exception as exc:  # noqa: BLE001 - surface per-badge errors
        state["error"] = str(exc)
    finally:
        state["loading"] = False


def _render_earners_section(badge_id: str, earners_count, df_player_badges, df_players) -> None:
    state = _get_badge_state(badge_id, earners_count)

    expander_label = f"Earners ({earners_count})" if earners_count is not None else "Earners"
    with st.expander(expander_label, expanded=False):
        if earners_count == 0:
            st.caption("No one has earned this badge yet.")
            return

        if state["error"]:
            st.error(f"Couldn’t load earners. {state['error']}")
            if st.button("Retry", key=f"badge_codex_retry_{badge_id}"):
                state["error"] = None
                with st.spinner("Loading earners..."):
                    _load_more_earners(state, badge_id, df_player_badges, df_players)
                state["loaded_once"] = True
                st.rerun()

        if not state["loaded_once"] and state["error"] is None:
            if st.button("Load earners", key=f"badge_codex_load_{badge_id}"):
                with st.spinner("Loading earners..."):
                    _load_more_earners(state, badge_id, df_player_badges, df_players)
                state["loaded_once"] = True
                st.rerun()
            return

        if state["loading"]:
            st.info("Loading earners...")

        if state["earners"]:
            st.caption(f"Earned by {state['total']} players" if state.get("total") is not None else "Earners")
            for earner in state["earners"]:
                st.markdown(f"- 👤 {earner['name']}")

        if state.get("total") == 0 and not state["loading"] and state["error"] is None:
            st.caption("No one has earned this badge yet.")

        if state.get("has_more") and not state["loading"]:
            if st.button("Load more", key=f"badge_codex_load_more_{badge_id}"):
                with st.spinner("Loading earners..."):
                    _load_more_earners(state, badge_id, df_player_badges, df_players)
                state["loaded_once"] = True
                st.rerun()


def _render_badge_details(selected_badge: dict, df_player_badges, df_players) -> None:
    badge_id = selected_badge.get("badge_id", "")
    name = selected_badge.get("name", "Badge")
    icon = badge_icon(badge_id, selected_badge.get("category"))
    requirements = display_requirement_text(selected_badge.get("requirements"))
    earners_count = selected_badge.get("earners_count")

    st.markdown(f"### {icon} {name}")
    st.markdown(requirements)
    st.caption(f"{earners_count} earners" if earners_count is not None else "Earners")
    _render_earners_section(badge_id, earners_count, df_player_badges, df_players)

    if st.button("Close", key=f"badge_codex_close_{badge_id}"):
        st.session_state["badge_details_open"] = False
        st.session_state["badge_details_badge_id"] = None
        st.rerun()


@st.dialog("Badge details")
def _render_badge_details_dialog(selected_badge: dict, df_player_badges, df_players) -> None:
    _render_badge_details(selected_badge, df_player_badges, df_players)


def render(ctx) -> None:
    st.header("Badge Codex")
    st.caption("A full ledger of badges, with reels for the ones already on tape.")

    if "badge_details_open" not in st.session_state:
        st.session_state["badge_details_open"] = False
    if "badge_details_badge_id" not in st.session_state:
        st.session_state["badge_details_badge_id"] = None

    st.markdown(
        """
        <style>
            .badge-card {
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 12px;
                min-height: 170px;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                background: var(--panel);
            }
            .badge-card__icon {
                font-size: 32px;
                line-height: 1;
            }
            .badge-card__name {
                font-weight: 600;
                font-size: 0.95rem;
                line-height: 1.2;
                display: -webkit-box;
                -webkit-line-clamp: 2;
                -webkit-box-orient: vertical;
                overflow: hidden;
                text-overflow: ellipsis;
                min-height: 2.4em;
            }
            .badge-card__req {
                font-size: 0.8rem;
                color: var(--text-muted);
                white-space: nowrap;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .badge-card__meta {
                font-size: 0.75rem;
                color: var(--text-muted);
                min-height: 1.2em;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    df_players = getattr(ctx, "df_players_all", None)

    badge_defs = getattr(ctx, "df_badges", None)
    player_badges = getattr(ctx, "df_player_badges", None)
    if badge_defs is None:
        st.info("Badge data is still loading.")
        return

    badges = get_all_badges(badge_defs, player_badges)
    if not badges:
        st.caption("No badges are available yet.")
        return

    sections = _group_badges(badges)

    selected_badge_id = st.session_state.get("badge_details_badge_id")
    details_open = st.session_state.get("badge_details_open")
    selected_badge = None
    if selected_badge_id:
        selected_badge = next((badge for badge in badges if badge.get("badge_id") == selected_badge_id), None)

    for section_name, items in sections:
        st.subheader(section_name)
        columns = st.columns(3)
        for idx, badge in enumerate(items):
            _render_badge_card(
                badge,
                columns[idx % 3],
                open_key=f"badge_codex_open_{badge.get('badge_id', '')}",
            )
        st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    if details_open and selected_badge:
        _render_badge_details_dialog(selected_badge, player_badges, df_players)
