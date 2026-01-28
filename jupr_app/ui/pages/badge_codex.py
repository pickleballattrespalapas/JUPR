from __future__ import annotations

import html
import logging

import pandas as pd
import streamlit as st

from jupr_app.domain.gamification.profile import build_gamification_summary
from jupr_app.ui.pages.players import badge_icon

logger = logging.getLogger(__name__)


def _hint_text(value: object | None) -> str:
    text = str(value or "").strip()
    return text if text else "No hint yet."


def _lore_text(value: object | None) -> str:
    text = str(value or "").strip()
    return text if text else "No story yet."


def _player_options(df_players: pd.DataFrame) -> list[tuple[str, int]]:
    if df_players is None or df_players.empty:
        return []
    if "id" not in df_players.columns or "name" not in df_players.columns:
        return []
    options = []
    for row in df_players.itertuples(index=False):
        try:
            pid = int(getattr(row, "id"))
        except Exception:
            continue
        name = str(getattr(row, "name", "") or "").strip() or f"Player {pid}"
        options.append((name, pid))
    return sorted(options, key=lambda x: x[0].lower())


def render(ctx) -> None:
    st.header("Badge Codex")
    st.caption("A full ledger of badges, with reels for the ones already on tape.")

    df_players = getattr(ctx, "df_players_active", None)
    if df_players is None or df_players.empty:
        df_players = getattr(ctx, "df_players_all", None)

    options = _player_options(df_players)
    if not options:
        st.info("No player list available.")
        return

    names = [name for name, _ in options]
    selected_name = st.selectbox("Player", names, index=0)
    player_id = next(pid for name, pid in options if name == selected_name)

    badge_defs = getattr(ctx, "df_badges", None)
    player_badges = getattr(ctx, "df_player_badges", None)
    if badge_defs is None or player_badges is None:
        st.info("Badge data is still loading.")
        return

    summary = build_gamification_summary(player_id, badge_defs, player_badges)
    unlocked_badges = summary.get("unlocked_badges", [])
    locked_badges = summary.get("locked_badges", [])

    stat_cols = st.columns(2)
    stat_cols[0].metric("Prestige", int(summary.get("prestige_total", 0)))
    stat_cols[1].metric(
        "Collection",
        f"{summary.get('collected_unique_count', 0)}/{summary.get('total_active_badge_types', 0)}",
    )

    all_badges = []
    for badge in unlocked_badges:
        badge_copy = dict(badge)
        badge_copy["status"] = "unlocked"
        all_badges.append(badge_copy)
    for badge in locked_badges:
        badge_copy = dict(badge)
        badge_copy["status"] = "locked"
        all_badges.append(badge_copy)

    categories = sorted({b.get("category") or "Other" for b in all_badges})
    rarities = sorted({b.get("rarity") or "common" for b in all_badges})

    with st.expander("Filters", expanded=False):
        selected_categories = st.multiselect("Category", categories, default=categories)
        selected_rarities = st.multiselect("Rarity", rarities, default=rarities)
        show_unlocked = st.checkbox("Show unlocked", value=True)
        show_locked = st.checkbox("Show locked", value=True)
        show_stackable = st.checkbox("Only stackable", value=False)

    def _visible(badge: dict) -> bool:
        category = badge.get("category") or "Other"
        rarity = badge.get("rarity") or "common"
        if category not in selected_categories or rarity not in selected_rarities:
            return False
        if badge.get("status") == "unlocked" and not show_unlocked:
            return False
        if badge.get("status") == "locked" and not show_locked:
            return False
        if show_stackable and not bool(badge.get("stack_count", 0) > 1 or badge.get("is_stackable")):
            return False
        return True

    visible_badges = [b for b in all_badges if _visible(b)]
    if not visible_badges:
        st.caption("No badges match the filters.")
        return

    grid_cols = st.columns(4)
    for idx, badge in enumerate(visible_badges):
        with grid_cols[idx % len(grid_cols)]:
            status = badge.get("status")
            name = badge.get("name", "Badge")
            prestige = int(badge.get("prestige", 0) or 0)
            if status == "locked":
                icon = "🔒"
            else:
                icon = badge_icon(badge.get("badge_id"), badge.get("category"))
            st.markdown(f"**{icon} {name}**")
            st.caption(f"Prestige {prestige}")
            lore = html.escape(_lore_text(badge.get("lore")))
            if lore:
                st.caption(lore)
            if status == "locked":
                hint = html.escape(_hint_text(badge.get("hint")))
                st.caption(hint)
