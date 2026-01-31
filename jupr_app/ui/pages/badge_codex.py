from __future__ import annotations

import logging

import streamlit as st

from jupr_app.ui.helpers import display_requirement_text
from jupr_app.ui.pages.players import badge_icon

logger = logging.getLogger(__name__)

EARNS_PAGE_SIZE = 25

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
    key = f"badge_codex_state_{badge_id}"
    if key not in st.session_state:
        st.session_state[key] = {
            "earners": [],
            "offset": 0,
            "total": total_known,
            "has_more": total_known is None or total_known > 0,
            "loading": False,
            "error": None,
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


def render(ctx) -> None:
    st.header("Badge Codex")
    st.caption("A full ledger of badges, with reels for the ones already on tape.")

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

    for badge in badges:
        badge_id = badge.get("badge_id", "")
        name = badge.get("name", "Badge")
        icon = badge_icon(badge_id, badge.get("category"))
        requirements = display_requirement_text(badge.get("requirements"))
        earners_count = badge.get("earners_count")

        with st.container():
            st.markdown(f"**{icon} {name}**")
            st.markdown(requirements)
            if earners_count is not None:
                st.caption(f"{earners_count} earners")

            expanded = st.toggle("Show earners", key=f"badge_codex_toggle_{badge_id}")
            if expanded:
                state = _get_badge_state(badge_id, earners_count)
                if earners_count == 0:
                    st.caption("No one has earned this badge yet.")

                if (
                    earners_count != 0
                    and not state["earners"]
                    and not state["loading"]
                    and state["error"] is None
                ):
                    _load_more_earners(state, badge_id, player_badges, df_players)

                if state["error"]:
                    st.error(f"Could not load earners. {state['error']}")
                    if st.button("Retry", key=f"badge_codex_retry_{badge_id}"):
                        state["error"] = None
                        _load_more_earners(state, badge_id, player_badges, df_players)

                if state["loading"]:
                    st.info("Loading earners...")

                if state["earners"]:
                    st.caption(
                        f"Earned by {state['total']} players" if state.get("total") is not None else "Earners"
                    )
                    for earner in state["earners"]:
                        st.markdown(f"- 👤 {earner['name']}")

                if earners_count != 0 and state.get("total") == 0 and not state["loading"] and state["error"] is None:
                    st.caption("No one has earned this badge yet.")

                if state.get("has_more") and not state["loading"]:
                    if st.button("Load more", key=f"badge_codex_load_more_{badge_id}"):
                        _load_more_earners(state, badge_id, player_badges, df_players)

            st.divider()
