from __future__ import annotations

import pandas as pd
import streamlit as st

from jupr_court_board import court_board


def sort_items(items: list[str], key: str = "sortable") -> list[str]:
    """Sortable list with drag-and-drop and an editor fallback."""
    cards = [
        {
            "player_id": str(idx),
            "name": str(item),
        }
        for idx, item in enumerate(items)
    ]
    payload = [{"court_id": "Court 1", "players": cards}]
    try:
        dragged = court_board(payload, key=f"{key}_dnd")
    except Exception:
        dragged = None

    if isinstance(dragged, dict):
        dragged_courts = dragged.get("courts")
        if isinstance(dragged_courts, list):
            by_id = {str(idx): item for idx, item in enumerate(items)}
            new_order: list[str] = []
            seen: set[str] = set()
            for court in dragged_courts:
                if not isinstance(court, dict):
                    continue
                for player in court.get("players", []):
                    if not isinstance(player, dict):
                        continue
                    pid = str(player.get("player_id", ""))
                    if pid in by_id and pid not in seen:
                        new_order.append(by_id[pid])
                        seen.add(pid)
            if len(new_order) == len(items):
                return new_order

    st.caption("Drag-and-drop unavailable here; use Rank to reorder.")
    rows = [{"Rank": idx + 1, "Player": item} for idx, item in enumerate(items)]
    df = pd.DataFrame(rows)
    edited = st.data_editor(
        df,
        key=f"{key}_editor",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", min_value=1, step=1),
            "Player": st.column_config.TextColumn("Player", disabled=True),
        },
    )
    if edited is None or edited.empty:
        return items
    ordered = edited.sort_values(["Rank", "Player"], ascending=[True, True])["Player"].astype(str).tolist()
    if len(ordered) != len(items):
        return items
    return ordered
