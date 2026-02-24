from __future__ import annotations

import pandas as pd
import streamlit as st


def sort_items(items: list[str], key: str = "sortable") -> list[str]:
    """Streamlit-native sortable list fallback.

    Users can change the Rank column to reorder items deterministically.
    """
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
