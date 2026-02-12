from __future__ import annotations

import pandas as pd
import streamlit as st

from jupr_app.ui.helpers import sanitize_story_text


def render_story_cards(story_df: pd.DataFrame) -> None:
    if story_df is None or story_df.empty:
        st.caption("No new stories in the tape room yet.")
        return

    highlights = story_df[story_df["story_type"].astype(str).str.startswith("highlight", na=False)].head(3)
    foreshadow = story_df[story_df["story_type"].astype(str).str.startswith("foreshadow", na=False)].head(3)

    highlight_col, foreshadow_col = st.columns(2)

    with highlight_col:
        st.markdown("**Highlights**")
        if highlights.empty:
            st.caption("No highlights yet.")
        else:
            for _, row in highlights.iterrows():
                title = sanitize_story_text(row.get("title")) or "Highlight"
                body = sanitize_story_text(row.get("body"))
                st.markdown(f"**{title}**")
                st.caption(body)

    with foreshadow_col:
        st.markdown("**Foreshadowing**")
        if foreshadow.empty:
            st.caption("No foreshadowing yet.")
        else:
            for _, row in foreshadow.iterrows():
                title = sanitize_story_text(row.get("title")) or "Foreshadowing"
                body = sanitize_story_text(row.get("body"))
                st.markdown(f"**{title}**")
                st.caption(body)
