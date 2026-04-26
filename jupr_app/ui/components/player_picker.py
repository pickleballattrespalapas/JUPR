from __future__ import annotations

import pandas as pd
import streamlit as st


MAX_MATCH_OPTIONS = 50


def _base_label_for_row(row: pd.Series) -> str:
    display_name = str(row.get("display_name") or "").strip()
    name = str(row.get("name") or "").strip()
    return display_name or name or f"Player #{int(row['id'])}"


def build_player_picker_df(players_df: pd.DataFrame, *, include_inactive: bool = True) -> pd.DataFrame:
    if players_df is None or players_df.empty or "id" not in players_df.columns:
        return pd.DataFrame(columns=["id", "display_label", "option_label", "search_text", "sort_label"])

    working = players_df.copy()
    working = working[working["id"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["id", "display_label", "option_label", "search_text", "sort_label"])

    working["id"] = pd.to_numeric(working["id"], errors="coerce")
    working = working[working["id"].notna()].copy()
    if working.empty:
        return pd.DataFrame(columns=["id", "display_label", "option_label", "search_text", "sort_label"])
    working["id"] = working["id"].astype(int)

    if not include_inactive:
        if "inactive_at" in working.columns:
            working = working[working["inactive_at"].isna()].copy()
        elif "active" in working.columns:
            working = working[working["active"] == True].copy()

    if working.empty:
        return pd.DataFrame(columns=["id", "display_label", "option_label", "search_text", "sort_label"])

    working["display_label"] = working.apply(_base_label_for_row, axis=1)
    working["option_label"] = working["display_label"] + working["id"].map(lambda pid: f" (#{int(pid)})")
    working["search_text"] = working["display_label"].astype(str).str.lower()
    working["sort_label"] = working["display_label"].astype(str).str.lower()

    deduped = (
        working.sort_values(["sort_label", "display_label", "id"])
        .drop_duplicates(subset=["id"], keep="first")
        .reset_index(drop=True)
    )
    return deduped[["id", "display_label", "option_label", "search_text", "sort_label"]]


def filter_player_picker_df(options_df: pd.DataFrame, query: str) -> pd.DataFrame:
    if options_df is None or options_df.empty:
        return pd.DataFrame(columns=getattr(options_df, "columns", []))

    clean_query = str(query or "").strip().lower()
    if not clean_query:
        return options_df.sort_values(["sort_label", "id"]).reset_index(drop=True)

    tokens = [tok for tok in clean_query.split() if tok]
    if not tokens:
        return options_df.sort_values(["sort_label", "id"]).reset_index(drop=True)

    mask = pd.Series([True] * len(options_df), index=options_df.index)
    haystack = options_df["search_text"].fillna("").astype(str).str.lower()
    for token in tokens:
        mask = mask & haystack.str.contains(token, regex=False)
    return options_df[mask].sort_values(["sort_label", "id"]).reset_index(drop=True)


def render_player_picker(
    players_df,
    *,
    label: str = "Player",
    key: str,
    default_player_id: int | None = None,
    include_inactive: bool = True,
    help: str | None = None,
    placeholder: str = "Search player by name…",
) -> int | None:
    options_df = build_player_picker_df(players_df, include_inactive=include_inactive)
    if options_df.empty:
        st.info("No players found.")
        return None

    search_key = f"{key}__search"
    select_key = f"{key}__selected"

    if search_key not in st.session_state:
        st.session_state[search_key] = ""

    valid_ids = set(options_df["id"].astype(int).tolist())
    if default_player_id is not None and int(default_player_id) in valid_ids and select_key not in st.session_state:
        st.session_state[select_key] = int(default_player_id)

    st.caption("Start typing a first or last name, then choose the player.")
    search_cols = st.columns([4, 1])
    with search_cols[0]:
        st.text_input(label, key=search_key, placeholder=placeholder, help=help)
    with search_cols[1]:
        st.write("")
        if st.button("Clear Search", key=f"{key}__clear"):
            st.session_state[search_key] = ""
            st.rerun()

    query = str(st.session_state.get(search_key, "") or "")
    filtered = filter_player_picker_df(options_df, query)
    if filtered.empty:
        st.info("No players match that search yet. Try a different first or last name.")
        return None

    option_ids = filtered["id"].astype(int).tolist()
    if len(option_ids) > MAX_MATCH_OPTIONS:
        option_ids = option_ids[:MAX_MATCH_OPTIONS]
        st.caption(f"Showing first {MAX_MATCH_OPTIONS} matches. Keep typing to narrow the list.")

    selected_id = st.session_state.get(select_key)
    if selected_id not in option_ids:
        if default_player_id is not None and int(default_player_id) in option_ids:
            selected_id = int(default_player_id)
        else:
            selected_id = option_ids[0]
        st.session_state[select_key] = selected_id

    selected = st.selectbox(
        "Matching players",
        options=option_ids,
        index=option_ids.index(int(selected_id)) if int(selected_id) in option_ids else 0,
        key=select_key,
        format_func=lambda pid: str(filtered.loc[filtered["id"] == int(pid), "option_label"].iloc[0]),
    )
    return int(selected) if selected is not None else None
