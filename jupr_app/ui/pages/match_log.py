import streamlit as st
import pandas as pd

from jupr_app.domain.dupes import canonical_dup_key
from jupr_app.domain.match_admin import preview_week_tag_update
from jupr_app.ui.layout import page_shell

def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("📝 Match Log", "Review and filter recorded matches.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    df_matches = getattr(ctx, "df_matches", None)
    if df_matches is None or df_matches.empty:
        st.info("No matches loaded.")
        st.stop()

    df = df_matches.copy()

    # Basic filters
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        filter_type = st.radio("Filter", ["All", "League", "Pop-Up"], horizontal=True)
    with c2:
        id_filter = st.number_input("Jump to Match ID", min_value=0, value=0, step=1)
    with c3:
        limit_rows = st.number_input("Display limit", min_value=100, max_value=5000, value=500, step=100)

    if filter_type == "League":
        df = df[df.get("match_type", "") != "PopUp"].copy()
    elif filter_type == "Pop-Up":
        df = df[df.get("match_type", "") == "PopUp"].copy()

    if id_filter > 0 and "id" in df.columns:
        df = df[df["id"].astype(int) == int(id_filter)].copy()

    df = df.head(int(limit_rows)).copy()

    st.divider()

    # Duplicate scanner
    st.subheader("🔎 Find Duplicate Matches")
    st.caption("Detects duplicates even if teammates or teams are swapped (scores normalized too).")

    if df.empty:
        st.info("No rows to scan.")
    else:
        df["dup_key"] = [canonical_dup_key(r, str(ctx.club_id)) for _, r in df.iterrows()]

        counts = df["dup_key"].value_counts()
        dup_keys = counts[counts > 1].index.tolist()

        if not dup_keys:
            st.success("✅ No duplicates found in the current view/filter.")
        else:
            st.error(f"⚠️ Found {len(dup_keys)} duplicate groups.")

            dup_only = df[df["dup_key"].isin(dup_keys)].copy()
            dup_only = dup_only.sort_values(["dup_key", "id"], ascending=[True, True])
            dup_only["dup_rank"] = dup_only.groupby("dup_key").cumcount() + 1
            dup_only["dup_count"] = dup_only.groupby("dup_key")["id"].transform("count")

            summary = (
                dup_only.groupby("dup_key")
                .agg(
                    dup_count=("id", "count"),
                    keep_id=("id", "min"),
                    delete_ids=("id", lambda x: ", ".join(map(str, sorted(x.tolist())[1:]))),
                    league=("league", "first"),
                    week_tag=("week_tag", "first"),
                    match_type=("match_type", "first"),
                )
                .reset_index(drop=True)
            )

            st.write("### Duplicate Groups (keep oldest, delete rest)")
            st.dataframe(summary, use_container_width=True, hide_index=True)

            st.write("### Duplicate Rows (detailed)")
            show_cols = [c for c in [
                "id", "date", "league", "week_tag", "match_type",
                "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2",
                "dup_rank", "dup_count"
            ] if c in dup_only.columns]
            st.dataframe(dup_only[show_cols], use_container_width=True, hide_index=True)

            delete_mode = st.radio(
                "Delete mode",
                ["Delete duplicates (keep oldest in each group)", "I’ll delete manually"],
                horizontal=True,
            )

            if delete_mode == "Delete duplicates (keep oldest in each group)":
                ids_to_delete = dup_only[dup_only["dup_rank"] > 1]["id"].astype(int).tolist()
                st.warning(
                    f"Ready to delete {len(ids_to_delete)} duplicated match rows "
                    f"(keeping the oldest copy per group)."
                )

                confirm = st.text_input("Type DELETE to confirm", value="", key="dup_delete_confirm")
                if st.button("🗑️ Delete duplicates now", type="primary", disabled=(confirm.strip().upper() != "DELETE")):
                    if ids_to_delete:
                        ctx.supabase.table("matches").delete().eq("club_id", str(ctx.club_id)).in_("id", ids_to_delete).execute()
                        st.success("Deleted duplicates. Now run Admin Tools → Replay History → ALL.")
                        st.rerun()

    st.divider()

    # Bulk edit week tag UI
    st.subheader("✏️ Bulk Edit Week Tag")
    st.caption("Filter league matches, select rows, and update week tags in bulk.")

    df_bulk = df_matches.copy()
    if "match_type" in df_bulk.columns:
        df_bulk["match_type"] = df_bulk["match_type"].fillna("").astype(str).str.strip()

    league_options = ["All"]
    if "league" in df_bulk.columns:
        league_options += sorted(
            df_bulk["league"].fillna("").astype(str).str.strip().replace("", "Unspecified").unique().tolist()
        )

    match_type_options = ["All"]
    if "match_type" in df_bulk.columns:
        match_type_options += sorted(
            df_bulk["match_type"].fillna("").astype(str).str.strip().replace("", "Unspecified").unique().tolist()
        )

    week_tag_options = ["All"]
    if "week_tag" in df_bulk.columns:
        week_tag_options += sorted(
            df_bulk["week_tag"].fillna("").astype(str).str.strip().replace("", "Unspecified").unique().tolist()
        )

    df_bulk["date_dt"] = pd.to_datetime(df_bulk.get("date", None), errors="coerce", utc=True)
    date_min = df_bulk["date_dt"].min()
    date_max = df_bulk["date_dt"].max()

    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
    with f1:
        bulk_league = st.selectbox("League", league_options, index=0, key="bulk_week_league")
    with f2:
        bulk_match_type = st.selectbox("Match type", match_type_options, index=0, key="bulk_week_match_type")
    with f3:
        bulk_week_tag = st.selectbox("Current week_tag", week_tag_options, index=0, key="bulk_week_tag")
    with f4:
        if pd.notna(date_min) and pd.notna(date_max):
            default_start = (date_max - pd.Timedelta(days=7)).date()
            default_end = date_max.date()
            date_range = st.date_input(
                "Date range",
                value=(default_start, default_end),
                key="bulk_week_date_range",
            )
        else:
            date_range = st.date_input("Date range", value=(), key="bulk_week_date_range")

    if bulk_league != "All" and "league" in df_bulk.columns:
        bulk_league_value = "" if bulk_league == "Unspecified" else bulk_league
        df_bulk = df_bulk[df_bulk["league"].fillna("").astype(str).str.strip() == bulk_league_value].copy()

    if bulk_match_type != "All" and "match_type" in df_bulk.columns:
        bulk_match_type_value = "" if bulk_match_type == "Unspecified" else bulk_match_type
        df_bulk = df_bulk[df_bulk["match_type"].fillna("").astype(str).str.strip() == bulk_match_type_value].copy()

    if bulk_week_tag != "All" and "week_tag" in df_bulk.columns:
        bulk_week_value = "" if bulk_week_tag == "Unspecified" else bulk_week_tag
        df_bulk = df_bulk[df_bulk["week_tag"].fillna("").astype(str).str.strip() == bulk_week_value].copy()

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            df_bulk = df_bulk[
                (df_bulk["date_dt"].dt.date >= start_date)
                & (df_bulk["date_dt"].dt.date <= end_date)
            ].copy()

    st.caption(f"{len(df_bulk)} match(es) match the filters.")

    bulk_cols = [c for c in [
        "id", "date", "league", "week_tag", "match_type",
        "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2"
    ] if c in df_bulk.columns]

    bulk_view = df_bulk[bulk_cols].copy()
    bulk_view.insert(0, "Select", False)

    current_filters = (
        bulk_league,
        bulk_match_type,
        bulk_week_tag,
        tuple(date_range) if isinstance(date_range, tuple) else date_range,
    )
    if "bulk_week_editor_version" not in st.session_state:
        st.session_state["bulk_week_editor_version"] = 0
    if st.session_state.get("bulk_week_filters") != current_filters:
        st.session_state["bulk_week_filters"] = current_filters
        st.session_state["bulk_week_df"] = bulk_view.copy()
        st.session_state["bulk_week_editor_version"] += 1
    if "bulk_week_df" not in st.session_state:
        st.session_state["bulk_week_df"] = bulk_view.copy()

    edited_bulk = st.data_editor(
        st.session_state["bulk_week_df"],
        column_config={"Select": st.column_config.CheckboxColumn(default=False)},
        hide_index=True,
        use_container_width=True,
        key=f"bulk_week_editor_{st.session_state['bulk_week_editor_version']}",
    )
    st.session_state["bulk_week_df"] = edited_bulk.copy()

    if st.button("Clear selection", key="bulk_week_clear"):
        st.session_state["bulk_week_df"] = bulk_view.copy()
        st.session_state["bulk_week_editor_version"] += 1
        st.rerun()

    selected_bulk = edited_bulk[edited_bulk["Select"] == True].copy()
    selected_ids = selected_bulk["id"].astype(int).tolist() if not selected_bulk.empty else []

    st.markdown("**New week_tag**")
    new_tag_options = [f"Week {i}" for i in range(1, 21)] + ["Custom..."]
    new_tag_choice = st.selectbox("Select new week_tag", new_tag_options, key="bulk_week_new_tag")
    if new_tag_choice == "Custom...":
        new_week_tag = st.text_input("Custom week_tag", value="", key="bulk_week_custom_tag").strip()
    else:
        new_week_tag = new_tag_choice

    preview = preview_week_tag_update(df_bulk, selected_ids, new_week_tag)
    old_tags = preview.get("old_tags", [])
    old_tags_text = ", ".join(old_tags) if old_tags else "(none)"
    st.info(
        f"You are about to update {preview.get('count', 0)} match(es) "
        f"from {old_tags_text} to {preview.get('new_tag', '') or '(blank)'}."
    )

    confirm_update = st.checkbox(
        "Confirm bulk update",
        value=False,
        help="Updates week_tag for the selected matches.",
        key="bulk_week_confirm",
    )

    update_disabled = not selected_ids or not new_week_tag or not confirm_update
    if st.button("Update week_tag for selected", type="primary", disabled=update_disabled):
        if not selected_ids:
            st.warning("Select at least one match to update.")
        elif not new_week_tag:
            st.warning("Enter a new week_tag before updating.")
        else:
            try:
                ctx.supabase.table("matches").update({"week_tag": new_week_tag}).eq(
                    "club_id", str(ctx.club_id)
                ).in_("id", selected_ids).execute()
                st.success(f"Updated {len(selected_ids)} match(es).")
                st.session_state["bulk_week_df"] = bulk_view.copy()
                st.session_state["bulk_week_editor_version"] += 1
                st.rerun()
            except Exception as exc:
                st.error(f"Unable to update week_tag: {exc}")

    st.divider()

    # Bulk delete UI
    st.subheader("🗑️ Bulk Delete (first N rows shown)")
    edit_cols = [c for c in [
        "id", "date", "league", "week_tag", "match_type",
        "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2"
    ] if c in df.columns]

    view = df[edit_cols].copy()
    view.insert(0, "Delete", False)

    edited = st.data_editor(
        view,
        column_config={"Delete": st.column_config.CheckboxColumn(default=False)},
        hide_index=True,
        use_container_width=True,
    )

    to_delete = edited[edited["Delete"] == True]
    if not to_delete.empty:
        st.warning(f"Ready to delete {len(to_delete)} match(es).")
        confirm2 = st.text_input("Type DELETE to confirm bulk delete", value="", key="bulk_delete_confirm")
        if st.button(f"Delete {len(to_delete)} Matches", type="primary", disabled=(confirm2.strip().upper() != "DELETE")):
            ctx.supabase.table("matches").delete().eq("club_id", str(ctx.club_id)).in_("id", to_delete["id"].astype(int).tolist()).execute()
            st.success("Deleted. Run Replay ALL if you want ratings to be consistent.")
            st.rerun()
