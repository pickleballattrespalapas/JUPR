import streamlit as st
import pandas as pd

from jupr_app.domain.dupes import canonical_dup_key
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
