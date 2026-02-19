import streamlit as st
import pandas as pd

# Match writes must go through match_pipeline.
from jupr_app.domain.dupes import canonical_dup_key
from jupr_app.domain.bulk_match_editor import apply_bulk_match_edits, compute_recompute_scope
from jupr_app.domain.player_activity import recompute_last_game_at_for_players
from jupr_app.ui.layout import page_shell
from jupr_app.domain.match_pipeline import delete_matches



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
                        deleted_rows = dup_only[dup_only["id"].astype(int).isin(ids_to_delete)]
                        deleted_pids = set()
                        for col in ["t1_p1", "t1_p2", "t2_p1", "t2_p2"]:
                            if col in deleted_rows.columns:
                                deleted_pids.update(deleted_rows[col].dropna().astype(int).tolist())
                        delete_matches(supabase=ctx.supabase, club_id=str(ctx.club_id), match_ids=ids_to_delete)
                        if deleted_pids:
                            recompute_last_game_at_for_players(
                                supabase=ctx.supabase,
                                club_id=str(ctx.club_id),
                                player_ids=deleted_pids,
                            )
                        st.success("Deleted duplicates. Now run Admin Tools → Replay History → ALL.")
                        st.rerun()

    st.divider()

    # Bulk match editor UI
    st.subheader("✏️ Bulk Match Editor")
    st.caption("Filter matches, select rows, edit league/date/week_tag/match_type/notes/is_active, preview impact, then apply safely.")

    df_bulk = df_matches.copy()

    # Normalize common string columns
    for col in ("league", "match_type", "week_tag"):
        if col in df_bulk.columns:
            df_bulk[col] = df_bulk[col].fillna("").astype(str).str.strip()

    # Build filter options from what's currently loaded
    league_options = ["All"]
    if "league" in df_bulk.columns:
        league_options += sorted(df_bulk["league"].replace("", "Unspecified").unique().tolist())

    match_type_options = ["All"]
    if "match_type" in df_bulk.columns:
        match_type_options += sorted(df_bulk["match_type"].replace("", "Unspecified").unique().tolist())

    week_tag_options = ["All"]
    if "week_tag" in df_bulk.columns:
        week_tag_options += sorted(df_bulk["week_tag"].replace("", "Unspecified").unique().tolist())

    # Date helpers
    df_bulk["date_dt"] = pd.to_datetime(df_bulk.get("date", None), errors="coerce", utc=True)
    date_min = df_bulk["date_dt"].min()
    date_max = df_bulk["date_dt"].max()

    f1, f2, f3, f4 = st.columns([2, 2, 2, 2])
    with f1:
        bulk_league = st.selectbox("League", league_options, index=0, key="bulk_match_league")
    with f2:
        bulk_match_type = st.selectbox("Match type", match_type_options, index=0, key="bulk_match_type")
    with f3:
        bulk_week_tag = st.selectbox("Current week_tag", week_tag_options, index=0, key="bulk_match_week_tag")
    with f4:
        if pd.notna(date_min) and pd.notna(date_max):
            default_start = (date_max - pd.Timedelta(days=7)).date()
            default_end = date_max.date()
            date_range = st.date_input(
                "Date range",
                value=(default_start, default_end),
                key="bulk_match_date_range",
            )
        else:
            date_range = st.date_input("Date range", value=(), key="bulk_match_date_range")

    # Apply filters
    if bulk_league != "All" and "league" in df_bulk.columns:
        v = "" if bulk_league == "Unspecified" else bulk_league
        df_bulk = df_bulk[df_bulk["league"] == v].copy()

    if bulk_match_type != "All" and "match_type" in df_bulk.columns:
        v = "" if bulk_match_type == "Unspecified" else bulk_match_type
        df_bulk = df_bulk[df_bulk["match_type"] == v].copy()

    if bulk_week_tag != "All" and "week_tag" in df_bulk.columns:
        v = "" if bulk_week_tag == "Unspecified" else bulk_week_tag
        df_bulk = df_bulk[df_bulk["week_tag"] == v].copy()

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            df_bulk = df_bulk[
                (df_bulk["date_dt"].dt.date >= start_date)
                & (df_bulk["date_dt"].dt.date <= end_date)
            ].copy()

    st.caption(f"{len(df_bulk)} match(es) match the filters.")

    # Build editable view
    base_cols = [c for c in ["id", "date_dt", "league", "week_tag", "match_type"] if c in df_bulk.columns]
    view = df_bulk[base_cols].copy()

    # Rename date_dt -> date for editor friendliness
    if "date_dt" in view.columns:
        view = view.rename(columns={"date_dt": "date"})

    # Optional editable columns: notes, is_active (may or may not exist in df_matches)
    # We'll allow them in the editor if present; otherwise they are simply not shown.
    extra_cols = [c for c in ["notes", "is_active"] if c in df_bulk.columns]
    for c in extra_cols:
        view[c] = df_bulk[c].copy()

    view.insert(0, "Select", False)

    current_filters = (
        bulk_league,
        bulk_match_type,
        bulk_week_tag,
        tuple(date_range) if isinstance(date_range, tuple) else date_range,
    )

    if "bulk_match_editor_version" not in st.session_state:
        st.session_state["bulk_match_editor_version"] = 0

    if st.session_state.get("bulk_match_filters") != current_filters:
        st.session_state["bulk_match_filters"] = current_filters
        st.session_state["bulk_match_baseline"] = view.copy(deep=True)
        st.session_state["bulk_match_df"] = view.copy(deep=True)
        st.session_state["bulk_match_editor_version"] += 1

    if "bulk_match_df" not in st.session_state:
        st.session_state["bulk_match_baseline"] = view.copy(deep=True)
        st.session_state["bulk_match_df"] = view.copy(deep=True)

    # Bulk apply controls
    st.markdown("**Bulk apply to selected rows**")
    b1, b2, b3 = st.columns([2, 2, 2])

    with b1:
        new_league = st.selectbox("Set league", ["(no change)"] + league_options[1:], key="bulk_match_new_league")
        new_week_mode = st.selectbox("week_tag action", ["(no change)", "Set", "Clear"], key="bulk_match_week_mode")
        new_week_tag = ""
        if new_week_mode == "Set":
            week_choices = [f"Week {i}" for i in range(1, 21)] + ["Tournament", "Custom..."]
            pick = st.selectbox("New week_tag", week_choices, key="bulk_match_new_week_pick")
            new_week_tag = st.text_input("Custom week_tag", value="", key="bulk_match_new_week_custom").strip() if pick == "Custom..." else pick

    with b2:
        new_match_type = st.selectbox("Set match_type", ["(no change)"] + match_type_options[1:], key="bulk_match_new_type")
        is_active_mode = "(no change)"
        if "is_active" in st.session_state["bulk_match_df"].columns:
            is_active_mode = st.selectbox("Set is_active", ["(no change)", "true", "false"], key="bulk_match_new_active")

    with b3:
        date_mode = st.selectbox("Date edit", ["(no change)", "Set date", "Shift days"], key="bulk_match_date_mode")
        set_date = st.date_input("New date", value=(date_max.date() if pd.notna(date_max) else pd.Timestamp.utcnow().date()),
                                 key="bulk_match_set_date", disabled=(date_mode != "Set date"))
        shift_days = st.number_input("Shift days (+/-)", value=0, step=1, key="bulk_match_shift_days",
                                     disabled=(date_mode != "Shift days"))

    new_notes = ""
    if "notes" in st.session_state["bulk_match_df"].columns:
        new_notes = st.text_area("Set notes (replaces existing; blank = no change)", value="", key="bulk_match_new_notes")

    if st.button("Stage bulk changes", key="bulk_match_stage"):
        df_edit = st.session_state["bulk_match_df"].copy()
        sel = df_edit["Select"] == True

        if not sel.any():
            st.warning("Select at least one match.")
        else:
            if new_league != "(no change)" and "league" in df_edit.columns:
                df_edit.loc[sel, "league"] = "" if new_league == "Unspecified" else new_league

            if new_week_mode == "Clear" and "week_tag" in df_edit.columns:
                df_edit.loc[sel, "week_tag"] = ""
            elif new_week_mode == "Set" and "week_tag" in df_edit.columns:
                df_edit.loc[sel, "week_tag"] = (new_week_tag or "").strip()

            if new_match_type != "(no change)" and "match_type" in df_edit.columns:
                df_edit.loc[sel, "match_type"] = "" if new_match_type == "Unspecified" else new_match_type

            if "is_active" in df_edit.columns and is_active_mode != "(no change)":
                df_edit.loc[sel, "is_active"] = (is_active_mode == "true")

            if date_mode == "Set date" and "date" in df_edit.columns:
                # set to midnight UTC for that date
                df_edit.loc[sel, "date"] = pd.to_datetime(set_date, utc=True)
            elif date_mode == "Shift days" and "date" in df_edit.columns:
                df_edit.loc[sel, "date"] = pd.to_datetime(df_edit.loc[sel, "date"], utc=True) + pd.to_timedelta(int(shift_days), unit="D")

            if "notes" in df_edit.columns and new_notes.strip() != "":
                df_edit.loc[sel, "notes"] = new_notes

            st.session_state["bulk_match_df"] = df_edit
            st.session_state["bulk_match_editor_version"] += 1
            st.success("Bulk changes staged (not yet saved).")

    # Editable grid (per-row overrides)
    editor_key = f"bulk_match_editor_{st.session_state['bulk_match_editor_version']}"
    edited = st.data_editor(
        st.session_state["bulk_match_df"],
        hide_index=True,
        use_container_width=True,
        key=editor_key,
        column_config={
            "Select": st.column_config.CheckboxColumn(default=False),
            "date": st.column_config.DatetimeColumn("date", help="UTC datetime"),
            "is_active": st.column_config.CheckboxColumn("is_active") if "is_active" in st.session_state["bulk_match_df"].columns else None,
        },
    )
    st.session_state["bulk_match_df"] = edited.copy()

    selected_count = int(edited["Select"].sum()) if "Select" in edited.columns else 0
    total_count = int(len(edited))

    c_select, c_clear, c_summary = st.columns([1, 1, 2])
    with c_select:
        if st.button("Select all", key="bulk_match_select_all", disabled=total_count == 0):
            df_select = st.session_state["bulk_match_df"].copy()
            df_select["Select"] = True
            st.session_state["bulk_match_df"] = df_select
            st.session_state["bulk_match_editor_version"] += 1
            st.rerun()
    with c_clear:
        if st.button("Clear selection", key="bulk_match_clear", disabled=total_count == 0):
            df_clear = st.session_state["bulk_match_df"].copy()
            df_clear["Select"] = False
            st.session_state["bulk_match_df"] = df_clear
            st.session_state["bulk_match_editor_version"] += 1
            st.rerun()
    with c_summary:
        st.caption(f"Selected: {selected_count} / {total_count}")

    # Build patches from baseline vs edited for selected rows
    baseline = st.session_state["bulk_match_baseline"].copy()
    base_by_id = baseline.set_index("id", drop=False)
    cur_by_id = edited.set_index("id", drop=False)

    selected_rows = edited[edited["Select"] == True].copy()
    selected_ids = selected_rows["id"].astype(int).tolist() if not selected_rows.empty else []

    patches = []
    for mid in selected_ids:
        b = base_by_id.loc[mid]
        a = cur_by_id.loc[mid]

        patch = {"id": int(mid)}

        # league
        if "league" in cur_by_id.columns and "league" in base_by_id.columns:
            if str(a.get("league", "")).strip() != str(b.get("league", "")).strip():
                patch["league"] = str(a.get("league", "")).strip()

        # week_tag
        if "week_tag" in cur_by_id.columns and "week_tag" in base_by_id.columns:
            bw = str(b.get("week_tag", "")).strip()
            aw = str(a.get("week_tag", "")).strip()
            if aw != bw:
                patch["week_tag"] = aw  # domain layer will blank->None

        # match_type
        if "match_type" in cur_by_id.columns and "match_type" in base_by_id.columns:
            if str(a.get("match_type", "")).strip() != str(b.get("match_type", "")).strip():
                patch["match_type"] = str(a.get("match_type", "")).strip()

        # notes
        if "notes" in cur_by_id.columns and "notes" in base_by_id.columns:
            bn = "" if pd.isna(b.get("notes")) else str(b.get("notes"))
            an = "" if pd.isna(a.get("notes")) else str(a.get("notes"))
            if an != bn:
                patch["notes"] = an

        # is_active
        if "is_active" in cur_by_id.columns and "is_active" in base_by_id.columns:
            if bool(a.get("is_active")) != bool(b.get("is_active")):
                patch["is_active"] = bool(a.get("is_active"))

        # date
        if "date" in cur_by_id.columns and "date" in base_by_id.columns:
            bd = pd.to_datetime(b.get("date"), errors="coerce", utc=True)
            ad = pd.to_datetime(a.get("date"), errors="coerce", utc=True)
            if pd.notna(ad) and (pd.isna(bd) or ad != bd):
                patch["date"] = ad.isoformat()

        if len(patch.keys()) > 1:
            patches.append(patch)

    scope = compute_recompute_scope(patches)

    # Preview
    st.markdown("**Impact preview**")
    st.info(
        f"Detected {len(patches)} changed match(es) among {len(selected_ids)} selected.\n\n"
        f"- Recompute likely needed: standings={scope['standings']}, ratings={scope['ratings']}\n\n"
        f"Safety rule: if you change league/date and do NOT explicitly set week_tag, week_tag will be auto-cleared."
    )

    # Apply
    st.markdown("**Apply changes**")
    actor = st.text_input("Actor (for audit log)", value="admin", key="bulk_match_actor")
    confirm = st.text_input("Type APPLY to confirm", value="", key="bulk_match_confirm")

    disabled = (confirm.strip().upper() != "APPLY") or (not patches)

    if st.button("Apply staged edits", type="primary", disabled=disabled, key="bulk_match_apply"):
        try:
            with st.spinner("Applying updates..."):
                result = apply_bulk_match_edits(
                    supabase=ctx.supabase,
                    club_id=str(ctx.club_id),
                    patches=patches,
                    actor=actor.strip() or "admin",
                )

            st.success(
                f"Updated {result['updated_count']} match(es). "
                f"Affected leagues: {', '.join(result.get('affected_leagues', [])) or '(unknown)'}"
            )

            if result.get("warnings"):
                st.warning("Warnings:\n- " + "\n- ".join(result["warnings"][:10]))

            if result["recompute_scope"]["ratings"]:
                st.warning("Ratings may be impacted. Run **Admin Tools → Replay History → ALL** (or the affected leagues) to fully re-sync ratings/standings.")
            else:
                st.info("Week/league views may change immediately. If anything looks off, run **Replay History**.")

            # Reset editor state
            st.session_state["bulk_match_df"] = view.copy(deep=True)
            st.session_state["bulk_match_baseline"] = view.copy(deep=True)
            st.session_state["bulk_match_editor_version"] += 1
            st.rerun()

        except Exception as exc:
            st.error(f"Unable to apply bulk edits: {exc}")


    st.info("Replay should only be run from Admin Tools.")

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
            delete_ids = to_delete["id"].astype(int).tolist()
            deleted_pids = set()
            for col in ["t1_p1", "t1_p2", "t2_p1", "t2_p2"]:
                if col in to_delete.columns:
                    deleted_pids.update(to_delete[col].dropna().astype(int).tolist())
            delete_matches(supabase=ctx.supabase, club_id=str(ctx.club_id), match_ids=delete_ids)
            if deleted_pids:
                recompute_last_game_at_for_players(
                    supabase=ctx.supabase,
                    club_id=str(ctx.club_id),
                    player_ids=deleted_pids,
                )
            st.success("Deleted. Run Replay ALL if you want ratings to be consistent.")
            st.rerun()
