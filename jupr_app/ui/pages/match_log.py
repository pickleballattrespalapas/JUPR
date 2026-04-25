import streamlit as st
import pandas as pd

from jupr_app.domain.dupes import canonical_dup_key
from jupr_app.domain.bulk_match_editor import apply_bulk_match_edits, compute_recompute_scope
from jupr_app.domain.live_social import (
    SocialTablesNotInstalledError,
    delete_social_matches,
    list_social_match_log_rows,
    update_social_match_row,
)
from jupr_app.ui.layout import page_shell
from jupr_app.domain.replay_history import FULL_RESET_LABEL, replay_history
from jupr_app.domain.match_delete import delete_rated_matches_with_replay

BULK_REPLAY_ERROR_STATE_KEY = "match_log_bulk_replay_error"


def _set_bulk_replay_error(message: str) -> None:
    st.session_state[BULK_REPLAY_ERROR_STATE_KEY] = str(message)


def _clear_bulk_replay_error() -> None:
    st.session_state.pop(BULK_REPLAY_ERROR_STATE_KEY, None)



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
    if st.session_state.get(BULK_REPLAY_ERROR_STATE_KEY):
        st.warning(
            "⚠️ Previous automatic Replay ALL failed after bulk edit. "
            f"{st.session_state[BULK_REPLAY_ERROR_STATE_KEY]}"
        )

    df = df_matches.copy()
    df["source_type"] = "rated"
    df["source_label"] = "Rated"

    st.markdown("### Match Sources")
    source_mode = st.radio(
        "Choose what to show",
        ["Rated only", "Include Club Social"],
        horizontal=True,
        key="match_log_source_mode",
    )
    include_social = source_mode == "Include Club Social"

    social_df = pd.DataFrame()
    social_error = None
    if include_social:
        try:
            social_df = list_social_match_log_rows(
                ctx.supabase,
                club_id=str(ctx.club_id),
                limit=5000,
            )
        except SocialTablesNotInstalledError as exc:
            social_error = str(exc)
        except Exception as exc:
            social_error = f"Unable to load Club Social rows: {exc}"

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

    if include_social and social_error:
        st.warning(social_error)

    if include_social and social_df is not None and not social_df.empty:
        social_df = social_df.head(int(limit_rows)).copy()
        combined_df = pd.concat([df, social_df], ignore_index=True, sort=False)
    else:
        combined_df = df.copy()

    st.subheader("📋 Match Log Rows")
    show_cols = [
        c
        for c in [
            "source_label",
            "id",
            "event_name",
            "date",
            "played_on",
            "league",
            "week_tag",
            "match_type",
            "match_key",
            "round_number",
            "court_number",
            "mini_round_number",
            "status",
            "submission_mode",
            "t1_p1",
            "t1_p2",
            "t2_p1",
            "t2_p2",
            "score_t1",
            "score_t2",
        ]
        if c in combined_df.columns
    ]
    if not combined_df.empty:
        st.dataframe(combined_df[show_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No rows for current filter/settings.")

    if include_social and social_df is not None and not social_df.empty:
        st.divider()
        st.subheader("🎾 Club Social Editor")
        st.caption(
            "Edit/delete unrated Club Social rows here. This updates only live_event_matches (and optionally live_events name)."
        )

        social_view = social_df.copy()
        social_view.insert(0, "Update", False)
        social_view.insert(1, "Delete", False)

        edit_cols = [
            c
            for c in [
                "Update",
                "Delete",
                "source_label",
                "social_match_id",
                "event_id",
                "event_name",
                "played_on",
                "round_number",
                "court_number",
                "mini_round_number",
                "score_t1",
                "score_t2",
                "status",
                "submission_mode",
                "match_key",
                "t1_p1",
                "t1_p2",
                "t2_p1",
                "t2_p2",
            ]
            if c in social_view.columns
        ]

        edited_social = st.data_editor(
            social_view[edit_cols],
            hide_index=True,
            use_container_width=True,
            key="social_match_editor",
            column_config={
                "Update": st.column_config.CheckboxColumn(default=False),
                "Delete": st.column_config.CheckboxColumn(default=False),
                "played_on": st.column_config.DateColumn("played_on"),
                "event_name": st.column_config.TextColumn("event_name"),
            },
            disabled=["source_label", "social_match_id", "event_id", "status", "submission_mode", "match_key", "t1_p1", "t1_p2", "t2_p1", "t2_p2"],
        )

        def _as_date_text(value):
            if pd.isna(value):
                return None
            ts = pd.to_datetime(value, errors="coerce")
            if pd.isna(ts):
                return None
            return ts.date().isoformat()

        original_by_id = social_df.set_index("social_match_id", drop=False)
        update_rows = edited_social[edited_social["Update"] == True] if "Update" in edited_social.columns else pd.DataFrame()
        delete_rows = edited_social[edited_social["Delete"] == True] if "Delete" in edited_social.columns else pd.DataFrame()

        c_save, c_delete = st.columns([1, 1])
        with c_save:
            if st.button("Save selected social edits", type="primary", key="social_save_btn"):
                updated_count = 0
                for _, row in update_rows.iterrows():
                    sid = str(row.get("social_match_id") or "").strip()
                    if not sid or sid not in original_by_id.index:
                        continue
                    before = original_by_id.loc[sid]
                    patch = {}
                    for fld in ("score_t1", "score_t2", "round_number", "court_number", "mini_round_number"):
                        old_v = None if pd.isna(before.get(fld)) else int(before.get(fld))
                        new_v = None if pd.isna(row.get(fld)) else int(row.get(fld))
                        if new_v != old_v:
                            patch[fld] = new_v
                    old_played = _as_date_text(before.get("played_on"))
                    new_played = _as_date_text(row.get("played_on"))
                    if new_played != old_played and new_played is not None:
                        patch["played_on"] = new_played
                    old_name = str(before.get("event_name") or "").strip()
                    new_name = str(row.get("event_name") or "").strip()
                    if new_name != old_name:
                        patch["event_name"] = new_name
                    if patch:
                        update_social_match_row(
                            ctx.supabase,
                            club_id=str(ctx.club_id),
                            social_match_id=sid,
                            patch=patch,
                        )
                        updated_count += 1
                if updated_count:
                    st.success(f"Updated {updated_count} Club Social match row(s).")
                    st.rerun()
                st.info("No social row edits detected.")
        with c_delete:
            if st.button("Delete selected social rows", key="social_delete_btn"):
                delete_ids = delete_rows["social_match_id"].dropna().astype(str).tolist() if not delete_rows.empty else []
                deleted = delete_social_matches(
                    ctx.supabase,
                    club_id=str(ctx.club_id),
                    social_match_ids=delete_ids,
                )
                if deleted:
                    st.success(f"Deleted {deleted} Club Social match row(s).")
                    st.rerun()
                st.info("No social rows deleted.")

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
                    f"Ready to exclude {len(ids_to_delete)} duplicated rated match rows "
                    f"(keeping the oldest copy per group) and rebuild ratings."
                )

                confirm = st.text_input("Type DELETE to confirm", value="", key="dup_delete_confirm")
                if st.button("🗑️ Delete duplicates now", type="primary", disabled=(confirm.strip().upper() != "DELETE")):
                    if ids_to_delete:
                        bar = st.progress(0.0)
                        with st.spinner("Excluding selected rated matches and rebuilding ratings..."):
                            delete_result = delete_rated_matches_with_replay(
                                supabase=ctx.supabase,
                                club_id=str(ctx.club_id),
                                match_ids=ids_to_delete,
                                df_meta=getattr(ctx, "df_meta", pd.DataFrame()),
                                progress_cb=lambda x: bar.progress(float(x)),
                                actor="match_log.duplicate_delete",
                                source="match_log",
                            )
                        if delete_result.get("warning"):
                            st.warning(delete_result["warning"])
                        if delete_result.get("replay_error"):
                            st.error(
                                "Matches were excluded, but Replay ALL failed. Ratings may be stale. "
                                "Run Admin Tools → Replay History → ALL immediately. "
                                f"Error: {delete_result['replay_error']}"
                            )
                        else:
                            st.success(
                                f"Excluded {delete_result['deleted_count']} rated match(es) from official history. "
                                "Ratings were rebuilt automatically via Replay ALL."
                            )
                        st.session_state["force_data_refresh"] = True
                        st.rerun()

    st.divider()

    # Bulk match editor UI
    st.subheader("✏️ Bulk Match Editor")
    st.caption("Filter matches, select rows, edit league/date/week_tag/match_type/players/scores/notes/is_active, preview impact, then apply safely.")

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
    base_cols = [c for c in ["id", "date_dt", "league", "week_tag", "match_type", "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2"] if c in df_bulk.columns]
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

    players_df = getattr(ctx, "df_players_all", pd.DataFrame()).copy()
    player_name_by_id = {}
    if players_df is not None and not players_df.empty and {"id", "name"}.issubset(players_df.columns):
        for _, prow in players_df.iterrows():
            try:
                player_name_by_id[int(prow["id"])] = str(prow["name"])
            except Exception:
                continue
    player_choices = sorted([(f"{name} (#{pid})", pid) for pid, name in player_name_by_id.items()], key=lambda x: x[0].lower())

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

    st.markdown("**Player correction helper**")
    pcol1, pcol2, pcol3 = st.columns([2, 3, 2])
    with pcol1:
        replace_slot = st.selectbox("Replace slot", ["t1_p1", "t1_p2", "t2_p1", "t2_p2"], key="bulk_player_replace_slot")
    with pcol2:
        replace_label = st.selectbox(
            "Replacement player",
            [label for label, _pid in player_choices] if player_choices else ["No players loaded"],
            key="bulk_player_replace_target",
        )
    with pcol3:
        if st.button("Stage player replacement", key="bulk_player_stage", disabled=not player_choices):
            selected_df = st.session_state["bulk_match_df"].copy()
            sel_mask = selected_df["Select"] == True
            if not sel_mask.any():
                st.warning("Select at least one row before staging player replacement.")
            else:
                replacement_pid = next(pid for label, pid in player_choices if label == replace_label)
                selected_df.loc[sel_mask, replace_slot] = int(replacement_pid)
                st.session_state["bulk_match_df"] = selected_df
                st.session_state["bulk_match_editor_version"] += 1
                st.success(f"Staged replacement in {replace_slot} for selected rows.")

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
            "score_t1": st.column_config.NumberColumn("score_t1", min_value=0, step=1),
            "score_t2": st.column_config.NumberColumn("score_t2", min_value=0, step=1),
            "is_active": st.column_config.CheckboxColumn("is_active") if "is_active" in st.session_state["bulk_match_df"].columns else None,
        },
        disabled=[c for c in ["t1_p1", "t1_p2", "t2_p1", "t2_p2"] if c in st.session_state["bulk_match_df"].columns],
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

        for player_slot in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            if player_slot in cur_by_id.columns and player_slot in base_by_id.columns:
                try:
                    old_pid = int(b.get(player_slot))
                except Exception:
                    old_pid = None
                try:
                    new_pid = int(a.get(player_slot))
                except Exception:
                    new_pid = None
                if new_pid != old_pid:
                    patch[player_slot] = new_pid

        for score_col in ("score_t1", "score_t2"):
            if score_col in cur_by_id.columns and score_col in base_by_id.columns:
                old_score = pd.to_numeric(b.get(score_col), errors="coerce")
                new_score = pd.to_numeric(a.get(score_col), errors="coerce")
                if pd.isna(new_score):
                    continue
                if pd.isna(old_score) or int(new_score) != int(old_score):
                    patch[score_col] = int(new_score)

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
    correction_note = st.text_input("Correction note (optional)", value="", key="bulk_match_correction_note")
    confirm = st.text_input("Type APPLY to confirm", value="", key="bulk_match_confirm")

    disabled = (confirm.strip().upper() != "APPLY") or (not patches)

    if st.button("Apply staged edits", type="primary", disabled=disabled, key="bulk_match_apply"):
        try:
            with st.spinner("Applying correction and rebuilding ratings..."):
                result = apply_bulk_match_edits(
                    supabase=ctx.supabase,
                    club_id=str(ctx.club_id),
                    patches=patches,
                    actor=actor.strip() or "admin",
                    correction_note=(correction_note.strip() or None),
                )

            replay_error = None
            if result["recompute_scope"]["ratings"]:
                bar = st.progress(0.0)
                try:
                    replay_history(
                        supabase=ctx.supabase,
                        club_id=str(ctx.club_id),
                        df_meta=getattr(ctx, "df_meta", pd.DataFrame()),
                        target_reset=FULL_RESET_LABEL,
                        progress_cb=lambda x: bar.progress(float(x)),
                    )
                except Exception as exc:
                    replay_error = str(exc)

            if replay_error:
                _set_bulk_replay_error(
                    "Edits were saved, but Replay ALL failed. Ratings may be stale. "
                    "Run Admin Tools → Replay History → ALL immediately. "
                    f"Error: {replay_error}"
                )
                st.error(
                    st.session_state[BULK_REPLAY_ERROR_STATE_KEY]
                )
            elif result["recompute_scope"]["ratings"]:
                _clear_bulk_replay_error()
                st.success(f"Updated {result['updated_count']} match(es). Ratings were rebuilt automatically via Replay ALL.")
            else:
                _clear_bulk_replay_error()
                st.success(
                    f"Updated {result['updated_count']} match(es). "
                    f"Affected leagues: {', '.join(result.get('affected_leagues', [])) or '(unknown)'}"
                )

            if result.get("warnings"):
                st.warning("Warnings:\n- " + "\n- ".join(result["warnings"][:10]))

            if not result["recompute_scope"]["ratings"]:
                st.info("Week/league views may change immediately. If anything looks off, run **Replay History**.")

            # Reset editor state
            st.session_state["bulk_match_df"] = view.copy(deep=True)
            st.session_state["bulk_match_baseline"] = view.copy(deep=True)
            st.session_state["bulk_match_editor_version"] += 1
            st.session_state["force_data_refresh"] = True
            if replay_error is None:
                st.rerun()

        except Exception as exc:
            st.error(f"Unable to apply bulk edits: {exc}")


        st.subheader("🔄 Quick Replay (Easy Button)")
        st.caption("Runs the same Replay History as Admin Tools. Use ALL after moving matches between leagues.")
    
        df_meta = getattr(ctx, "df_meta", pd.DataFrame())
        league_opts = [FULL_RESET_LABEL]
        if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
            league_opts += sorted(df_meta["league_name"].dropna().astype(str).unique().tolist())
    
        target_reset_quick = st.selectbox("Replay scope", league_opts, key="match_log_replay_scope")
        confirm_replay = st.text_input("Type REPLAY to confirm", value="", key="match_log_replay_confirm")
    
        if st.button("🔄 Replay Now", type="primary", disabled=(confirm_replay.strip().upper() != "REPLAY"), key="match_log_replay_btn"):
            bar = st.progress(0.0)
            with st.spinner("Replaying..."):
                result = replay_history(
                    supabase=ctx.supabase,
                    club_id=str(ctx.club_id),
                    df_meta=df_meta,
                    target_reset=str(target_reset_quick),
                    progress_cb=lambda x: bar.progress(float(x)),
                )
            _clear_bulk_replay_error()
            st.success("Replay complete.")
            st.info(f"Skipped incomplete doubles rows: {result['skipped_incomplete']}")
            st.info(f"Matches rewritten: {result['matches_rewritten']}")
            st.info(f"League ratings rows rebuilt: {result['league_ratings_rows']}")
            st.session_state["force_data_refresh"] = True
            st.rerun()

    st.divider()


    # Bulk delete UI
    st.subheader("🗑️ Exclude Rated Matches (first N rows shown)")
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
        st.warning(f"Exclude selected rated matches and rebuild ratings: {len(to_delete)} match(es).")
        confirm2 = st.text_input("Type DELETE to confirm exclusion", value="", key="bulk_delete_confirm")
        if st.button(f"Exclude {len(to_delete)} Rated Matches", type="primary", disabled=(confirm2.strip().upper() != "DELETE")):
            delete_ids = to_delete["id"].astype(int).tolist()
            bar = st.progress(0.0)
            with st.spinner("Excluding selected rated matches and rebuilding ratings..."):
                delete_result = delete_rated_matches_with_replay(
                    supabase=ctx.supabase,
                    club_id=str(ctx.club_id),
                    match_ids=delete_ids,
                    df_meta=getattr(ctx, "df_meta", pd.DataFrame()),
                    progress_cb=lambda x: bar.progress(float(x)),
                    actor="match_log.bulk_delete",
                    source="match_log",
                )
            if delete_result.get("warning"):
                st.warning(delete_result["warning"])
            if delete_result.get("replay_error"):
                st.error(
                    "Matches were excluded, but Replay ALL failed. Ratings may be stale. "
                    "Run Admin Tools → Replay History → ALL immediately. "
                    f"Error: {delete_result['replay_error']}"
                )
            else:
                st.success(
                    f"Excluded {delete_result['deleted_count']} rated match(es) from official history. "
                    "Ratings were rebuilt automatically via Replay ALL."
                )
            st.session_state["force_data_refresh"] = True
            st.rerun()
