# match_uploader.py

import time
from datetime import datetime

import pandas as pd
import streamlit as st

from jupr_app.domain.schedule import get_match_schedule
from jupr_app.domain.match_processing import process_matches
from jupr_app.ui.layout import page_shell


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("📝 Match Uploader", "Quick entry for pop-up or league matches.", mode_label=mode_label)
    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin only.")
        return

    # --- Resolve ctx fields we actually need ---
    df_players = getattr(ctx, "df_players_active", None)
    if df_players is None:
        df_players = getattr(ctx, "df_players", None)

    df_players_all = getattr(ctx, "df_players_all", None)
    df_leagues = getattr(ctx, "df_leagues", None)
    df_meta = getattr(ctx, "df_meta", None)

    # Supabase + club id are required to write
    supabase = getattr(ctx, "supabase", None)
    club_id = getattr(ctx, "club_id", None)
    name_to_id = getattr(ctx, "name_to_id", None)

    if supabase is None or not club_id or name_to_id is None:
        st.error("Match Uploader missing required ctx fields: supabase, club_id, or name_to_id.")
        return

    # ---- Top controls ----
    c1, c2, c3 = st.columns(3)
    ctx_type = c1.radio("Context", ["🏆 Official League", "🎉 Pop-Up"], horizontal=True)

    if ctx_type == "🏆 Official League":
        if df_meta is not None and not df_meta.empty and "is_active" in df_meta.columns:
            active = df_meta[df_meta["is_active"] == True]
            opts = sorted(active["league_name"].dropna().astype(str).tolist())
            if not opts:
                opts = ["Default"]
        else:
            opts = ["Default"]

        selected_league = c2.selectbox("Select League", opts, key="mu_selected_league")
        match_type_db = "Live Match"
        is_popup = False
    else:
        selected_league = c2.text_input("Event Name", "Saturday Social", key="mu_event_name")
        match_type_db = "PopUp"
        is_popup = True

    if is_popup:
        selected_league = (selected_league or "").strip() or "Pop-Up Event"
        selected_league = f"POPUP::{selected_league}"

    week_tag = c3.selectbox(
        "Week / Session",
        [f"Week {i}" for i in range(1, 13)] + ["Playoffs", "Finals", "Event"],
        key="mu_week_tag",
    )

    st.divider()

    entry_method = st.radio(
        "Entry Method",
        ["📋 Manual / Batch", "🏟️ Single Round Robin"],
        horizontal=True,
        key="mu_entry_method",
    )
    st.write("")

    # Player list for selectboxes
    if df_players is not None and not df_players.empty and "name" in df_players.columns:
        player_list = sorted(df_players["name"].astype(str).tolist())
    else:
        player_list = []

    # -------------------------
    # Manual / Batch
    # -------------------------
    if entry_method == "📋 Manual / Batch":
        if "mu_batch_df" not in st.session_state:
            st.session_state.mu_batch_df = pd.DataFrame(
                [
                    {
                        "T1_P1": None,
                        "T1_P2": None,
                        "Score_1": 0,
                        "Score_2": 0,
                        "T2_P1": None,
                        "T2_P2": None,
                    }
                    for _ in range(5)
                ]
            )

        edited_batch = st.data_editor(
            st.session_state.mu_batch_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "T1_P1": st.column_config.SelectboxColumn("T1 P1", options=player_list),
                "T1_P2": st.column_config.SelectboxColumn("T1 P2", options=player_list),
                "T2_P1": st.column_config.SelectboxColumn("T2 P1", options=player_list),
                "T2_P2": st.column_config.SelectboxColumn("T2 P2", options=player_list),
            },
            key="mu_batch_editor",
        )

        # Keep the edited df around for reruns
        st.session_state.mu_batch_df = edited_batch.copy()

        if st.button("Submit Batch", key="mu_submit_batch"):
            valid_batch = []
            for _, row in edited_batch.iterrows():
                try:
                    s1 = int(row.get("Score_1") or 0)
                    s2 = int(row.get("Score_2") or 0)
                except Exception:
                    s1, s2 = 0, 0

                if row.get("T1_P1") and row.get("T2_P1") and (s1 + s2 > 0):
                    valid_batch.append(
                        {
                            "t1_p1": row.get("T1_P1"),
                            "t1_p2": row.get("T1_P2"),
                            "t2_p1": row.get("T2_P1"),
                            "t2_p2": row.get("T2_P2"),
                            "s1": s1,
                            "s2": s2,
                            "date": str(datetime.now()),
                            "league": selected_league,
                            "match_type": match_type_db,
                            "week_tag": week_tag,
                            "is_popup": is_popup,
                        }
                    )

            if not valid_batch:
                st.warning("No valid rows found. Add at least T1_P1, T2_P1, and a non-zero score.")
                return

            process_matches(
                valid_batch,
                supabase=supabase,
                club_id=str(club_id),
                name_to_id=name_to_id,
                df_players_all=df_players_all,
                df_leagues=df_leagues,
                df_meta=df_meta,
            )
            st.success("✅ Processed!")
            time.sleep(0.8)
            st.rerun()

    # -------------------------
    # Single Round Robin
    # -------------------------
    else:
        if "mu_lc_courts" not in st.session_state:
            st.session_state.mu_lc_courts = 1

        st.session_state.mu_lc_courts = st.number_input(
            "Courts",
            min_value=1,
            max_value=10,
            value=int(st.session_state.mu_lc_courts),
            key="mu_lc_courts_input",
        )

        with st.form("mu_setup_lc"):
            c_data = []
            for i in range(int(st.session_state.mu_lc_courts)):
                cc1, cc2 = st.columns([1, 3])
                t = cc1.selectbox(
                    f"Format C{i+1}",
                    ["4-Player", "5-Player", "6-Player", "8-Player", "12-Player"],
                    key=f"mu_fmt_{i}",
                )
                n = cc2.text_area(
                    f"Players C{i+1}",
                    height=70,
                    key=f"mu_names_{i}",
                )
                c_data.append({"type": t, "names": n})

            st.markdown("---")
            custom_sched = st.text_area(
                "Overrides: Paste Custom Schedule Here (e.g., '1 2 3 4')",
                help="Overrides the format selection above.",
                key="mu_custom_sched",
            )

            if st.form_submit_button("Generate"):
                st.session_state.mu_lc_schedule = []
                st.session_state.mu_active_lg = selected_league
                st.session_state.mu_active_wk = week_tag
                st.session_state.mu_active_is_popup = is_popup
                st.session_state.mu_active_mt = match_type_db

                for idx, c in enumerate(c_data):
                    pl = [x.strip() for x in c["names"].replace("\n", ",").split(",") if x.strip()]
                    st.session_state.mu_lc_schedule.append(
                        {
                            "c": idx + 1,
                            "m": get_match_schedule(c["type"], pl, custom_text=custom_sched),
                        }
                    )
                st.rerun()

        if "mu_lc_schedule" in st.session_state:
            with st.form("mu_scores_lc"):
                all_res = []
                for c in st.session_state.mu_lc_schedule:
                    st.markdown(f"**Court {c['c']}**")
                    for i, m in enumerate(c["m"]):
                        cc1, cc2, cc3, cc4 = st.columns([3, 1, 1, 3])
                        cc1.text(f"{m['t1'][0]}/{m['t1'][1]}")

                        s1 = cc2.number_input("S1", min_value=0, value=0, key=f"mu_s1_{c['c']}_{i}")
                        s2 = cc3.number_input("S2", min_value=0, value=0, key=f"mu_s2_{c['c']}_{i}")

                        cc4.text(f"{m['t2'][0]}/{m['t2'][1]}")

                        all_res.append(
                            {
                                "t1_p1": m["t1"][0],
                                "t1_p2": m["t1"][1],
                                "t2_p1": m["t2"][0],
                                "t2_p2": m["t2"][1],
                                "s1": int(s1),
                                "s2": int(s2),
                                "date": str(datetime.now()),
                                "league": st.session_state.mu_active_lg,
                                "match_type": st.session_state.mu_active_mt,
                                "week_tag": st.session_state.mu_active_wk,
                                "is_popup": bool(st.session_state.mu_active_is_popup),
                            }
                        )

                if st.form_submit_button("Submit"):
                    payload = [x for x in all_res if (x["s1"] > 0 or x["s2"] > 0)]
                    if payload:
                        try:
                            process_matches(
                                payload,
                                supabase=supabase,
                                club_id=str(club_id),
                                name_to_id=name_to_id,
                                df_players_all=df_players_all,
                                df_leagues=df_leagues,
                                df_meta=df_meta,
                            )
                        except Exception as e:
                            st.error("Failed to submit matches.")
                            st.exception(e)
                            st.stop()

                    st.success("✅ Done!")
                    if "mu_lc_schedule" in st.session_state:
                        del st.session_state.mu_lc_schedule
                    time.sleep(0.8)
                    st.rerun()
