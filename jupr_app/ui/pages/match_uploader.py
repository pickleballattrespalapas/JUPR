# match_uploader.py

import time
import re
from datetime import datetime

import pandas as pd
import streamlit as st

from jupr_app.domain.events import upsert_or_get_active_event
from jupr_app.domain.schedule import get_match_schedule
from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.player_ops import safe_add_player
from jupr_app.ui.layout import page_shell


def _parse_week_num(week_tag: str) -> int | None:
    if week_tag is None:
        return None
    match = re.search(r"(\d+)", str(week_tag))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


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

    if "mu_pending_new_players" not in st.session_state:
        st.session_state.mu_pending_new_players = []
    if "mu_pending_courts_data" not in st.session_state:
        st.session_state.mu_pending_courts_data = None
    if "mu_new_players_editor_seed" not in st.session_state:
        st.session_state.mu_new_players_editor_seed = None

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
        popup_event_name = None
    else:
        popup_event_name = c2.text_input("Event Name", "Saturday Social", key="mu_event_name")
        match_type_db = "PopUp"
        is_popup = True

    if is_popup:
        popup_event_name = (popup_event_name or "").strip() or "Pop-Up Event"
        selected_league = "POPUP"
        if st.session_state.get("mu_event_id_name") != popup_event_name:
            st.session_state.mu_event_id = None
        if st.session_state.get("mu_event_id"):
            st.caption(f"Event ID: {st.session_state.mu_event_id}")

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
            event_id = None
            if is_popup:
                event_id = upsert_or_get_active_event(
                    supabase,
                    club_id=str(club_id),
                    name=popup_event_name,
                )
                st.session_state.mu_event_id = event_id
                st.session_state.mu_event_id_name = popup_event_name
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
                            "context_type": "event" if is_popup else None,
                            "context_id": event_id if is_popup else None,
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
            st.session_state["force_data_refresh"] = True
            time.sleep(0.8)
            st.rerun()

    # -------------------------
    # Single Round Robin
    # -------------------------
    else:
        if "mu_lc_courts" not in st.session_state:
            st.session_state.mu_lc_courts = 1

        pending_new_players = st.session_state.get("mu_pending_new_players") or []
        if pending_new_players:
            st.subheader("New players found — create profiles to continue")

            seed_df = st.session_state.get("mu_new_players_editor_seed")
            if seed_df is None or seed_df.empty or len(seed_df) != len(pending_new_players):
                seed_df = pd.DataFrame(
                    {
                        "Name": [str(x) for x in pending_new_players],
                        "Starting JUPR": [3.5] * len(pending_new_players),
                    }
                )
                st.session_state.mu_new_players_editor_seed = seed_df

            edited_new = st.data_editor(
                seed_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Starting JUPR": st.column_config.NumberColumn(
                        "Starting JUPR",
                        min_value=1.0,
                        max_value=7.0,
                        step=0.1,
                    )
                },
                key="mu_new_players_editor",
            )

            if st.button("Create Players & Continue", type="primary"):
                errors = []
                for _, row in edited_new.iterrows():
                    nm = str(row["Name"]).strip()
                    jupr = float(row["Starting JUPR"])
                    ok, err = safe_add_player(
                        supabase=ctx.supabase,
                        club_id=str(ctx.club_id),
                        name=nm,
                        rating_jupr=jupr,
                    )
                    if not ok:
                        errors.append(f"{nm}: {err}")

                if errors:
                    for err in errors:
                        st.error(f"Could not add {err}")
                    st.stop()

                fetch = (
                    ctx.supabase.table("players")
                    .select("id,name,rating")
                    .eq("club_id", str(ctx.club_id))
                    .execute()
                )
                all_rows = fetch.data or []
                refreshed_name_to_id = {
                    str(row["name"]).strip(): int(row["id"])
                    for row in all_rows
                    if str(row.get("name", "")).strip()
                }
                refreshed_id_to_name = {
                    int(row["id"]): str(row["name"]).strip()
                    for row in all_rows
                    if row.get("id") is not None and str(row.get("name", "")).strip()
                }

                if isinstance(ctx.name_to_id, dict):
                    ctx.name_to_id.update(refreshed_name_to_id)
                if isinstance(getattr(ctx, "id_to_name", None), dict):
                    ctx.id_to_name.update(refreshed_id_to_name)
                st.session_state["force_data_refresh"] = True

                pending_data = st.session_state.get("mu_pending_courts_data") or {}
                st.session_state.mu_pending_new_players = []
                st.session_state.mu_pending_courts_data = None
                st.session_state.mu_new_players_editor_seed = None

                if pending_data.get("is_popup"):
                    event_id = upsert_or_get_active_event(
                        supabase,
                        club_id=str(club_id),
                        name=pending_data.get("popup_event_name") or "Pop-Up Event",
                    )
                    st.session_state.mu_event_id = event_id
                    st.session_state.mu_event_id_name = pending_data.get("popup_event_name") or "Pop-Up Event"

                st.session_state.mu_lc_schedule = []
                st.session_state.mu_active_lg = pending_data.get("selected_league")
                st.session_state.mu_active_wk = pending_data.get("week_tag")
                st.session_state.mu_active_is_popup = pending_data.get("is_popup")
                st.session_state.mu_active_mt = pending_data.get("match_type_db")

                pending_c_data = pending_data.get("c_data") or []
                pending_custom_sched = pending_data.get("custom_sched") or ""
                for idx, c in enumerate(pending_c_data):
                    pl = [x.strip() for x in c["names"].replace("\n", ",").split(",") if x.strip()]
                    st.session_state.mu_lc_schedule.append(
                        {
                            "c": idx + 1,
                            "m": get_match_schedule(c["type"], pl, custom_text=pending_custom_sched),
                        }
                    )
                st.rerun()

            st.stop()

        format_expected_games = {
            "4-Player": 3,
            "5-Player": 5,
            "6-Player": 9,
            "8-Player": 14,
            "9-Player": 18,
            "12-Player": 33,
        }

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
                    ["4-Player", "5-Player", "6-Player", "8-Player", "9-Player", "12-Player"],
                    key=f"mu_fmt_{i}",
                )
                expected_games = format_expected_games.get(t)
                if expected_games is not None:
                    cc1.caption(f"Expected games for this format: {expected_games}")
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
                def normalize_name(name: str) -> str:
                    return " ".join(str(name or "").replace("\u00A0", " ").split())

                normalized_name_to_id = {
                    normalize_name(k): v
                    for k, v in (name_to_id or {}).items()
                    if normalize_name(k)
                }

                all_names = set()
                for c in c_data:
                    pl = [x.strip() for x in c["names"].replace("\n", ",").split(",") if x.strip()]
                    all_names.update(pl)

                missing_names = []
                for nm in sorted(all_names):
                    if nm in name_to_id:
                        continue
                    normalized_nm = normalize_name(nm)
                    if normalized_nm and normalized_name_to_id.get(normalized_nm) is not None:
                        continue
                    missing_names.append(nm)

                if missing_names:
                    st.session_state.mu_pending_new_players = missing_names
                    st.session_state.mu_pending_courts_data = {
                        "c_data": c_data,
                        "custom_sched": custom_sched,
                        "popup_event_name": popup_event_name,
                        "week_tag": week_tag,
                        "match_type_db": match_type_db,
                        "selected_league": selected_league,
                        "is_popup": is_popup,
                    }
                    st.rerun()

                if is_popup:
                    event_id = upsert_or_get_active_event(
                        supabase,
                        club_id=str(club_id),
                        name=popup_event_name,
                    )
                    st.session_state.mu_event_id = event_id
                    st.session_state.mu_event_id_name = popup_event_name
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
                                "context_type": "event" if st.session_state.mu_active_is_popup else None,
                                "context_id": st.session_state.get("mu_event_id")
                                if st.session_state.mu_active_is_popup
                                else None,
                            }
                        )

                if st.form_submit_button("Submit"):
                    per_court_counts = ", ".join(
                        f"C{c['c']}={len(c['m'])}" for c in st.session_state.mu_lc_schedule
                    )
                    st.caption(
                        f"Rendered matches per court: {per_court_counts}. "
                        f"Total rendered matches: {len(all_res)}."
                    )
                    payload = [x for x in all_res if (x["s1"] > 0 or x["s2"] > 0)]
                    total_games = len(all_res)
                    scored_games = len(payload)
                    st.info(f"Generated {total_games} games. Submitting {scored_games} scored games.")

                    zero_score_count = len(all_res) - len(payload)
                    unmapped_rows = []

                    def is_unmapped(name):
                        if name is None:
                            return True
                        name_str = str(name).strip()
                        if not name_str:
                            return True
                        if name_str.isdigit():
                            return False
                        return name_to_id.get(name_str) is None

                    for idx, match in enumerate(payload, start=1):
                        t1_p1 = match.get("t1_p1")
                        t1_p2 = match.get("t1_p2")
                        t2_p1 = match.get("t2_p1")
                        t2_p2 = match.get("t2_p2")
                        names = [t1_p1, t1_p2, t2_p1, t2_p2]
                        if any(is_unmapped(name) for name in names):
                            unmapped_rows.append(
                                {
                                    "match_index": idx,
                                    "t1": f"{t1_p1}/{t1_p2}",
                                    "t2": f"{t2_p1}/{t2_p2}",
                                }
                            )

                    if payload:
                        try:
                            res = process_matches(
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
                        st.success(
                            f"Inserted {res.get('inserted', 0)} • "
                            f"Skipped incomplete {res.get('skipped_incomplete', 0)} • "
                            f"Skipped empty {res.get('skipped_empty', 0)}"
                        )
                        if res.get("skipped_incomplete", 0) > 0:
                            with st.expander("Unmapped names in submitted matches", expanded=False):
                                if unmapped_rows:
                                    st.table(pd.DataFrame(unmapped_rows))
                                else:
                                    st.write("No unmapped names detected in the submitted payload.")
                    else:
                        st.warning("No non-zero score matches to submit.")

                    if "mu_lc_schedule" in st.session_state:
                        del st.session_state.mu_lc_schedule
                    st.session_state["force_data_refresh"] = True
                    time.sleep(0.8)
                    st.rerun()
