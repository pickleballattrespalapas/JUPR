# jupr_app/ui/pages/league_manager.py
from __future__ import annotations

import time
from datetime import datetime, timezone
from jupr_app.domain.player_ops import safe_add_player

import pandas as pd
import streamlit as st

from jupr_court_board import court_board

from jupr_app.domain.constants import DEFAULT_K_FACTOR
from jupr_app.domain.live_ladder import build_movement_preview, compute_round_stats, validate_courts
from jupr_app.domain.league_night_roster import (
    RosterChangeError,
    apply_roster_change,
    roster_change_availability,
    suggest_court_sizes,
)
from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.roster import (
    compress_courts,
    courts_to_roster_df,
    move_player_to_court,
    move_within_court,
    normalize_slots,
    roster_df_to_courts,
    swap_players,
)
from jupr_app.domain.schedule import get_match_schedule
from jupr_app.ui.layout import page_shell


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _league_options(df_meta: pd.DataFrame) -> list[str]:
    if df_meta is not None and not df_meta.empty and "is_active" in df_meta.columns and "league_name" in df_meta.columns:
        opts = sorted(df_meta[df_meta["is_active"] == True]["league_name"].dropna().astype(str).tolist())
        return opts if opts else ["Default"]
    return ["Default"]


def _seed_rating_for_player(pid: int, league_name: str, df_players_all: pd.DataFrame, df_leagues: pd.DataFrame) -> float:
    # League rating if exists, else overall
    if df_leagues is not None and not df_leagues.empty:
        hit = df_leagues[
            (df_leagues["player_id"].astype(int) == int(pid))
            & (df_leagues["league_name"].astype(str).str.strip() == str(league_name).strip())
        ]
        if not hit.empty:
            return float(hit.iloc[0].get("rating", 1200.0) or 1200.0)

    hit2 = df_players_all[df_players_all["id"].astype(int) == int(pid)]
    if not hit2.empty:
        return float(hit2.iloc[0].get("rating", 1200.0) or 1200.0)

    return 1200.0


def _summarize_roster(roster_df: pd.DataFrame) -> pd.DataFrame:
    if roster_df is None or roster_df.empty:
        return pd.DataFrame()
    cols = [c for c in ["court", "slot", "name", "player_id", "rating"] if c in roster_df.columns]
    df = roster_df[cols].copy()
    df["court"] = pd.to_numeric(df.get("court"), errors="coerce").fillna(0).astype(int)
    df["slot"] = pd.to_numeric(df.get("slot"), errors="coerce").fillna(0).astype(int)
    return df.sort_values(["court", "slot"]).reset_index(drop=True)


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("🏟️ League Manager", "Run live events and manage ladders.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    # ---- state init ----
    st.session_state.setdefault("ladder_state", "SETUP")
    st.session_state.setdefault("ladder_round_num", 1)
    st.session_state.setdefault("ladder_total_rounds", 5)
    st.session_state.setdefault("ladder_roster", [])
    st.session_state.setdefault("ladder_court_sizes", [])

    df_players_all = ctx.df_players_all
    df_leagues = ctx.df_leagues
    df_meta = ctx.df_meta
    id_to_name = ctx.id_to_name
    name_to_id = ctx.name_to_id

    tabs = st.tabs(["🏃‍♂️ Run Live Event (Ladder)", "⚙️ Settings"])

    # ============================================================
    # TAB 1: LIVE LADDER
    # ============================================================
    with tabs[0]:
        st.subheader("Ladder Management")

        # -------------------------
        # 1) SETUP
        # -------------------------
        if st.session_state.ladder_state == "SETUP":
            st.markdown("#### Step 1: Select League & Roster")

            opts = _league_options(df_meta)
            lg_select = st.selectbox("Select League", opts, key="ladder_lg")
            week_select = st.selectbox("Week", [f"Week {i}" for i in range(1, 13)] + ["Playoffs"], key="ladder_wk")
            num_rounds = st.number_input(
                "Total Rounds to Play",
                1, 20,
                value=int(st.session_state.get("ladder_total_rounds", 5)),
                step=1,
                key="ladder_total_rounds_input",
            )
            raw = st.text_area("Paste Player List (one per line)", height=150, key="ladder_raw_input")

            if st.button("Analyze & Seed"):
                st.session_state.saved_ladder_lg = str(lg_select)
                st.session_state.saved_ladder_wk = str(week_select)
                st.session_state.ladder_total_rounds = int(num_rounds)

                parsed = [x.strip() for x in (raw or "").replace("\n", ",").split(",") if x.strip()]
                roster_data = []
                new_ps = []

                for n in parsed:
                    if n in name_to_id:
                        pid = int(name_to_id[n])
                        r = _seed_rating_for_player(pid, lg_select, df_players_all, df_leagues)
                        roster_data.append({"name": n, "rating": float(r), "id": pid})
                    else:
                        new_ps.append(n)

                st.session_state.ladder_temp_roster = roster_data
                st.session_state.ladder_temp_new = new_ps
                st.session_state.ladder_state = "REVIEW_ROSTER"
                st.session_state.ladder_round_num = 1
                st.rerun()

        # -------------------------
        # 2) REVIEW / NEW PLAYERS
        # -------------------------
        if st.session_state.ladder_state == "REVIEW_ROSTER":
            c_back, _ = st.columns([1, 5])
            if c_back.button("⬅️ Back (edit league/week/rounds/roster)"):
                st.session_state.ladder_state = "SETUP"
                st.rerun()

            st.markdown("#### Step 2: Confirm Roster")

            new_names = st.session_state.get("ladder_temp_new", [])
            # Re-check DB for "missing" names (cache/name_to_id may be stale)
            if new_names:
                normalized = [str(x).strip() for x in new_names if str(x).strip()]
                resp = (
                    ctx.supabase.table("players")
                    .select("id,name,rating")
                    .eq("club_id", str(ctx.club_id))
                    .in_("name", normalized)
                    .execute()
                )
                existing = resp.data or []
                if existing:
                    # Move found ones into the roster immediately
                    base_roster = st.session_state.get("ladder_temp_roster", []) or []
                    base_names = {str(x.get("name", "")).strip() for x in base_roster}

                    for row in existing:
                        nm = str(row["name"]).strip()
                        if nm in base_names:
                            continue
                        base_roster.append({
                            "name": nm,
                            "rating": float(row.get("rating", 1200.0) or 1200.0),
                            "id": int(row["id"]),
                        })

                    st.session_state.ladder_temp_roster = base_roster

                    # Remove those from the "new" list
                    existing_names = {str(r["name"]).strip() for r in existing}
                    st.session_state.ladder_temp_new = [n for n in normalized if str(n).strip() not in existing_names]
                    new_names = st.session_state.ladder_temp_new

                st.caption("Set a starting JUPR for each new player, then click Save & Continue. (This creates the accounts.)")

                df_new = pd.DataFrame(
                    {"Name": [str(x) for x in new_names], "Starting JUPR": [3.5] * len(new_names)}
                )

                edited_new = st.data_editor(
                    df_new,
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
                    key="lm_new_players_editor",
                )

                c1, c2 = st.columns([1, 3])
                if c1.button("Save New Players & Continue", type="primary"):
                    errs = 0
                    for _, r in edited_new.iterrows():
                        nm = str(r["Name"]).strip()
                        jupr = float(r["Starting JUPR"])
                        ok, err = safe_add_player(
                            supabase=ctx.supabase,
                            club_id=str(ctx.club_id),
                            name=nm,
                            rating_jupr=jupr,
                        )
                        if not ok:
                            errs += 1
                            st.error(f"Could not add {nm}: {err}")

                    # If any inserts failed, stop here (do NOT continue)
                    if errs > 0:
                        st.stop()

                    # -------------------------
                    # SUCCESS PATH: fetch + advance
                    # -------------------------
                    created_names = [str(x).strip() for x in edited_new["Name"].tolist() if str(x).strip()]
                    resp = (
                        ctx.supabase.table("players")
                        .select("id,name,rating")
                        .eq("club_id", str(ctx.club_id))
                        .execute()
                    )
                    all_rows = resp.data or []

                    def norm(s: str) -> str:
                        return str(s or "").strip().lower()

                    wanted = {norm(x) for x in created_names}
                    created_rows = [r for r in all_rows if norm(r.get("name")) in wanted]

                    if not created_rows:
                        st.error("Players were inserted, but could not be re-fetched. Try Refresh and re-run Analyze & Seed.")
                        st.stop()

                    created_by_name = {str(rr["name"]).strip(): rr for rr in created_rows}

                    base_roster = st.session_state.get("ladder_temp_roster", []) or []
                    base_roster_names = {str(x.get("name", "")).strip() for x in base_roster}

                    for nm in created_names:
                        if nm in base_roster_names:
                            continue
                        row = created_by_name.get(nm)
                        if not row:
                            continue
                        base_roster.append({
                            "name": str(row["name"]).strip(),
                            "rating": float(row.get("rating", 1200.0) or 1200.0),
                            "id": int(row["id"]),
                        })

                    # Clear "new players" list so Step 2 doesn't trap you again
                    st.session_state.ladder_temp_roster = base_roster
                    st.session_state.ladder_temp_new = []

                    # Promote to real roster and advance
                    st.session_state.ladder_roster = sorted(
                        base_roster,
                        key=lambda x: float(x.get("rating", 1200.0)),
                        reverse=True,
                    )
                    st.session_state.ladder_state = "CONFIG_COURTS"

                    # Optional: refresh global cached data so other pages see them immediately
                    st.session_state["force_data_refresh"] = True

                    st.success("New players created. Continuing to court setup…")
                    st.rerun()

                # Keep this to prevent falling through while new players exist
                st.stop()



            st.success("All players found.")
            if st.button("Proceed to Court Setup"):
                st.session_state.ladder_roster = sorted(
                    st.session_state.get("ladder_temp_roster", []),
                    key=lambda x: float(x.get("rating", 1200.0)),
                    reverse=True,
                )
                st.session_state.ladder_state = "CONFIG_COURTS"
                st.session_state.pop("current_schedule", None)
                st.session_state.pop("current_schedule_round", None)
                st.rerun()


        # -------------------------
        # 3) CONFIG COURTS
        # -------------------------
        if st.session_state.ladder_state == "CONFIG_COURTS":
            c_back, _ = st.columns([1, 5])
            if c_back.button("⬅️ Back (edit roster)"):
                st.session_state.ladder_state = "REVIEW_ROSTER"
                st.rerun()

            st.markdown("#### Step 3: Configure Courts")
            total_p = len(st.session_state.ladder_roster)
            st.info(f"Total Players: {total_p}")

            auto = suggest_court_sizes(total_p)
            use_auto = st.checkbox("Use suggested setup (4s/5s only)", value=True, key="ladder_use_auto_courts")

            if use_auto and auto["ok"]:
                court_sizes = auto["sizes"]
                if auto.get("bench", 0) > 0:
                    st.warning(f"Auto setup requires {auto['bench']} bench player(s). Bench players will not be scheduled.")
                st.caption(auto["note"])
            else:
                num_courts = st.number_input("Number of Courts", 1, 10, key="ladder_num_courts", value=3)
                court_sizes = []
                cols = st.columns(int(num_courts))
                for i in range(int(num_courts)):
                    with cols[i]:
                        s = st.number_input(f"Ct {i+1} Size", 4, 12, 4, key=f"cs_{i}")
                        court_sizes.append(int(s))

            if sum(court_sizes) != total_p:
                st.error(f"Court sizes sum to {sum(court_sizes)}, but you have {total_p} players.")
            else:
                if st.button("Preview Assignments"):
                    current_idx = 0
                    final_assignments = []

                    for c_idx, size in enumerate(court_sizes):
                        group = st.session_state.ladder_roster[current_idx: current_idx + int(size)]
                        for pl in group:
                            final_assignments.append({
                                "player_id": int(pl["id"]),
                                "name": str(pl["name"]),
                                "rating": float(pl["rating"]),
                                "court": int(c_idx + 1),
                            })
                        current_idx += int(size)

                    final_roster = pd.DataFrame(final_assignments)
                    final_roster = final_roster.sort_values(["court", "rating"], ascending=[True, False]).copy()
                    final_roster["slot"] = final_roster.groupby("court").cumcount() + 1

                    st.session_state.ladder_live_roster = final_roster[["player_id", "name", "rating", "court", "slot"]].copy()
                    st.session_state.ladder_court_sizes = list(map(int, court_sizes))

                    # Printable sheet
                    print_df = final_roster.copy()
                    print_df["JUPR"] = (print_df["rating"].astype(float) / 400.0).round(3)
                    print_df = print_df.sort_values(["court", "slot"], ascending=[True, True])
                    print_df = print_df[["court", "slot", "name", "JUPR"]].rename(
                        columns={"court": "Court", "slot": "Slot", "name": "Player", "JUPR": "JUPR"}
                    )
                    st.session_state.ladder_print_sheet = print_df

                    st.session_state.ladder_state = "CONFIRM_START"
                    st.rerun()

        # -------------------------
        # 3.5) CONFIRM START (Court Board)
        # -------------------------
        if st.session_state.ladder_state == "CONFIRM_START":
            c_back, _ = st.columns([1, 5])
            if c_back.button("⬅️ Back (edit courts)"):
                st.session_state.pop("ladder_live_roster", None)
                st.session_state.ladder_state = "CONFIG_COURTS"
                st.rerun()

            st.markdown("#### Step 4: Court Board Preview (Drag & Drop)")
            st.caption("Use the Court Board to make final adjustments. Bench players will not be scheduled.")

            ps = st.session_state.get("ladder_print_sheet", None)
            if isinstance(ps, pd.DataFrame) and not ps.empty:
                csv_bytes = ps.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Court Sheet (CSV)",
                    data=csv_bytes,
                    file_name=f"court_sheet_{st.session_state.get('saved_ladder_lg','league')}_{st.session_state.get('saved_ladder_wk','week')}.csv",
                    mime="text/csv",
                )
                st.dataframe(ps, use_container_width=True, hide_index=True)

            roster_df = compress_courts(normalize_slots(st.session_state.ladder_live_roster.copy()))
            st.session_state.ladder_live_roster = roster_df

            courts_payload = roster_df_to_courts(roster_df, ladder_court_sizes=st.session_state.get("ladder_court_sizes"))

            round_num = int(st.session_state.get("ladder_round_num", 1))
            result = court_board(courts_payload, key=f"court_board_confirm_start_r{round_num}")

            if result and isinstance(result, dict) and "courts" in result:
                updated_courts = result["courts"]
                new_df = courts_to_roster_df(updated_courts, roster_df)
                if not new_df.equals(st.session_state.ladder_live_roster):
                    st.session_state.ladder_live_roster = new_df
                    st.session_state.pop("current_schedule", None)
                    st.session_state.pop("current_schedule_round", None)
                    st.rerun()

            # Validation
            target_sizes = st.session_state.get("ladder_court_sizes", None)
            v = validate_courts(roster_df, min_players_per_court=4, target_sizes=target_sizes)

            if v["warnings"]:
                st.info("Court sizes don't need to be perfect, but double-check:\n\n- " + "\n- ".join(v["warnings"]))
            if v["problems"]:
                st.warning("Fix these before starting:\n\n- " + "\n- ".join(v["problems"]))

            can_start = bool(v["can_start"])
            total_r = int(st.session_state.get("ladder_total_rounds", 1))
            start_label = "✅ Start Event (Round 1)" if round_num == 1 else f"✅ Start Round {round_num} / {total_r}"

            if st.button(start_label, disabled=not can_start, key=f"start_round_btn_{round_num}"):
                st.session_state.ladder_state = "PLAY_ROUND"
                st.session_state.pop("current_schedule", None)
                st.session_state.pop("current_schedule_round", None)
                st.rerun()

        # -------------------------
        # 4) PLAY ROUND (scoring + save + movement)
        # -------------------------
        if st.session_state.ladder_state == "PLAY_ROUND":
            current_r = int(st.session_state.get("ladder_round_num", 1))
            total_r = int(st.session_state.get("ladder_total_rounds", 1))
            st.markdown(f"### 🎾 Round {current_r} / {total_r}")

            roster_now = compress_courts(normalize_slots(st.session_state.ladder_live_roster.copy()))
            st.session_state.ladder_live_roster = roster_now

            # Quick edits
            with st.expander("✏️ Quick court edits (before scoring)", expanded=False):
                roster_df = roster_now.copy()
                names_now = roster_df["name"].astype(str).tolist()
                court_list = sorted(roster_df["court"].astype(int).unique().tolist())

                cA, cB, cC = st.columns([2, 2, 1])
                a = cA.selectbox("Swap Player A", names_now, key=f"swap_a_r{current_r}")
                b = cB.selectbox("with Player B", names_now, key=f"swap_b_r{current_r}", index=1 if len(names_now) > 1 else 0)
                if cC.button("Swap", key=f"swap_btn_r{current_r}"):
                    st.session_state.ladder_live_roster = compress_courts(swap_players(roster_df, a, b))
                    st.session_state.pop("current_schedule", None)
                    st.rerun()

                st.divider()

                c1, c2, c3 = st.columns([2, 2, 1])
                chosen_court = c1.selectbox("Court to reorder", court_list, key=f"re_ct_r{current_r}")
                court_players = roster_df[roster_df["court"].astype(int) == int(chosen_court)].sort_values("slot")["name"].tolist()
                p = c2.selectbox("Player", court_players, key=f"re_p_r{current_r}")
                new_pos = c3.number_input("New position", min_value=1, max_value=max(1, len(court_players)), value=1, step=1, key=f"re_pos_r{current_r}")
                if st.button("Apply reorder", key=f"re_btn_r{current_r}"):
                    st.session_state.ladder_live_roster = compress_courts(move_within_court(roster_df, p, int(new_pos)))
                    st.session_state.pop("current_schedule", None)
                    st.rerun()

                st.divider()

                st.markdown("#### 🔁 Move player to a different court")
                m1, m2, m3, m4 = st.columns([2, 1, 1, 1])
                mv_player = m1.selectbox("Player to move", names_now, key=f"mv_p_r{current_r}")
                target_court = m2.selectbox("To court", court_list, key=f"mv_ct_r{current_r}")
                target_names = roster_df[roster_df["court"].astype(int) == int(target_court)].sort_values("slot")["name"].tolist()
                target_pos = m3.number_input("Insert pos", min_value=1, max_value=max(1, len(target_names) + 1), value=1, step=1, key=f"mv_pos_r{current_r}")
                if m4.button("Move", key=f"mv_btn_r{current_r}"):
                    st.session_state.ladder_live_roster = move_player_to_court(roster_df, mv_player, int(target_court), int(target_pos))
                    st.session_state.pop("current_schedule", None)
                    st.rerun()

            # Build schedule once per round unless roster changed
            if ("current_schedule" not in st.session_state) or (st.session_state.get("current_schedule_round") != current_r):
                schedule = []
                for c_num in sorted(roster_now["court"].astype(int).unique().tolist()):
                    court_df = roster_now[roster_now["court"].astype(int) == int(c_num)].sort_values("slot")
                    pids = court_df["player_id"].astype(int).tolist()
                    fmt = f"{len(pids)}-Player"
                    matches = get_match_schedule(fmt, pids)
                    schedule.append({"c": int(c_num), "matches": matches})
                st.session_state.current_schedule = schedule
                st.session_state.current_schedule_round = current_r

            all_results = []
            with st.form("round_score_form"):
                for c_data in st.session_state.current_schedule:
                    st.markdown(f"### Court {c_data['c']}")
                    for m_idx, mm in enumerate(c_data["matches"]):
                        label = mm.get("desc", f"Game {m_idx+1}")
                        t1 = mm["t1"]
                        t2 = mm["t2"]

                        p1 = id_to_name.get(int(t1[0]), f"#{t1[0]}")
                        p2 = id_to_name.get(int(t1[1]), f"#{t1[1]}")
                        p3 = id_to_name.get(int(t2[0]), f"#{t2[0]}")
                        p4 = id_to_name.get(int(t2[1]), f"#{t2[1]}")

                        c1, c2, c3, c4 = st.columns([3, 1, 1, 3])
                        c1.text(f"{label}: {p1} & {p2}")
                        s1 = c2.number_input("S1", 0, 99, 0, 1, key=f"s1_r{current_r}_c{c_data['c']}_{m_idx}")
                        s2 = c3.number_input("S2", 0, 99, 0, 1, key=f"s2_r{current_r}_c{c_data['c']}_{m_idx}")
                        c4.text(f"{p3} & {p4}")

                        all_results.append({
                            "t1_p1": int(t1[0]),
                            "t1_p2": int(t1[1]),
                            "t2_p1": int(t2[0]),
                            "t2_p2": int(t2[1]),
                            "s1": int(s1),
                            "s2": int(s2),
                        })

                submitted = st.form_submit_button("Submit Round & Calculate Movement")

            if submitted:
                # Build match payload
                valid_matches = []
                for r in all_results:
                    if r["s1"] > 0 or r["s2"] > 0:
                        valid_matches.append({
                            **r,
                            "date": _utc_iso_now(),
                            "league": st.session_state.get("saved_ladder_lg", ""),
                            "match_type": "Live Match",
                            "week_tag": st.session_state.get("saved_ladder_wk", ""),
                            "is_popup": False,
                        })

                if not valid_matches:
                    st.warning("No scores entered (all matches 0–0).")
                    st.stop()

                # Save matches (refactored signature)
                res = process_matches(
                    valid_matches,
                    supabase=ctx.supabase,
                    club_id=str(ctx.club_id),
                    name_to_id=name_to_id,
                    df_players_all=ctx.df_players_all,
                    df_leagues=ctx.df_leagues,
                    df_meta=ctx.df_meta,
                )
                st.success(f"Matches saved ({res['inserted']}). Skipped incomplete: {res['skipped_incomplete']}.")

                # Compute movement preview
                roster_pids = roster_now["player_id"].astype(int).tolist()
                stats = compute_round_stats(valid_matches, roster_pids)
                max_court = int(roster_now["court"].max())
                preview = build_movement_preview(roster_now, stats, max_court=max_court)

                st.session_state.ladder_movement_preview = preview
                st.session_state.ladder_state = "CONFIRM_MOVEMENT"
                st.session_state.pop("current_schedule", None)
                st.rerun()

        # -------------------------
        # 5) CONFIRM MOVEMENT
        # -------------------------
        if st.session_state.ladder_state == "CONFIRM_MOVEMENT":
            st.markdown("#### Round Results & Movement")

            preview_df = st.session_state.get("ladder_movement_preview", pd.DataFrame())
            if preview_df is None or preview_df.empty:
                st.error("No movement preview found.")
                st.session_state.ladder_state = "PLAY_ROUND"
                st.rerun()

            # Display per court
            for c_num in sorted(preview_df["court"].astype(int).unique()):
                st.markdown(f"### Court {int(c_num)} Results")
                c_players = preview_df[preview_df["court"].astype(int) == int(c_num)]

                for _, p in c_players.iterrows():
                    move = "➖"
                    if int(p["Proposed Court"]) < int(p["court"]):
                        move = f"⬆️ To Ct {int(p['Proposed Court'])}"
                    elif int(p["Proposed Court"]) > int(p["court"]):
                        move = f"⬇️ To Ct {int(p['Proposed Court'])}"

                    st.write(f"**{p['name']}** — W:{int(p['Round Wins'])} | Diff:{int(p['Round Diff'])} | Pts:{int(p['Round Pts'])} — {move}")

                st.divider()

            st.markdown("#### 🛠️ Manual Override")
            st.info("If the arrows look right, click Start Next Round. Otherwise edit Proposed Court below.")

            edit_view = preview_df[["player_id", "name", "rating", "court", "Proposed Court"]].copy()
            editor_df = st.data_editor(
                edit_view,
                column_config={
                    "player_id": st.column_config.NumberColumn("ID", disabled=True),
                    "court": st.column_config.NumberColumn("Old Ct", disabled=True),
                    "Proposed Court": st.column_config.NumberColumn("New Ct", min_value=1, max_value=10, step=1),
                },
                hide_index=True,
                use_container_width=True,
            )

            current_r = int(st.session_state.get("ladder_round_num", 1))
            total_r = int(st.session_state.get("ladder_total_rounds", 1))
            btn_label = "Start Next Round" if current_r < total_r else "🏁 Finish League Night"

            base_next_roster = editor_df.copy()
            base_next_roster["court"] = base_next_roster["Proposed Court"].astype(int)
            base_next_roster = base_next_roster.sort_values(["court", "rating"], ascending=[True, False]).copy()
            base_next_roster["slot"] = base_next_roster.groupby("court").cumcount() + 1
            base_next_roster = base_next_roster[["player_id", "name", "rating", "court", "slot"]].copy()

            if st.session_state.get("ladder_next_roster_override") is not None:
                st.info("Roster changes queued for the next round.")
                st.dataframe(
                    _summarize_roster(st.session_state.ladder_next_roster_override),
                    use_container_width=True,
                    hide_index=True,
                )
                if st.session_state.get("ladder_roster_change_note"):
                    st.caption(str(st.session_state.ladder_roster_change_note))
                bench_ids = st.session_state.get("ladder_roster_bench_ids") or []
                if bench_ids:
                    bench_names = [id_to_name.get(int(pid), f"#{pid}") for pid in bench_ids]
                    st.warning(f"Sit-out this round: {', '.join(bench_names)}")

            st.markdown("#### Roster Changes (Between Rounds)")
            st.caption("Roster changes apply to the next round only. Completed rounds stay unchanged.")

            roster_change_enabled, roster_change_msg = roster_change_availability(
                ladder_state=st.session_state.get("ladder_state"),
                current_round=current_r,
                total_rounds=total_r,
                is_admin=bool(ctx.admin_logged_in),
            )
            if st.button(
                "Substitute / Add Player",
                key="ladder_roster_change_btn",
                disabled=not roster_change_enabled,
            ):
                st.session_state.ladder_show_roster_change_dialog = True
            if not roster_change_enabled and roster_change_msg:
                st.caption(roster_change_msg)

            if st.session_state.get("ladder_show_roster_change_dialog", False):
                @st.dialog("Roster Changes (Next Round)")
                def roster_changes_dialog() -> None:
                    next_round = int(current_r + 1)
                    final_round = int(total_r)
                    tabs_rc = st.tabs(["Substitute", "Add Player"])

                    close_cols = st.columns([1, 1, 6])
                    with close_cols[0]:
                        if st.button("Close", key="ladder_roster_change_close"):
                            st.session_state.ladder_show_roster_change_dialog = False
                            st.rerun()

                    def _player_options() -> tuple[list[str], dict[str, dict]]:
                        rows = ctx.df_players_all if ctx.df_players_all is not None else pd.DataFrame()
                        options = []
                        mapping = {}
                        if rows is None or rows.empty:
                            return options, mapping
                        for _, row in rows.iterrows():
                            pid = int(row.get("id"))
                            nm = str(row.get("name", "")).strip()
                            if not nm:
                                continue
                            label = f"{nm} (#{pid})"
                            options.append(label)
                            mapping[label] = {
                                "id": pid,
                                "name": nm,
                                "rating": float(row.get("rating", 1200.0) or 1200.0),
                            }
                        return options, mapping

                    options, option_map = _player_options()

                    with tabs_rc[0]:
                        st.markdown("Replace player (active roster):")
                        active_names = preview_df["name"].astype(str).tolist()
                        active_map = {
                            str(r["name"]): {
                                "id": int(r["player_id"]),
                                "name": str(r["name"]),
                                "rating": float(r.get("rating", 1200.0)),
                            }
                            for _, r in preview_df.iterrows()
                        }
                        replaced_name = st.selectbox("Replace player", active_names, key="ladder_sub_out")
                        replace_info = active_map.get(replaced_name, {})

                        st.markdown("Substitute with:")
                        use_guest_sub = st.checkbox("Create guest substitute", value=False, key="ladder_sub_guest")
                        if use_guest_sub:
                            guest_name = st.text_input("Guest name", key="ladder_sub_guest_name")
                            guest_rating = st.number_input(
                                "Guest starting JUPR", min_value=1.0, max_value=7.0, step=0.1, value=3.5, key="ladder_sub_guest_rating"
                            )
                            sub_player = {"name": guest_name.strip(), "rating": float(guest_rating)}
                        else:
                            sub_label = st.selectbox("Select substitute", options, key="ladder_sub_in")
                            sub_player = option_map.get(sub_label, {})

                        st.text(f"Effective round: {next_round}")
                        st.warning(
                            f"This will regenerate matchups for rounds {next_round}–{final_round}. Completed rounds will not change."
                        )
                        confirm_sub = st.checkbox("I understand and want to apply this substitute.", key="ladder_sub_confirm")

                        if st.button("Apply Substitute", type="primary", key="ladder_sub_apply"):
                            if not confirm_sub:
                                st.error("Please confirm the roster change before applying.")
                                st.stop()
                            try:
                                if use_guest_sub:
                                    if not sub_player.get("name"):
                                        raise RosterChangeError("Guest name is required.")
                                    ok, err = safe_add_player(
                                        supabase=ctx.supabase,
                                        club_id=str(ctx.club_id),
                                        name=sub_player["name"],
                                        rating_jupr=float(sub_player.get("rating", 3.5)),
                                    )
                                    if not ok:
                                        raise RosterChangeError(f"Could not add guest: {err}")

                                    fetch = (
                                        ctx.supabase.table("players")
                                        .select("id,name,rating")
                                        .eq("club_id", str(ctx.club_id))
                                        .eq("name", sub_player["name"])
                                        .execute()
                                    )
                                    rows = fetch.data or []
                                    if not rows:
                                        raise RosterChangeError("Guest player was created but could not be loaded.")
                                    row = rows[0]
                                    sub_player = {
                                        "id": int(row["id"]),
                                        "name": str(row["name"]),
                                        "rating": float(row.get("rating", 1200.0) or 1200.0),
                                    }
                                    id_to_name[int(row["id"])] = str(row["name"])
                                    name_to_id[str(row["name"])] = int(row["id"])

                                result = apply_roster_change(
                                    roster_df=base_next_roster,
                                    change_type="substitute",
                                    replaced_player_id=int(replace_info.get("id")),
                                    new_player=sub_player,
                                    court_sizes=st.session_state.get("ladder_court_sizes"),
                                    roster_locked=False,
                                )
                                st.session_state.ladder_next_roster_override = result.roster_df
                                st.session_state.ladder_next_court_sizes = result.court_sizes
                                st.session_state.ladder_roster_change_note = result.note
                                st.session_state.ladder_roster_bench_ids = result.bench_ids
                                st.session_state.ladder_show_roster_change_dialog = False
                                st.success("Substitution queued for next round.")
                                st.rerun()
                            except RosterChangeError as exc:
                                st.error(str(exc))

                    with tabs_rc[1]:
                        st.markdown("Add player (late arrival):")
                        use_guest_add = st.checkbox("Create guest player", value=False, key="ladder_add_guest")
                        if use_guest_add:
                            guest_name = st.text_input("Guest name", key="ladder_add_guest_name")
                            guest_rating = st.number_input(
                                "Guest starting JUPR",
                                min_value=1.0,
                                max_value=7.0,
                                step=0.1,
                                value=3.5,
                                key="ladder_add_guest_rating",
                            )
                            add_player = {"name": guest_name.strip(), "rating": float(guest_rating)}
                        else:
                            add_label = st.selectbox("Select player", options, key="ladder_add_player")
                            add_player = option_map.get(add_label, {})

                        st.text(f"Effective round: {next_round}")
                        st.warning(
                            f"This will regenerate matchups for rounds {next_round}–{final_round}. Completed rounds will not change."
                        )
                        confirm_add = st.checkbox("I understand and want to add this player.", key="ladder_add_confirm")

                        if st.button("Apply Add Player", type="primary", key="ladder_add_apply"):
                            if not confirm_add:
                                st.error("Please confirm the roster change before applying.")
                                st.stop()
                            try:
                                if use_guest_add:
                                    if not add_player.get("name"):
                                        raise RosterChangeError("Guest name is required.")
                                    ok, err = safe_add_player(
                                        supabase=ctx.supabase,
                                        club_id=str(ctx.club_id),
                                        name=add_player["name"],
                                        rating_jupr=float(add_player.get("rating", 3.5)),
                                    )
                                    if not ok:
                                        raise RosterChangeError(f"Could not add guest: {err}")

                                    fetch = (
                                        ctx.supabase.table("players")
                                        .select("id,name,rating")
                                        .eq("club_id", str(ctx.club_id))
                                        .eq("name", add_player["name"])
                                        .execute()
                                    )
                                    rows = fetch.data or []
                                    if not rows:
                                        raise RosterChangeError("Guest player was created but could not be loaded.")
                                    row = rows[0]
                                    add_player = {
                                        "id": int(row["id"]),
                                        "name": str(row["name"]),
                                        "rating": float(row.get("rating", 1200.0) or 1200.0),
                                    }
                                    id_to_name[int(row["id"])] = str(row["name"])
                                    name_to_id[str(row["name"])] = int(row["id"])

                                result = apply_roster_change(
                                    roster_df=base_next_roster,
                                    change_type="add",
                                    new_player=add_player,
                                    court_sizes=st.session_state.get("ladder_court_sizes"),
                                    roster_locked=False,
                                )
                                st.session_state.ladder_next_roster_override = result.roster_df
                                st.session_state.ladder_next_court_sizes = result.court_sizes
                                st.session_state.ladder_roster_change_note = result.note
                                st.session_state.ladder_roster_bench_ids = result.bench_ids
                                st.session_state.ladder_show_roster_change_dialog = False
                                st.success("Player added for next round.")
                                st.rerun()
                            except RosterChangeError as exc:
                                st.error(str(exc))

                roster_changes_dialog()

            if st.button(btn_label):
                if current_r >= total_r:
                    st.success("League Night Complete! All matches saved.")
                    for k in [
                        "ladder_round_num",
                        "ladder_live_roster",
                        "ladder_court_sizes",
                        "ladder_movement_preview",
                        "current_schedule",
                        "current_schedule_round",
                        "ladder_temp_roster",
                        "ladder_temp_new",
                        "ladder_roster",
                        "ladder_print_sheet",
                    ]:
                        st.session_state.pop(k, None)
                    st.session_state.ladder_state = "SETUP"
                    st.rerun()

                override = st.session_state.get("ladder_next_roster_override")
                if override is None:
                    new_roster = base_next_roster
                else:
                    if hasattr(override, "empty"):
                        new_roster = override if not override.empty else base_next_roster
                    else:
                        try:
                            new_roster = override if len(override) > 0 else base_next_roster
                        except Exception:
                            new_roster = override
                new_court_sizes = st.session_state.get("ladder_next_court_sizes") or [
                    int(x) for x in new_roster["court"].value_counts().sort_index().tolist()
                ]

                st.session_state.ladder_live_roster = new_roster.copy()
                st.session_state.ladder_court_sizes = new_court_sizes
                st.session_state.pop("ladder_next_roster_override", None)
                st.session_state.pop("ladder_next_court_sizes", None)
                st.session_state.pop("ladder_roster_change_note", None)
                st.session_state.pop("ladder_roster_bench_ids", None)

                st.session_state.ladder_round_num = current_r + 1
                st.session_state.ladder_state = "CONFIRM_START"
                st.session_state.pop("current_schedule", None)
                st.rerun()

    # ============================================================
    # TAB 2: SETTINGS
    # ============================================================
    with tabs[1]:
        st.subheader("Settings")

        with st.expander("➕ Create New League", expanded=False):
            default_k = int(DEFAULT_K_FACTOR) if DEFAULT_K_FACTOR is not None else 32
            league_name = st.text_input("League name (required)", key="create_league_name")
            description = st.text_area("Description (optional)", key="create_league_description")
            min_games = st.number_input("Minimum games", min_value=0, step=1, value=6, key="create_league_min_games")
            k_factor = st.number_input(
                "K-factor",
                min_value=1,
                step=1,
                value=default_k,
                key="create_league_k_factor",
            )
            is_active = st.checkbox("Active", value=True, key="create_league_active")

            if st.button("Create League", type="primary"):
                trimmed_name = str(league_name or "").strip()
                if not trimmed_name:
                    st.error("League name is required.")
                    st.stop()

                normalized_input = trimmed_name.lower()
                if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
                    existing = (
                        df_meta["league_name"]
                        .dropna()
                        .astype(str)
                        .str.strip()
                        .str.lower()
                        .tolist()
                    )
                    if normalized_input in existing:
                        st.error("A league with that name already exists for this club.")
                        st.stop()

                payload = {
                    "club_id": str(ctx.club_id),
                    "league_name": trimmed_name,
                    "description": str(description or "").strip(),
                    "min_games": int(min_games or 0),
                    "k_factor": int(k_factor or default_k),
                    "is_active": bool(is_active),
                }

                try:
                    resp = ctx.supabase.table("leagues_metadata").insert(payload).execute()
                except Exception as exc:
                    st.error(f"Could not create league: {exc}")
                    st.stop()

                if getattr(resp, "error", None):
                    st.error(f"Could not create league: {resp.error}")
                    st.stop()

                st.session_state["force_data_refresh"] = True
                st.success("League created.")
                st.rerun()

        if df_meta is None or df_meta.empty:
            st.info("No league metadata loaded.")
            return

        cols = [c for c in ["id", "league_name", "is_active", "min_games", "description", "k_factor"] if c in df_meta.columns]
        editor = st.data_editor(
            df_meta[cols],
            disabled=["id", "league_name"],
            hide_index=True,
            use_container_width=True,
        )

        if st.button("Save Config"):
            for _, r in editor.iterrows():
                ctx.supabase.table("leagues_metadata").update(
                    {
                        "is_active": bool(r.get("is_active", True)),
                        "min_games": int(r.get("min_games", 0) or 0),
                        "k_factor": int(r.get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR),
                        "description": str(r.get("description", "") or ""),
                    }
                ).eq("id", int(r["id"])).eq("club_id", str(ctx.club_id)).execute()

            st.success("Saved.")
            time.sleep(0.3)
            st.rerun()
