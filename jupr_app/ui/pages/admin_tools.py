import streamlit as st
import pandas as pd
import time
from datetime import datetime, timezone

from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.domain.constants import DEFAULT_K_FACTOR
from jupr_app.ui.layout import page_shell

def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("⚙️ Admin Tools", "Diagnostics, replays, and system maintenance.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    supabase = ctx.supabase
    club_id = str(ctx.club_id)

    st.subheader("🏥 System Health Check")

    if st.button("Run Diagnostics"):
        with st.status("Checking System...", expanded=True) as status:
            try:
                sample = supabase.table("matches").select("*").eq("club_id", club_id).limit(1).execute()
                if sample.data:
                    keys = sample.data[0].keys()
                    needed = {"t1_p1_r", "t1_p2_r", "t2_p1_r", "t2_p2_r", "t1_p1_r_end", "t1_p2_r_end", "t2_p1_r_end", "t2_p2_r_end"}
                    if needed.issubset(set(keys)):
                        st.success("✅ Snapshot columns exist.")
                    else:
                        st.error(f"❌ Snapshot columns missing: {sorted(list(needed - set(keys)))}")
                else:
                    st.warning("⚠️ No matches found.")

                # Find null snapshot rows (sample range)
                null_snaps = (
                    supabase.table("matches")
                    .select("id,t1_p1_r,t1_p1_r_end")
                    .eq("club_id", club_id)
                    .is_("t1_p1_r", None)
                    .limit(5000)
                    .execute()
                )
                if null_snaps.data:
                    st.error(f"❌ Found {len(null_snaps.data)} matches with empty snapshots (in sampled range). Run Replay.")
                else:
                    st.success("✅ No null snapshots found (in sampled range).")

            except Exception as e:
                st.error("Diagnostics failed.")
                st.exception(e)

            status.update(label="Complete", state="complete")

    st.divider()

    # -------------------------
    # Replay History
    # -------------------------
    st.subheader("🔄 Recalculate / Replay History")
    st.caption("This rebuilds snapshots and (optionally) players + league_ratings based on match history order.")

    df_meta = getattr(ctx, "df_meta", pd.DataFrame())
    league_opts = ["ALL (Full System Reset)"]
    if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
        league_opts += sorted(df_meta["league_name"].dropna().astype(str).unique().tolist())

    target_reset = st.selectbox("Replay scope", league_opts)

    if st.button(f"⚠️ Replay History for: {target_reset}"):
        with st.spinner("Crunching..."):
            all_players = supabase.table("players").select("*").eq("club_id", club_id).execute().data or []
            all_matches = (
                supabase.table("matches")
                .select("*")
                .eq("club_id", club_id)
                .order("date", desc=False)
                .order("id", desc=False)
                .execute()
                .data
            ) or []

            # K map
            k_map = {}
            if df_meta is not None and not df_meta.empty:
                for _, r in df_meta.iterrows():
                    try:
                        k_map[str(r["league_name"])] = int(r.get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
                    except Exception:
                        pass

            def k_for(lg: str) -> int:
                return int(k_map.get(str(lg), DEFAULT_K_FACTOR))

            # init overall ratings from starting_rating (fallback rating)
            p_map = {}
            for p in all_players:
                base = p.get("starting_rating", None)
                if base is None:
                    base = p.get("rating", 1200.0)
                p_map[int(p["id"])] = {"r": float(base), "w": 0, "l": 0, "mp": 0}

            island_map = {}  # (pid, league) -> stats
            matches_to_update = []
            skipped_incomplete = 0

            def gr(pid):
                return float(p_map[int(pid)]["r"])

            def gir(pid, lg):
                key = (int(pid), str(lg))
                if key not in island_map:
                    island_map[key] = {"r": float(p_map[int(pid)]["r"]), "w": 0, "l": 0, "mp": 0}
                return float(island_map[key]["r"])

            for m in all_matches:
                lg = str(m.get("league", "") or "").strip()
                if target_reset != "ALL (Full System Reset)" and lg != str(target_reset).strip():
                    continue

                p1, p2, p3, p4 = m.get("t1_p1"), m.get("t1_p2"), m.get("t2_p1"), m.get("t2_p2")
                if p1 is None or p2 is None or p3 is None or p4 is None:
                    skipped_incomplete += 1
                    continue

                s1 = int(m.get("score_t1", 0) or 0)
                s2 = int(m.get("score_t2", 0) or 0)

                # overall start snaps
                sr1, sr2, sr3, sr4 = gr(p1), gr(p2), gr(p3), gr(p4)

                do1, do2 = calculate_hybrid_elo((sr1 + sr2) / 2, (sr3 + sr4) / 2, s1, s2, k_factor=DEFAULT_K_FACTOR)

                win = s1 > s2
                for pid, d, won_flag in [
                    (p1, do1, win),
                    (p2, do1, win),
                    (p3, do2, not win),
                    (p4, do2, not win),
                ]:
                    pid = int(pid)
                    p_map[pid]["r"] += float(d)
                    p_map[pid]["mp"] += 1
                    if won_flag:
                        p_map[pid]["w"] += 1
                    else:
                        p_map[pid]["l"] += 1

                # overall end snaps
                er1, er2, er3, er4 = gr(p1), gr(p2), gr(p3), gr(p4)

                # league replay skip PopUp
                if str(m.get("match_type", "")) != "PopUp":
                    ir1, ir2, ir3, ir4 = gir(p1, lg), gir(p2, lg), gir(p3, lg), gir(p4, lg)
                    di1, di2 = calculate_hybrid_elo((ir1 + ir2) / 2, (ir3 + ir4) / 2, s1, s2, k_factor=k_for(lg))
                    for pid, d, won_flag in [
                        (p1, di1, win),
                        (p2, di1, win),
                        (p3, di2, not win),
                        (p4, di2, not win),
                    ]:
                        key = (int(pid), lg)
                        island_map[key]["r"] += float(d)
                        island_map[key]["mp"] += 1
                        if won_flag:
                            island_map[key]["w"] += 1
                        else:
                            island_map[key]["l"] += 1

                stored_elo_delta = abs(do1) if win else abs(do2)
                matches_to_update.append({
                    "id": int(m["id"]),
                    "elo_delta": float(stored_elo_delta),
                    "t1_p1_r": float(sr1), "t1_p2_r": float(sr2), "t2_p1_r": float(sr3), "t2_p2_r": float(sr4),
                    "t1_p1_r_end": float(er1), "t1_p2_r_end": float(er2), "t2_p1_r_end": float(er3), "t2_p2_r_end": float(er4),
                })

            st.info(f"Skipped incomplete doubles rows: {skipped_incomplete}")
            st.info(f"Matches to rewrite snapshots for: {len(matches_to_update)}")

            # Update players (only on full reset)
            if target_reset == "ALL (Full System Reset)":
                for pid, s in p_map.items():
                    supabase.table("players").update(
                        {"rating": s["r"], "wins": s["w"], "losses": s["l"], "matches_played": s["mp"]}
                    ).eq("club_id", club_id).eq("id", int(pid)).execute()

            # Rebuild league_ratings
            if target_reset != "ALL (Full System Reset)":
                supabase.table("league_ratings").delete().eq("club_id", club_id).eq("league_name", str(target_reset)).execute()
            else:
                supabase.table("league_ratings").delete().eq("club_id", club_id).execute()

            new_rows = []
            for (pid, lg), s in island_map.items():
                if target_reset == "ALL (Full System Reset)" or str(lg).strip() == str(target_reset).strip():
                    # starting_rating: use players.starting_rating if present
                    start_base = 1200.0
                    for p in all_players:
                        if int(p["id"]) == int(pid):
                            start_base = float(p.get("starting_rating", p.get("rating", 1200.0)) or 1200.0)
                            break
                    new_rows.append({
                        "club_id": club_id,
                        "player_id": int(pid),
                        "league_name": str(lg),
                        "rating": float(s["r"]),
                        "wins": int(s["w"]),
                        "losses": int(s["l"]),
                        "matches_played": int(s["mp"]),
                        "starting_rating": float(start_base),
                    })

            # insert in chunks
            for i in range(0, len(new_rows), 1000):
                supabase.table("league_ratings").insert(new_rows[i:i+1000]).execute()

            # Rewrite match snapshots (chunked, but still row updates)
            bar = st.progress(0.0)
            total = max(1, len(matches_to_update))
            for i, u in enumerate(matches_to_update):
                supabase.table("matches").update(u).eq("club_id", club_id).eq("id", int(u["id"])).execute()
                bar.progress((i + 1) / total)

            st.success("Replay complete.")
            time.sleep(0.6)
            st.rerun()

    st.divider()

    # -------------------------
    # Reports (simple, fast)
    # -------------------------
    st.subheader("📊 Reports & Exports")

    df_players_active = getattr(ctx, "df_players_active", pd.DataFrame())
    df_leagues = getattr(ctx, "df_leagues", pd.DataFrame())

    league_names = ["OVERALL"]
    if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
        league_names += sorted(df_meta["league_name"].dropna().astype(str).unique().tolist())

    report_league = st.selectbox("Select League for Report", league_names)

    if st.button("Generate Report"):
        if report_league == "OVERALL":
            rep_df = df_players_active.copy()
            if "starting_rating" not in rep_df.columns:
                rep_df["starting_rating"] = rep_df.get("rating", 1200.0)
            rep_df["name"] = rep_df.get("name", "")
        else:
            rep_df = df_leagues[df_leagues["league_name"].astype(str).str.strip() == str(report_league)].copy()
            rep_df["name"] = rep_df["player_id"].map(getattr(ctx, "id_to_name", {}))
            if "starting_rating" not in rep_df.columns:
                rep_df["starting_rating"] = rep_df["rating"]

        if rep_df is None or rep_df.empty:
            st.error("No data found for this league.")
        else:
            rep_df["JUPR"] = rep_df["rating"].astype(float) / 400.0
            rep_df["Win %"] = (rep_df["wins"].astype(float) / rep_df["matches_played"].replace(0, 1).astype(float)) * 100.0
            rep_df["Gain"] = (rep_df["rating"].astype(float) - rep_df["starting_rating"].astype(float)) / 400.0

            st.dataframe(
                rep_df.sort_values("rating", ascending=False)[["name", "JUPR", "wins", "losses", "matches_played", "Win %", "Gain"]],
                use_container_width=True,
                hide_index=True,
            )

            csv = rep_df[["name", "JUPR", "wins", "losses", "matches_played", "Win %", "Gain"]].to_csv(index=False).encode("utf-8")
            st.download_button(
                label=f"Download {report_league} Report (CSV)",
                data=csv,
                file_name=f"{report_league}_report_{datetime.now(timezone.utc).date().isoformat()}.csv",
                mime="text/csv",
            )
