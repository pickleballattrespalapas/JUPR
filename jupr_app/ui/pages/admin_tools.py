import streamlit as st
import pandas as pd
import time
from datetime import datetime, timezone
from jupr_app.domain.replay_history import replay_history


from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.domain.constants import DEFAULT_K_FACTOR
from jupr_app.domain.gamification.badge_engine import compute_candidates_for_club
from jupr_app.domain.gamification.badges_repo import upsert_player_badges
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
            bar = st.progress(0.0)
            with st.spinner("Crunching..."):
                result = replay_history(
                    supabase=supabase,
                    club_id=club_id,
                    df_meta=df_meta,
                    target_reset=str(target_reset),
                    progress_cb=lambda x: bar.progress(float(x)),
                )
    
            st.info(f"Skipped incomplete doubles rows: {result['skipped_incomplete']}")
            st.info(f"Matches to rewrite snapshots for: {result['matches_rewritten']}")
            st.info(f"League ratings rows rebuilt: {result['league_ratings_rows']}")
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

    st.divider()

    # -------------------------
    # Badge Backfill
    # -------------------------
    st.subheader("🎖️ Badge Backfill")
    st.caption("Compute badge candidates and write any missing awards for this club.")

    league_options = ["All leagues"]
    if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
        league_options += sorted(df_meta["league_name"].dropna().astype(str).unique().tolist())

    league_choice = st.selectbox("League scope", league_options, key="badge_backfill_league")
    use_as_of = st.checkbox("Run as-of date", value=False, key="badge_backfill_asof")
    as_of_date = None
    if use_as_of:
        as_of_date = st.date_input("As-of date", value=datetime.now(timezone.utc).date(), key="badge_backfill_date")

    if st.button("Run Badge Backfill", key="badge_backfill_run"):
        with st.spinner("Computing badge candidates..."):
            league_id = None if league_choice == "All leagues" else str(league_choice).strip()
            as_of_dt = None
            if as_of_date is not None:
                as_of_dt = datetime.combine(as_of_date, datetime.min.time(), tzinfo=timezone.utc)

            candidates = list(
                compute_candidates_for_club(
                    club_id=club_id,
                    league_id=league_id,
                    as_of=as_of_dt,
                    ctx=ctx,
                )
            )
            if not candidates:
                st.info("No badge candidates found.")
            else:
                created = upsert_player_badges(supabase, club_id, candidates)
                if not created:
                    st.success("No new badges to award.")
                else:
                    summary = (
                        pd.Series([c.badge_id for c in created], name="badge_id")
                        .value_counts()
                        .reset_index(name="new_awards")
                        .sort_values("new_awards", ascending=False)
                    )
                    st.success(f"Awarded {len(created)} new badges.")
                    st.caption("Developer summary: new awards by badge ID.")
                    st.dataframe(summary, use_container_width=True, hide_index=True)
