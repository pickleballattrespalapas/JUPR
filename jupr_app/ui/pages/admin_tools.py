import streamlit as st
import pandas as pd
from datetime import datetime, timezone

from postgrest.exceptions import APIError

from jupr_app.domain.gamification.ensure_badges import ensure_badges
from jupr_app.domain.gamification.badge_state import ALLOWED_BADGE_STATES, can_transition_badge_state
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue
from jupr_app.domain.replay_lock import is_replay_running
from jupr_app.data.sb_write import sb_update
from jupr_app.ui.layout import page_shell


REQUIRED_SCHEMA_VERSION = "rebuild_phase1_alignment"


def _get_api_error_code(exc: APIError) -> str | None:
    code = getattr(exc, "code", None)
    if code:
        return code
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("code")
    return None


def _is_replay_schema_valid(supabase) -> bool:
    try:
        response = (
            supabase.table("schema_version")
            .select("version")
            .eq("version", REQUIRED_SCHEMA_VERSION)
            .limit(1)
            .execute()
        )
    except Exception:
        return False
    return bool(response.data)


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
    # Badge Eval Queue (opportunistic worker)
    # -------------------------
    st.subheader("🧵 Badge Eval Queue")
    st.caption("Process queued badge evaluations without blocking the UI (short time budget).")
    if st.button("Process queued badge evaluations", key="badge_eval_queue_process"):
        with st.spinner("Processing queued badge evaluations..."):
            try:
                result = process_badge_eval_queue(supabase, max_jobs=5, time_budget_seconds=2)
            except APIError as exc:
                code = _get_api_error_code(exc)
                if code in {"PGRST205", "42P01"}:
                    st.error(
                        "Missing table badge_eval_queue; apply migrations/20260705_badge_eval_queue.sql and "
                        "run NOTIFY pgrst, 'reload schema';"
                    )
                    st.code(
                        "-- Apply migrations/20260705_badge_eval_queue.sql\nNOTIFY pgrst, 'reload schema';",
                        language="sql",
                    )
                else:
                    st.error("Failed to process queued badge evaluations.")
                    st.exception(exc)
            except Exception as exc:  # noqa: BLE001 - UI should not crash
                st.error("Failed to process queued badge evaluations.")
                st.exception(exc)
            else:
                st.success(f"Processed {result['processed']} job(s); {result['errored']} error(s).")

    # -------------------------
    # Replay History
    # -------------------------
    st.subheader("🔄 Recalculate / Replay History")
    st.caption("Heavy replay jobs are not run from Streamlit page handlers.")
    st.warning("Replay must be run via CLI or background job.")

    if _is_replay_schema_valid(supabase):
        st.caption("Schema check: ready for CLI replay.")
    else:
        st.caption("Schema check: migrations required before CLI replay.")

    st.code(
        f"python -m jupr_app.cli.replay_ratings --club-id {club_id}",
        language="bash",
    )

    replay_runs = None
    try:
        replay_runs = (
            supabase.table("replay_runs")
            .select("started_at,finished_at,status,summary")
            .eq("club_id", club_id)
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception:
        replay_runs = None

    if replay_runs and replay_runs.data:
        latest = replay_runs.data[0]
        st.caption("Last replay run (read-only)")
        st.json(latest)

    replay_lock_info = None
    try:
        replay_lock_info = is_replay_running(supabase, club_id)
    except Exception:
        replay_lock_info = None

    if replay_lock_info is None:
        st.caption("Replay lock: not running")
    else:
        st.caption("Replay lock: running")
        st.json({"club_id": replay_lock_info.club_id, "started_at": replay_lock_info.started_at, "status": replay_lock_info.status})

    st.divider()

    df_meta = getattr(ctx, "df_meta", pd.DataFrame())

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

            created = ensure_badges(ctx, club_id=club_id, league_id=league_id, as_of=as_of_dt)
            if not created:
                st.info("No badge candidates found.")
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

    st.divider()

    # -------------------------
    # Badge State Controls
    # -------------------------
    st.subheader("🧊 Badge State Controls")
    st.caption("Update badge awardability with a required reason. Allowed path: live → frozen → deprecated.")

    df_badges = getattr(ctx, "df_badges", pd.DataFrame())
    if df_badges is None or df_badges.empty or "badge_id" not in df_badges.columns:
        st.info("Badge definitions are unavailable.")
        return

    df_badges = df_badges.copy()
    df_badges["badge_id"] = df_badges["badge_id"].astype(str)
    df_badges["name"] = df_badges.get("name", "Badge")
    df_badges["state"] = df_badges.get("state", "live").fillna("live").astype(str)

    badge_rows = df_badges.sort_values("name")
    badge_id = st.selectbox(
        "Badge",
        badge_rows["badge_id"].tolist(),
        format_func=lambda bid: f"{badge_rows.loc[badge_rows['badge_id'] == bid, 'name'].iloc[0]} ({bid})",
        key="badge_state_badge_id",
    )
    current_state = (
        badge_rows.loc[badge_rows["badge_id"] == badge_id, "state"].iloc[0]
        if badge_id
        else "live"
    )
    st.caption(f"Current state: **{current_state}**")

    new_state = st.selectbox(
        "New state",
        ALLOWED_BADGE_STATES,
        index=ALLOWED_BADGE_STATES.index(current_state)
        if current_state in ALLOWED_BADGE_STATES
        else 0,
        key="badge_state_new_state",
    )
    reason = st.text_input("Reason for change", key="badge_state_reason")
    force = st.checkbox("Force transition (admin override)", value=False, key="badge_state_force")

    if st.button("Update Badge State", key="badge_state_update"):
        if not reason.strip():
            st.error("Please provide a reason for the state change.")
        else:
            transition = can_transition_badge_state(current_state, new_state, force=force)
            if not transition.allowed:
                st.error(transition.reason or "Transition not allowed.")
            else:
                try:
                    sb_update(
                        supabase,
                        "badges",
                        {
                            "state": new_state,
                            "state_changed_at": datetime.now(timezone.utc).isoformat(),
                            "state_change_reason": reason.strip(),
                        },
                        filters={"badge_id": badge_id},
                    )
                    st.success(f"Updated {badge_id} → {new_state}.")
                except Exception as exc:
                    st.error("Failed to update badge state.")
                    st.exception(exc)
