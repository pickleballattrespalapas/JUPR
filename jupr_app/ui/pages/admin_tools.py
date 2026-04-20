import streamlit as st
import pandas as pd
import time
import json
from datetime import datetime, timezone
from jupr_app.domain.replay_history import replay_history

from postgrest.exceptions import APIError

from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.domain.constants import DEFAULT_K_FACTOR
from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.tournament_match_payload import build_tournament_match_payload
from jupr_app.domain.gamification.ensure_badges import ensure_badges
from jupr_app.domain.gamification.badge_audit import (
    build_badge_audit_report,
    build_high_roller_diagnostic_report,
)
from jupr_app.domain.gamification.recompute import run_badge_recompute
from jupr_app.domain.gamification.badge_state import ALLOWED_BADGE_STATES, can_transition_badge_state
from jupr_app.domain.gamification.evaluators import build_evaluation_context
from jupr_app.domain.gamification.badge_worker import (
    process_badge_eval_queue,
    process_badge_eval_queue_until_empty,
)
from jupr_app.domain.live_social import (
    list_social_submissions_for_review,
    moderate_social_submission,
)
from jupr_app.ui.layout import page_shell

def _get_api_error_code(exc: APIError) -> str | None:
    code = getattr(exc, "code", None)
    if code:
        return code
    if exc.args and isinstance(exc.args[0], dict):
        return exc.args[0].get("code")
    return None


def _get_api_error_message(exc: APIError) -> str:
    if exc.args and isinstance(exc.args[0], dict):
        payload = exc.args[0]
        return str(payload.get("message") or payload.get("details") or payload.get("hint") or exc)
    return str(exc)

def _badge_queue_preflight(supabase, club_id: str) -> bool:
    try:
        supabase.table("badge_eval_queue").select("id").eq("club_id", club_id).limit(1).execute()
        supabase.table("player_badge_facts").select("club_id").eq("club_id", club_id).limit(1).execute()
        return True
    except APIError as exc:
        code = _get_api_error_code(exc) or ""
        message = _get_api_error_message(exc)

        if code in {"42P01", "PGRST205"}:
            st.error("Badge queue prerequisite table is missing.")
        elif code == "42703":
            st.error("Badge queue prerequisite column mismatch detected.")
        elif code == "42501":
            st.error("Badge queue prerequisite permission error (missing grants/RLS access).")
        else:
            st.error("Badge queue prerequisites are missing or inaccessible.")

        st.code(
            "-- Apply migrations/20260705_badge_eval_queue.sql (tables)\n"
            "-- Apply migrations/20260801_badge_queue_and_facts_grants.sql (grants)\n"
            "NOTIFY pgrst, 'reload schema';",
            language="sql",
        )
        st.caption(f"Details: {code or 'unknown'} | {message}")
        return False
    except Exception as exc:  # noqa: BLE001 - diagnostics only
        st.error("Could not verify badge queue prerequisites.")
        st.exception(exc)
        return False

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
    badge_queue_ready = _badge_queue_preflight(supabase, club_id)

    col_pending, col_jobs = st.columns(2)
    with col_pending:
        if st.button("Show pending queue count", key="badge_eval_queue_pending_count") and badge_queue_ready:
            pending_resp = (
                supabase.table("badge_eval_queue")
                .select("id", count="exact")
                .eq("club_id", club_id)
                .eq("status", "pending")
                .execute()
            )
            st.info(f"Pending jobs: {int(pending_resp.count or 0)}")
    with col_jobs:
        if st.button("Show last 10 pending jobs", key="badge_eval_queue_pending_jobs") and badge_queue_ready:
            pending_rows = (
                supabase.table("badge_eval_queue")
                .select("id,created_at,event_type,match_id,attempts,player_ids,context_id")
                .eq("club_id", club_id)
                .eq("status", "pending")
                .order("created_at", desc=True)
                .limit(10)
                .execute()
            )
            st.dataframe(pd.DataFrame(pending_rows.data or []), use_container_width=True, hide_index=True)

    drain_col1, drain_col2 = st.columns(2)
    with drain_col1:
        drain_wall_clock = st.selectbox(
            "Drain max wall clock",
            options=[30, 60, 90],
            index=2,
            format_func=lambda v: f"{int(v)}s",
            key="badge_eval_queue_drain_max_wall_clock",
            disabled=not badge_queue_ready,
        )
    with drain_col2:
        drain_batch_size = st.selectbox(
            "Drain batch size",
            options=[5, 10, 20],
            index=1,
            key="badge_eval_queue_drain_batch_size",
            disabled=not badge_queue_ready,
        )

    if st.button("Process queued badge evaluations", key="badge_eval_queue_process"):
        if not badge_queue_ready:
            st.info("Run required queue migrations before processing jobs.")
        else:
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
                    if int(result.get("errored") or 0) > 0:
                        errored_rows = (
                            supabase.table("badge_eval_queue")
                            .select("id,created_at,event_type,match_id,attempts,last_error")
                            .eq("club_id", club_id)
                            .eq("status", "error")
                            .order("created_at", desc=True)
                            .limit(5)
                            .execute()
                        )
                        rows = errored_rows.data or []
                        if rows:
                            st.warning("Recent queue errors")
                            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                            st.code(str(rows[0].get("last_error") or ""), language="text")

    if st.button("Drain badge queue (run all)", key="badge_eval_queue_drain"):
        if not badge_queue_ready:
            st.info("Run required queue migrations before processing jobs.")
        else:
            try:
                pending_resp = (
                    supabase.table("badge_eval_queue")
                    .select("id", count="exact")
                    .eq("club_id", club_id)
                    .eq("status", "pending")
                    .execute()
                )
                pending_count = int(pending_resp.count or 0)
                st.info(f"Pending jobs before drain: {pending_count}")

                progress = st.progress(0.0)
                progress_line = st.empty()
                target = max(1, pending_count)

                def _on_progress(payload: dict[str, int | float | str]) -> None:
                    processed_total = int(payload.get("total_processed") or 0)
                    errored_total = int(payload.get("total_errored") or 0)
                    completed_total = processed_total + errored_total
                    ratio = min(1.0, completed_total / target)
                    progress.progress(ratio)
                    progress_line.write(
                        f"Loop {int(payload.get('loop') or 0)}: processed {completed_total}/{target}, "
                        f"errors {errored_total}"
                    )

                drain_result = process_badge_eval_queue_until_empty(
                    supabase,
                    club_id,
                    max_total_jobs=500,
                    batch_max_jobs=int(drain_batch_size),
                    per_batch_time_budget_seconds=2.0,
                    max_wall_clock_seconds=float(drain_wall_clock),
                    max_errors=10,
                    progress_cb=_on_progress,
                )
                progress.progress(1.0)
                st.success(
                    "Drain complete: "
                    f"processed {int(drain_result.get('total_processed') or 0)}, "
                    f"errors {int(drain_result.get('total_errored') or 0)}, "
                    f"loops {int(drain_result.get('loops') or 0)}, "
                    f"reason {drain_result.get('stopped_reason')}, "
                    f"duration {float(drain_result.get('duration_seconds') or 0):.2f}s."
                )

                if int(drain_result.get("total_errored") or 0) > 0:
                    errored_rows = (
                        supabase.table("badge_eval_queue")
                        .select("id,event_type,attempts,last_error")
                        .eq("club_id", club_id)
                        .eq("status", "error")
                        .order("created_at", desc=True)
                        .limit(5)
                        .execute()
                    )
                    rows = errored_rows.data or []
                    st.warning("Last 5 errored jobs")
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            except APIError as exc:
                st.error("Failed to drain badge queue.")
                st.exception(exc)
            except Exception as exc:  # noqa: BLE001 - UI should not crash
                st.error("Failed to drain badge queue.")
                st.exception(exc)

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
    # Tournament Match Backfill
    # -------------------------
    st.subheader("🛠️ Tournament Match Backfill")
    st.caption("Insert missing public match rows for finalized tournament games.")

    if st.button("Backfill Missing Tournament Matches", key="tournament_match_backfill"):
        summary = _run_tournament_match_backfill(ctx)
        st.info(
            "Backfill summary: "
            f"attempted={summary['attempted']}, "
            f"inserted={summary['inserted']}, "
            f"skipped_incomplete={summary['skipped_incomplete']}, "
            f"skipped_empty={summary['skipped_empty']}, "
            f"errors={summary['errors']}"
        )


    st.divider()

    # -------------------------
    # Badge Backfill
    # -------------------------
    st.subheader("🎖️ Badge Backfill")
    st.caption(
        "Compute badge candidates and write any missing awards for this club. "
        "Includes eligible legacy-safe match rows for hybrid-safe badges while keeping canonical-only badges strict."
    )

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
    _render_badge_audit_section(ctx, club_id)

    st.divider()
    _render_high_roller_diagnostic_section(ctx, club_id)

    st.divider()
    _render_badge_recompute_section(ctx, club_id)

    st.divider()
    _render_club_social_review(ctx)

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
                    supabase.table("badges").update(
                        {
                            "state": new_state,
                            "state_changed_at": datetime.now(timezone.utc).isoformat(),
                            "state_change_reason": reason.strip(),
                        }
                    ).eq("badge_id", badge_id).execute()
                    st.success(f"Updated {badge_id} → {new_state}.")
                except Exception as exc:
                    st.error("Failed to update badge state.")
                    st.exception(exc)


def _render_club_social_review(ctx) -> None:
    st.subheader("🧾 Club Social Review")
    st.caption("Review pending club social submissions for this club context.")
    status_filter = st.selectbox(
        "Queue",
        ["pending", "saved", "rejected"],
        index=0,
        key="club_social_review_status_filter",
    )
    try:
        rows = list_social_submissions_for_review(ctx, status=status_filter, limit=100)
    except Exception as exc:
        st.error("Unable to load Club Social review queue.")
        st.exception(exc)
        return
    if not rows:
        st.info(f"No {status_filter} Club Social submissions found.")
        return

    for row in rows:
        summary = row.get("summary_json") or {}
        leader = summary.get("leader") if isinstance(summary, dict) else None
        leader_text = ""
        if isinstance(leader, dict) and leader.get("name"):
            leader_text = (
                f" • Leader: {leader.get('name')} "
                f"({int(leader.get('wins') or 0)}W, diff {int(leader.get('differential') or 0)})"
            )
        st.markdown(
            f"**{row.get('name') or 'Untitled'}** ({row.get('event_type')}) — {row.get('event_date')}  \n"
            f"Submitted by: `{row.get('submitted_by_name') or 'unknown'}` "
            f"({row.get('submission_mode') or 'unknown'}) • "
            f"Participants: {int(summary.get('participant_count') or 0)} • "
            f"Matches: {int(summary.get('match_count') or 0)}{leader_text}  \n"
            f"Created: {row.get('created_at') or '—'} • Updated: {row.get('updated_at') or '—'}"
        )
        if row.get("status") == "rejected" and row.get("rejection_reason"):
            st.caption(f"Rejection reason: {row.get('rejection_reason')}")

        action_cols = st.columns([1.2, 1.2, 3])
        rejection_reason = action_cols[2].text_input(
            "Rejection reason",
            key=f"club_social_reject_reason_{row.get('id')}",
        )
        if action_cols[0].button(
            "Approve",
            key=f"club_social_approve_{row.get('id')}",
            disabled=row.get("status") == "saved",
        ):
            try:
                moderate_social_submission(
                    ctx,
                    event_id=str(row.get("id")),
                    action="approve",
                )
                st.session_state["force_data_refresh"] = True
                st.success("Submission approved.")
                st.rerun()
            except Exception as exc:
                st.error(f"Approve failed: {exc}")
        if action_cols[1].button(
            "Reject",
            key=f"club_social_reject_{row.get('id')}",
            disabled=row.get("status") == "rejected",
        ):
            try:
                moderate_social_submission(
                    ctx,
                    event_id=str(row.get("id")),
                    action="reject",
                    rejection_reason=str(rejection_reason or ""),
                )
                st.session_state["force_data_refresh"] = True
                st.success("Submission rejected.")
                st.rerun()
            except Exception as exc:
                st.error(f"Reject failed: {exc}")
        with st.expander("View summary_json", expanded=False):
            st.json(summary)
        with st.expander("View raw_event_json", expanded=False):
            st.json(row.get("raw_event_json") or {})
        st.divider()


def _render_badge_audit_section(ctx, club_id: str) -> None:
    st.subheader("🎯 Badge Audit")
    st.caption("Compare expected badge rows vs actual rows for targeted diagnostics and troubleshooting.")

    df_players = getattr(ctx, "df_players_all", pd.DataFrame())
    player_options = [""] + sorted(df_players.get("id", pd.Series(dtype=int)).dropna().astype(int).unique().tolist())
    df_badges = getattr(ctx, "df_badges", pd.DataFrame())
    badge_options = [""] + sorted(df_badges.get("badge_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    df_meta = getattr(ctx, "df_meta", pd.DataFrame())
    league_options = [""] + sorted(df_meta.get("league_name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())

    col1, col2, col3 = st.columns(3)
    with col1:
        player_id = st.selectbox("Player", options=player_options, key="admin_badge_audit_player")
    with col2:
        badge_id = st.selectbox("Badge", options=badge_options, key="admin_badge_audit_badge")
    with col3:
        league_id = st.selectbox("League (optional)", options=league_options, key="admin_badge_audit_league")

    include_revoked = st.checkbox("Include revoked rows", value=False, key="admin_badge_audit_include_revoked")
    include_non_live = st.checkbox("Include non-live badges", value=False, key="admin_badge_audit_include_non_live")

    if st.button("Run Badge Audit", key="admin_badge_audit_run"):
        report = build_badge_audit_report(
            ctx.supabase,
            club_id=club_id,
            ctx=ctx,
            player_id=int(player_id) if str(player_id).strip() else None,
            badge_id=str(badge_id).strip() or None,
            league_id=str(league_id).strip() or None,
            include_revoked=bool(include_revoked),
            include_non_live=bool(include_non_live),
        )
        st.session_state["admin_badge_audit_report"] = report
        st.session_state["admin_badge_audit_inputs"] = {
            "player_id": int(player_id) if str(player_id).strip() else None,
            "badge_id": str(badge_id).strip() or None,
            "league_id": str(league_id).strip() or None,
        }

    report = st.session_state.get("admin_badge_audit_report")
    if not report:
        return

    counts = report.get("counts", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected", int(counts.get("expected_count", 0)))
    c2.metric("Actual Active", int(counts.get("actual_active_count", 0)))
    c3.metric("Missing", int(counts.get("missing_count", 0)))
    c4.metric("Stale", int(counts.get("stale_count", 0)))

    c5, c6, c7 = st.columns(3)
    c5.metric("Context Drift", int(counts.get("context_drift_exact_row_count", 0)))
    c6.metric("Revoked", int(counts.get("revoked_count", 0)))
    c7.metric("Duplicates", int(counts.get("duplicate_count", 0)))

    tabs = st.tabs(["Expected", "Actual Active", "Revoked", "Missing", "Stale", "Context Drift"])
    with tabs[0]:
        st.dataframe(pd.DataFrame(report.get("expected_rows", [])), use_container_width=True, hide_index=True)
    with tabs[1]:
        st.dataframe(pd.DataFrame(report.get("actual_active_rows", [])), use_container_width=True, hide_index=True)
    with tabs[2]:
        st.dataframe(pd.DataFrame(report.get("revoked_rows", [])), use_container_width=True, hide_index=True)
    with tabs[3]:
        st.dataframe(pd.DataFrame(report.get("missing_rows", [])), use_container_width=True, hide_index=True)
    with tabs[4]:
        st.dataframe(pd.DataFrame(report.get("stale_rows", [])), use_container_width=True, hide_index=True)
    with tabs[5]:
        st.dataframe(pd.DataFrame(report.get("context_drift_rows", [])), use_container_width=True, hide_index=True)

    selected_badge = (st.session_state.get("admin_badge_audit_inputs", {}) or {}).get("badge_id")
    selected_player = (st.session_state.get("admin_badge_audit_inputs", {}) or {}).get("player_id")
    selected_league = (st.session_state.get("admin_badge_audit_inputs", {}) or {}).get("league_id")
    if selected_badge == "high_roller" and selected_player is not None:
        _render_high_roller_diagnostics(ctx, club_id=club_id, player_id=int(selected_player), league_id=selected_league)


def _render_high_roller_diagnostics(ctx, *, club_id: str, player_id: int, league_id: str | None) -> None:
    st.markdown("**High Roller diagnostics**")
    eval_ctx = build_evaluation_context(ctx, club_id=club_id, league_id=league_id, as_of=None)
    facts = eval_ctx.facts_hybrid if eval_ctx.facts_hybrid is not None else eval_ctx.facts
    wins = facts[(facts["player_id"] == int(player_id)) & (facts["win"] == True)].dropna(subset=["match_id"])
    unique_match_ids = sorted(wins["match_id"].astype(str).unique().tolist())
    win_count = len(unique_match_ids)
    threshold_met = win_count >= 100

    d1, d2, d3 = st.columns(3)
    d1.metric("Computed wins", win_count)
    d2.metric("Threshold (100) met", "Yes" if threshold_met else "No")
    d3.metric("Unique winning match_ids", win_count)

    if unique_match_ids:
        st.caption("First 5 winning match_ids counted by the badge engine")
        st.code(", ".join(unique_match_ids[:5]), language="text")
        st.caption("Last 5 winning match_ids counted by the badge engine")
        st.code(", ".join(unique_match_ids[-5:]), language="text")


def _render_high_roller_diagnostic_section(ctx, club_id: str) -> None:
    st.subheader("🕵️ High Roller Diagnostic")
    st.caption("Run focused diagnostics for High Roller counting and match-filter removals.")

    df_players = getattr(ctx, "df_players_all", pd.DataFrame()).copy()
    if df_players is None or df_players.empty or "id" not in df_players.columns:
        st.info("No players available for High Roller diagnostic.")
        return

    df_players["id"] = pd.to_numeric(df_players["id"], errors="coerce")
    df_players = df_players.dropna(subset=["id"])
    if df_players.empty:
        st.info("No players available for High Roller diagnostic.")
        return
    df_players["id"] = df_players["id"].astype(int)

    player_options = sorted(df_players["id"].unique().tolist())
    player_id = st.selectbox("Diagnostic player", player_options, key="high_roller_diag_player")
    if st.button("Run High Roller Diagnostic", key="high_roller_diag_run"):
        report = build_high_roller_diagnostic_report(
            ctx.supabase,
            club_id=club_id,
            player_id=int(player_id),
            match_limit=5000,
            ctx=ctx,
        )
        st.session_state["high_roller_diag_report"] = report

    report = st.session_state.get("high_roller_diag_report")
    if not report:
        return

    selected = report.get("selected_player", {}) or {}
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Canonical wins", int(selected.get("canonical_unique_win_match_ids", 0)))
    col2.metric("Hybrid wins", int(selected.get("hybrid_unique_win_match_ids", 0)))
    col3.metric("Qualifies canonical", "Yes" if selected.get("qualifies_high_roller_canonical") else "No")
    col4.metric("Qualifies hybrid", "Yes" if selected.get("qualifies_high_roller_hybrid") else "No")

    filter_steps = ((report.get("filter_steps", {}) or {}).get("steps")) or []
    if filter_steps:
        st.caption("Filter-step removal summary")
        step_df = pd.DataFrame(filter_steps)
        st.dataframe(
            step_df[["step_name", "before_count", "after_count", "removed_count"]],
            use_container_width=True,
            hide_index=True,
        )

    top_df = pd.DataFrame(report.get("top_20_players_by_hybrid_unique_win_count", []))
    if not top_df.empty:
        st.caption("Top players by hybrid unique win count")
        st.dataframe(top_df, use_container_width=True, hide_index=True)

    ui_win_columns = [name for name in ("wins", "win_count", "total_wins", "lifetime_wins") if name in df_players.columns]
    if ui_win_columns and selected:
        ui_col = ui_win_columns[0]
        player_row = df_players[df_players["id"] == int(selected.get("player_id", -1))]
        if not player_row.empty:
            ui_wins = int(pd.to_numeric(player_row.iloc[0][ui_col], errors="coerce") or 0)
            hybrid_wins = int(selected.get("hybrid_unique_win_match_ids", 0))
            if abs(ui_wins - hybrid_wins) >= 10:
                st.warning(
                    f"Material mismatch: UI wins={ui_wins}, High Roller hybrid wins={hybrid_wins}. "
                    "Badge and UI counts may be out of sync."
                )


def _render_badge_recompute_section(ctx, club_id: str) -> None:
    st.subheader("🧹 Badge Recompute / Cleanup")
    st.caption("Run scoped badge recompute. Strict mode can revoke stale rows when safely scoped.")

    df_players = getattr(ctx, "df_players_all", pd.DataFrame())
    player_options = [""] + sorted(df_players.get("id", pd.Series(dtype=int)).dropna().astype(int).unique().tolist())
    df_badges = getattr(ctx, "df_badges", pd.DataFrame())
    badge_options = [""] + sorted(df_badges.get("badge_id", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    df_meta = getattr(ctx, "df_meta", pd.DataFrame())
    league_options = [""] + sorted(df_meta.get("league_name", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())

    mode = st.selectbox("Mode", ["dry-run", "append-only", "strict"], key="admin_badge_recompute_mode")
    col1, col2, col3 = st.columns(3)
    with col1:
        player_id = st.selectbox("Player (optional)", options=player_options, key="admin_badge_recompute_player")
    with col2:
        badge_id = st.selectbox("Badge (optional)", options=badge_options, key="admin_badge_recompute_badge")
    with col3:
        league_id = st.selectbox("League (optional)", options=league_options, key="admin_badge_recompute_league")
    context_id = st.text_input("Context ID (optional advanced)", key="admin_badge_recompute_context")
    include_non_live = st.checkbox("Include non-live badges", value=False, key="admin_badge_recompute_include_non_live")
    revoke_reason = st.text_input("Revoke reason (required for strict)", value="", key="admin_badge_recompute_reason")
    match_limit = st.number_input("Match limit", min_value=100, max_value=200000, step=100, value=5000, key="admin_badge_recompute_match_limit")

    if st.button("Run Badge Recompute", key="admin_badge_recompute_run"):
        scoped = any(
            (
                str(player_id).strip(),
                str(badge_id).strip(),
                str(league_id).strip(),
                str(context_id).strip(),
            )
        )
        if mode == "strict":
            if not scoped:
                st.error("Strict mode requires at least one scope filter: player_id, badge_id, league_id, or context_id.")
                return
            if not revoke_reason.strip():
                st.error("Revoke reason is required for strict mode.")
                return

        result = run_badge_recompute(
            ctx.supabase,
            club_id=club_id,
            mode=mode,
            player_id=int(player_id) if str(player_id).strip() else None,
            badge_id=str(badge_id).strip() or None,
            league_id=str(league_id).strip() or None,
            context_id=str(context_id).strip() or None,
            include_non_live=bool(include_non_live),
            revoke_reason=str(revoke_reason).strip() or None,
            allow_strict_global=False,
            match_limit=int(match_limit),
            ctx=ctx,
            created_by="admin_tools_ui",
        )
        st.success("Recompute finished.")
        st.code(json.dumps(result, indent=2), language="json")



def _run_tournament_match_backfill(ctx):
    supabase = ctx.supabase
    club_id = str(ctx.club_id)
    df_players_all = ctx.df_players_all
    df_leagues = ctx.df_leagues
    df_meta = ctx.df_meta
    name_to_id = ctx.name_to_id

    missing_games = _load_finalized_tournament_games_missing_matches(supabase, club_id)
    attempted = len(missing_games)
    inserted = 0
    skipped_incomplete = 0
    skipped_empty = 0
    errors = 0

    for game in missing_games:
        tournament_id = game.get("tournament_id")
        if not tournament_id:
            skipped_incomplete += 1
            continue

        tournament_resp = supabase.table("tournaments").select("id,name").eq("id", tournament_id).limit(1).execute()
        tournaments = tournament_resp.data or []
        if not tournaments:
            skipped_incomplete += 1
            continue
        tournament = tournaments[0]

        teams_resp = supabase.table("tournament_teams").select("*").eq("tournament_id", tournament_id).execute()
        teams = teams_resp.data or []
        teams_by_id = {row["id"]: row for row in teams if row.get("id")}

        score_a = int(game.get("score_a") or 0)
        score_b = int(game.get("score_b") or 0)
        if score_a + score_b <= 0:
            skipped_empty += 1
            continue

        payload = build_tournament_match_payload(
            tournament,
            game,
            teams_by_id,
            score_a=score_a,
            score_b=score_b,
        )

        if any(payload.get(k) is None for k in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")):
            skipped_incomplete += 1
            continue

        try:
            result = process_matches(
                [payload],
                supabase=supabase,
                club_id=club_id,
                name_to_id=name_to_id,
                df_players_all=df_players_all,
                df_leagues=df_leagues,
                df_meta=df_meta,
            )
        except Exception as exc:
            errors += 1
            st.error(f"Backfill failed for tournament game {game.get('id')}: {exc}")
            continue

        inserted += int(result.get("inserted", 0) or 0)
        skipped_incomplete += int(result.get("skipped_incomplete", 0) or 0)
        skipped_empty += int(result.get("skipped_empty", 0) or 0)

    return {
        "attempted": attempted,
        "inserted": inserted,
        "skipped_incomplete": skipped_incomplete,
        "skipped_empty": skipped_empty,
        "errors": errors,
    }


def _load_finalized_tournament_games_missing_matches(supabase, club_id: str) -> list[dict]:
    games_resp = (
        supabase.table("tournament_games")
        .select("id,tournament_id,team_a_id,team_b_id,score_a,score_b,finalized_at")
        .execute()
    )
    games = [row for row in (games_resp.data or []) if row.get("finalized_at") is not None]
    if not games:
        return []

    game_ids = [row.get("id") for row in games if row.get("id")]
    matches_resp = (
        supabase.table("matches")
        .select("tournament_game_id")
        .eq("club_id", club_id)
        .in_("tournament_game_id", game_ids)
        .execute()
    )
    existing_ids = {row.get("tournament_game_id") for row in (matches_resp.data or []) if row.get("tournament_game_id")}
    return [row for row in games if row.get("id") not in existing_ids]
