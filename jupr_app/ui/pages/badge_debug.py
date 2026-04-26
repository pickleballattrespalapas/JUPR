from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from jupr_app.domain.gamification.badge_debug import build_badge_debug_report
from jupr_app.domain.match_filters import apply_match_filters_with_audit
from jupr_app.ui.components.player_picker import render_player_picker
from jupr_app.ui.layout import page_shell


@st.cache_data(show_spinner=False)
def _cached_match_filter_audit(df_matches: pd.DataFrame, club_id: str, league_id: str | None):
    _ = league_id
    return apply_match_filters_with_audit(df_matches, {"club_id": club_id, "exclude_popups": True})


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("🧪 Badge Debug", "Trace why a player earned (or missed) a badge.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    df_players = getattr(ctx, "df_players_all", pd.DataFrame())
    df_badges = getattr(ctx, "df_badges", pd.DataFrame())
    df_meta = getattr(ctx, "df_meta", pd.DataFrame())
    df_player_badges = getattr(ctx, "df_player_badges", pd.DataFrame())

    club_id = str(ctx.club_id)

    st.subheader("Inputs")
    club_options = [club_id]
    st.selectbox("Club", club_options, index=0, disabled=len(club_options) == 1)

    league_options = ["All leagues"]
    if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
        league_options += sorted(df_meta["league_name"].dropna().astype(str).unique().tolist())
    league_choice = st.selectbox("League", league_options)
    league_id = None if league_choice == "All leagues" else str(league_choice).strip()

    if df_players is None or df_players.empty or "id" not in df_players.columns:
        st.warning("Player data is still loading or unavailable.")
        st.stop()

    player_id = render_player_picker(
        df_players,
        label="Search player",
        key="badge_debug_player",
        include_inactive=True,
    )
    if player_id is None:
        st.info("Choose a player to run badge debug.")
        return

    if df_badges is None or df_badges.empty or "badge_id" not in df_badges.columns:
        st.warning("Badge definitions are unavailable.")
        st.stop()

    badge_ids = df_badges["badge_id"].dropna().astype(str).unique().tolist()
    badge_name_map = {
        str(row.badge_id): str(getattr(row, "name", "Badge"))
        for row in df_badges.itertuples(index=False)
    }
    badge_id = st.selectbox(
        "Badge",
        sorted(badge_ids),
        format_func=lambda bid: f"{badge_name_map.get(str(bid), 'Badge')} ({bid})",
    )

    run = st.button("Run Badge Debug", type="primary")

    if not run:
        st.caption("Run the report to inspect evaluator outputs, raw matches, and filter removals.")
        return

    with st.spinner("Building badge debug report..."):
        filtered_matches, audit = _cached_match_filter_audit(
            getattr(ctx, "df_matches", pd.DataFrame()),
            club_id,
            league_id,
        )
        report = build_badge_debug_report(
            ctx,
            club_id=club_id,
            league_id=league_id,
            player_id=int(player_id),
            badge_id=str(badge_id),
            filtered_matches=filtered_matches,
            match_audit=audit,
        )

    st.subheader("Truth Table")
    awarded = False
    if df_player_badges is not None and not df_player_badges.empty:
        awarded = not df_player_badges[
            (df_player_badges.get("player_id") == int(player_id))
            & (df_player_badges.get("badge_id").astype(str) == str(badge_id))
        ].empty

    truth_row = {
        "club_id": report.club_id,
        "league_id": report.league_id or "ALL",
        "player_id": report.player_id,
        "badge_id": report.badge_id,
        "raw_matches_count": len(report.matches_raw),
        "filtered_matches_count": len(report.matches_filtered),
        "candidates_count": len(report.candidates),
        "awarded?": awarded,
    }
    st.dataframe(pd.DataFrame([truth_row]), use_container_width=True, hide_index=True)

    if report.errors:
        st.error("Evaluator error encountered.")
        for idx, err in enumerate(report.errors, start=1):
            with st.expander(f"Error {idx}"):
                st.code(err)

    st.subheader("Candidate Rows")
    if not report.candidates:
        st.info("No candidates returned by evaluator.")
        _render_no_candidate_diagnostics(report)
    else:
        candidate_df = pd.DataFrame(report.candidates)
        display_cols = [c for c in ["context_id", "match_id", "value_json"] if c in candidate_df.columns]
        for col in ["context_type", "value_num"]:
            if col in candidate_df.columns and col not in display_cols:
                display_cols.append(col)
        candidate_df["value_json"] = candidate_df["value_json"].apply(_format_value_json)
        st.dataframe(candidate_df[display_cols], use_container_width=True, hide_index=True)

        st.markdown("**Candidate details**")
        for idx, row in candidate_df.iterrows():
            with st.expander(f"Candidate {idx + 1} • context_id={row.get('context_id')}"):
                st.json(report.candidates[idx].get("value_json", {}))

    st.subheader("Badge Diagnostics")
    if report.diagnostics:
        _render_badge_diagnostics(report)
    else:
        st.caption("No badge-specific diagnostics available for this badge yet.")

    st.subheader("Match Filtering Audit")
    if not report.filter_audit_steps:
        st.info("No filter audit steps recorded.")
    else:
        audit_table = pd.DataFrame(
            [
                {
                    "step_name": step.step_name,
                    "before_count": step.before_count,
                    "after_count": step.after_count,
                    "removed_count": len(step.removed_match_ids),
                }
                for step in report.filter_audit_steps
            ]
        )
        st.dataframe(audit_table, use_container_width=True, hide_index=True)

        for idx, step in enumerate(report.filter_audit_steps):
            removed_ids = step.removed_match_ids
            with st.expander(f"{step.step_name} removed {len(removed_ids)} match(es)"):
                _render_id_list(removed_ids, key=f"removed_{idx}")

    st.subheader("Raw vs Filtered Match IDs")
    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**Raw match IDs**")
        with st.expander(f"{len(report.matches_raw)} total"):
            _render_id_list(report.matches_raw, key="raw_ids")
    with col_right:
        st.markdown("**Filtered match IDs**")
        with st.expander(f"{len(report.matches_filtered)} total"):
            _render_id_list(report.matches_filtered, key="filtered_ids")

    raw_set = set(report.matches_raw)
    filtered_set = set(report.matches_filtered)
    removed = sorted(raw_set - filtered_set)
    added = sorted(filtered_set - raw_set)

    with st.expander("Diff: Raw \\ Filtered"):
        _render_id_list(removed, key="raw_minus_filtered")
    with st.expander("Diff: Filtered \\ Raw"):
        _render_id_list(added, key="filtered_minus_raw")


def _format_value_json(value) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, indent=2, sort_keys=True)
    except Exception:
        return str(value)


def _render_id_list(ids: list[str], *, key: str, sample_size: int = 50) -> None:
    if not ids:
        st.caption("None")
        return
    show_all = st.checkbox("Show full list", value=len(ids) <= sample_size, key=f"{key}_show_all")
    if show_all or len(ids) <= sample_size:
        st.write(ids)
    else:
        st.write(ids[:sample_size])
        st.caption(f"Showing first {sample_size} of {len(ids)} IDs.")


def _render_no_candidate_diagnostics(report) -> None:
    if len(report.candidates) > 0 or not report.diagnostics:
        return
    st.markdown("**Why no candidate?**")
    _render_badge_diagnostics(report)


def _render_badge_diagnostics(report) -> None:
    diagnostics = dict(report.diagnostics or {})
    badge_id = diagnostics.get("badge_id")
    if badge_id == "high_roller":
        distinct_wins = int(diagnostics.get("filtered_player_distinct_win_match_ids", 0))
        threshold = int(diagnostics.get("threshold_required", 100))
        qualifies = bool(diagnostics.get("qualifies_boolean", False))
        st.write(f"Distinct qualifying wins: {distinct_wins}")
        st.write(f"Threshold: {threshold}")
        st.write(f"Qualifies: {'Yes' if qualifies else 'No'}")
        st.caption("High Roller uses canonical filtered match history, not the OVERALL player aggregate row.")
    elif badge_id in {"dedicated_participant_50", "lifetime_participant_200"}:
        distinct_matches = int(diagnostics.get("filtered_player_distinct_match_ids", 0))
        threshold = int(diagnostics.get("threshold_required", 0))
        qualifies = bool(diagnostics.get("qualifies_boolean", False))
        st.write(f"Distinct qualifying matches: {distinct_matches}")
        st.write(f"Threshold: {threshold}")
        st.write(f"Qualifies: {'Yes' if qualifies else 'No'}")


    reconciliation = diagnostics.get("player_aggregate_reconciliation")
    if isinstance(reconciliation, dict) and reconciliation:
        st.markdown("### Player Aggregate Reconciliation")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown("**Leaderboard / players row**")
            st.write(f"wins: {int(reconciliation.get('players_table_wins', 0))}")
            st.write(f"losses: {int(reconciliation.get('players_table_losses', 0))}")
            st.write(f"matches_played: {int(reconciliation.get('players_table_matches_played', 0))}")
        with col_b:
            st.markdown("**Canonical filtered match history**")
            st.write(f"distinct matches: {int(reconciliation.get('filtered_match_distinct_match_ids', 0))}")
            st.write(f"distinct wins: {int(reconciliation.get('filtered_match_distinct_win_match_ids', 0))}")
            st.write(f"distinct losses: {int(reconciliation.get('filtered_match_distinct_loss_match_ids', 0))}")
        with col_c:
            st.markdown("**Difference (players - canonical)**")
            st.write(f"wins delta: {int(reconciliation.get('wins_delta', 0))}")
            st.write(f"losses delta: {int(reconciliation.get('losses_delta', 0))}")
            st.write(f"matches delta: {int(reconciliation.get('matches_delta', 0))}")

        aux_rows = {
            "raw_player_match_rows": int(reconciliation.get("raw_player_match_rows", 0)),
            "filtered_player_match_rows": int(reconciliation.get("filtered_player_match_rows", 0)),
            "filtered_match_win_rows": int(reconciliation.get("filtered_match_win_rows", 0)),
            "filtered_match_loss_rows": int(reconciliation.get("filtered_match_loss_rows", 0)),
            "popup_match_count_for_player": int(reconciliation.get("popup_match_count_for_player", 0)),
            "tournament_context_match_count_for_player": int(reconciliation.get("tournament_context_match_count_for_player", 0)),
            "invalid_or_void_match_count_for_player": int(reconciliation.get("invalid_or_void_match_count_for_player", 0)),
            "invalid_or_missing_score_match_count_for_player": int(reconciliation.get("invalid_or_missing_score_match_count_for_player", 0)),
            "matches_missing_required_player_slots_for_facts": int(reconciliation.get("matches_missing_required_player_slots_for_facts", 0)),
            "filtered_duplicate_match_rows_for_player": int(reconciliation.get("filtered_duplicate_match_rows_for_player", 0)),
        }
        st.dataframe(
            pd.DataFrame([aux_rows]).T.rename(columns={0: "count"}),
            use_container_width=True,
        )

        by_step = reconciliation.get("excluded_match_count_by_filter_step", {})
        if by_step:
            st.markdown("**Excluded by filter step**")
            step_df = pd.DataFrame(
                [{"step": str(k), "removed": int(v)} for k, v in by_step.items()]
            )
            st.dataframe(step_df, use_container_width=True, hide_index=True)

        if bool(reconciliation.get("aggregate_out_of_sync_warning", False)):
            st.warning("players aggregate appears out of sync with canonical match history")

    with st.expander("Diagnostics payload"):
        st.json(diagnostics)
