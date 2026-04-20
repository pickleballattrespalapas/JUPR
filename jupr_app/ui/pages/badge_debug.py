from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from jupr_app.domain.gamification.badge_debug import build_badge_debug_report
from jupr_app.domain.match_filters import apply_match_filters_with_audit
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

    player_ids = sorted(df_players["id"].dropna().astype(int).unique().tolist())
    id_to_name = getattr(ctx, "id_to_name", {})
    player_id = st.selectbox(
        "Player",
        player_ids,
        format_func=lambda pid: f"{id_to_name.get(int(pid), 'Player')} (#{pid})",
    )

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
