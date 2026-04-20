from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from jupr_app.domain.match_canonical_audit import build_match_canonical_audit
from jupr_app.domain.match_canonical_migration import normalize_legacy_matches_for_canonical
from jupr_app.ui.layout import page_shell

AUDIT_STATE_KEY = "match_canonical_audit_report"


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell(
        "🧩 Match Canonical Audit",
        "Compare profile-visible matches vs canonical facts visibility and normalize legacy rows safely.",
        mode_label=mode_label,
    )

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    club_id = str(ctx.club_id)
    player_options = _player_options(ctx)
    selected_player = st.selectbox(
        "Player",
        options=player_options,
        format_func=lambda pid: f"{pid} — {ctx.id_to_name.get(int(pid), f'Player {pid}')}",
    )

    league_options = [""] + _league_options(ctx)
    selected_league = st.selectbox(
        "League (optional)",
        options=league_options,
        format_func=lambda value: "All leagues" if not value else value,
    )

    limit = st.number_input("Audit row limit", min_value=100, max_value=5000, value=1200, step=100)
    if st.button("Run Audit", type="primary"):
        with st.spinner("Building canonical visibility audit..."):
            report = build_match_canonical_audit(
                ctx,
                club_id=club_id,
                player_id=int(selected_player),
                league_id=(selected_league or None),
                limit=int(limit),
            )
        st.session_state[AUDIT_STATE_KEY] = report

    report = st.session_state.get(AUDIT_STATE_KEY)
    if not report:
        st.info("Run the audit to compare profile-visible vs canonical-visible matches.")
        return

    counts = report.get("counts", {})
    cols = st.columns(5)
    cols[0].metric("Profile-visible", int(counts.get("profile_visible", 0)))
    cols[1].metric("Canonical-visible", int(counts.get("canonical_visible", 0)))
    cols[2].metric("Only in Profile", int(counts.get("only_in_profile", 0)))
    cols[3].metric("Only in Canonical", int(counts.get("only_in_canonical", 0)))
    cols[4].metric("Shared", int(counts.get("shared", 0)))

    tab_only_profile, tab_only_canonical, tab_shared, tab_reasons = st.tabs(
        ["Only in Profile", "Only in Canonical", "Shared", "Exclusion Reasons Summary"]
    )
    with tab_only_profile:
        rows = report.get("excluded_only_in_profile", [])
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    with tab_only_canonical:
        st.dataframe(
            pd.DataFrame({"match_id": report.get("only_in_canonical", [])}),
            use_container_width=True,
            hide_index=True,
        )
    with tab_shared:
        st.dataframe(
            pd.DataFrame({"match_id": report.get("shared_ids", [])}),
            use_container_width=True,
            hide_index=True,
        )
    with tab_reasons:
        st.dataframe(
            pd.DataFrame(report.get("exclusion_reasons_summary", [])),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Normalize Legacy Rows")
    st.caption("Safe, explicit migration only. No auto-submit.")
    col_dry, col_apply = st.columns(2)

    if col_dry.button("Dry Run Normalize"):
        with st.spinner("Generating normalization plan..."):
            result = normalize_legacy_matches_for_canonical(
                ctx.supabase,
                ctx=ctx,
                club_id=club_id,
                player_id=int(selected_player),
                dry_run=True,
            )
        st.code(json.dumps(result, indent=2), language="json")

    if col_apply.button("Apply Normalize"):
        with st.spinner("Applying normalization updates..."):
            result = normalize_legacy_matches_for_canonical(
                ctx.supabase,
                ctx=ctx,
                club_id=club_id,
                player_id=int(selected_player),
                dry_run=False,
            )
        st.success(f"Applied {int(result.get('applied_update_count', 0))} updates.")
        st.code(json.dumps(result, indent=2), language="json")
        st.session_state["force_data_refresh"] = True
        st.rerun()

    st.subheader("Next operator sequence")
    st.markdown(
        "\n".join(
            [
                "1. Run Match Canonical Audit for the player.",
                "2. Inspect excluded legacy rows in **Only in Profile**.",
                "3. Run **Dry Run Normalize**.",
                "4. Run **Apply Normalize**.",
                "5. Re-run Badge Debug (`high_roller`) for this player.",
                "6. If canonical counts make sense, run aggregate repair and badge recompute.",
            ]
        )
    )


def _league_options(ctx) -> list[str]:
    df_matches = getattr(ctx, "df_matches", pd.DataFrame())
    if df_matches is None or df_matches.empty or "league" not in df_matches.columns:
        return []
    return sorted(df_matches["league"].dropna().astype(str).str.strip().replace("", pd.NA).dropna().unique().tolist())


def _player_options(ctx) -> list[int]:
    df_players = getattr(ctx, "df_players_all", pd.DataFrame())
    if df_players is None or df_players.empty or "id" not in df_players.columns:
        return []
    return sorted(df_players["id"].dropna().astype(int).unique().tolist())
