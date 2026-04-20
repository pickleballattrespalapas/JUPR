from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from jupr_app.domain.gamification.badge_audit import build_badge_audit_report
from jupr_app.domain.gamification.recompute import run_badge_recompute
from jupr_app.ui.layout import page_shell


BADGE_AUDIT_FILTERS_FORM_KEY = "badge_audit_filters_form"
BADGE_AUDIT_FILTERS_STATE_KEY = "badge_audit_filters_state"
BADGE_AUDIT_REPORT_STATE_KEY = "badge_audit_report"


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("🧾 Badge Audit", "Audit and repair badge awards against the current engine.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    club_id = str(ctx.club_id)
    defaults = {
        "badge_id": "",
        "player_id": "",
        "league_id": "",
        "context_id": "",
        "since": "",
        "until": "",
        "include_non_live": False,
        "include_revoked": False,
    }
    st.session_state.setdefault(BADGE_AUDIT_FILTERS_STATE_KEY, defaults)

    with st.form(BADGE_AUDIT_FILTERS_FORM_KEY):
        st.caption(f"Club: {club_id}")
        col1, col2, col3 = st.columns(3)

        badge_options = [""] + _badge_options(ctx)
        with col1:
            badge_id = st.selectbox(
                "Badge ID",
                options=badge_options,
                index=_safe_index(badge_options, st.session_state[BADGE_AUDIT_FILTERS_STATE_KEY].get("badge_id", "")),
                format_func=lambda v: "All" if not v else v,
            )
            player_options = [""] + _player_options(ctx)
            saved_player = st.session_state[BADGE_AUDIT_FILTERS_STATE_KEY].get("player_id", "")
            saved_player_value = int(saved_player) if str(saved_player).strip().isdigit() else ""
            player_id = st.selectbox(
                "Player ID",
                options=player_options,
                index=_safe_index(player_options, saved_player_value),
                format_func=lambda v: "All" if v == "" else str(v),
            )

        with col2:
            league_id = st.selectbox(
                "League ID",
                options=[""] + _league_options(ctx),
                index=_safe_index([""] + _league_options(ctx), st.session_state[BADGE_AUDIT_FILTERS_STATE_KEY].get("league_id", "")),
                format_func=lambda v: "All" if not v else v,
            )
            context_id = st.text_input("Context ID", value=st.session_state[BADGE_AUDIT_FILTERS_STATE_KEY].get("context_id", ""))

        with col3:
            since = st.text_input("Since (optional)", value=st.session_state[BADGE_AUDIT_FILTERS_STATE_KEY].get("since", ""))
            until = st.text_input("Until (optional)", value=st.session_state[BADGE_AUDIT_FILTERS_STATE_KEY].get("until", ""))

        include_non_live = st.checkbox(
            "Include non-live badges?",
            value=bool(st.session_state[BADGE_AUDIT_FILTERS_STATE_KEY].get("include_non_live", False)),
        )
        include_revoked = st.checkbox(
            "Include revoked rows in detail?",
            value=bool(st.session_state[BADGE_AUDIT_FILTERS_STATE_KEY].get("include_revoked", False)),
        )

        run_audit = st.form_submit_button("Run Audit", type="primary")

    if run_audit:
        filters = {
            "badge_id": badge_id or None,
            "player_id": int(player_id) if str(player_id).strip() else None,
            "league_id": league_id or None,
            "context_id": context_id.strip() or None,
            "since": since.strip() or None,
            "until": until.strip() or None,
            "include_non_live": bool(include_non_live),
            "include_revoked": bool(include_revoked),
        }
        st.session_state[BADGE_AUDIT_FILTERS_STATE_KEY] = {
            "badge_id": badge_id,
            "player_id": str(player_id),
            "league_id": league_id,
            "context_id": context_id,
            "since": since,
            "until": until,
            "include_non_live": include_non_live,
            "include_revoked": include_revoked,
        }
        with st.spinner("Running badge audit..."):
            report = build_badge_audit_report(
                ctx.supabase,
                club_id=club_id,
                ctx=ctx,
                **filters,
            )
        st.session_state[BADGE_AUDIT_REPORT_STATE_KEY] = report

    report = st.session_state.get(BADGE_AUDIT_REPORT_STATE_KEY)
    if not report:
        st.info("Run an audit to view summary, details, and repair actions.")
        return

    counts = report.get("counts", {})
    st.caption(
        "Exact = strict player_id + badge_id + context_id. "
        "Soft = player_id + badge_id only. "
        "Context Drift = same player+badge exists on both sides but context_ids differ."
    )
    st.markdown("**Exact Match**")
    exact_cols = st.columns(4)
    exact_cols[0].metric("Expected Exact", int(counts.get("expected_exact_count", 0)))
    exact_cols[1].metric("Actual Exact", int(counts.get("actual_active_exact_count", 0)))
    exact_cols[2].metric("Missing Exact", int(counts.get("missing_exact_count", 0)))
    exact_cols[3].metric("Stale Exact", int(counts.get("stale_exact_count", 0)))

    st.markdown("**Soft Match**")
    soft_cols = st.columns(5)
    soft_cols[0].metric("Expected Soft", int(counts.get("expected_soft_count", 0)))
    soft_cols[1].metric("Actual Soft", int(counts.get("actual_active_soft_count", 0)))
    soft_cols[2].metric("Missing Soft", int(counts.get("missing_soft_count", 0)))
    soft_cols[3].metric("Stale Soft", int(counts.get("stale_soft_count", 0)))
    soft_cols[4].metric("Context Drift", int(counts.get("context_drift_soft_key_count", 0)))

    diag_cols = st.columns(3)
    diag_cols[0].metric("Context Drift Rows", int(counts.get("context_drift_exact_row_count", 0)))
    diag_cols[1].metric("Duplicates", int(counts.get("duplicate_count", 0)))
    diag_cols[2].metric("Revoked", int(counts.get("revoked_count", 0)))

    if report.get("schema_degraded"):
        st.warning(report.get("schema_degraded_reason") or "Schema degraded; badge details may be limited.")

    st.subheader("Per-badge Summary")
    st.dataframe(pd.DataFrame(report.get("per_badge_summary", [])), use_container_width=True, hide_index=True)

    tab_missing, tab_stale, tab_context_drift, tab_duplicates, tab_revoked, tab_active, tab_expected = st.tabs(
        ["Missing Exact", "Stale Exact", "Context Drift", "Duplicates", "Revoked", "Actual Active", "Expected"]
    )
    with tab_missing:
        st.dataframe(pd.DataFrame(report.get("missing_rows", [])), use_container_width=True, hide_index=True)
    with tab_stale:
        st.dataframe(pd.DataFrame(report.get("stale_rows", [])), use_container_width=True, hide_index=True)
    with tab_context_drift:
        st.dataframe(pd.DataFrame(report.get("context_drift_rows", [])), use_container_width=True, hide_index=True)
    with tab_duplicates:
        st.dataframe(pd.DataFrame(report.get("duplicate_rows", [])), use_container_width=True, hide_index=True)
        if report.get("duplicate_groups"):
            st.caption("Duplicate groups")
            st.json(report.get("duplicate_groups"))
    with tab_revoked:
        st.dataframe(pd.DataFrame(report.get("revoked_rows", [])), use_container_width=True, hide_index=True)
    with tab_active:
        st.dataframe(pd.DataFrame(report.get("actual_active_rows", [])), use_container_width=True, hide_index=True)
    with tab_expected:
        st.dataframe(pd.DataFrame(report.get("expected_rows", [])), use_container_width=True, hide_index=True)

    st.subheader("Repair")
    with st.form("badge_repair_form"):
        mode = st.selectbox("Mode", ["dry-run", "append-only", "strict"], format_func=_format_mode)
        created_by = st.text_input("Created by / operator note", value="")
        allow_strict_global = st.checkbox("Allow strict global repair", value=False)
        revoke_reason = st.text_input("Revoke reason", value="badge audit strict repair")

        run_repair = st.form_submit_button("Run Repair", type="primary")

    if run_repair:
        scope = report.get("scope", {})
        scope_for_run = {
            "league_id": scope.get("league_id"),
            "badge_id": scope.get("badge_id"),
            "player_id": scope.get("player_id"),
            "context_id": scope.get("context_id"),
            "since": scope.get("since"),
            "until": scope.get("until"),
        }

        strict_scope_present = any(scope_for_run.get(k) for k in ("league_id", "badge_id", "player_id", "context_id"))
        if mode == "strict":
            if not revoke_reason.strip():
                st.error("Revoke reason is required for strict mode.")
                st.stop()
            if not strict_scope_present and not allow_strict_global:
                st.error("Strict global repair is blocked unless you explicitly check 'Allow strict global repair'.")
                st.stop()
            if strict_scope_present and not scope.get("badge_id"):
                st.warning("Strict repair on broad scopes can revoke many rows. Double-check your filters.")

        with st.spinner("Running repair..."):
            result = run_badge_recompute(
                ctx.supabase,
                club_id=club_id,
                mode=mode,
                league_id=scope_for_run["league_id"],
                badge_id=scope_for_run["badge_id"],
                player_id=scope_for_run["player_id"],
                context_id=scope_for_run["context_id"],
                since=scope_for_run["since"],
                until=scope_for_run["until"],
                created_by=created_by.strip() or None,
                revoke_reason=revoke_reason.strip() or "badge audit strict repair",
                allow_strict_global=bool(allow_strict_global),
                ctx=ctx,
            )
        st.success("Repair completed.")
        st.code(json.dumps(result, indent=2), language="json")
        if mode in {"append-only", "strict"}:
            st.session_state["force_data_refresh"] = True
            st.rerun()


def _badge_options(ctx) -> list[str]:
    df_badges = getattr(ctx, "df_badges", pd.DataFrame())
    if df_badges is None or df_badges.empty or "badge_id" not in df_badges.columns:
        return []
    return sorted(df_badges["badge_id"].dropna().astype(str).unique().tolist())


def _league_options(ctx) -> list[str]:
    df_meta = getattr(ctx, "df_meta", pd.DataFrame())
    if df_meta is None or df_meta.empty or "league_name" not in df_meta.columns:
        return []
    return sorted(df_meta["league_name"].dropna().astype(str).unique().tolist())


def _player_options(ctx) -> list[int]:
    df_players = getattr(ctx, "df_players_all", pd.DataFrame())
    if df_players is None or df_players.empty or "id" not in df_players.columns:
        return []
    return sorted(df_players["id"].dropna().astype(int).unique().tolist())


def _safe_index(options: list, value):
    try:
        return options.index(value)
    except Exception:
        return 0


def _format_mode(value: str) -> str:
    mapping = {
        "dry-run": "Dry run",
        "append-only": "Append only",
        "strict": "Strict",
    }
    return mapping.get(value, value)
