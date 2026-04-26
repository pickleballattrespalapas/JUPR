import streamlit as st
import pandas as pd
import math
import altair as alt

from jupr_app.ui.helpers import qp_get
from jupr_app.domain.ratings import calculate_hybrid_elo
from jupr_app.domain.constants import DEFAULT_K_FACTOR, MIN_WIN_DELTA_ELO, CAP_LOSER_GAIN_ELO
from jupr_app.ui.components.player_picker import render_player_picker
from jupr_app.ui.layout import page_shell


def render(ctx):
    df_players_all = ctx.df_players_all
    df_players = ctx.df_players_active
    df_leagues = ctx.df_leagues
    df_meta = ctx.df_meta
    id_to_name = ctx.id_to_name
    PUBLIC_MODE = bool(ctx.public_mode)

    mode_label = "Public" if PUBLIC_MODE else "Admin"
    page_shell(
        "🎯 Match Explorer",
        "Preview win odds and projected JUPR movement.",
        mode_label=mode_label,
    )

    # -------- Helpers --------
    def win_label(p: float) -> str:
        if p >= 0.70:
            return "Heavy Favorite"
        if p >= 0.55:
            return "Favored"
        if p >= 0.45:
            return "Toss-up"
        if p >= 0.30:
            return "Underdog"
        return "Heavy Underdog"

    def qp_int(key: str, default: int | None = None) -> int | None:
        v = qp_get(key, "")
        if not v:
            return default
        try:
            return int(v)
        except Exception:
            return default

    # -------- League options --------
    if df_meta is not None and not df_meta.empty and "is_active" in df_meta.columns:
        active_meta = df_meta[df_meta["is_active"] == True].copy()
        league_opts = ["OVERALL"] + sorted(active_meta["league_name"].dropna().unique().tolist())
    else:
        league_opts = ["OVERALL"]

    # Preselect league from URL if provided
    pre_league = qp_get("league", "").strip()
    default_idx = league_opts.index(pre_league) if (pre_league and pre_league in league_opts) else 0

    # -------- Apply deep-link params (only when they change) --------
    sig = "|".join([
        qp_get("ctx", ""),
        qp_get("me", ""),
        qp_get("partner", ""),
        qp_get("opp1", ""),
        qp_get("opp2", ""),
        qp_get("sy", ""),
        qp_get("so", ""),
    ])

    if st.session_state.get("mx_qp_sig", "") != sig:
        # ctx param maps to league context
        ctx_q = qp_get("ctx", "").strip()
        if ctx_q and (ctx_q in league_opts):
            st.session_state["mx_ctx_name"] = ctx_q

        # IDs from URL
        me_q = qp_int("me")
        partner_q = qp_int("partner")
        opp1_q = qp_int("opp1")
        opp2_q = qp_int("opp2")

        if me_q and int(me_q) in id_to_name:
            st.session_state["mx_me_id"] = int(me_q)
        if partner_q and int(partner_q) in id_to_name:
            st.session_state["mx_partner_id"] = int(partner_q)
        if opp1_q and int(opp1_q) in id_to_name:
            st.session_state["mx_opp1_id"] = int(opp1_q)
        if opp2_q and int(opp2_q) in id_to_name:
            st.session_state["mx_opp2_id"] = int(opp2_q)

        st.session_state["mx_sy"] = int(qp_int("sy", 11) or 11)
        st.session_state["mx_so"] = int(qp_int("so", 9) or 9)

        st.session_state["mx_qp_sig"] = sig

    # -------- Context select (do NOT overwrite ctx object) --------
    if "mx_ctx_name" not in st.session_state:
        st.session_state["mx_ctx_name"] = league_opts[default_idx]

    ctx_name = st.selectbox("Rating context", league_opts, index=league_opts.index(st.session_state["mx_ctx_name"]), key="mx_ctx_name")
    st.caption("If you select a league, calculations and the graph use league ratings only (overall ratings shown for reference).")

    # -------- Active players list --------
    if df_players is None or df_players.empty:
        st.info("No active players found.")
        return

    # Build active player ID list (stable identity)
    p = df_players.copy()
    if "id" not in p.columns:
        st.error("Players dataframe is missing 'id' column.")
        return
    p["id"] = p["id"].astype(int)


    # -------- K factor --------
    def get_k_for_context(context_name: str) -> int:
        if context_name == "OVERALL":
            return int(DEFAULT_K_FACTOR)
        if df_meta is None or df_meta.empty:
            return int(DEFAULT_K_FACTOR)
        row = df_meta[df_meta["league_name"] == context_name]
        if not row.empty:
            try:
                return int(row.iloc[0].get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
            except Exception:
                return int(DEFAULT_K_FACTOR)
        return int(DEFAULT_K_FACTOR)

    # -------- Ratings lookups --------
    def get_overall_elo(pid: int) -> float:
        row = df_players_all[df_players_all["id"] == pid]
        if not row.empty:
            return float(row.iloc[0].get("rating", 1200.0) or 1200.0)
        return 1200.0

    def get_league_elo(pid: int, league_name: str) -> float:
        if df_leagues is not None and not df_leagues.empty:
            r = df_leagues[(df_leagues["player_id"] == pid) & (df_leagues["league_name"] == league_name)]
            if not r.empty:
                return float(r.iloc[0].get("rating", 1200.0) or 1200.0)
        return get_overall_elo(pid)

    # -------- Player selection (ID-safe) --------
    st.subheader("Your matchup (doubles)")

    me_id = render_player_picker(
        p,
        label="I am",
        key="mx_me_id",
        default_player_id=st.session_state.get("mx_me_id"),
        include_inactive=False,
    )

    partner_df = p[p["id"].astype(int) != int(me_id)].copy() if me_id else p.copy()
    partner_id = render_player_picker(
        partner_df,
        label="My partner",
        key="mx_partner_id",
        default_player_id=st.session_state.get("mx_partner_id"),
        include_inactive=False,
    )

    exclude_ids = {int(x) for x in [me_id, partner_id] if x}
    opp1_df = p[~p["id"].astype(int).isin(exclude_ids)].copy() if exclude_ids else p.copy()
    opp1_id = render_player_picker(
        opp1_df,
        label="Opponent 1",
        key="mx_opp1_id",
        default_player_id=st.session_state.get("mx_opp1_id"),
        include_inactive=False,
    )

    exclude_ids = {int(x) for x in [me_id, partner_id, opp1_id] if x}
    opp2_df = p[~p["id"].astype(int).isin(exclude_ids)].copy() if exclude_ids else p.copy()
    opp2_id = render_player_picker(
        opp2_df,
        label="Opponent 2",
        key="mx_opp2_id",
        default_player_id=st.session_state.get("mx_opp2_id"),
        include_inactive=False,
    )

    if not (me_id and partner_id and opp1_id and opp2_id):
        st.info("Select yourself, your partner, and both opponents.")
        return

    me_id = int(me_id)
    partner_id = int(partner_id)
    opp1_id = int(opp1_id)
    opp2_id = int(opp2_id)

    me_name = id_to_name.get(me_id, f"#{me_id}")
    partner_name = id_to_name.get(partner_id, f"#{partner_id}")
    opp1_name = id_to_name.get(opp1_id, f"#{opp1_id}")
    opp2_name = id_to_name.get(opp2_id, f"#{opp2_id}")

    # Ratings for display
    r_me_overall = get_overall_elo(me_id)
    r_partner_overall = get_overall_elo(partner_id)
    r_opp1_overall = get_overall_elo(opp1_id)
    r_opp2_overall = get_overall_elo(opp2_id)

    if ctx_name != "OVERALL":
        r_me_ctx = get_league_elo(me_id, ctx_name)
        r_partner_ctx = get_league_elo(partner_id, ctx_name)
        r_opp1_ctx = get_league_elo(opp1_id, ctx_name)
        r_opp2_ctx = get_league_elo(opp2_id, ctx_name)
    else:
        r_me_ctx, r_partner_ctx, r_opp1_ctx, r_opp2_ctx = r_me_overall, r_partner_overall, r_opp1_overall, r_opp2_overall

    team_you_avg = (r_me_ctx + r_partner_ctx) / 2.0
    team_opp_avg = (r_opp1_ctx + r_opp2_ctx) / 2.0

    expected_you = 1.0 / (1.0 + 10 ** ((team_opp_avg - team_you_avg) / 400.0))
    label = win_label(float(expected_you))
    k_val = get_k_for_context(ctx_name)

    # Ratings table
    rows = [
        {"Role": "You", "Player": me_name, "Overall JUPR": r_me_overall / 400.0, "League JUPR": (r_me_ctx / 400.0) if ctx_name != "OVERALL" else None},
        {"Role": "Partner", "Player": partner_name, "Overall JUPR": r_partner_overall / 400.0, "League JUPR": (r_partner_ctx / 400.0) if ctx_name != "OVERALL" else None},
        {"Role": "Opponent 1", "Player": opp1_name, "Overall JUPR": r_opp1_overall / 400.0, "League JUPR": (r_opp1_ctx / 400.0) if ctx_name != "OVERALL" else None},
        {"Role": "Opponent 2", "Player": opp2_name, "Overall JUPR": r_opp2_overall / 400.0, "League JUPR": (r_opp2_ctx / 400.0) if ctx_name != "OVERALL" else None},
    ]
    df_view = pd.DataFrame(rows)

    show_cols = ["Role", "Player", "Overall JUPR"]
    if ctx_name != "OVERALL":
        show_cols.append("League JUPR")

    st.dataframe(
        df_view[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Overall JUPR": st.column_config.NumberColumn(format="%.3f"),
            "League JUPR": st.column_config.NumberColumn(format="%.3f"),
        },
    )

    st.divider()

    h1, h2 = st.columns([2, 1])
    with h1:
        st.markdown(f"## Your Expected Win Rate = **{expected_you*100:.0f}%**")
        st.caption(f"{label} • Context: {ctx_name}")
        if ctx_name != "OVERALL":
            st.info("Graph + projected movement below are computed using LEAGUE ratings only. Overall ratings above are for reference.")
    with h2:
        st.metric("Opponents win %", f"{(1.0-expected_you)*100:.0f}%")

    st.divider()

    st.subheader("Hypothetical Score")

    if "mx_sy" not in st.session_state:
        st.session_state["mx_sy"] = 11
    if "mx_so" not in st.session_state:
        st.session_state["mx_so"] = 9

    scol1, scol2 = st.columns(2)
    with scol1:
        s_you = st.number_input("Your points", min_value=0, max_value=99, step=1, key="mx_sy", value=int(st.session_state.get("mx_sy", 11)))
    with scol2:
        s_opp = st.number_input("Opp points", min_value=0, max_value=99, step=1, key="mx_so", value=int(st.session_state.get("mx_so", 9)))

    d_you_elo, d_opp_elo = calculate_hybrid_elo(
        team_you_avg,
        team_opp_avg,
        int(s_you),
        int(s_opp),
        k_factor=int(k_val),
        min_win_delta=float(MIN_WIN_DELTA_ELO),
        cap_loser_gain=float(CAP_LOSER_GAIN_ELO),
    )

    d_you_jupr = float(d_you_elo) / 400.0
    d_opp_jupr = float(d_opp_elo) / 400.0

    total_pts = int(s_you) + int(s_opp)
    share_you = None
    if total_pts > 0 and int(s_you) != int(s_opp):
        share_you = float(s_you) / float(total_pts)

    beat_pp = None
    if share_you is not None:
        beat_pp = (share_you - expected_you) * 100.0

    t1, t2, t3, t4 = st.columns(4)

    if beat_pp is not None:
        t1.metric("Beat expectation", f"{beat_pp:+.0f} pp")
        t1.caption(f"Your share {share_you*100:.1f}% vs expected {expected_you*100:.1f}%")
    else:
        t1.metric("Beat expectation", "—")
        t1.caption("No movement on ties / empty scores.")

    t2.metric("Your projected JUPR change", f"{d_you_jupr:+.4f}")
    t2.caption(f"Context: {ctx_name}")

    t3.metric("Partner projected JUPR change", f"{d_you_jupr:+.4f}")
    t3.caption("Same delta as you (team-based update).")

    t4.metric("Opponents projected JUPR change", f"{d_opp_jupr:+.4f}")
    t4.caption("Preview only — nothing is saved.")

    st.divider()

    def share_to_score11_label(share: float) -> str:
        if share is None:
            return "—"
        if abs(share - 0.5) < 1e-12:
            return "11–11"
        if share < 0.5:
            my_pts = int(round(11.0 * share / (1.0 - share)))
            return f"{my_pts}–11"
        opp_pts = int(round(11.0 * (1.0 - share) / share))
        return f"11–{opp_pts}"

    st.subheader("Rating Impact Predictor")

    share_chart = None
    if total_pts > 0:
        share_chart = float(s_you) / float(total_pts)

    def delta_you_from_share(share: float, expected: float, k: float) -> float:
        if abs(share - 0.5) < 1e-12:
            return 0.0
        d = float(k) * 2.0 * (float(share) - float(expected))
        if share > 0.5:
            if d <= 0:
                d = float(MIN_WIN_DELTA_ELO)
        else:
            if d > 0:
                d = min(d, float(CAP_LOSER_GAIN_ELO))
        return d

    xs, ys, score11s = [], [], []
    for i in range(0, 101):
        sh = i / 100.0
        xs.append(sh)
        ys.append(delta_you_from_share(sh, expected_you, k_val) / 400.0)
        score11s.append(share_to_score11_label(sh))

    curve_df = pd.DataFrame({"share": xs, "delta": ys, "score11": score11s})

    tick_vals = [
        0.0,
        3.0 / 14.0,
        6.0 / 17.0,
        9.0 / 20.0,
        0.5,
        11.0 / 20.0,
        11.0 / 17.0,
        11.0 / 14.0,
        1.0,
    ]

    label_expr = (
        "datum.value==0.5 ? '11–11' : "
        "(datum.value<0.5 ? "
        "(round(11*datum.value/(1-datum.value)) + '–11') : "
        "('11–' + round(11*(1-datum.value)/datum.value)))"
    )

    base = (
        alt.Chart(curve_df)
        .mark_line()
        .encode(
            x=alt.X("share:Q", title="Score (to 11 scale)", axis=alt.Axis(values=tick_vals, labelExpr=label_expr)),
            y=alt.Y("delta:Q", title=f"Projected JUPR change (you) — {ctx_name}", axis=alt.Axis(format="+.4f")),
            tooltip=[
                alt.Tooltip("score11:N", title="Score (to 11)"),
                alt.Tooltip("delta:Q", title="Δ JUPR", format="+.4f"),
            ],
        )
    )

    layers = [base]

    exp_df = pd.DataFrame({"share": [float(expected_you)], "score11": [share_to_score11_label(float(expected_you))]})
    exp_rule = (
        alt.Chart(exp_df)
        .mark_rule(strokeDash=[6, 4])
        .encode(
            x="share:Q",
            tooltip=[
                alt.Tooltip("score11:N", title="Expected (to 11)"),
                alt.Tooltip("share:Q", title="Expected share", format=".3f"),
            ],
        )
    )
    layers.append(exp_rule)

    if share_chart is not None:
        sel_df = pd.DataFrame(
            {
                "share": [float(share_chart)],
                "delta": [float(d_you_jupr)],
                "score_actual": [f"{int(s_you)}–{int(s_opp)}"],
                "score11": [share_to_score11_label(float(share_chart))],
            }
        )

        sel_pt = (
            alt.Chart(sel_df)
            .mark_point(size=140, filled=True)
            .encode(
                x="share:Q",
                y="delta:Q",
                tooltip=[
                    alt.Tooltip("score_actual:N", title="Actual score"),
                    alt.Tooltip("score11:N", title="Equivalent (to 11)"),
                    alt.Tooltip("delta:Q", title="Δ JUPR", format="+.4f"),
                ],
            )
        )
        layers.append(sel_pt)

    st.altair_chart(alt.layer(*layers).properties(height=360).interactive(), use_container_width=True)
