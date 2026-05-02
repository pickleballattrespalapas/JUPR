from __future__ import annotations

import random
import re
from datetime import datetime
from uuid import uuid4

import pandas as pd
import streamlit as st

from jupr_app.domain.schedule import get_match_schedule
from jupr_app.services import ServiceContext, submit_match_batch
from jupr_app.data.load import load_data
from jupr_app.ui.layout import page_shell


def mb_expected_share(t1_avg: float, t2_avg: float) -> float:
    try:
        return 1.0 / (1.0 + 10 ** ((float(t2_avg) - float(t1_avg)) / 400.0))
    except Exception:
        return 0.5


def mb_expected_scoreline_from_share(p: float, goal_points: int = 11) -> tuple[int, int, int]:
    if p is None:
        return goal_points, goal_points, 0

    p = max(0.0001, min(0.9999, float(p)))
    if abs(p - 0.5) < 1e-12:
        return goal_points, goal_points, 0

    if p > 0.5:
        opp = int(round(goal_points * (1.0 - p) / p))
        opp = max(0, min(goal_points, opp))
        return goal_points, opp, goal_points - opp

    me = int(round(goal_points * p / (1.0 - p)))
    me = max(0, min(goal_points, me))
    return me, goal_points, me - goal_points


def mb_get_ctx_elo(pid: int, ctx_name: str, df_players_all: pd.DataFrame, df_leagues: pd.DataFrame) -> float:
    row = df_players_all[df_players_all["id"] == int(pid)]
    overall = 1200.0
    if not row.empty:
        overall = float(row.iloc[0].get("rating", 1200.0) or 1200.0)

    if ctx_name == "OVERALL":
        return overall

    if df_leagues is not None and not df_leagues.empty:
        league_row = df_leagues[
            (df_leagues["player_id"] == int(pid)) & (df_leagues["league_name"] == str(ctx_name))
        ]
        if not league_row.empty:
            return float(league_row.iloc[0].get("rating", overall) or overall)

    return overall


def mb_build_schedule_df_8(player_ids: list[int], id_to_name: dict[int, str]) -> pd.DataFrame:
    schedule = get_match_schedule("8-Player", player_ids)
    rows = []
    for idx, match in enumerate(schedule, start=1):
        desc = str(match.get("desc", ""))
        rnd = None
        court = None
        rnd_match = re.search(r"Rnd\s*(\d+)", desc)
        if rnd_match:
            rnd = int(rnd_match.group(1))
        court_match = re.search(r"Ct\s*(\d+)", desc)
        if court_match:
            court = int(court_match.group(1))

        t1 = match.get("t1", [])
        t2 = match.get("t2", [])
        t1_p1, t1_p2 = int(t1[0]), int(t1[1])
        t2_p1, t2_p2 = int(t2[0]), int(t2[1])

        t1_name = f"{id_to_name.get(t1_p1, f'#{t1_p1}')} / {id_to_name.get(t1_p2, f'#{t1_p2}')}"
        t2_name = f"{id_to_name.get(t2_p1, f'#{t2_p1}')} / {id_to_name.get(t2_p2, f'#{t2_p2}')}"

        rows.append(
            {
                "Match #": idx,
                "Round": rnd or "",
                "Court": court or "",
                "Team 1": t1_name,
                "S1": 0,
                "S2": 0,
                "Team 2": t2_name,
                "t1_p1": t1_p1,
                "t1_p2": t1_p2,
                "t2_p1": t2_p1,
                "t2_p2": t2_p2,
            }
        )

    return pd.DataFrame(rows)


def mb_add_expectations(schedule_df: pd.DataFrame, ctx_ratings: dict[int, float], goal_points: int) -> pd.DataFrame:
    df = schedule_df.copy()
    exp_p = []
    exp_margin = []
    exp_score = []

    for _, row in df.iterrows():
        t1_avg = (
            float(ctx_ratings.get(int(row["t1_p1"]), 1200.0))
            + float(ctx_ratings.get(int(row["t1_p2"]), 1200.0))
        ) / 2.0
        t2_avg = (
            float(ctx_ratings.get(int(row["t2_p1"]), 1200.0))
            + float(ctx_ratings.get(int(row["t2_p2"]), 1200.0))
        ) / 2.0

        p = mb_expected_share(t1_avg, t2_avg)
        s1, s2, margin = mb_expected_scoreline_from_share(p, goal_points=goal_points)
        exp_p.append(float(p))
        exp_margin.append(int(margin))
        exp_score.append(f"{s1}–{s2}")

    df["exp_p_t1"] = exp_p
    df["Expected Win% for Team1"] = [round(p * 100.0, 1) for p in exp_p]
    df["Expected Score (to 11 scale)"] = exp_score
    df["Expected Margin"] = exp_margin
    return df


def mb_compute_totals(
    schedule_df: pd.DataFrame,
    win_rate: float,
    point_rate: float,
) -> tuple[pd.DataFrame, list[int]]:
    player_ids: list[int] = sorted(
        set(
            schedule_df[["t1_p1", "t1_p2", "t2_p1", "t2_p2"]]
            .astype(int)
            .values
            .ravel()
            .tolist()
        )
    )

    stats = {
        pid: {
            "Player": pid,
            "GP": 0,
            "Wins": 0,
            "Losses": 0,
            "PD": 0,
            "Exp Wins": 0.0,
            "Exp PD": 0.0,
        }
        for pid in player_ids
    }

    tie_matches: list[int] = []

    for _, row in schedule_df.iterrows():
        t1 = [int(row["t1_p1"]), int(row["t1_p2"])]
        t2 = [int(row["t2_p1"]), int(row["t2_p2"])]

        p_t1 = float(row.get("exp_p_t1", 0.5))
        margin = int(row.get("Expected Margin", 0) or 0)
        for pid in t1:
            stats[pid]["Exp Wins"] += p_t1
            stats[pid]["Exp PD"] += margin
        for pid in t2:
            stats[pid]["Exp Wins"] += 1.0 - p_t1
            stats[pid]["Exp PD"] += -margin

        s1 = int(row.get("S1", 0) or 0)
        s2 = int(row.get("S2", 0) or 0)
        if (s1 + s2) <= 0:
            continue
        if s1 == s2:
            tie_matches.append(int(row.get("Match #", 0)))
            continue

        t1_win = s1 > s2
        pd = s1 - s2
        for pid in t1:
            stats[pid]["GP"] += 1
            stats[pid]["PD"] += pd
            if t1_win:
                stats[pid]["Wins"] += 1
            else:
                stats[pid]["Losses"] += 1
        for pid in t2:
            stats[pid]["GP"] += 1
            stats[pid]["PD"] += -pd
            if t1_win:
                stats[pid]["Losses"] += 1
            else:
                stats[pid]["Wins"] += 1

    rows = []
    for pid, s in stats.items():
        win_delta = float(s["Wins"]) - float(s["Exp Wins"])
        pd_delta = float(s["PD"]) - float(s["Exp PD"])
        money = (win_delta * float(win_rate)) + (pd_delta * float(point_rate))
        rows.append(
            {
                "Player": pid,
                "GP": s["GP"],
                "Wins": s["Wins"],
                "Losses": s["Losses"],
                "PD": s["PD"],
                "Exp Wins": s["Exp Wins"],
                "Exp PD": s["Exp PD"],
                "Win Δ": win_delta,
                "PD Δ": pd_delta,
                "Net $": money,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["Net $"] = df["Net $"].round(2)
        drift = round(-float(df["Net $"].sum()), 2)
        if abs(drift) >= 0.01:
            idx = df["Net $"].abs().idxmax()
            df.loc[idx, "Net $"] = round(float(df.loc[idx, "Net $"]) + drift, 2)

    return df, tie_matches


def render(ctx):
    mode_label = "Public" if bool(ctx.public_mode) else "Admin"
    page_shell("💰 Moneyball", "Run an 8-player Moneyball round robin.", mode_label=mode_label)

    if not bool(getattr(ctx, "admin_logged_in", False)):
        st.error("Admin login required.")
        st.stop()

    df_players_all = ctx.df_players_all
    df_players = ctx.df_players_active
    df_leagues = ctx.df_leagues
    df_meta = ctx.df_meta
    id_to_name = ctx.id_to_name
    name_to_id = ctx.name_to_id
    supabase = ctx.supabase
    club_id = str(ctx.club_id)

    if df_players is None or df_players.empty:
        st.info("No active players found.")
        return
    if "id" not in df_players.columns:
        st.error("Players dataframe is missing 'id' column.")
        return

    if df_meta is not None and not df_meta.empty and "is_active" in df_meta.columns:
        active_meta = df_meta[df_meta["is_active"] == True].copy()
        ctx_options = ["OVERALL"] + sorted(active_meta["league_name"].dropna().unique().tolist())
    else:
        ctx_options = ["OVERALL"]

    league_store_options = ["Moneyball"] + [x for x in ctx_options if x != "OVERALL"]

    if "mb_ctx_name" not in st.session_state:
        st.session_state["mb_ctx_name"] = ctx_options[0]
    if "mb_win_rate" not in st.session_state:
        st.session_state["mb_win_rate"] = 5.0
    if "mb_point_rate" not in st.session_state:
        st.session_state["mb_point_rate"] = 2.0
    if "mb_league_name_to_store" not in st.session_state:
        st.session_state["mb_league_name_to_store"] = league_store_options[0]
    if "mb_week_tag" not in st.session_state:
        st.session_state["mb_week_tag"] = f"Moneyball {datetime.now().date().isoformat()}"
    if "mb_match_type" not in st.session_state:
        st.session_state["mb_match_type"] = "Moneyball RR"

    st.subheader("Setup")
    ctx_name = st.selectbox(
        "Rating context",
        ctx_options,
        index=ctx_options.index(st.session_state["mb_ctx_name"]),
        key="mb_ctx_name",
    )
    st.caption("Expectations are computed from frozen ratings at event start.")

    r1, r2 = st.columns(2)
    r1.number_input("WIN_RATE", min_value=0.0, step=0.5, key="mb_win_rate")
    r2.number_input("POINT_RATE", min_value=0.0, step=0.5, key="mb_point_rate")

    c1, c2, c3 = st.columns(3)
    c1.selectbox("league_name_to_store", league_store_options, key="mb_league_name_to_store")
    c2.text_input("week_tag", key="mb_week_tag")
    c3.text_input("match_type", key="mb_match_type")

    active_ids = sorted(df_players["id"].astype(int).tolist())

    def fmt_pid(pid: int) -> str:
        return f"{id_to_name.get(int(pid), f'#{int(pid)}')}  (#{int(pid)})"

    selected_ids = st.multiselect(
        "Select exactly 8 players",
        options=active_ids,
        default=st.session_state.get("mb_selected_players", []),
        format_func=fmt_pid,
        key="mb_selected_players",
    )

    if "mb_player_order" not in st.session_state:
        st.session_state["mb_player_order"] = []

    if selected_ids:
        selected_ids = [int(x) for x in selected_ids]
        if set(selected_ids) != set(st.session_state["mb_player_order"]):
            st.session_state["mb_player_order"] = selected_ids

    order_ids = st.session_state.get("mb_player_order", [])

    st.markdown("**Player Order (P1..P8)**")
    if len(selected_ids) == 8:
        btn1, btn2 = st.columns([1, 1])
        if btn1.button("Sort by rating (context)"):
            order_ids = sorted(
                selected_ids,
                key=lambda pid: mb_get_ctx_elo(pid, ctx_name, df_players_all, df_leagues),
                reverse=True,
            )
            st.session_state["mb_player_order"] = order_ids
            for i, pid in enumerate(order_ids):
                st.session_state[f"mb_order_{i}"] = pid

        if btn2.button("Shuffle"):
            order_ids = selected_ids[:]
            random.shuffle(order_ids)
            st.session_state["mb_player_order"] = order_ids
            for i, pid in enumerate(order_ids):
                st.session_state[f"mb_order_{i}"] = pid

        order_cols = st.columns(4)
        for i in range(8):
            col = order_cols[i % 4]
            with col:
                st.selectbox(
                    f"P{i+1}",
                    options=selected_ids,
                    format_func=fmt_pid,
                    key=f"mb_order_{i}",
                )

        order_ids = [int(st.session_state.get(f"mb_order_{i}", selected_ids[i])) for i in range(8)]
        st.session_state["mb_player_order"] = order_ids

        order_df = pd.DataFrame(
            {
                "Order": list(range(1, 9)),
                "Player": [id_to_name.get(pid, f"#{pid}") for pid in order_ids],
            }
        )
        st.dataframe(order_df, hide_index=True, use_container_width=True)
    else:
        st.info("Select 8 players to set the player order.")

    order_unique = len(set(order_ids)) == 8
    if len(selected_ids) == 8 and not order_unique:
        st.error("Player order must contain each player exactly once.")

    current_sig = (ctx_name, tuple(order_ids))
    if st.session_state.get("mb_event_signature") and st.session_state.get("mb_event_signature") != current_sig:
        st.warning("Context or player list changed. Starting a new event will reset scores.")

    def reset_event_state() -> None:
        for key in [
            "mb_schedule_df",
            "mb_ctx_ratings",
            "mb_saved",
            "mb_event_id",
            "mb_event_signature",
        ]:
            st.session_state.pop(key, None)

    start_disabled = (len(selected_ids) != 8) or (not order_unique)
    if st.button("Start Event", disabled=start_disabled):
        reset_event_state()
        ctx_ratings = {
            pid: mb_get_ctx_elo(pid, ctx_name, df_players_all, df_leagues)
            for pid in order_ids
        }
        schedule_df = mb_build_schedule_df_8(order_ids, id_to_name)
        schedule_df = mb_add_expectations(schedule_df, ctx_ratings, goal_points=11)
        st.session_state["mb_ctx_ratings"] = ctx_ratings
        st.session_state["mb_schedule_df"] = schedule_df
        st.session_state["mb_saved"] = False
        st.session_state["mb_event_id"] = str(uuid4())
        st.session_state["mb_event_signature"] = current_sig
        st.rerun()

    if "mb_schedule_df" not in st.session_state:
        return

    st.divider()
    st.subheader("Live Scoring")

    schedule_df = st.session_state["mb_schedule_df"].copy()
    display_cols = [
        "Match #",
        "Round",
        "Court",
        "Team 1",
        "S1",
        "S2",
        "Team 2",
        "Expected Win% for Team1",
        "Expected Score (to 11 scale)",
        "Expected Margin",
    ]
    edited = st.data_editor(
        schedule_df[display_cols],
        hide_index=True,
        use_container_width=True,
        column_config={
            "S1": st.column_config.NumberColumn(min_value=0, step=1),
            "S2": st.column_config.NumberColumn(min_value=0, step=1),
            "Expected Win% for Team1": st.column_config.NumberColumn(format="%.1f"),
        },
        disabled=[
            "Match #",
            "Round",
            "Court",
            "Team 1",
            "Team 2",
            "Expected Win% for Team1",
            "Expected Score (to 11 scale)",
            "Expected Margin",
        ],
        key="mb_scores_editor",
    )

    schedule_df["S1"] = edited["S1"].fillna(0).astype(int)
    schedule_df["S2"] = edited["S2"].fillna(0).astype(int)
    st.session_state["mb_schedule_df"] = schedule_df

    st.divider()
    st.subheader("Live Standings + Moneyball Leaderboard")

    standings_df, tie_matches = mb_compute_totals(
        schedule_df,
        win_rate=float(st.session_state["mb_win_rate"]),
        point_rate=float(st.session_state["mb_point_rate"]),
    )

    if tie_matches:
        tie_list = ", ".join([f"#{m}" for m in tie_matches])
        st.warning(f"Ties are ignored for settlement. Matches with ties: {tie_list}.")

    standings_df["Player"] = standings_df["Player"].apply(lambda pid: id_to_name.get(int(pid), f"#{int(pid)}"))
    standings_df = standings_df.sort_values(by="Net $", ascending=False)

    st.dataframe(
        standings_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Exp Wins": st.column_config.NumberColumn(format="%.2f"),
            "Exp PD": st.column_config.NumberColumn(format="%.1f"),
            "Win Δ": st.column_config.NumberColumn(format="%.2f"),
            "PD Δ": st.column_config.NumberColumn(format="%.1f"),
            "Net $": st.column_config.NumberColumn(format="$%.2f"),
        },
    )

    owes = standings_df[standings_df["Net $"] < 0][["Player", "Net $"]]
    receives = standings_df[standings_df["Net $"] > 0][["Player", "Net $"]]

    p1, p2 = st.columns(2)
    with p1:
        st.markdown("**Owes**")
        if owes.empty:
            st.caption("—")
        else:
            for _, row in owes.iterrows():
                st.caption(f"{row['Player']}: ${abs(row['Net $']):.2f}")
    with p2:
        st.markdown("**Receives**")
        if receives.empty:
            st.caption("—")
        else:
            for _, row in receives.iterrows():
                st.caption(f"{row['Player']}: ${row['Net $']:.2f}")

    st.divider()
    st.subheader("Finalize Night (Save to JUPR)")

    scored = schedule_df[(schedule_df["S1"] + schedule_df["S2"]) > 0]
    scored_valid = scored[scored["S1"] != scored["S2"]]
    st.caption(f"Scored matches: {len(scored_valid)} (ties ignored)")

    if st.session_state.get("mb_saved", False):
        st.success("Scores already saved for this event.")
    else:
        if st.button("Finalize Night (Save to JUPR)"):
            if scored_valid.empty:
                st.warning("No valid scored matches to save.")
            else:
                payload = []
                for _, row in scored_valid.iterrows():
                    payload.append(
                        {
                            "t1_p1": int(row["t1_p1"]),
                            "t1_p2": int(row["t1_p2"]),
                            "t2_p1": int(row["t2_p1"]),
                            "t2_p2": int(row["t2_p2"]),
                            "s1": int(row["S1"]),
                            "s2": int(row["S2"]),
                            "date": datetime.now(),
                            "league": str(st.session_state["mb_league_name_to_store"]),
                            "match_type": str(st.session_state["mb_match_type"]),
                            "week_tag": str(st.session_state["mb_week_tag"]),
                            "is_popup": False,
                        }
                    )

                service_ctx = ServiceContext(
                    supabase=supabase,
                    club_id=club_id,
                    actor_email=st.session_state.get("admin_email"),
                    actor_role=st.session_state.get("admin_role"),
                    source="moneyball",
                    public_base_url=st.session_state.get("public_base_url"),
                )
                result = submit_match_batch(
                    service_ctx,
                    payload,
                    name_to_id=name_to_id,
                    df_players_all=df_players_all,
                    df_leagues=df_leagues,
                    df_meta=df_meta,
                )
                if not result.ok:
                    st.error("; ".join(result.errors) or "Unable to save matches.")
                    return
                st.session_state["mb_saved"] = True
                st.session_state["mb_event_id"] = st.session_state.get("mb_event_id") or str(uuid4())
                load_data(supabase, club_id, match_limit=5000)
                st.session_state["force_data_refresh"] = True
                st.success("Saved to JUPR. Leaderboards will refresh on next load.")

    if not st.session_state.get("mb_saved", False):
        return

    st.divider()
    st.subheader("Printout")

    summary_cols = ["Player", "Wins", "Losses", "PD", "Exp Wins", "Exp PD", "Win Δ", "PD Δ", "Net $"]
    summary_df = standings_df[summary_cols].copy()

    header_html = (
        f"<h2>Moneyball Round Robin</h2>"
        f"<p><strong>Date:</strong> {datetime.now().date().isoformat()}<br/>"
        f"<strong>Context:</strong> {ctx_name}<br/>"
        f"<strong>Rates:</strong> WIN_RATE {float(st.session_state['mb_win_rate']):.2f}, "
        f"POINT_RATE {float(st.session_state['mb_point_rate']):.2f}</p>"
    )

    table_html = summary_df.to_html(index=False, float_format="%.2f", border=0)
    html_body = (
        "<html><head><style>"
        "body{font-family:Arial, sans-serif;"
        "color:var(--text-primary);"
        "background:var(--bg);}"
        "table{border-collapse:collapse;width:100%;"
        "background:var(--panel);color:var(--text-primary);}"
        "th,td{border:1px solid var(--border);padding:6px;text-align:center;}"
        "th{background:var(--table-stripe);color:var(--text-muted);}"
        "</style></head><body>"
        f"{header_html}{table_html}"
        "</body></html>"
    )

    st.markdown(html_body, unsafe_allow_html=True)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, file_name="moneyball_summary.csv", mime="text/csv")
    st.download_button("Download printable HTML", html_body.encode("utf-8"), file_name="moneyball_summary.html", mime="text/html")
