import streamlit as st
import pandas as pd

from jupr_app.ui.helpers import qp_get, build_match_explorer_link

try:
    import altair as alt
except Exception:
    alt = None


# -------------------------
# Data fetch (with snapshot fallback)
# -------------------------
@st.cache_data(ttl=30)
def fetch_player_matches(_supabase, club_id: str, pid: int, limit: int = 400) -> pd.DataFrame:
    """
    Tries to fetch snapshot columns (t*_r / t*_r_end). If your matches table
    doesn't have them, it falls back gracefully to a base select.
    """
    base_select = (
        "id,date,league,match_type,score_t1,score_t2,"
        "t1_p1,t1_p2,t2_p1,t2_p2,"
        "elo_delta"
    )

    snap_select = (
        base_select
        + ",t1_p1_r,t1_p1_r_end,t1_p2_r,t1_p2_r_end,"
          "t2_p1_r,t2_p1_r_end,t2_p2_r,t2_p2_r_end"
    )

    def _run(select_cols: str):
        resp = (
            _supabase.table("matches")
            .select(select_cols)
            .eq("club_id", str(club_id))
            .or_(f"t1_p1.eq.{pid},t1_p2.eq.{pid},t2_p1.eq.{pid},t2_p2.eq.{pid}")
            .order("date", desc=True)
            .order("id", desc=True)
            .limit(int(limit))
            .execute()
        )
        return pd.DataFrame(resp.data or [])

    # Try snapshots first; fall back if columns don't exist
    try:
        df = _run(snap_select)
        return df
    except Exception:
        return _run(base_select)


def render(ctx):
    st.header("🔍 Player Search")

    df_players_all = ctx.df_players_all
    if df_players_all is None or df_players_all.empty:
        st.info("No players found.")
        return

    # Active-only list for selection
    players_df = df_players_all.copy()
    if "active" in players_df.columns:
        players_df = players_df[players_df["active"] == True].copy()

    if players_df.empty:
        st.info("No active players.")
        return

    players_df["id"] = players_df["id"].astype(int)

    # Deep-link support: ?pid=<id> (apply once per pid value)
    pid_q = qp_get("pid", "").strip()
    pid_sig = f"pid:{pid_q}" if pid_q else ""
    last_sig = st.session_state.get("player_pid_sig_applied", "")

    if pid_q.isdigit() and pid_sig != last_sig:
        pid_int = int(pid_q)
        hit = players_df[players_df["id"] == pid_int]
        if not hit.empty:
            st.session_state["player_search_id"] = int(hit.iloc[0]["id"])
            try:
                st.query_params.pop("pid", None)
            except Exception:
                pass
        st.session_state["player_pid_sig_applied"] = pid_sig

    # Sort and build ID-based selector (safe for duplicate names)
    players_df = players_df.sort_values("name").copy()
    options = [""] + players_df["id"].tolist()

    def _fmt(x):
        if x == "":
            return ""
        r = players_df[players_df["id"] == int(x)]
        if r.empty:
            return f"#{x}"
        return f"{str(r.iloc[0]['name'])}  (#{int(x)})"

    pick_id = st.selectbox(
        "Select a player",
        options=options,
        format_func=_fmt,
        key="player_search_id",
    )

    if pick_id == "":
        st.info("Select a player to view details.")
        return

    pid = int(pick_id)
    row = players_df[players_df["id"] == pid].iloc[0]
    pick_name = str(row["name"])

    # current OVERALL in Elo stored as rating; JUPR is Elo/400.0
    try:
        current_overall_elo = float(row.get("rating", 1200.0) or 1200.0)
    except Exception:
        current_overall_elo = 1200.0
    current_jupr = current_overall_elo / 400.0

    c1, c2 = st.columns(2)
    c1.metric("Player", pick_name)
    c2.metric("Overall JUPR", f"{current_jupr:.3f}")

    st.divider()

    _supabase = ctx.supabase
    club_id = ctx.club_id
    matches = fetch_player_matches(_supabase, club_id, pid, limit=400)

    if matches.empty:
        st.info("No matches recorded for this player.")
        return

    # -------------------------
    # Helpers
    # -------------------------
    def _safe_int(x, default=None):
        try:
            if x is None or str(x).strip() == "":
                return default
            return int(x)
        except Exception:
            return default

    def _safe_float(x, default=None):
        try:
            if x is None or str(x).strip() == "":
                return default
            return float(x)
        except Exception:
            return default

    def score_for_player(r):
        try:
            t1p1 = _safe_int(r.get("t1_p1"))
            t1p2 = _safe_int(r.get("t1_p2"))
            s1 = _safe_int(r.get("score_t1"), 0) or 0
            s2 = _safe_int(r.get("score_t2"), 0) or 0
        except Exception:
            return ""
        if t1p1 == pid or t1p2 == pid:
            return f"{s1}-{s2}"
        return f"{s2}-{s1}"

    def explain_link(r):
        try:
            t1p1 = _safe_int(r.get("t1_p1"))
            t1p2 = _safe_int(r.get("t1_p2"))
            t2p1 = _safe_int(r.get("t2_p1"))
            t2p2 = _safe_int(r.get("t2_p2"))
            s1 = _safe_int(r.get("score_t1"), 0) or 0
            s2 = _safe_int(r.get("score_t2"), 0) or 0
        except Exception:
            return ""

        if t1p1 == pid or t1p2 == pid:
            partner = t1p1 if t1p2 == pid else t1p2
            opp1, opp2 = t2p1, t2p2
            sy, so = s1, s2
        elif t2p1 == pid or t2p2 == pid:
            partner = t2p1 if t2p2 == pid else t2p2
            opp1, opp2 = t1p1, t1p2
            sy, so = s2, s1
        else:
            return ""

        return build_match_explorer_link(
            ctx="OVERALL",
            me=int(pid),
            partner=int(partner),
            opp1=int(opp1),
            opp2=int(opp2),
            sy=int(sy),
            so=int(so),
            public=bool(ctx.public_mode),
        )

    def get_overall_snap(r: dict, pid_: int):
        """
        Read OVERALL start/end Elo from snapshot columns if present.
        Returns (start_elo, end_elo) or (None, None).
        """
        pid_ = int(pid_)
        try:
            t1p1 = _safe_int(r.get("t1_p1"))
            t1p2 = _safe_int(r.get("t1_p2"))
            t2p1 = _safe_int(r.get("t2_p1"))
            t2p2 = _safe_int(r.get("t2_p2"))
        except Exception:
            return None, None

        if t1p1 == pid_:
            return _safe_float(r.get("t1_p1_r")), _safe_float(r.get("t1_p1_r_end"))
        if t1p2 == pid_:
            return _safe_float(r.get("t1_p2_r")), _safe_float(r.get("t1_p2_r_end"))
        if t2p1 == pid_:
            return _safe_float(r.get("t2_p1_r")), _safe_float(r.get("t2_p1_r_end"))
        if t2p2 == pid_:
            return _safe_float(r.get("t2_p2_r")), _safe_float(r.get("t2_p2_r_end"))
        return None, None

    def signed_delta_from_elo_delta(r: dict, pid_: int):
        """
        Legacy fallback when snapshots don't exist:
        Determine the sign of elo_delta from the player's perspective.
        Returns delta_elo (float) or None.
        """
        pid_ = int(pid_)
        raw = _safe_float(r.get("elo_delta"), None)
        if raw is None:
            return None

        s1 = _safe_int(r.get("score_t1"), 0) or 0
        s2 = _safe_int(r.get("score_t2"), 0) or 0
        if s1 == s2:
            return 0.0

        t1 = {_safe_int(r.get("t1_p1")), _safe_int(r.get("t1_p2"))}
        t2 = {_safe_int(r.get("t2_p1")), _safe_int(r.get("t2_p2"))}
        on_t1 = pid_ in t1
        on_t2 = pid_ in t2
        if not on_t1 and not on_t2:
            return None

        winner_team = 1 if s1 > s2 else 2
        my_team = 1 if on_t1 else 2
        if winner_team == my_team:
            return abs(float(raw))
        return -abs(float(raw))

    # -------------------------
    # Build display rows + rating series
    # -------------------------
    matches = matches.copy()
    matches["Score"] = matches.apply(score_for_player, axis=1)
    matches["Explain"] = matches.apply(explain_link, axis=1)

    # Normalize date
    matches["date_dt"] = pd.to_datetime(matches.get("date", None), errors="coerce", utc=True)
    matches = matches.dropna(subset=["date_dt"]).copy()

    # Process rows newest -> oldest (as fetched), then we’ll compute series
    processed = []
    for _, r in matches.iterrows():
        r = dict(r)

        start_elo, end_elo = get_overall_snap(r, pid)
        after_jupr = None
        delta_jupr = None

        if start_elo is not None and end_elo is not None:
            try:
                delta_jupr = (float(end_elo) - float(start_elo)) / 400.0
                after_jupr = float(end_elo) / 400.0
            except Exception:
                delta_jupr = None
                after_jupr = None
        else:
            d_elo = signed_delta_from_elo_delta(r, pid)
            if d_elo is not None:
                delta_jupr = float(d_elo) / 400.0

        processed.append(
            {
                "id": _safe_int(r.get("id")),
                "Date": r.get("date_dt"),
                "league": str(r.get("league", "") or "").strip(),
                "match_type": str(r.get("match_type", "") or "").strip(),
                "Score": r.get("Score", ""),
                "Overall Δ": delta_jupr,
                "Overall After": after_jupr,
                "Explain": r.get("Explain", ""),
            }
        )

    df = pd.DataFrame(processed)
    if df.empty:
        st.info("No matches available.")
        return

    # Sort oldest -> newest for chart logic
    df = df.sort_values(["Date", "id"], ascending=[True, True]).reset_index(drop=True)

    # Backfill missing "Overall After" by walking forward from the earliest known,
    # or (most common) walking backward from current overall.
    #
    # Strategy:
    #   1) If we have ANY after values, fill forward where possible.
    #   2) Otherwise, compute after values by starting from current_jupr and walking backward using deltas.
    df["Overall Δ"] = pd.to_numeric(df["Overall Δ"], errors="coerce")
    df["Overall After"] = pd.to_numeric(df["Overall After"], errors="coerce")

    if df["Overall After"].notna().any():
        # Fill missing after values using deltas if adjacent known values exist
        # Forward fill: if previous after exists and delta exists, derive current after
        for i in range(len(df)):
            if pd.isna(df.loc[i, "Overall After"]):
                if i > 0 and pd.notna(df.loc[i - 1, "Overall After"]) and pd.notna(df.loc[i, "Overall Δ"]):
                    df.loc[i, "Overall After"] = float(df.loc[i - 1, "Overall After"]) + float(df.loc[i, "Overall Δ"])
        # Backward fill: if next after exists and next delta exists, derive current after
        for i in range(len(df) - 2, -1, -1):
            if pd.isna(df.loc[i, "Overall After"]):
                if pd.notna(df.loc[i + 1, "Overall After"]) and pd.notna(df.loc[i + 1, "Overall Δ"]):
                    df.loc[i, "Overall After"] = float(df.loc[i + 1, "Overall After"]) - float(df.loc[i + 1, "Overall Δ"])
    else:
        # No snapshot afters at all: derive by walking backward from current overall.
        # First compute a running backtrack from the end using deltas.
        df_rev = df.sort_values(["Date", "id"], ascending=[False, False]).reset_index(drop=True)
        df_rev["Overall After"] = pd.NA

        running = 0.0
        for i in range(len(df_rev)):
            # after at this row = current_jupr - running
            df_rev.loc[i, "Overall After"] = float(current_jupr) - float(running)
            d = df_rev.loc[i, "Overall Δ"]
            if pd.notna(d):
                running += float(d)

        df = df_rev.sort_values(["Date", "id"], ascending=[True, True]).reset_index(drop=True)

    # -------------------------
    # Chart: Overall trend (restored)
    # -------------------------
    st.subheader("Overall JUPR Trend")

    chart_df = df.copy()
    chart_df = chart_df.dropna(subset=["Overall After"]).copy()
    if chart_df.empty:
        st.info("No chartable rating data for this player.")
    else:
        chart_df["Match #"] = range(1, len(chart_df) + 1)
        chart_df["DateStr"] = pd.to_datetime(chart_df["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        chart_df["DeltaStr"] = chart_df["Overall Δ"].map(lambda x: f"{float(x):+.4f}" if pd.notna(x) else "")
        chart_df["AfterStr"] = chart_df["Overall After"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")

        tail = chart_df.tail(60).copy()

        if alt is not None:
            line = (
                alt.Chart(tail)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Match #:Q", axis=alt.Axis(tickMinStep=1), title="Match Order"),
                    y=alt.Y(
                        "Overall After:Q",
                        axis=alt.Axis(format=".3f"),
                        title="JUPR After Match",
                        scale=alt.Scale(zero=False),
                    ),
                    tooltip=[
                        alt.Tooltip("DateStr:N", title="Date"),
                        alt.Tooltip("league:N", title="League"),
                        alt.Tooltip("Score:N", title="Score"),
                        alt.Tooltip("AfterStr:N", title="After"),
                        alt.Tooltip("DeltaStr:N", title="Δ"),
                    ],
                )
                .interactive()
            )
            st.altair_chart(line, use_container_width=True)
        else:
            # Minimal fallback if altair is unavailable
            st.line_chart(tail.set_index("Match #")["Overall After"])

    st.divider()
    st.subheader("Match History")

    # Table for display (newest first)
    show = df.sort_values(["Date", "id"], ascending=[False, False]).copy()
    show["date"] = pd.to_datetime(show["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
    show["Overall Δ"] = show["Overall Δ"].map(lambda x: f"{float(x):+.4f}" if pd.notna(x) else "")
    show["Overall After"] = show["Overall After"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")

    show = show.rename(columns={"league": "League"})
    show = show[["date", "League", "Score", "match_type", "Overall Δ", "Overall After", "Explain"]]

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Explain": st.column_config.LinkColumn("Explain", display_text="Explain"),
        },
    )

