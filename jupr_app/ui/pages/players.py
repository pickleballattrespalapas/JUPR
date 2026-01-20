import streamlit as st
import pandas as pd

from jupr_app.ui.helpers import qp_get, build_match_explorer_link


@st.cache_data(ttl=30)
def fetch_player_matches(_supabase, club_id: str, pid: int, limit: int = 200) -> pd.DataFrame:
    resp = (
        _supabase.table("matches")
        .select("id,date,league,match_type,score_t1,score_t2,t1_p1,t1_p2,t2_p1,t2_p2")
        .eq("club_id", str(club_id))
        .or_(f"t1_p1.eq.{pid},t1_p2.eq.{pid},t2_p1.eq.{pid},t2_p2.eq.{pid}")
        .order("date", desc=True)
        .order("id", desc=True)
        .limit(int(limit))
        .execute()
    )
    return pd.DataFrame(resp.data or [])


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

    # Normalize id type
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
            # Optional: remove pid so user can browse freely afterward
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
    jupr = float(row.get("rating", 1200.0) or 1200.0) / 400.0

    c1, c2 = st.columns(2)
    c1.metric("Player", pick_name)
    c2.metric("Overall JUPR", f"{jupr:.3f}")

    st.divider()
    st.subheader("Match History")

    _supabase = ctx.supabase
    club_id = ctx.club_id

    matches = fetch_player_matches(_supabase, club_id, pid, limit=200)

    if matches.empty:
        st.info("No matches recorded for this player.")
        return

    # Build score from the player's perspective
    def score_for_player(r):
        try:
            t1p1 = int(r["t1_p1"])
            t1p2 = int(r["t1_p2"])
            s1 = int(r.get("score_t1", 0) or 0)
            s2 = int(r.get("score_t2", 0) or 0)
        except Exception:
            return ""

        if t1p1 == pid or t1p2 == pid:
            return f"{s1}-{s2}"
        return f"{s2}-{s1}"

    matches["Score"] = matches.apply(score_for_player, axis=1)

    # Explain link per match
    def explain_link(r):
        try:
            t1p1 = int(r["t1_p1"])
            t1p2 = int(r["t1_p2"])
            t2p1 = int(r["t2_p1"])
            t2p2 = int(r["t2_p2"])
            s1 = int(r.get("score_t1", 0) or 0)
            s2 = int(r.get("score_t2", 0) or 0)
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

    matches["Explain"] = matches.apply(explain_link, axis=1)

    show = matches[["date", "league", "Score", "match_type", "Explain"]].copy()
    show["date"] = pd.to_datetime(show["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    st.dataframe(
        show,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Explain": st.column_config.LinkColumn("Explain", display_text="Explain"),
        },
    )
