import logging

import streamlit as st
import pandas as pd

from jupr_app.ui.helpers import qp_get, build_match_explorer_link
from jupr_app.ui.layout import page_shell

logger = logging.getLogger(__name__)

try:
    import altair as alt
except Exception:
    alt = None

# Optional: only needed for "league replay" trend charts
try:
    from jupr_app.domain.ratings import calculate_hybrid_elo
    from jupr_app.domain.constants import DEFAULT_K_FACTOR, MIN_WIN_DELTA_ELO, CAP_LOSER_GAIN_ELO
    _LEAGUE_REPLAY_AVAILABLE = True
except Exception:
    calculate_hybrid_elo = None
    DEFAULT_K_FACTOR = 32
    MIN_WIN_DELTA_ELO = 1.0
    CAP_LOSER_GAIN_ELO = 16.0
    _LEAGUE_REPLAY_AVAILABLE = False


@st.cache_data(ttl=30)
def fetch_player_matches(_supabase, club_id: str, pid: int, limit: int = 600) -> pd.DataFrame:
    """
    Tries to fetch snapshot columns (t*_r / t*_r_end). Falls back gracefully if missing.
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

    try:
        return _run(snap_select)
    except Exception:
        return _run(base_select)


@st.cache_data(ttl=60)
def fetch_player_badges(_supabase, club_id: str, pid: int) -> pd.DataFrame:
    try:
        resp = (
            _supabase.table("player_badges")
            .select("badge_id,earned_at,context_type,context_id,match_id,value_num,value_json")
            .eq("club_id", str(club_id))
            .eq("player_id", int(pid))
            .execute()
        )
        pb_df = pd.DataFrame(resp.data or [])
    except Exception:
        logger.exception("Failed to load player_badges")
        return pd.DataFrame()

    if pb_df.empty or "badge_id" not in pb_df.columns:
        return pd.DataFrame()

    badge_ids = pb_df["badge_id"].dropna().astype(str).unique().tolist()
    if not badge_ids:
        return pd.DataFrame()

    try:
        b_resp = (
            _supabase.table("badges")
            .select("badge_id,name,prestige,category")
            .in_("badge_id", badge_ids)
            .execute()
        )
        badges_df = pd.DataFrame(b_resp.data or [])
    except Exception:
        logger.exception("Failed to load badges definitions")
        return pd.DataFrame()

    if badges_df.empty:
        return pd.DataFrame()

    return pb_df.merge(badges_df, on="badge_id", how="left")


BADGE_ICONS = {
    "participant": "🎟️",
    "dedicated_participant_50": "🧭",
    "lifetime_participant_200": "🏅",
    "mountain_climber": "🧗",
    "breakthrough": "🚀",
    "above_expectations": "⭐",
    "clutch_performer": "⚡",
    "dominant_run": "🔥",
    "high_output": "💥",
    "battle_tested": "🛡️",
    "consistency": "🎯",
    "giant_slayer": "🗡️",
    "upset_champion": "👑",
}


def badge_icon(badge_id: str, category: str | None = None) -> str:
    return BADGE_ICONS.get(str(badge_id), "🏆")


@st.cache_data(ttl=300)
def build_league_snapshot_map(_supabase, club_id: str, league_name: str, df_meta: pd.DataFrame | None, df_players_all: pd.DataFrame | None) -> dict:
    """
    Optional “full restore”: replay league-island Elo across matches in that league.
    Returns snap_map[match_id][player_id] = (start_elo, end_elo)
    Only runs if domain imports exist; otherwise returns {}.
    """
    if not _LEAGUE_REPLAY_AVAILABLE:
        return {}

    lg = str(league_name or "").strip()
    if not lg:
        return {}

    base_select = "id,date,league,match_type,score_t1,score_t2,t1_p1,t1_p2,t2_p1,t2_p2"
    snap_select = base_select + ",t1_p1_r,t1_p2_r,t2_p1_r,t2_p2_r"

    rows = []
    used_snap_select = True
    try:
        resp = (
            _supabase.table("matches")
            .select(snap_select)
            .eq("club_id", str(club_id))
            .order("date", desc=False)
            .order("id", desc=False)
            .execute()
        )
        rows = resp.data or []
    except Exception:
        used_snap_select = False
        resp = (
            _supabase.table("matches")
            .select(base_select)
            .eq("club_id", str(club_id))
            .order("date", desc=False)
            .order("id", desc=False)
            .execute()
        )
        rows = resp.data or []

    if not rows:
        return {}

    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    df["league"] = df.get("league", "").fillna("").astype(str).str.strip()
    df["match_type"] = df.get("match_type", "").fillna("").astype(str).str.strip()

    df = df[df["league"] == lg].copy()
    if df.empty:
        return {}

    # exclude PopUp only; allow NULL/blank match_type
    df = df[df["match_type"] != "PopUp"].copy()
    if df.empty:
        return {}

    df["date"] = pd.to_datetime(df.get("date", None), utc=True, errors="coerce")
    df = df.dropna(subset=["date"])
    if df.empty:
        return {}

    # K factor from meta
    k_val = int(DEFAULT_K_FACTOR)
    try:
        if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
            hit = df_meta[df_meta["league_name"].astype(str).str.strip() == lg]
            if not hit.empty:
                k_val = int(hit.iloc[0].get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
    except Exception:
        k_val = int(DEFAULT_K_FACTOR)

    # Seed from current overall Elo if needed
    overall_seed = {}
    try:
        if df_players_all is not None and not df_players_all.empty:
            overall_seed = dict(zip(df_players_all["id"].astype(int), df_players_all["rating"].astype(float)))
    except Exception:
        overall_seed = {}

    island = {}    # pid -> league elo
    snap_map = {}  # match_id -> {pid: (start,end)}

    def _safe_int(x, default=None):
        try:
            if x is None or str(x).strip() == "":
                return default
            return int(x)
        except Exception:
            return default

    def seed_from_row(row, pid: int) -> float:
        pid = int(pid)
        if used_snap_select:
            try:
                if pid == _safe_int(row.get("t1_p1")):
                    v = row.get("t1_p1_r", None)
                elif pid == _safe_int(row.get("t1_p2")):
                    v = row.get("t1_p2_r", None)
                elif pid == _safe_int(row.get("t2_p1")):
                    v = row.get("t2_p1_r", None)
                elif pid == _safe_int(row.get("t2_p2")):
                    v = row.get("t2_p2_r", None)
                else:
                    v = None
                if v is not None and str(v).strip() != "":
                    return float(v)
            except Exception:
                pass
        return float(overall_seed.get(pid, 1200.0))

    def get_r(row, pid: int) -> float:
        pid = int(pid)
        if pid not in island:
            island[pid] = seed_from_row(row, pid)
        return float(island[pid])

    df = df.sort_values(["date", "id"], ascending=[True, True])

    for _, m in df.iterrows():
        try:
            mid = int(m["id"])
            p1, p2, p3, p4 = int(m["t1_p1"]), int(m["t1_p2"]), int(m["t2_p1"]), int(m["t2_p2"])
            s1 = int(m.get("score_t1", 0) or 0)
            s2 = int(m.get("score_t2", 0) or 0)
        except Exception:
            continue

        if (s1 + s2) <= 0:
            continue

        r1, r2, r3, r4 = get_r(m, p1), get_r(m, p2), get_r(m, p3), get_r(m, p4)

        d1, d2 = calculate_hybrid_elo(
            (r1 + r2) / 2.0,
            (r3 + r4) / 2.0,
            s1,
            s2,
            k_factor=int(k_val),
            min_win_delta=float(MIN_WIN_DELTA_ELO),
            cap_loser_gain=float(CAP_LOSER_GAIN_ELO),
        )

        island[p1] = r1 + float(d1)
        island[p2] = r2 + float(d1)
        island[p3] = r3 + float(d2)
        island[p4] = r4 + float(d2)

        snap_map[mid] = {
            p1: (r1, island[p1]),
            p2: (r2, island[p2]),
            p3: (r3, island[p3]),
            p4: (r4, island[p4]),
        }

    return snap_map


def render(ctx):
    PUBLIC_MODE = bool(getattr(ctx, "public_mode", False))
    mode_label = "Public" if PUBLIC_MODE else "Admin"
    page_shell("🔍 Player Search", "Find players and view ratings.", mode_label=mode_label)

    df_players_all = ctx.df_players_all
    df_leagues = getattr(ctx, "df_leagues", None)
    df_meta = getattr(ctx, "df_meta", None)

    if df_players_all is None or df_players_all.empty:
        st.info("No players found.")
        return

    players_df = df_players_all.copy()
    if "active" in players_df.columns:
        players_df = players_df[players_df["active"] == True].copy()

    if players_df.empty:
        st.info("No active players.")
        return

    players_df["id"] = players_df["id"].astype(int)

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

    try:
        current_overall_elo = float(row.get("rating", 1200.0) or 1200.0)
    except Exception:
        current_overall_elo = 1200.0
    current_jupr = current_overall_elo / 400.0

    c1, c2 = st.columns(2)
    c1.metric("Player", pick_name)
    c2.metric("Overall JUPR", f"{current_jupr:.3f}")

    # -------------------------
    # Restore: Ratings by active league (table)
    # -------------------------
    st.markdown("### Ratings by active league")

    active_leagues = []
    if df_meta is not None and isinstance(df_meta, pd.DataFrame) and not df_meta.empty:
        if "is_active" in df_meta.columns and "league_name" in df_meta.columns:
            active_leagues = (
                df_meta[df_meta["is_active"] == True]["league_name"]
                .dropna()
                .astype(str)
                .str.strip()
                .tolist()
            )

    lr_rows = pd.DataFrame()
    if df_leagues is not None and isinstance(df_leagues, pd.DataFrame) and not df_leagues.empty:
        if "player_id" in df_leagues.columns:
            lr_rows = df_leagues[df_leagues["player_id"].astype(int) == int(pid)].copy()

    if not lr_rows.empty:
        if "league_name" in lr_rows.columns:
            lr_rows["league_name"] = lr_rows["league_name"].astype(str).str.strip()

        if active_leagues and "league_name" in lr_rows.columns:
            lr_rows = lr_rows[lr_rows["league_name"].isin(active_leagues)].copy()

        if "is_active" in lr_rows.columns:
            lr_rows = lr_rows[lr_rows["is_active"] == True].copy()

        if lr_rows.empty:
            st.caption("No active league ratings found for this player.")
        else:
            if "rating" in lr_rows.columns:
                lr_rows["League JUPR"] = lr_rows["rating"].astype(float) / 400.0

            cols = ["league_name", "League JUPR", "wins", "losses", "matches_played"]
            cols = [c for c in cols if c in lr_rows.columns]

            if "League JUPR" in lr_rows.columns:
                lr_rows = lr_rows.sort_values("League JUPR", ascending=False)

            st.dataframe(
                lr_rows[cols].rename(
                    columns={"league_name": "League", "wins": "W", "losses": "L", "matches_played": "MP"}
                ),
                use_container_width=True,
                hide_index=True,
                column_config={"League JUPR": st.column_config.NumberColumn(format="%.3f")},
            )
    else:
        st.caption("No league ratings table entries found for this player yet.")

    st.markdown("### Badges")
    badges_df = pd.DataFrame()
    if getattr(ctx, "df_player_badges", None) is not None and getattr(ctx, "df_badges", None) is not None:
        pb_df = ctx.df_player_badges
        b_df = ctx.df_badges
        if isinstance(pb_df, pd.DataFrame) and isinstance(b_df, pd.DataFrame) and not pb_df.empty:
            pb_df = pb_df[pb_df.get("player_id") == int(pid)].copy()
            if not pb_df.empty and "badge_id" in pb_df.columns:
                badges_df = pb_df.merge(b_df, on="badge_id", how="left")

    if badges_df.empty:
        try:
            badges_df = fetch_player_badges(_supabase, club_id, pid)
        except Exception:
            logger.exception("Failed to fetch badges for player view")
            badges_df = pd.DataFrame()

    if badges_df.empty:
        st.caption("No badges earned yet.")
    else:
        badges_df["earned_at_dt"] = pd.to_datetime(badges_df.get("earned_at", None), utc=True, errors="coerce")
        badges_df["prestige"] = pd.to_numeric(badges_df.get("prestige", 0), errors="coerce").fillna(0)
        badges_df = badges_df.sort_values(["prestige", "earned_at_dt"], ascending=[False, False])

        participation_order = ["lifetime_participant_200", "dedicated_participant_50", "participant"]
        participation_df = badges_df[badges_df["badge_id"].isin(participation_order)].copy()
        participation_badge = None
        if not participation_df.empty:
            participation_df["tier_rank"] = participation_df["badge_id"].map(
                {badge_id: idx for idx, badge_id in enumerate(participation_order)}
            )
            participation_df = participation_df.sort_values(["tier_rank", "earned_at_dt"])
            participation_badge = participation_df.iloc[0]

        st.subheader("Participation")
        if participation_badge is None:
            st.caption("No participation badge earned yet.")
        else:
            icon = badge_icon(participation_badge.get("badge_id"), participation_badge.get("category"))
            st.markdown(f"**{icon} {participation_badge.get('name', 'Badge')}**")
            st.caption(f"Prestige {int(participation_badge.get('prestige', 0) or 0)}")

        st.subheader("All badges")
        top_badges = badges_df.head(3).copy()
        cols = st.columns(len(top_badges))
        for idx, (_, row) in enumerate(top_badges.iterrows()):
            with cols[idx]:
                icon = badge_icon(row.get("badge_id"), row.get("category"))
                st.markdown(f"**{icon} {row.get('name', 'Badge')}**")
                st.caption(f"Prestige {int(row.get('prestige', 0) or 0)}")

        with st.expander("View all badges", expanded=False):
            show_cols = ["name", "prestige", "category", "earned_at_dt"]
            show_df = badges_df[show_cols].rename(
                columns={
                    "name": "Badge",
                    "prestige": "Prestige",
                    "category": "Category",
                    "earned_at_dt": "Earned",
                }
            )
            st.dataframe(
                show_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Prestige": st.column_config.NumberColumn(format="%d"),
                    "Earned": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
                },
            )

    st.divider()

    _supabase = ctx.supabase
    club_id = ctx.club_id
    matches = fetch_player_matches(_supabase, club_id, pid, limit=600)

    if matches.empty:
        st.info("No matches recorded for this player.")
        return

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

    def result_for_player(r):
        try:
            t1p1 = _safe_int(r.get("t1_p1"))
            t1p2 = _safe_int(r.get("t1_p2"))
            s1 = _safe_int(r.get("score_t1"), 0) or 0
            s2 = _safe_int(r.get("score_t2"), 0) or 0
        except Exception:
            return ""

        if s1 == s2:
            return "DRAW"
        on_t1 = pid in {t1p1, t1p2}
        winner = "WIN" if s1 > s2 else "LOSS"
        if not on_t1:
            winner = "WIN" if s2 > s1 else "LOSS"
        return winner

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
        pid_ = int(pid_)
        t1p1 = _safe_int(r.get("t1_p1"))
        t1p2 = _safe_int(r.get("t1_p2"))
        t2p1 = _safe_int(r.get("t2_p1"))
        t2p2 = _safe_int(r.get("t2_p2"))

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
        return abs(float(raw)) if winner_team == my_team else -abs(float(raw))

    # Normalize date + league strings
    matches = matches.copy()
    matches["date_dt"] = pd.to_datetime(matches.get("date", None), errors="coerce", utc=True)
    matches = matches.dropna(subset=["date_dt"]).copy()
    matches["league"] = matches.get("league", "").fillna("").astype(str).str.strip()
    matches["match_type"] = matches.get("match_type", "").fillna("").astype(str).str.strip()

    # Build overall series rows
    processed = []
    for _, r0 in matches.iterrows():
        r = dict(r0)

        start_elo, end_elo = get_overall_snap(r, pid)
        after_jupr = None
        delta_jupr = None

        if start_elo is not None and end_elo is not None:
            try:
                delta_jupr = (float(end_elo) - float(start_elo)) / 400.0
                after_jupr = float(end_elo) / 400.0
            except Exception:
                pass
        else:
            d_elo = signed_delta_from_elo_delta(r, pid)
            if d_elo is not None:
                delta_jupr = float(d_elo) / 400.0

        processed.append(
            {
                "id": _safe_int(r.get("id")),
                "Date": r.get("date_dt"),
                "League": str(r.get("league", "") or "").strip(),
                "match_type": str(r.get("match_type", "") or "").strip(),
                "Score": score_for_player(r),
                "Result": result_for_player(r),
                "Overall Δ": delta_jupr,
                "Overall After": after_jupr,
                "Explain": explain_link(r),
            }
        )

    df = pd.DataFrame(processed)
    if df.empty:
        st.info("No matches available.")
        return

    df = df.sort_values(["Date", "id"], ascending=[True, True]).reset_index(drop=True)
    df["Overall Δ"] = pd.to_numeric(df["Overall Δ"], errors="coerce")
    df["Overall After"] = pd.to_numeric(df["Overall After"], errors="coerce")

    # Backfill overall-after if needed
    if df["Overall After"].notna().any():
        for i in range(len(df)):
            if pd.isna(df.loc[i, "Overall After"]):
                if i > 0 and pd.notna(df.loc[i - 1, "Overall After"]) and pd.notna(df.loc[i, "Overall Δ"]):
                    df.loc[i, "Overall After"] = float(df.loc[i - 1, "Overall After"]) + float(df.loc[i, "Overall Δ"])
        for i in range(len(df) - 2, -1, -1):
            if pd.isna(df.loc[i, "Overall After"]):
                if pd.notna(df.loc[i + 1, "Overall After"]) and pd.notna(df.loc[i + 1, "Overall Δ"]):
                    df.loc[i, "Overall After"] = float(df.loc[i + 1, "Overall After"]) - float(df.loc[i + 1, "Overall Δ"])
    else:
        df_rev = df.sort_values(["Date", "id"], ascending=[False, False]).reset_index(drop=True)
        running = 0.0
        after_vals = []
        for i in range(len(df_rev)):
            after_vals.append(float(current_jupr) - float(running))
            d = df_rev.loc[i, "Overall Δ"]
            if pd.notna(d):
                running += float(d)
        df_rev["Overall After"] = after_vals
        df = df_rev.sort_values(["Date", "id"], ascending=[True, True]).reset_index(drop=True)

    # -------------------------
    # Restore: tabs for Overall + each league
    # -------------------------
    leagues_in_matches = sorted(
        [x for x in df["League"].fillna("").astype(str).str.strip().unique().tolist() if x and x.upper() != "OVERALL"]
    )
    tab_labels = ["Overall"] + [f"League: {lg}" for lg in leagues_in_matches]
    tabs = st.tabs(tab_labels)

    def render_chart_and_table(view_df: pd.DataFrame, title_prefix: str, *, league_trend: bool = False, league_name: str = ""):
        st.subheader(f"{title_prefix} JUPR Trend")

        chart_df = view_df.copy().dropna(subset=["Overall After"]).sort_values(["Date", "id"]).reset_index(drop=True)
        if chart_df.empty:
            st.info("No chartable rating data in this view.")
        else:
            chart_df["Match #"] = range(1, len(chart_df) + 1)
            chart_df["DateStr"] = pd.to_datetime(chart_df["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
            chart_df["DeltaStr"] = chart_df["Overall Δ"].map(lambda x: f"{float(x):+.4f}" if pd.notna(x) else "")
            chart_df["AfterStr"] = chart_df["Overall After"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")

            tail = chart_df.tail(60).copy()

            # Optional full restore: show league replay trend (if available)
            if league_trend and league_name and _LEAGUE_REPLAY_AVAILABLE:
                snap_map = build_league_snapshot_map(_supabase, club_id, league_name, df_meta, df_players_all)
                if snap_map:
                    # Build a league-after series from snap_map for this player
                    tmp = view_df.copy()
                    tmp["League After"] = pd.NA
                    tmp["League Δ"] = pd.NA
                    for i in range(len(tmp)):
                        mid = tmp.iloc[i].get("id", None)
                        if mid is None:
                            continue
                        hit = snap_map.get(int(mid), {}).get(int(pid), None)
                        if hit:
                            ls, le = hit
                            tmp.at[tmp.index[i], "League Δ"] = (float(le) - float(ls)) / 400.0
                            tmp.at[tmp.index[i], "League After"] = float(le) / 400.0

                    tmp2 = tmp.dropna(subset=["League After"]).sort_values(["Date", "id"]).reset_index(drop=True)
                    if not tmp2.empty:
                        tmp2["Match #"] = range(1, len(tmp2) + 1)
                        tmp2["DateStr"] = pd.to_datetime(tmp2["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
                        tmp2["DeltaStr"] = tmp2["League Δ"].map(lambda x: f"{float(x):+.4f}" if pd.notna(x) else "")
                        tmp2["AfterStr"] = tmp2["League After"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")
                        tail = tmp2.tail(60).copy()
                        y_col = "League After"
                        y_title = "League JUPR After Match"
                    else:
                        y_col = "Overall After"
                        y_title = "JUPR After Match (Overall)"
                else:
                    y_col = "Overall After"
                    y_title = "JUPR After Match (Overall)"
            else:
                y_col = "Overall After"
                y_title = "JUPR After Match (Overall)"

            if alt is not None:
                line = (
                    alt.Chart(tail)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Match #:Q", axis=alt.Axis(tickMinStep=1), title="Match Order"),
                        y=alt.Y(
                            f"{y_col}:Q",
                            axis=alt.Axis(format=".3f"),
                            title=y_title,
                            scale=alt.Scale(zero=False),
                        ),
                        tooltip=[
                            alt.Tooltip("DateStr:N", title="Date"),
                            alt.Tooltip("League:N", title="League"),
                            alt.Tooltip("Score:N", title="Score"),
                            alt.Tooltip("AfterStr:N", title="After"),
                            alt.Tooltip("DeltaStr:N", title="Δ"),
                        ],
                    )
                    .interactive()
                )
                st.altair_chart(line, use_container_width=True)
            else:
                st.line_chart(tail.set_index("Match #")[y_col])

        st.divider()
        st.subheader(f"{title_prefix} Match History")

        show = view_df.sort_values(["Date", "id"], ascending=[False, False]).copy()
        show["date"] = pd.to_datetime(show["Date"], utc=True, errors="coerce").dt.strftime("%Y-%m-%d")
        show["delta_raw"] = pd.to_numeric(show["Overall Δ"], errors="coerce")
        show["Overall Δ"] = show["delta_raw"].map(lambda x: f"{float(x):+.4f}" if pd.notna(x) else "")
        show["Overall After"] = show["Overall After"].map(lambda x: f"{float(x):.3f}" if pd.notna(x) else "")

        def result_badge(result: str) -> str:
            label = str(result or "").strip().upper() or "—"
            normalized = label.upper()
            if normalized in {"W", "WIN", "WON"}:
                variant = "win"
            elif normalized in {"L", "LOSS", "LOST"}:
                variant = "loss"
            else:
                variant = "draw"
            return f"<span class='jupr-result-badge {variant}'>{label}</span>"

        def delta_span(delta_str: str, delta_raw: float | None) -> str:
            if not delta_str:
                return ""
            kind = "zero"
            try:
                delta_val = float(delta_raw)
            except (TypeError, ValueError):
                delta_val = 0.0
            if delta_val > 0:
                kind = "pos"
            elif delta_val < 0:
                kind = "neg"
            return f"<span class='jupr-delta {kind}'>{delta_str}</span>"

        show["Result"] = show["Result"].map(result_badge)
        show["Overall Δ"] = show.apply(lambda row: delta_span(row["Overall Δ"], row["delta_raw"]), axis=1)
        show["Explain"] = show["Explain"].map(
            lambda url: f"<a href='{url}' target='_self'>Explain</a>" if url else ""
        )

        show = show[["date", "League", "Score", "Result", "match_type", "Overall Δ", "Overall After", "Explain"]]

        html_table = show.to_html(index=False, escape=False)

        st.markdown(
            f"""
            <div class="match-history-table">
              {html_table}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tabs[0]:
        render_chart_and_table(df, "Overall", league_trend=False)

    for i, lg in enumerate(leagues_in_matches, start=1):
        with tabs[i]:
            df_lg = df[df["League"].astype(str).str.strip() == lg].copy()
            # Show league replay trend if available; otherwise overall trend filtered to that league’s matches.
            render_chart_and_table(df_lg, f"League: {lg}", league_trend=True, league_name=lg)
