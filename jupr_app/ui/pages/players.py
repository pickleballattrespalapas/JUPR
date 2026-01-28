import html
import logging
import math
import re
import textwrap

import streamlit as st
import pandas as pd
from streamlit.components.v1 import html as st_html

from jupr_app.ui.helpers import qp_get, build_match_explorer_link
from jupr_app.ui.layout import page_shell
from jupr_app.domain.gamification.profile import (
    build_gamification_summary,
)

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
            .select("player_id,badge_id,earned_at,context_type,context_id,match_id,value_num,value_json")
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


@st.cache_data(ttl=120)
def fetch_badge_definitions(_supabase) -> pd.DataFrame:
    try:
        resp = _supabase.table("badges").select("*").execute()
        return pd.DataFrame(resp.data or [])
    except Exception:
        logger.exception("Failed to load badge definitions")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_player_stories(_supabase, club_id: str, pid: int, limit: int = 6) -> pd.DataFrame:
    try:
        resp = (
            _supabase.table("player_stories")
            .select("story_type,context_id,created_at,title,body,importance,match_id")
            .eq("club_id", str(club_id))
            .eq("player_id", int(pid))
            .order("created_at", desc=True)
            .limit(int(limit) * 3)
            .execute()
        )
        return pd.DataFrame(resp.data or [])
    except Exception:
        logger.exception("Failed to load player stories")
        return pd.DataFrame()


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


def _season_sort_key(league_name: str) -> tuple[int, int] | None:
    name = str(league_name or "").strip()
    if not name:
        return None
    lowered = name.lower()
    season_order = {"winter": 1, "spring": 2, "summer": 3, "fall": 4}
    season_rank = None
    for season, rank in season_order.items():
        if season in lowered:
            season_rank = rank
            break
    years = [int(y) for y in re.findall(r"\b(?:19|20)\d{2}\b", lowered)]
    year = max(years) if years else None
    if year is None and season_rank is None:
        return None
    return (year or 0, season_rank or 0)


def compute_top_finisher_trophies(pid: int, df_leagues: pd.DataFrame | None, df_meta: pd.DataFrame | None) -> list[dict]:
    if df_leagues is None or df_leagues.empty:
        return []

    leagues_df = df_leagues.copy()
    league_col = "league_name" if "league_name" in leagues_df.columns else "league" if "league" in leagues_df.columns else None
    if not league_col or "player_id" not in leagues_df.columns:
        return []

    leagues_df[league_col] = leagues_df[league_col].astype(str).str.strip()
    leagues_df["player_id"] = pd.to_numeric(leagues_df["player_id"], errors="coerce").fillna(-1).astype(int)

    ended_leagues: list[str] = []
    if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
        if "is_active" in df_meta.columns:
            meta = df_meta.copy()
            meta["league_name"] = meta["league_name"].astype(str).str.strip()
            meta["is_active"] = meta["is_active"].fillna(False).astype(bool)
            ended_leagues = meta[meta["is_active"] != True]["league_name"].dropna().unique().tolist()

    if not ended_leagues:
        season_pattern = r"\b(?:fall|spring|summer|winter)\b|\b(?:19|20)\d{2}\b"
        season_mask = leagues_df[league_col].str.contains(season_pattern, case=False, regex=True, na=False)
        ended_leagues = leagues_df.loc[season_mask, league_col].dropna().unique().tolist()

    if not ended_leagues:
        ended_leagues = leagues_df[league_col].dropna().unique().tolist()

    trophies: list[dict] = []
    icon_map = {1: "🥇", 2: "🥈", 3: "🥉"}
    for league in ended_leagues:
        league_rows = leagues_df[leagues_df[league_col] == league].copy()
        if league_rows.empty:
            continue

        league_rows["wins_calc"] = pd.to_numeric(league_rows.get("wins", 0), errors="coerce").fillna(0)
        league_rows["losses_calc"] = pd.to_numeric(league_rows.get("losses", 0), errors="coerce").fillna(0)
        league_rows["matches_calc"] = pd.to_numeric(
            league_rows.get("matches_played", league_rows["wins_calc"] + league_rows["losses_calc"]),
            errors="coerce",
        ).fillna(league_rows["wins_calc"] + league_rows["losses_calc"])
        if "win_pct" in league_rows.columns:
            league_rows["win_pct_calc"] = pd.to_numeric(league_rows.get("win_pct", 0), errors="coerce").fillna(0)
        else:
            total_matches = league_rows["wins_calc"] + league_rows["losses_calc"]
            league_rows["win_pct_calc"] = league_rows["wins_calc"] / total_matches.where(total_matches > 0, 1)
        league_rows["rating_calc"] = pd.to_numeric(league_rows.get("rating", 0), errors="coerce").fillna(0)

        league_rows = league_rows.sort_values(
            ["wins_calc", "win_pct_calc", "rating_calc", "matches_calc"],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)

        player_rows = league_rows[league_rows["player_id"] == int(pid)]
        if player_rows.empty:
            continue
        place = int(player_rows.index[0]) + 1
        if place > 3:
            continue
        player_row = player_rows.iloc[0]
        wins = int(player_row.get("wins_calc", 0))
        losses = int(player_row.get("losses_calc", 0))
        matches = int(player_row.get("matches_calc", wins + losses))
        win_pct = float(player_row.get("win_pct_calc", 0)) if matches else 0.0
        rating_val = player_row.get("rating_calc", 0)
        rating_text = f"{float(rating_val):.0f}" if rating_val else "—"
        subtitle = f"{place} place in {league}"
        body = f"Record {wins}-{losses} • {win_pct:.0%} win rate • League JUPR {rating_text}"
        trophies.append(
            {
                "league_name": league,
                "place": place,
                "icon": icon_map.get(place, "🏆"),
                "title": "Top Finisher",
                "subtitle": subtitle,
                "body": body,
                "_season_sort": _season_sort_key(league),
            }
        )

    if not trophies:
        return []

    seasonal = [t for t in trophies if t.get("_season_sort") is not None]
    non_seasonal = [t for t in trophies if t.get("_season_sort") is None]
    seasonal_sorted = sorted(
        seasonal,
        key=lambda t: (t["_season_sort"][0], t["_season_sort"][1], -t["place"]),
        reverse=True,
    )
    non_seasonal_sorted = sorted(non_seasonal, key=lambda t: t["place"])
    ordered = seasonal_sorted + non_seasonal_sorted
    for trophy in ordered:
        trophy.pop("_season_sort", None)
    return ordered[:3]


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
    if "inactive_at" in players_df.columns:
        players_df = players_df[players_df["inactive_at"].isna()].copy()
    elif "active" in players_df.columns:
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
    _supabase = ctx.supabase
    club_id = ctx.club_id

    try:
        current_overall_elo = float(row.get("rating", 1200.0) or 1200.0)
    except Exception:
        current_overall_elo = 1200.0
    current_jupr = current_overall_elo / 400.0

    c1, c2 = st.columns(2)
    c1.metric("Player", pick_name)
    c2.metric("Overall JUPR", f"{current_jupr:.3f}")

    tape_tab, ratings_tab = st.tabs(["Trophy Room", "Ratings"])

    with tape_tab:
        debug_render = False
        if bool(getattr(ctx, "admin_logged_in", False)):
            debug_render = st.toggle("Debug badge render", value=False)

        def _debug_html_warning(label: str, fn_name: str, text: str) -> None:
            if not debug_render:
                return
            if "<div" in text or "badge-card" in text:
                snippet = textwrap.shorten(text.replace("\n", " "), width=140, placeholder="…")
                st.warning(f"Badge render debug ({label}) via {fn_name}: {snippet}")

        def badge_markdown(text: str, *, label: str) -> None:
            _debug_html_warning(label, "markdown", text)
            st.markdown(text)

        def badge_write(text: str, *, label: str) -> None:
            _debug_html_warning(label, "write", text)
            st.write(text)

        def badge_caption(text: str, *, label: str) -> None:
            _debug_html_warning(label, "caption", text)
            st.caption(text)

        def badge_code(text: str, *, label: str) -> None:
            _debug_html_warning(label, "code", text)
            st.code(text)

        badge_markdown("### Badges & Trophies", label="badges.header")

        badge_css = """
        .badge-summary {
            display: flex;
            flex-wrap: wrap;
            gap: 0.75rem;
            align-items: stretch;
            margin-bottom: 0.75rem;
        }
        .badge-stat {
            background: var(--panel, rgba(255,255,255,0.04));
            border: 1px solid var(--border, rgba(255,255,255,0.10));
            box-shadow: var(--shadow, none);
            border-radius: 0.75rem;
            padding: 0.75rem 0.9rem;
            min-width: 120px;
            color: var(--text-primary, rgba(255,255,255,0.92));
        }
        .badge-stat-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary, rgba(255,255,255,0.80));
        }
        .badge-stat-value {
            font-size: 1.6rem;
            font-weight: 700;
        }
        .badge-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
            align-items: center;
        }
        .badge-chip {
            display: inline-flex;
            gap: 0.35rem;
            align-items: center;
            padding: 0.25rem 0.5rem;
            border-radius: 999px;
            border: 1px solid var(--border, rgba(255,255,255,0.10));
            background: var(--pill-bg, rgba(255,255,255,0.04));
            font-size: 0.8rem;
            max-width: 180px;
            color: var(--text-primary, rgba(255,255,255,0.92));
        }
        .trophy-section {
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
            margin-bottom: 0.6rem;
        }
        .trophy-label {
            font-size: 0.7rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--text-secondary, rgba(255,255,255,0.80));
        }
        .trophy-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            align-items: stretch;
        }
        .trophy-chip {
            display: inline-flex;
            gap: 0.45rem;
            align-items: flex-start;
            padding: 0.35rem 0.6rem;
            border-radius: 0.75rem;
            border: 1px solid var(--border, rgba(255,255,255,0.10));
            background: var(--panel, rgba(255,255,255,0.04));
            box-shadow: var(--shadow, none);
            font-size: 0.8rem;
            max-width: 320px;
            color: var(--text-primary, rgba(255,255,255,0.92));
        }
        .trophy-text {
            display: flex;
            flex-direction: column;
            gap: 0.1rem;
            min-width: 0;
        }
        .trophy-title {
            font-weight: 600;
            font-size: 0.85rem;
        }
        .trophy-body {
            font-size: 0.7rem;
            color: var(--text-muted, rgba(255,255,255,0.65));
        }
        .badge-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 0.75rem;
        }
        .featured-grid .badge-card:nth-child(n+4) {
            display: none;
        }
        @media (max-width: 900px) {
            .featured-grid .badge-card:nth-child(n+3) {
                display: none;
            }
        }
        @media (max-width: 640px) {
            .featured-grid .badge-card:nth-child(n+2) {
                display: none;
            }
        }
        .badge-card {
            border-radius: 0.8rem;
            border: 1px solid var(--border, rgba(255,255,255,0.10));
            background: var(--panel, rgba(255,255,255,0.04));
            box-shadow: var(--shadow, none);
            padding: 0.7rem 0.8rem;
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
            color: var(--text-primary, rgba(255,255,255,0.92));
        }
        .badge-card.silhouette {
            background: var(--panel, rgba(255,255,255,0.04));
            opacity: 0.7;
        }
        .badge-card-header {
            display: flex;
            align-items: center;
            gap: 0.4rem;
            font-weight: 600;
        }
        .badge-subtext {
            font-size: 0.75rem;
            color: var(--text-muted, rgba(255,255,255,0.65));
        }
        .truncate-1 {
            display: -webkit-box;
            -webkit-line-clamp: 1;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
        .truncate-2 {
            display: -webkit-box;
            -webkit-line-clamp: 2;
            -webkit-box-orient: vertical;
            overflow: hidden;
        }
    """

        def _estimate_badge_height(cleaned: str) -> int:
            card_count = cleaned.count("badge-card")
            if card_count <= 0:
                return 120
            cards_per_row = 3 if "featured-grid" in cleaned else 4
            rows = max(1, math.ceil(card_count / cards_per_row))
            return 110 + rows * 150

        def render_badge_html(html_block: str, *, label: str, height: int | None = None) -> None:
            cleaned = textwrap.dedent(html_block).strip()
            _debug_html_warning(label, "st_html", cleaned)
            doc = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <style>{badge_css}</style>
  </head>
  <body>{cleaned}</body>
</html>"""
            resolved_height = height if height is not None else _estimate_badge_height(cleaned)
            st_html(doc, height=resolved_height, scrolling=False)
        badge_defs = getattr(ctx, "df_badges", None)
        if badge_defs is None or (isinstance(badge_defs, pd.DataFrame) and badge_defs.empty):
            badge_defs = fetch_badge_definitions(_supabase)

        player_badges = getattr(ctx, "df_player_badges", None)
        if player_badges is None or (isinstance(player_badges, pd.DataFrame) and player_badges.empty):
            try:
                player_badges = fetch_player_badges(_supabase, club_id, pid)
            except Exception:
                logger.exception("Failed to fetch badges for player view")
                player_badges = pd.DataFrame()

        summary = build_gamification_summary(pid, badge_defs, player_badges)
        prestige_total = summary.get("prestige_total", 0)
        collected_unique = summary.get("collected_unique_count", 0)
        total_active = summary.get("total_active_badge_types", 0)

        unlocked_badges = summary.get("unlocked_badges", [])
        locked_badges = summary.get("locked_badges", [])

        trophies = compute_top_finisher_trophies(pid, df_leagues, df_meta)
        if trophies:
            trophy_items = []
            for trophy in trophies:
                trophy_items.append(
                    "<span class='trophy-chip'>"
                    f"<span>{html.escape(trophy.get('icon', '🏆'))}</span>"
                    "<span class='trophy-text'>"
                    f"<span class='trophy-title'>{html.escape(trophy.get('subtitle', 'Top Finisher'))}</span>"
                    f"<span class='trophy-body'>{html.escape(trophy.get('body', ''))}</span>"
                    "</span>"
                    "</span>"
                )
            trophy_html = f"""
            <div class="trophy-section">
                <div class="trophy-label">Top Finisher</div>
                <div class="trophy-chip-row">{''.join(trophy_items)}</div>
            </div>
            """
            render_badge_html(trophy_html, label="trophies.top_finisher", height=120 + len(trophies) * 35)

        top_prestige_key = f"top_prestige_{pid}"
        if top_prestige_key in st.session_state:
            top_prestige = st.session_state[top_prestige_key]
        else:
            prestige_sorted = sorted(
                unlocked_badges,
                key=lambda b: (
                    int(b.get("prestige", 0) or 0),
                    pd.to_datetime(b.get("last_earned_at"), utc=True, errors="coerce"),
                ),
                reverse=True,
            )
            top_prestige = prestige_sorted[:5]
            st.session_state[top_prestige_key] = top_prestige

        chip_items = []
        for badge in top_prestige:
            icon = badge_icon(badge.get("badge_id"), badge.get("category"))
            stack = badge.get("stack_count", 1)
            stack_text = f" ×{stack}" if stack and stack > 1 else ""
            chip_items.append(
                f"<span class='badge-chip'><span>{html.escape(icon)}</span>"
                f"<span class='truncate-1'>{html.escape(str(badge.get('name', 'Badge')))}{stack_text}</span></span>"
            )

        summary_html = f"""
        <div class="badge-summary">
            <div class="badge-stat">
                <div class="badge-stat-label">Prestige</div>
                <div class="badge-stat-value">{int(prestige_total)}</div>
            </div>
            <div class="badge-stat">
                <div class="badge-stat-label">Collection</div>
                <div class="badge-stat-value">{collected_unique}/{total_active}</div>
            </div>
            <div class="badge-stat" style="flex:1; min-width: 220px;">
                <div class="badge-stat-label">Top Prestige</div>
                <div class="badge-chip-row">{''.join(chip_items) or "<span class='badge-subtext'>No reels yet.</span>"}</div>
            </div>
        </div>
    """
        render_badge_html(summary_html, label="badges.summary")

        if not unlocked_badges and not locked_badges:
            badge_caption("No badges available yet.", label="badges.empty")
        else:
            badge_markdown("#### Featured Cuts", label="badges.featured.header")
            prestige_sorted = sorted(
                unlocked_badges,
                key=lambda b: (
                    int(b.get("prestige", 0) or 0),
                    pd.to_datetime(b.get("last_earned_at"), utc=True, errors="coerce"),
                ),
                reverse=True,
            )
            non_participant = [b for b in prestige_sorted if b.get("badge_id") != "participant"]
            if len(non_participant) >= 3:
                featured = non_participant[:3]
            else:
                featured = non_participant[:]
                remaining_slots = 3 - len(featured)
                if remaining_slots > 0:
                    participant_badges = [
                        b for b in prestige_sorted if b.get("badge_id") == "participant"
                    ]
                    featured.extend(participant_badges[:remaining_slots])
            if not featured:
                badge_caption(
                    "The trophy room is quiet—new reels arrive after the next run.",
                    label="badges.featured.empty",
                )
            else:
                featured_cards = []
                for badge in featured:
                    icon = badge_icon(badge.get("badge_id"), badge.get("category"))
                    stack = badge.get("stack_count", 1)
                    stack_text = f" ×{stack}" if stack and stack > 1 else ""
                    excerpt = html.escape(str(badge.get("latest_tape_excerpt") or ""))
                    featured_cards.append(
                        f"""
                    <div class="badge-card">
                        <div class="badge-card-header">
                            <span>{html.escape(icon)}</span>
                            <span class="truncate-1">{html.escape(str(badge.get('name', 'Badge')))}{stack_text}</span>
                        </div>
                        <div class="badge-subtext">Prestige {int(badge.get('prestige', 0) or 0)}</div>
                        <div class="badge-subtext truncate-1">{excerpt}</div>
                    </div>
                    """
                    )
                render_badge_html(
                    f"<div class='badge-grid featured-grid'>{''.join(featured_cards)}</div>",
                    label="badges.featured.grid",
                )

            with st.expander("Open Cabinet", expanded=False):
                filter_cols = st.columns(2)
                show_unlocked = filter_cols[0].checkbox("Unlocked", value=True, key="badge_filter_unlocked")
                show_locked = filter_cols[0].checkbox("Locked", value=True, key="badge_filter_locked")

                all_badges = []
                for badge in unlocked_badges:
                    badge_copy = dict(badge)
                    badge_copy["status"] = "unlocked"
                    all_badges.append(badge_copy)
                for badge in locked_badges:
                    badge_copy = dict(badge)
                    badge_copy["status"] = "locked"
                    all_badges.append(badge_copy)

                categories = sorted({b.get("category") or "Other" for b in all_badges})
                rarities = sorted({b.get("rarity") or "common" for b in all_badges})
                selected_categories = filter_cols[1].multiselect(
                    "Category",
                    categories,
                    default=categories,
                    key="badge_filter_categories",
                )
                selected_rarities = filter_cols[1].multiselect(
                    "Rarity",
                    rarities,
                    default=rarities,
                    key="badge_filter_rarities",
                )

                def _visible(badge: dict) -> bool:
                    category = badge.get("category") or "Other"
                    rarity = badge.get("rarity") or "common"
                    if badge.get("status") == "unlocked" and not show_unlocked:
                        return False
                    if badge.get("status") == "locked" and not show_locked:
                        return False
                    if category not in selected_categories:
                        return False
                    if rarity not in selected_rarities:
                        return False
                    return True

                visible_badges = [b for b in all_badges if _visible(b)]
                if not visible_badges:
                    badge_caption("No badges match the filters.", label="badges.filters.empty")
                else:
                    card_items = []
                    for badge in visible_badges:
                        status = badge.get("status")
                        name = html.escape(str(badge.get("name", "Badge")))
                        prestige = int(badge.get("prestige", 0) or 0)
                        icon = badge_icon(badge.get("badge_id"), badge.get("category"))
                        stack = badge.get("stack_count", 1)
                        stack_text = f" ×{stack}" if stack and stack > 1 else ""
                        if status == "locked":
                            hint = html.escape(str(badge.get("hint") or "A reel still missing."))
                            card_items.append(
                                f"""
                            <div class="badge-card silhouette">
                                <div class="badge-card-header">
                                    <span>⬛</span>
                                    <span class="truncate-1">{name}{stack_text}</span>
                                </div>
                                <div class="badge-subtext">Prestige {prestige}</div>
                                <div class="badge-subtext truncate-2">{hint}</div>
                            </div>
                            """
                            )
                        else:
                            excerpt = html.escape(str(badge.get("latest_tape_excerpt") or ""))
                            card_items.append(
                                f"""
                            <div class="badge-card">
                                <div class="badge-card-header">
                                    <span>{html.escape(icon)}</span>
                                    <span class="truncate-1">{name}{stack_text}</span>
                                </div>
                                <div class="badge-subtext">Prestige {prestige}</div>
                                <div class="badge-subtext truncate-2">{excerpt}</div>
                            </div>
                            """
                            )
                    render_badge_html(
                        f"<div class='badge-grid'>{''.join(card_items)}</div>",
                        label="badges.cabinet.grid",
                    )

                details_view = st.toggle("Details view", value=False, key="badge_details_view")
                if details_view and unlocked_badges:
                    summary_df = pd.DataFrame(unlocked_badges)
                    summary_df["last_earned_at_dt"] = pd.to_datetime(
                        summary_df.get("last_earned_at", None), utc=True, errors="coerce"
                    )
                    summary_df = summary_df.sort_values(
                        ["last_earned_at_dt", "prestige"], ascending=[False, False]
                    )
                    show_df = summary_df[
                        ["name", "category", "prestige", "stack_count", "last_earned_at_dt"]
                    ].rename(
                        columns={
                            "name": "Badge",
                            "category": "Category",
                            "prestige": "Prestige",
                            "stack_count": "Count",
                            "last_earned_at_dt": "Last Earned",
                        }
                    )
                    st.dataframe(
                        show_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Prestige": st.column_config.NumberColumn(format="%d"),
                            "Last Earned": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
                        },
                    )

                    admin_debug = False
                    if bool(getattr(ctx, "admin_logged_in", False)):
                        admin_debug = st.toggle("Show debug columns", value=False, key="badge_debug_columns")

                    if isinstance(player_badges, pd.DataFrame) and not player_badges.empty:
                        pb_df = player_badges.copy()
                        pb_df = pb_df[pb_df.get("player_id") == int(pid)].copy()
                        pb_df["earned_at_dt"] = pd.to_datetime(
                            pb_df.get("earned_at", None), utc=True, errors="coerce"
                        )
                        for badge in summary_df.itertuples(index=False):
                            badge_id = getattr(badge, "badge_id", "")
                            badge_name = getattr(badge, "name", "Badge")
                            stack = getattr(badge, "stack_count", 1)
                            stack_text = f" x{stack}" if stack and stack > 1 else ""
                            icon = badge_icon(badge_id, getattr(badge, "category", None))
                            with st.expander(f"{icon} {badge_name}{stack_text}", expanded=False):
                                rows = pb_df[pb_df.get("badge_id") == badge_id].copy()
                                rows = rows.sort_values("earned_at_dt", ascending=False)
                                cols = ["earned_at_dt", "match_id"]
                                if admin_debug:
                                    cols.append("context_id")
                                show_rows = rows[cols].rename(
                                    columns={
                                        "earned_at_dt": "Earned",
                                        "match_id": "Match",
                                        "context_id": "Context",
                                    }
                                )
                                st.dataframe(
                                    show_rows,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Earned": st.column_config.DatetimeColumn(format="YYYY-MM-DD"),
                                    },
                                )
            st.subheader("Story Cards")
            story_df = fetch_player_stories(_supabase, club_id, pid, limit=6)
            if story_df.empty:
                st.caption("No new stories in the tape room yet.")
            else:
                story_df = story_df.drop_duplicates(subset=["story_type", "context_id"], keep="first")
                story_df = story_df.sort_values("created_at", ascending=False)
                highlights = story_df[story_df["story_type"].str.startswith("highlight", na=False)].head(3)
                foreshadow = story_df[story_df["story_type"].str.startswith("foreshadow", na=False)].head(3)
                highlight_col, foreshadow_col = st.columns(2)
                with highlight_col:
                    st.markdown("**Highlights**")
                    if highlights.empty:
                        st.caption("No highlights yet.")
                    else:
                        for _, row in highlights.iterrows():
                            title = html.escape(str(row.get("title") or "Highlight"))
                            body = html.escape(str(row.get("body") or ""))
                            st.markdown(f"**{title}**")
                            st.caption(body)
                with foreshadow_col:
                    st.markdown("**Foreshadowing**")
                    if foreshadow.empty:
                        st.caption("No foreshadowing yet.")
                    else:
                        for _, row in foreshadow.iterrows():
                            title = html.escape(str(row.get("title") or "Foreshadowing"))
                            body = html.escape(str(row.get("body") or ""))
                            st.markdown(f"**{title}**")
                            st.caption(body)

    def render_ratings_tab():
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

        st.divider()

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

    with ratings_tab:
        render_ratings_tab()
