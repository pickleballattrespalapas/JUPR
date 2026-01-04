# =========================
# JUPR Leagues (PART 1 / 2)
# Paste this first half, then I’ll send PART 2 when you ask.
# =========================

import streamlit as st
import pandas as pd
from supabase import create_client
import time
from datetime import datetime
import re
import altair as alt
import urllib.parse

def sb_retry(fn, retries: int = 4, base_sleep: float = 0.6):
    """
    Retries transient Supabase/httpx failures.
    """
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2 ** attempt))
    raise last

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="JUPR Leagues", layout="wide", page_icon="🌵")

# Clear cache once per session (helps when you add columns in Supabase)
if "cache_cleared" not in st.session_state:
    st.cache_resource.clear()
    st.session_state.cache_cleared = True

# Custom CSS
st.markdown(
    """
<style>
    .stDataFrame { font-size: 1.05rem; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; color: #00C0F2; }
    .court-header { background-color: #f0f2f6; padding: 10px; border-radius: 5px; font-weight: bold; margin-top: 20px; font-size: 1.2rem; }
    .move-up { color: #28a745; font-weight: bold; font-size: 1.1rem; }
    .move-down { color: #dc3545; font-weight: bold; font-size: 1.1rem; }
    .move-stay { color: #6c757d; font-weight: bold; }
    div[data-testid="stVerticalBlock"] > div { gap: 0.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

# --- CONFIGURATION ---
DEFAULT_K_FACTOR = 32
CLUB_ID = "tres_palapas"

# --- QUERY PARAM HELPERS / DEEP LINKS ---
def qp_get(key: str, default: str = "") -> str:
    """Streamlit query params can be str or list depending on version."""
    try:
        v = st.query_params.get(key, default)
    except Exception:
        return default
    if isinstance(v, list):
        return v[0] if v else default
    return str(v) if v is not None else default

# --- MAGIC LINK LOGIN ---
admin_key = qp_get("admin_key", "")
if admin_key == st.secrets["supabase"]["admin_password"]:
        st.session_state.admin_logged_in = True

PUBLIC_MODE = qp_get("public", "0").lower() in ("1", "true", "yes", "y")
DEEP_PAGE = qp_get("page", "").lower().strip()
DEEP_LEAGUE = qp_get("league", "").strip()

PAGE_MAP = {
    "leaderboards": "🏆 Leaderboards",
    "players": "🔍 Player Search",
    "faqs": "❓ FAQs",
}

# Apply deep-links ONLY once per session, otherwise it overrides sidebar clicks on every rerun
# Apply deep-links ONLY once per session
if "deep_link_applied" not in st.session_state:
    st.session_state.deep_link_applied = False

if PUBLIC_MODE:
    st.session_state.admin_logged_in = False
    st.session_state["main_nav"] = "🏆 Leaderboards"
    st.markdown("<style>[data-testid='stSidebar']{display:none;} header{visibility:hidden;}</style>", unsafe_allow_html=True)
    st.session_state.deep_link_applied = True
else:
    if not st.session_state.deep_link_applied:
        if DEEP_PAGE in PAGE_MAP:
            st.session_state["main_nav"] = PAGE_MAP[DEEP_PAGE]
        if DEEP_LEAGUE:
            st.session_state["preselect_league"] = DEEP_LEAGUE
        st.session_state.deep_link_applied = True

def build_standings_link(league_name: str, public: bool = True) -> str:
    """
    Returns a shareable URL for public leaderboards, pre-selected to a league.

    Requires:
      - st.secrets["PUBLIC_BASE_URL"] (recommended), e.g. "https://juprleagues.com"
        If missing, it falls back to a relative URL like "?page=leaderboards&league=..."
    """
    try:
        base = str(st.secrets.get("PUBLIC_BASE_URL", "") or "").rstrip("/")
    except Exception:
        base = ""

    params = {
        "page": "leaderboards",
        "league": str(league_name),
    }
    if public:
        params["public"] = "1"

    q = urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)
    return f"{base}/?{q}" if base else f"?{q}"



# --- NAV <-> URL SYNC HELPERS ---
NAV_TO_PAGE = {
    "🏆 Leaderboards": "leaderboards",
    "🔍 Player Search": "players",
    "❓ FAQs": "faqs",
    "🏟️ League Manager": "league_manager",
    "📝 Match Uploader": "match_uploader",
    "👥 Player Editor": "player_editor",
    "📝 Match Log": "match_log",
    "⚙️ Admin Tools": "admin_tools",
    "📘 Admin Guide": "admin_guide",
}

PAGE_TO_NAV = {v: k for k, v in NAV_TO_PAGE.items()}

def sync_url_from_nav(selected_nav: str):
    """Write the current page into the URL (but don't force sidebar selection)."""
    try:
        st.query_params["page"] = NAV_TO_PAGE.get(selected_nav, "leaderboards")
    except Exception:
        pass

# --- DATABASE CONNECTION ---
@st.cache_resource
def init_supabase():
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)

try:
    supabase = init_supabase()
except Exception as e:
    st.error(f"❌ Connection Error: {e}")
    st.stop()

# -------------------------
# LOGIC ENGINES
# -------------------------
def calculate_hybrid_elo(t1_avg, t2_avg, score_t1, score_t2, k_factor=32):
    """
    Returns (delta_for_team1_players, delta_for_team2_players) in ELO points (not JUPR).
    Deltas returned are "as applied" (winner positive, loser negative), never both positive.
    """
    expected_t1 = 1 / (1 + 10 ** ((t2_avg - t1_avg) / 400))
    expected_t2 = 1 - expected_t1

    total_points = score_t1 + score_t2
    if total_points == 0:
        return 0.0, 0.0

    raw_delta_t1 = k_factor * 2 * ((score_t1 / total_points) - expected_t1)
    raw_delta_t2 = k_factor * 2 * ((score_t2 / total_points) - expected_t2)

    # enforce win/loss direction
    if score_t1 > score_t2:
        d1 = max(0.0, raw_delta_t1)
        if d1 == 0.0:
            d1 = 1.0
        d2 = -abs(raw_delta_t2) if raw_delta_t2 != 0 else -1.0
        return d1, d2

    if score_t2 > score_t1:
        d2 = max(0.0, raw_delta_t2)
        if d2 == 0.0:
            d2 = 1.0
        d1 = -abs(raw_delta_t1) if raw_delta_t1 != 0 else -1.0
        return d1, d2

    # tie
    return 0.0, 0.0

# -------------------------
# DATA LOADER
# -------------------------
def load_data():
    """
    Loads club-scoped tables and returns:
      df_players_all, df_players_active, df_leagues, df_matches, df_meta, name_to_id, id_to_name
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Players: pull ALL for id/name mapping + audit (including inactive)
            p_resp = (
                supabase.table("players")
                .select("*")
                .eq("club_id", CLUB_ID)
                .execute()
            )
            df_players_all = pd.DataFrame(p_resp.data)

            # Active players for menus / leaderboards
            if not df_players_all.empty and "active" in df_players_all.columns:
                df_players_active = df_players_all[df_players_all["active"] == True].copy()
            else:
                df_players_active = df_players_all.copy()

            # League ratings + meta + matches are always club-scoped
            l_resp = (
                supabase.table("league_ratings")
                .select("*")
                .eq("club_id", CLUB_ID)
                .execute()
            )
            df_leagues = pd.DataFrame(l_resp.data)

            m_resp = (
                supabase.table("matches")
                .select("*")
                .eq("club_id", CLUB_ID)
                .order("id", desc=True)
                .limit(5000)
                .execute()
            )
            df_matches = pd.DataFrame(m_resp.data)

            meta_resp = (
                supabase.table("leagues_metadata")
                .select("*")
                .eq("club_id", CLUB_ID)
                .execute()
            )
            df_meta = pd.DataFrame(meta_resp.data)

            # Mappings
            if not df_players_all.empty and "id" in df_players_all.columns and "name" in df_players_all.columns:
                id_to_name = dict(zip(df_players_all["id"], df_players_all["name"]))
                name_to_id = dict(zip(df_players_all["name"], df_players_all["id"]))
            else:
                id_to_name, name_to_id = {}, {}
                df_players_all = pd.DataFrame(columns=["id", "name", "rating", "wins", "losses", "matches_played", "active"])
                df_players_active = df_players_all.copy()

            # Match display helper columns
            if not df_matches.empty and id_to_name:
                for col_src, col_out in [
                    ("t1_p1", "p1"),
                    ("t1_p2", "p2"),
                    ("t2_p1", "p3"),
                    ("t2_p2", "p4"),
                ]:
                    if col_src in df_matches.columns:
                        df_matches[col_out] = df_matches[col_src].map(id_to_name)

            return (
                df_players_all,
                df_players_active,
                df_leagues,
                df_matches,
                df_meta,
                name_to_id,
                id_to_name,
            )

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            st.error(f"⚠️ Network Error: {e}")
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                {},
                {},
            )

# -------------------------
# HELPERS
# -------------------------
def elo_to_jupr(elo_score):
    try:
        return float(elo_score) / 400.0
    except Exception:
        return 0.0

def safe_add_player(name, rating_jupr):
    """
    rating_jupr is in JUPR (e.g. 3.5). Stored as ELO (x400).
    """
    try:
        supabase.table("players").insert(
            {
                "club_id": CLUB_ID,
                "name": str(name).strip(),
                "rating": float(rating_jupr) * 400.0,
                "starting_rating": float(rating_jupr) * 400.0,
                "wins": 0,
                "losses": 0,
                "matches_played": 0,
                "active": True,
            }
        ).execute()
        return True, ""
    except Exception as e:
        return False, str(e)

def get_match_schedule(format_type, players, custom_text=None):
    p = players

    # custom schedule override
    if custom_text and len(custom_text.strip()) > 5:
        matches = []
        lines = custom_text.strip().split("\n")
        r_num = 1
        for line in lines:
            nums = [int(x) for x in re.findall(r"\d+", line)]
            if len(nums) >= 4:
                idx = [n - 1 for n in nums[:4]]
                if all(0 <= i < len(p) for i in idx):
                    matches.append(
                        {"t1": [p[idx[0]], p[idx[1]]], "t2": [p[idx[2]], p[idx[3]]], "desc": f"Game {r_num}"}
                    )
                    r_num += 1
        if matches:
            return matches

    # standard templates
    needed = int(format_type.split("-")[0])
    if len(p) < needed:
        return []

    if format_type == "4-Player":
        return [
            {"t1": [p[1], p[0]], "t2": [p[2], p[3]], "desc": "Rnd 1"},
            {"t1": [p[3], p[1]], "t2": [p[0], p[2]], "desc": "Rnd 2"},
            {"t1": [p[3], p[0]], "t2": [p[1], p[2]], "desc": "Rnd 3"},
        ]
    if format_type == "5-Player":
        return [
            {"t1": [p[0], p[1]], "t2": [p[2], p[3]], "desc": "Rnd 1"},
            {"t1": [p[1], p[3]], "t2": [p[2], p[4]], "desc": "Rnd 2"},
            {"t1": [p[0], p[4]], "t2": [p[1], p[2]], "desc": "Rnd 3"},
            {"t1": [p[0], p[2]], "t2": [p[3], p[4]], "desc": "Rnd 4"},
            {"t1": [p[0], p[3]], "t2": [p[1], p[4]], "desc": "Rnd 5"},
        ]
    if format_type == "6-Player":
        return [
            {"t1": [p[0], p[1]], "t2": [p[2], p[4]], "desc": "R1"},
            {"t1": [p[2], p[5]], "t2": [p[0], p[4]], "desc": "R2"},
            {"t1": [p[1], p[3]], "t2": [p[4], p[5]], "desc": "R3"},
            {"t1": [p[0], p[5]], "t2": [p[1], p[2]], "desc": "R4"},
            {"t1": [p[0], p[3]], "t2": [p[1], p[4]], "desc": "R5"},
        ]
    if format_type == "8-Player":
        return [
            {"t1": [p[0], p[5]], "t2": [p[1], p[4]], "desc": "Rnd 1 (Ct 1)"},
            {"t1": [p[2], p[7]], "t2": [p[3], p[6]], "desc": "Rnd 1 (Ct 2)"},
            {"t1": [p[1], p[2]], "t2": [p[4], p[7]], "desc": "Rnd 2 (Ct 1)"},
            {"t1": [p[0], p[3]], "t2": [p[5], p[6]], "desc": "Rnd 2 (Ct 2)"},
            {"t1": [p[0], p[7]], "t2": [p[2], p[5]], "desc": "Rnd 3 (Ct 1)"},
            {"t1": [p[1], p[6]], "t2": [p[3], p[4]], "desc": "Rnd 3 (Ct 2)"},
            {"t1": [p[0], p[1]], "t2": [p[2], p[3]], "desc": "Rnd 4 (Ct 1)"},
            {"t1": [p[4], p[5]], "t2": [p[6], p[7]], "desc": "Rnd 4 (Ct 2)"},
            {"t1": [p[0], p[6]], "t2": [p[1], p[7]], "desc": "Rnd 5 (Ct 1)"},
            {"t1": [p[2], p[4]], "t2": [p[3], p[5]], "desc": "Rnd 5 (Ct 2)"},
            {"t1": [p[1], p[5]], "t2": [p[2], p[6]], "desc": "Rnd 6 (Ct 1)"},
            {"t1": [p[0], p[4]], "t2": [p[3], p[7]], "desc": "Rnd 6 (Ct 2)"},
            {"t1": [p[1], p[3]], "t2": [p[5], p[7]], "desc": "Rnd 7 (Ct 1)"},
            {"t1": [p[0], p[2]], "t2": [p[4], p[6]], "desc": "Rnd 7 (Ct 2)"},
        ]
    if format_type == "12-Player":
        # unchanged (your long template)
        return [
            {"t1": [p[2], p[5]], "t2": [p[3], p[10]], "desc": "Rnd 1 (Ct 1)"},
            {"t1": [p[4], p[6]], "t2": [p[8], p[9]], "desc": "Rnd 1 (Ct 2)"},
            {"t1": [p[11], p[0]], "t2": [p[1], p[7]], "desc": "Rnd 1 (Ct 3)"},
            {"t1": [p[5], p[8]], "t2": [p[6], p[2]], "desc": "Rnd 2 (Ct 1)"},
            {"t1": [p[7], p[9]], "t2": [p[0], p[1]], "desc": "Rnd 2 (Ct 2)"},
            {"t1": [p[11], p[3]], "t2": [p[4], p[10]], "desc": "Rnd 2 (Ct 3)"},
            {"t1": [p[10], p[1]], "t2": [p[3], p[4]], "desc": "Rnd 3 (Ct 1)"},
            {"t1": [p[11], p[6]], "t2": [p[7], p[2]], "desc": "Rnd 3 (Ct 2)"},
            {"t1": [p[8], p[0]], "t2": [p[9], p[5]], "desc": "Rnd 3 (Ct 3)"},
            {"t1": [p[11], p[9]], "t2": [p[10], p[5]], "desc": "Rnd 4 (Ct 1)"},
            {"t1": [p[0], p[3]], "t2": [p[1], p[8]], "desc": "Rnd 4 (Ct 2)"},
            {"t1": [p[2], p[4]], "t2": [p[6], p[7]], "desc": "Rnd 4 (Ct 3)"},
            {"t1": [p[3], p[6]], "t2": [p[4], p[0]], "desc": "Rnd 5 (Ct 1)"},
            {"t1": [p[5], p[7]], "t2": [p[9], p[10]], "desc": "Rnd 5 (Ct 2)"},
            {"t1": [p[11], p[1]], "t2": [p[2], p[8]], "desc": "Rnd 5 (Ct 3)"},
            {"t1": [p[8], p[10]], "t2": [p[1], p[2]], "desc": "Rnd 6 (Ct 1)"},
            {"t1": [p[11], p[4]], "t2": [p[5], p[0]], "desc": "Rnd 6 (Ct 2)"},
            {"t1": [p[6], p[9]], "t2": [p[7], p[3]], "desc": "Rnd 6 (Ct 3)"},
            {"t1": [p[11], p[7]], "t2": [p[8], p[3]], "desc": "Rnd 7 (Ct 1)"},
            {"t1": [p[9], p[1]], "t2": [p[10], p[6]], "desc": "Rnd 7 (Ct 2)"},
            {"t1": [p[0], p[2]], "t2": [p[4], p[5]], "desc": "Rnd 7 (Ct 3)"},
            {"t1": [p[1], p[4]], "t2": [p[2], p[9]], "desc": "Rnd 8 (Ct 1)"},
            {"t1": [p[3], p[5]], "t2": [p[7], p[8]], "desc": "Rnd 8 (Ct 2)"},
            {"t1": [p[11], p[10]], "t2": [p[0], p[6]], "desc": "Rnd 8 (Ct 3)"},
            {"t1": [p[6], p[8]], "t2": [p[10], p[0]], "desc": "Rnd 9 (Ct 1)"},
            {"t1": [p[4], p[7]], "t2": [p[5], p[1]], "desc": "Rnd 9 (Ct 2)"},
            {"t1": [p[11], p[2]], "t2": [p[3], p[9]], "desc": "Rnd 9 (Ct 3)"},
            {"t1": [p[11], p[5]], "t2": [p[6], p[1]], "desc": "Rnd 10 (Ct 1)"},
            {"t1": [p[9], p[0]], "t2": [p[2], p[3]], "desc": "Rnd 10 (Ct 2)"},
            {"t1": [p[7], p[10]], "t2": [p[8], p[4]], "desc": "Rnd 10 (Ct 3)"},
            {"t1": [p[10], p[2]], "t2": [p[0], p[7]], "desc": "Rnd 11 (Ct 1)"},
            {"t1": [p[11], p[8]], "t2": [p[9], p[4]], "desc": "Rnd 11 (Ct 2)"},
            {"t1": [p[1], p[3]], "t2": [p[5], p[6]], "desc": "Rnd 11 (Ct 3)"},
        ]

    return []

# -------------------------
# PROCESSOR (MATCH SAVE + SNAPSHOTS)
# -------------------------
def process_matches(match_list, name_to_id, df_players_all, df_leagues, df_meta):
    """
    - Applies overall rating updates to players table
    - Applies league rating updates to league_ratings table (skips PopUp)
    - Inserts match rows with snapshot start/end ratings for each player in that match
    """
    db_matches = []
    overall_updates = {}
    island_updates = {}

    def get_k(league_name):
        if df_meta is None or df_meta.empty:
            return DEFAULT_K_FACTOR
        row = df_meta[df_meta["league_name"] == league_name]
        if not row.empty:
            try:
                return int(row.iloc[0].get("k_factor", DEFAULT_K_FACTOR))
            except Exception:
                return DEFAULT_K_FACTOR
        return DEFAULT_K_FACTOR

    def get_player_row(pid):
        row = df_players_all[df_players_all["id"] == pid]
        if row.empty:
            return None
        return row.iloc[0]

    def get_overall_r(pid):
        if pid in overall_updates:
            return overall_updates[pid]["r"]
        pr = get_player_row(pid)
        if pr is None:
            return 1200.0
        return float(pr.get("rating", 1200.0) or 1200.0)

    def get_island_r(pid, league_name):
        key = (pid, league_name)
        if key in island_updates:
            return island_updates[key]["r"]

        if df_leagues is not None and not df_leagues.empty:
            m = df_leagues[(df_leagues["player_id"] == pid) & (df_leagues["league_name"] == league_name)]
            if not m.empty:
                return float(m.iloc[0].get("rating", 1200.0) or 1200.0)

        # fallback: current overall
        return get_overall_r(pid)

    def ensure_overall_entry(pid):
        if pid in overall_updates:
            return
        pr = get_player_row(pid)
        if pr is None:
            overall_updates[pid] = {"r": 1200.0, "w": 0, "l": 0, "mp": 0}
            return
        overall_updates[pid] = {
            "r": float(pr.get("rating", 1200.0) or 1200.0),
            "w": int(pr.get("wins", 0) or 0),
            "l": int(pr.get("losses", 0) or 0),
            "mp": int(pr.get("matches_played", 0) or 0),
        }

    def ensure_island_entry(pid, league_name):
        key = (pid, league_name)
        if key in island_updates:
            return
        start = float(get_island_r(pid, league_name))  # BEFORE this batch applies changes
        island_updates[key] = {"r": start, "start": start, "w": 0, "l": 0, "mp": 0}


    for m in match_list:
        # map names -> ids
        p1 = name_to_id.get(m.get("t1_p1"))
        p2 = name_to_id.get(m.get("t1_p2"))
        p3 = name_to_id.get(m.get("t2_p1"))
        p4 = name_to_id.get(m.get("t2_p2"))

        if not p1 or not p3:
            continue

        s1 = int(m.get("s1", 0) or 0)
        s2 = int(m.get("s2", 0) or 0)
        league_name = str(m.get("league", "") or "").strip()
        week_tag = str(m.get("week_tag", "") or "")
        match_type = str(m.get("match_type", "") or "")
        is_popup = bool(m.get("is_popup", False)) or (match_type == "PopUp")

        # --- snapshots start (overall only; league snapshots are computed same way for storage consistency) ---
        ro1, ro2, ro3, ro4 = get_overall_r(p1), get_overall_r(p2), get_overall_r(p3), get_overall_r(p4)

        # overall deltas
        do1, do2 = calculate_hybrid_elo((ro1 + ro2) / 2, (ro3 + ro4) / 2, s1, s2, k_factor=DEFAULT_K_FACTOR)

        # league deltas (if official league)
        di1, di2 = 0.0, 0.0
        if not is_popup:
            k_val = get_k(league_name)
            ri1, ri2, ri3, ri4 = (
                get_island_r(p1, league_name),
                get_island_r(p2, league_name),
                get_island_r(p3, league_name),
                get_island_r(p4, league_name),
            )
            di1, di2 = calculate_hybrid_elo((ri1 + ri2) / 2, (ri3 + ri4) / 2, s1, s2, k_factor=k_val)

        win_team1 = s1 > s2

        def apply_updates(pid, d_ov, d_isl, won):
            if pid is None:
                return 1200.0
            ensure_overall_entry(pid)
            overall_updates[pid]["r"] += float(d_ov)
            overall_updates[pid]["mp"] += 1
            if won:
                overall_updates[pid]["w"] += 1
            else:
                overall_updates[pid]["l"] += 1

            if not is_popup:
                ensure_island_entry(pid, league_name)
                key = (pid, league_name)
                island_updates[key]["r"] += float(d_isl)
                island_updates[key]["mp"] += 1
                if won:
                    island_updates[key]["w"] += 1
                else:
                    island_updates[key]["l"] += 1

            return float(overall_updates[pid]["r"])

        # end snapshots
        end_r1 = apply_updates(p1, do1, di1, win_team1)
        end_r2 = apply_updates(p2, do1, di1, win_team1)
        end_r3 = apply_updates(p3, do2, di2, not win_team1)
        end_r4 = apply_updates(p4, do2, di2, not win_team1)

        # store elo_delta as magnitude for reference; snapshots are the source of truth for player audit
        stored_elo_delta = abs(do1) if win_team1 else abs(do2)

        db_matches.append(
            {
                "club_id": CLUB_ID,
                "date": m.get("date"),
                "league": league_name,
                "t1_p1": p1,
                "t1_p2": p2,
                "t2_p1": p3,
                "t2_p2": p4,
                "score_t1": s1,
                "score_t2": s2,
                "elo_delta": stored_elo_delta,
                "match_type": match_type,
                "week_tag": week_tag,
                # snapshots (overall)
                "t1_p1_r": ro1,
                "t1_p2_r": ro2,
                "t2_p1_r": ro3,
                "t2_p2_r": ro4,
                "t1_p1_r_end": end_r1,
                "t1_p2_r_end": end_r2,
                "t2_p1_r_end": end_r3,
                "t2_p2_r_end": end_r4,
            }
        )

    # Write matches
    if db_matches:
    CHUNK_M = 300
    for i in range(0, len(db_matches), CHUNK_M):
        chunk = db_matches[i : i + CHUNK_M]
        sb_retry(lambda chunk=chunk: supabase.table("matches").insert(chunk).execute())

    # ---- Overall player updates (BULK, chunked) ----
    def upsert_players_chunk(chunk_rows):
        # supabase-py versions differ slightly; handle both
        try:
            return supabase.table("players").upsert(chunk_rows, on_conflict="id").execute()
        except TypeError:
            return supabase.table("players").upsert(chunk_rows).execute()

    player_rows = []
    for pid, stats in overall_updates.items():
        player_rows.append(
            {
                "id": int(pid),
                "club_id": CLUB_ID,  # keep RLS / filters happy
                "rating": float(stats["r"]),
                "wins": int(stats["w"]),
                "losses": int(stats["l"]),
                "matches_played": int(stats["mp"]),
            }
        )

    CHUNK = 200
    for i in range(0, len(player_rows), CHUNK):
        chunk = player_rows[i : i + CHUNK]
        sb_retry(lambda chunk=chunk: upsert_players_chunk(chunk))

    # ---- League ratings updates (your existing code continues) ----
    if island_updates:
        for (pid, league_name), stats in island_updates.items():
            payload = {
                "club_id": CLUB_ID,
                "player_id": int(pid),
                "league_name": league_name,
                "rating": float(stats["r"]),
                "wins": int(stats["w"]),
                "losses": int(stats["l"]),
                "matches_played": int(stats["mp"]),
            }
            # ... keep the rest of your existing league upsert logic ...


            existing = sb_retry(lambda: (
                supabase.table("league_ratings")
                .select("id,wins,losses,matches_played,starting_rating")
                .eq("club_id", CLUB_ID)
                .eq("player_id", int(pid))
                .eq("league_name", league_name)
                .limit(1)
                .execute()
            ))


            if existing.data:
                cur = existing.data[0]

                payload["wins"] += int(cur.get("wins", 0) or 0)
                payload["losses"] += int(cur.get("losses", 0) or 0)
                payload["matches_played"] += int(cur.get("matches_played", 0) or 0)

                # preserve starting_rating if present, otherwise use the true league-entry snapshot we captured
                if cur.get("starting_rating") is not None:
                    payload["starting_rating"] = float(cur["starting_rating"])
                else:
                    payload["starting_rating"] = float(island_updates[(pid, league_name)].get("start", get_overall_r(int(pid))))

                supabase.table("league_ratings").update(payload).eq("id", int(cur["id"])).execute()
            else:
                payload["starting_rating"] = float(island_updates[(pid, league_name)].get("start", get_overall_r(int(pid))))
                supabase.table("league_ratings").insert(payload).execute()


# -------------------------
# MAIN APP LOAD
# -------------------------
df_players_all, df_players, df_leagues, df_matches, df_meta, name_to_id, id_to_name = load_data()

# -------------------------
# SIDEBAR (LOGIN)
# -------------------------
st.sidebar.title("JUPR Leagues 🌵")

if not st.session_state.admin_logged_in:
    with st.sidebar.expander("🔒 Admin Login"):
        pwd = st.text_input("Password", type="password")
        if st.button("Login"):
            if pwd == st.secrets["supabase"]["admin_password"]:
                st.session_state.admin_logged_in = True
                st.rerun()
else:
    st.sidebar.success("Logged In: Admin")
    if st.sidebar.button("Log Out"):
        st.session_state.admin_logged_in = False
        st.rerun()

# -------------------------
# NAVIGATION
# -------------------------
nav = ["🏆 Leaderboards", "🔍 Player Search", "❓ FAQs"]
if st.session_state.admin_logged_in:
    nav += ["──────────", "🏟️ League Manager", "📝 Match Uploader", "👥 Player Editor", "📝 Match Log", "⚙️ Admin Tools", "📘 Admin Guide"]

sel = st.sidebar.radio("Go to:", nav, key="main_nav")

if not PUBLIC_MODE:
    sync_url_from_nav(sel)

# =========================
# UI PAGES (PART 1)
# =========================

if sel == "🏆 Leaderboards":
    st.header("🏆 Leaderboards")

    if df_meta is not None and not df_meta.empty and "is_active" in df_meta.columns:
        active_meta = df_meta[df_meta["is_active"] == True]
        available_leagues = ["OVERALL"] + sorted(active_meta["league_name"].dropna().unique().tolist())
    else:
        available_leagues = ["OVERALL"]

    # Preselect league if the URL provided one
    pre = st.session_state.get("preselect_league", "")
    default_idx = 0
    if pre and pre in available_leagues:
        default_idx = available_leagues.index(pre)

    target_league = st.selectbox("Select View", available_leagues, index=default_idx, key="lb_league")

    # Keep URL in sync (handy even in private mode)
    try:
        st.query_params["page"] = "leaderboards"
        st.query_params["league"] = target_league
        if PUBLIC_MODE:
            st.query_params["public"] = "1"
    except Exception:
        pass

    # Shareable link
    st.caption("Share standings:")
    share_link = build_standings_link(target_league, public=True)
    st.text_input("Public standings link", value=share_link)
    try:
        st.link_button("Open Public Standings", share_link)
    except Exception:
        pass

    # min games
    min_games_req = 0
    if target_league != "OVERALL" and df_meta is not None and not df_meta.empty:
        cfg = df_meta[df_meta["league_name"] == target_league]
        if not cfg.empty:
            try:
                min_games_req = int(cfg.iloc[0].get("min_games", 0) or 0)
            except Exception:
                min_games_req = 0

    # build display df
    if target_league == "OVERALL":
        display_df = df_players.copy() if df_players is not None else pd.DataFrame()
        if not display_df.empty and "name" in display_df.columns:
            # ensure starting_rating exists
            if "starting_rating" not in display_df.columns:
                display_df["starting_rating"] = display_df.get("rating", 1200.0)
            # normalize required columns
            for c in ["wins", "losses", "matches_played", "rating"]:
                if c not in display_df.columns:
                    display_df[c] = 0
    else:
        if df_leagues is None or df_leagues.empty:
            display_df = pd.DataFrame()
        else:
            display_df = df_leagues[df_leagues["league_name"] == target_league].copy()
            display_df["name"] = display_df["player_id"].map(id_to_name)
            if "starting_rating" not in display_df.columns:
                # fallback if not loaded yet
                display_df["starting_rating"] = display_df["rating"]

    if display_df is None or display_df.empty or "rating" not in display_df.columns:
        st.info("No data.")
    else:
        # computed columns
        display_df["JUPR"] = display_df["rating"].astype(float) / 400.0
        mp = display_df["matches_played"].replace(0, 1).astype(float)
        display_df["Win %"] = (display_df["wins"].astype(float) / mp) * 100.0

        # Gain = current - starting (ELO points)
        def calc_gain(row):
            cur = float(row.get("rating", 0) or 0)
            start = float(row.get("starting_rating", cur) or cur)
            return cur - start

        display_df["rating_gain"] = display_df.apply(calc_gain, axis=1)

        if target_league != "OVERALL":
            qualified_df = display_df[display_df["matches_played"].astype(int) >= int(min_games_req)].copy()
            if not qualified_df.empty:
                st.markdown(f"### 🏅 Top Performers (Min {min_games_req} Games)")
                c1, c2, c3, c4 = st.columns(4)

                with c1:
                    st.markdown("**👑 Highest Rating**")
                    top = qualified_df.sort_values("rating", ascending=False).head(5)
                    for _, r in top.iterrows():
                        st.markdown(f"**{float(r['JUPR']):.3f}** - {r['name']}")

                with c2:
                    st.markdown("**🔥 Most Improved**")
                    top = qualified_df.sort_values("rating_gain", ascending=False).head(5)
                    for _, r in top.iterrows():
                        st.markdown(f"**{(float(r['rating_gain'])/400.0):+.3f}** - {r['name']}")

                with c3:
                    st.markdown("**🎯 Best Win %**")
                    top = qualified_df.sort_values("Win %", ascending=False).head(5)
                    for _, r in top.iterrows():
                        st.markdown(f"**{float(r['Win %']):.1f}%** - {r['name']}")

                with c4:
                    st.markdown("**🚜 Most Wins**")
                    top = qualified_df.sort_values("wins", ascending=False).head(5)
                    for _, r in top.iterrows():
                        st.markdown(f"**{int(r['wins'])} Wins** - {r['name']}")

                st.divider()

        st.markdown("### 📊 Standings")
        final_view = display_df.sort_values("rating", ascending=False).copy()
        final_view["Rank"] = range(1, len(final_view) + 1)
        final_view["Rank"] = final_view["Rank"].apply(
            lambda r: "🥇" if r == 1 else "🥈" if r == 2 else "🥉" if r == 3 else str(r)
        )
        final_view["Gain"] = (final_view["rating_gain"].astype(float) / 400.0).map("{:+.3f}".format)

        cols_to_show = ["Rank", "name", "JUPR", "Gain", "matches_played", "wins", "losses", "Win %"]
        cols_to_show = [c for c in cols_to_show if c in final_view.columns]

        st.dataframe(final_view[cols_to_show], use_container_width=True, hide_index=True)

elif sel == "🔍 Player Search":
    st.header("🕵️ Player Search & Audit")

    # helper to read snapshots for a specific pid from a match row
    def get_player_snap(match_row, pid):
        if match_row.get("t1_p1") == pid:
            return match_row.get("t1_p1_r"), match_row.get("t1_p1_r_end")
        if match_row.get("t1_p2") == pid:
            return match_row.get("t1_p2_r"), match_row.get("t1_p2_r_end")
        if match_row.get("t2_p1") == pid:
            return match_row.get("t2_p1_r"), match_row.get("t2_p1_r_end")
        if match_row.get("t2_p2") == pid:
            return match_row.get("t2_p2_r"), match_row.get("t2_p2_r_end")
        return None, None

    # Active players (club-scoped)
    players_resp = (
        supabase.table("players")
        .select("id,name,rating")
        .eq("club_id", CLUB_ID)
        .eq("active", True)
        .execute()
    )
    players_df = pd.DataFrame(players_resp.data)

    if players_df.empty:
        st.warning("No active players found.")
    else:
        player_names = sorted(players_df["name"].astype(str).tolist())
        selected_name = st.selectbox("Select a Player:", player_names)

        selected_player = players_df[players_df["name"] == selected_name].iloc[0]
        p_id = int(selected_player["id"])

        raw_elo = float(selected_player["rating"])
        current_jupr_rating = elo_to_jupr(raw_elo)

        c1, c2 = st.columns(2)
        c1.metric("Player Name", selected_name)
        c2.metric("Current JUPR", f"{current_jupr_rating:.3f}")

        # Match history (club-scoped)
        resp = (
            supabase.table("matches")
            .select("*")
            .eq("club_id", CLUB_ID)
            .or_(f"t1_p1.eq.{p_id},t1_p2.eq.{p_id},t2_p1.eq.{p_id},t2_p2.eq.{p_id}")
            .order("date", desc=True)
            .order("id", desc=True)
            .execute()
        )
        matches_data = resp.data

        if not matches_data:
            st.info("This player has no recorded matches yet.")
        else:
            processed = []
            for match in matches_data:
                s1 = int(match.get("score_t1", 0) or 0)
                s2 = int(match.get("score_t2", 0) or 0)
                display_score = f"{s1}-{s2}"

                start_elo, end_elo = get_player_snap(match, p_id)

                if start_elo is not None and end_elo is not None:
                    start_elo = float(start_elo)
                    end_elo = float(end_elo)
                    jupr_change = (end_elo - start_elo) / 400.0
                    rating_after = end_elo / 400.0
                else:
                    # fallback for any legacy rows without snapshots
                    raw_delta = float(match.get("elo_delta", 0) or 0)
                    my_team = 1 if (match.get("t1_p1") == p_id or match.get("t1_p2") == p_id) else 2
                    winner_team = 1 if s1 > s2 else 2 if s2 > s1 else 0
                    if winner_team == 0:
                        signed_elo = 0.0
                    elif winner_team == my_team:
                        signed_elo = abs(raw_delta)
                    else:
                        signed_elo = -abs(raw_delta)

                    jupr_change = signed_elo / 400.0
                    rating_after = None

                processed.append(
                    {
                        "Date": match.get("date"),
                        "Score": display_score,
                        "JUPR Change": jupr_change,
                        "Rating After Match": rating_after,
                    }
                )

            display_df = pd.DataFrame(processed)
            display_df["Date"] = pd.to_datetime(display_df["Date"], errors="coerce")
            display_df = display_df.dropna(subset=["Date"])

            if display_df.empty:
                st.warning("No valid dated matches found for this player.")
            else:
                # Backfill missing "Rating After Match" by backtracking from current rating
                if display_df["Rating After Match"].isna().any():
                    tmp = display_df.copy()
                    tmp["Undo"] = tmp["JUPR Change"].shift(1).fillna(0)
                    tmp["Backtrack"] = tmp["Undo"].cumsum()
                    tmp.loc[tmp["Rating After Match"].isna(), "Rating After Match"] = current_jupr_rating - tmp["Backtrack"]
                    display_df = tmp

                # Graph order oldest -> newest
                graph_df = display_df.iloc[::-1].reset_index(drop=True)
                graph_df["Match Sequence"] = graph_df.index + 1

                st.subheader("Rating Trend")
                st.caption("JUPR rating progression over recent matches")

                chart = (
                    alt.Chart(graph_df.tail(30))
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("Match Sequence", axis=alt.Axis(tickMinStep=1), title="Match Order"),
                        y=alt.Y(
                            "Rating After Match",
                            axis=alt.Axis(format=".3f"),
                            title="JUPR Rating",
                            scale=alt.Scale(zero=False),
                        ),
                        tooltip=[
                            "Match Sequence",
                            alt.Tooltip("Date", title="Date"),
                            "Score",
                            alt.Tooltip("Rating After Match", format=".3f"),
                            alt.Tooltip("JUPR Change", format=".4f"),
                        ],
                    )
                    .interactive()
                )

                st.altair_chart(chart, use_container_width=True)

                st.subheader("Match Log")
                table_df = display_df.copy()
                table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
                table_df["JUPR Change"] = table_df["JUPR Change"].map(lambda x: f"{float(x):+.4f}")
                table_df["Rating After Match"] = table_df["Rating After Match"].map(lambda x: f"{float(x):.3f}")

                st.dataframe(
                    table_df[["Date", "Score", "JUPR Change", "Rating After Match"]],
                    use_container_width=True,
                    hide_index=True,
                )

elif sel == "❓ FAQs":
    st.header("❓ FAQs")
    st.markdown(
        """
**What is JUPR?**  
JUPR is our internal rating system for Tres Palapas. Ratings update from match results and are displayed as a 1.000–7.000 style number.

**Do pop-up events affect official league ratings?**  
No. Pop-ups update overall ratings only (unless you decide otherwise later).

**Why do I see rating changes even in close games?**  
Expected outcome is based on both teams’ average ratings and the score ratio, not just win/loss.
"""
    )
# =========================
# JUPR Leagues (PART 2 / 2)
# Paste this after PART 1.
# =========================
def normalize_slots(roster_df: pd.DataFrame) -> pd.DataFrame:
    df = roster_df.copy()
    if "slot" not in df.columns:
        df["slot"] = 1
    df["court"] = df["court"].astype(int)
    df["slot"] = df["slot"].astype(int)

    for c in sorted(df["court"].unique()):
        idx = df[df["court"] == c].sort_values("slot").index
        df.loc[idx, "slot"] = list(range(1, len(idx) + 1))
    return df.sort_values(["court", "slot"]).reset_index(drop=True)

def swap_players(roster_df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    df = roster_df.copy()
    ia = df.index[df["name"] == a]
    ib = df.index[df["name"] == b]
    if len(ia) != 1 or len(ib) != 1:
        return df
    ia, ib = int(ia[0]), int(ib[0])

    # Swap court+slot (keeps court sizes constant)
    ca, sa = int(df.at[ia, "court"]), int(df.at[ia, "slot"])
    cb, sb = int(df.at[ib, "court"]), int(df.at[ib, "slot"])
    df.at[ia, "court"], df.at[ia, "slot"] = cb, sb
    df.at[ib, "court"], df.at[ib, "slot"] = ca, sa
    return normalize_slots(df)

def move_within_court(roster_df: pd.DataFrame, player: str, new_slot: int) -> pd.DataFrame:
    df = roster_df.copy()
    if player not in df["name"].tolist():
        return df
    row = df[df["name"] == player].iloc[0]
    c = int(row["court"])

    grp = df[df["court"] == c].sort_values("slot").copy()
    names = grp["name"].tolist()
    if player not in names:
        return df

    names.remove(player)
    new_slot = max(1, min(int(new_slot), len(names) + 1))
    names.insert(new_slot - 1, player)

    # Write back slot order
    for i, nm in enumerate(names, start=1):
        df.loc[(df["court"] == c) & (df["name"] == nm), "slot"] = i

    return normalize_slots(df)

# -------------------------
# PAGE: LEAGUE MANAGER
# -------------------------
if sel == "🏟️ League Manager":
    st.header("🏟️ League Manager")
    tabs = st.tabs(["🏃‍♂️ Run Live Event (Ladder)", "⚙️ Settings"])

    # ---------- TAB 1: LIVE LADDER ----------
    with tabs[0]:
        st.subheader("Ladder Management")

        if "ladder_state" not in st.session_state:
            st.session_state.ladder_state = "SETUP"
        if "ladder_roster" not in st.session_state:
            st.session_state.ladder_roster = []
        if "ladder_total_rounds" not in st.session_state:
            st.session_state.ladder_total_rounds = 5

        # ---- 1) SETUP ----
        if st.session_state.ladder_state == "SETUP":
            st.markdown("#### Step 1: Select League & Roster")

            if df_meta is not None and not df_meta.empty and "is_active" in df_meta.columns:
                opts = sorted(df_meta[df_meta["is_active"] == True]["league_name"].dropna().tolist())
                if not opts:
                    opts = ["Default"]
            else:
                opts = ["Default"]

            lg_select = st.selectbox("Select League", opts, key="ladder_lg")
            week_select = st.selectbox("Week", [f"Week {i}" for i in range(1, 13)] + ["Playoffs"], key="ladder_wk")
            num_rounds = st.number_input(
                "Total Rounds to Play",
                1, 20,
                value=int(st.session_state.get("ladder_total_rounds", 5)),
                step=1,
                key="ladder_total_rounds_input",
            )

            raw = st.text_area("Paste Player List (one per line)", height=150, key="ladder_raw_input")



            if st.button("Analyze & Seed"):
                st.session_state.saved_ladder_lg = st.session_state.ladder_lg
                st.session_state.saved_ladder_wk = st.session_state.ladder_wk

                # ✅ store the rounds value from the widget into your real variable
                st.session_state.ladder_total_rounds = int(st.session_state.get("ladder_total_rounds_input", 5))

                raw_txt = st.session_state.get("ladder_raw_input", "") or ""
                parsed = [x.strip() for x in raw_txt.replace("\n", ",").split(",") if x.strip()]

                
                roster_data = []
                new_ps = []

                for n in parsed:
                    if n in name_to_id:
                        pid = name_to_id[n]

                        # start seeding with league rating if exists, else overall
                        r = 1200.0
                        if df_leagues is not None and not df_leagues.empty:
                            row = df_leagues[(df_leagues["player_id"] == pid) & (df_leagues["league_name"] == lg_select)]
                            if not row.empty:
                                r = float(row.iloc[0].get("rating", 1200.0) or 1200.0)
                            else:
                                row_g = df_players_all[df_players_all["id"] == pid]
                                if not row_g.empty:
                                    r = float(row_g.iloc[0].get("rating", 1200.0) or 1200.0)
                        else:
                            row_g = df_players_all[df_players_all["id"] == pid]
                            if not row_g.empty:
                                r = float(row_g.iloc[0].get("rating", 1200.0) or 1200.0)

                        roster_data.append({"name": n, "rating": r, "id": pid, "status": "Found"})
                    else:
                        new_ps.append(n)

                st.session_state.ladder_temp_roster = roster_data
                st.session_state.ladder_temp_new = new_ps
                st.session_state.ladder_state = "REVIEW_ROSTER"
                st.rerun()

        # ---- 2) REVIEW / NEW PLAYERS ----
        if st.session_state.ladder_state == "REVIEW_ROSTER":
            c_back, _ = st.columns([1, 5])
            if c_back.button("⬅️ Back (edit league/week/rounds/roster)"):
                st.session_state.ladder_state = "SETUP"
                st.rerun()
            st.markdown("#### Step 2: Confirm Roster")

            if st.session_state.ladder_temp_new:
                st.warning(f"Found {len(st.session_state.ladder_temp_new)} new players.")
                df_new = pd.DataFrame({"Name": st.session_state.ladder_temp_new, "Rating": [3.5] * len(st.session_state.ladder_temp_new)})
                edited_new = st.data_editor(
                    df_new,
                    column_config={"Rating": st.column_config.NumberColumn(min_value=1.0, max_value=7.0, step=0.1)},
                    hide_index=True,
                    use_container_width=True,
                )

                if st.button("Save New Players & Continue"):
                    for _, r in edited_new.iterrows():
                        ok, err = safe_add_player(r["Name"], r["Rating"])
                        if not ok:
                            st.error(f"Could not add {r['Name']}: {err}")

                    # reload data so name_to_id includes new players
                    (
                        df_players_all,
                        df_players,
                        df_leagues,
                        df_matches,
                        df_meta,
                        name_to_id,
                        id_to_name,
                    ) = load_data()

                    # rebuild roster list with ids
                    rebuilt = []
                    for item in st.session_state.ladder_temp_roster:
                        rebuilt.append(item)

                    for nm in st.session_state.ladder_temp_new:
                        pid = name_to_id.get(nm)
                        # seed new players from their current overall (or default)
                        base = 1200.0
                        if pid:
                            row_g = df_players_all[df_players_all["id"] == pid]
                            if not row_g.empty:
                                base = float(row_g.iloc[0].get("rating", 1200.0) or 1200.0)
                        rebuilt.append({"name": nm, "rating": base, "id": pid, "status": "New"})

                    st.session_state.ladder_roster = sorted(rebuilt, key=lambda x: float(x["rating"]), reverse=True)
                    st.session_state.ladder_state = "CONFIG_COURTS"
                    st.rerun()

            else:
                st.success("All players found.")
                if st.button("Proceed to Court Setup"):
                    st.session_state.ladder_roster = sorted(st.session_state.ladder_temp_roster, key=lambda x: float(x["rating"]), reverse=True)
                    st.session_state.ladder_state = "CONFIG_COURTS"
                    st.rerun()

        # ---- 3) CONFIG COURTS ----
        if st.session_state.ladder_state == "CONFIG_COURTS":
            c_back, _ = st.columns([1, 5])
            if c_back.button("⬅️ Back (edit roster)"):
                st.session_state.ladder_state = "REVIEW_ROSTER"
                st.rerun()

            st.markdown("#### Step 3: Configure Courts")
            total_p = len(st.session_state.ladder_roster)
            st.info(f"Total Players: {total_p}")

            num_courts = st.number_input("Number of Courts", 1, 10, key="ladder_num_courts", value=st.session_state.get("ladder_num_courts", 3))
            court_sizes = []
            cols = st.columns(int(num_courts))
            for i in range(int(num_courts)):
                with cols[i]:
                    s = st.number_input(f"Ct {i+1} Size", 4, 12, 4, key=f"cs_{i}")
                    court_sizes.append(int(s))

            if sum(court_sizes) != total_p:
                st.error(f"Court sizes sum to {sum(court_sizes)}, but you have {total_p} players.")
            else:
                if st.button("Preview Assignments"):
                    current_idx = 0
                    final_assignments = []
                    for c_idx, size in enumerate(court_sizes):
                        group = st.session_state.ladder_roster[current_idx : current_idx + size]
                        for pl in group:
                            final_assignments.append({"name": pl["name"], "rating": pl["rating"], "court": c_idx + 1})
                        current_idx += size

                    final_roster = pd.DataFrame(final_assignments)

                    final_roster = final_roster.sort_values(["court", "rating"], ascending=[True, False]).copy()

                    final_roster["slot"] = final_roster.groupby("court").cumcount() + 1

                    st.session_state.ladder_live_roster = final_roster[["name", "rating", "court", "slot"]].copy()
                    st.session_state.ladder_court_sizes = court_sizes
                    st.session_state.ladder_state = "CONFIRM_START"
                    st.rerun()


        # ---- 3.5) CONFIRM START ----
        if st.session_state.ladder_state == "CONFIRM_START":
            c_back, _ = st.columns([1, 5])
            if c_back.button("⬅️ Back (edit courts)"):
        # optional: clear preview so it forces a fresh preview next time
                if "ladder_live_roster" in st.session_state:
                    del st.session_state.ladder_live_roster
                st.session_state.ladder_state = "CONFIG_COURTS"
                st.rerun()

            st.markdown("#### Step 4: Confirm Starting Positions")
            st.markdown("Verify the court assignments. Ratings shown in **JUPR**.")

            display_roster = st.session_state.ladder_live_roster.copy()
            display_roster["JUPR Rating"] = display_roster["rating"].astype(float) / 400.0

            edited_roster = st.data_editor(
                display_roster[["name", "JUPR Rating", "court"]],
                column_config={
                    "court": st.column_config.NumberColumn("Court", min_value=1, max_value=10, step=1),
                    "JUPR Rating": st.column_config.NumberColumn("JUPR Rating", format="%.3f", disabled=True),
                },
                hide_index=True,
                use_container_width=True,
            )

            if st.button("✅ Start Event (Round 1)"):
                final_roster = edited_roster.copy()
                final_roster["rating"] = final_roster["JUPR Rating"].astype(float) * 400.0

                # IMPORTANT: enforce stable ordering + slots so schedule is deterministic
                final_roster = final_roster.sort_values(["court", "name"]).copy()
                final_roster["slot"] = final_roster.groupby("court").cumcount() + 1

                new_sizes = final_roster["court"].value_counts().sort_index().tolist()
                st.session_state.ladder_court_sizes = new_sizes
                st.session_state.ladder_live_roster = final_roster[["name", "rating", "court", "slot"]].copy()

                st.session_state.ladder_round_num = 1
                st.session_state.ladder_state = "PLAY_ROUND"
                if "current_schedule" in st.session_state:
                    del st.session_state.current_schedule
                st.rerun()


        # ---- 4) PLAY ROUND ----
        if st.session_state.ladder_state == "PLAY_ROUND":
            current_r = int(st.session_state.ladder_round_num)
            total_r = int(st.session_state.ladder_total_rounds)

            st.markdown(f"### 🎾 Round {current_r} / {total_r}")
            # Ensure roster has slots (older sessions)
            if "slot" not in st.session_state.ladder_live_roster.columns:
                tmp = st.session_state.ladder_live_roster.copy()
                tmp = tmp.sort_values(["court", "name"]).copy()
                tmp["slot"] = tmp.groupby("court").cumcount() + 1
                st.session_state.ladder_live_roster = tmp

            with st.expander("✏️ Quick court edits (before scoring)", expanded=False):
                roster_df = normalize_slots(st.session_state.ladder_live_roster.copy())
                names_now = roster_df["name"].tolist()

                cA, cB, cC = st.columns([2, 2, 1])
                a = cA.selectbox("Swap Player A", names_now, key=f"swap_a_r{current_r}")
                b = cB.selectbox("with Player B", names_now, key=f"swap_b_r{current_r}", index=1 if len(names_now) > 1 else 0)
                if cC.button("Swap", key=f"swap_btn_r{current_r}"):
                    st.session_state.ladder_live_roster = swap_players(roster_df, a, b)
                    if "current_schedule" in st.session_state:
                        del st.session_state.current_schedule
                    st.rerun()

                st.divider()

                c1, c2, c3 = st.columns([2, 2, 1])
                court_list = sorted(roster_df["court"].unique().tolist())
                chosen_court = c1.selectbox("Court to reorder", court_list, key=f"re_ct_r{current_r}")
                court_players = roster_df[roster_df["court"] == int(chosen_court)].sort_values("slot")["name"].tolist()

                p = c2.selectbox("Player", court_players, key=f"re_p_r{current_r}")
                new_pos = c3.number_input("New position", min_value=1, max_value=max(1, len(court_players)), value=1, step=1, key=f"re_pos_r{current_r}")

                if st.button("Apply reorder", key=f"re_btn_r{current_r}"):
                    st.session_state.ladder_live_roster = move_within_court(roster_df, p, int(new_pos))
                    if "current_schedule" in st.session_state:
                        del st.session_state.current_schedule
                    st.rerun()

            if "current_schedule" not in st.session_state:
                schedule = []
                for c_idx in range(len(st.session_state.ladder_court_sizes)):
                    c_num = c_idx + 1
                    court_df = st.session_state.ladder_live_roster[st.session_state.ladder_live_roster["court"] == c_num].copy()
                    if "slot" in court_df.columns:
                        court_df = court_df.sort_values("slot")
                    players = court_df["name"].tolist()

                    fmt = f"{len(players)}-Player"
                    matches = get_match_schedule(fmt, players)
                    schedule.append({"c": c_num, "matches": matches})
                st.session_state.current_schedule = schedule

            all_results = []
            with st.form("round_score_form"):
                for c_data in st.session_state.current_schedule:
                    st.markdown(f"<div class='court-header'>Court {c_data['c']}</div>", unsafe_allow_html=True)
                    for m_idx, mm in enumerate(c_data["matches"]):
                        c1, c2, c3, c4 = st.columns([3, 1, 1, 3])
                        c1.text(f"{mm['t1'][0]} & {mm['t1'][1]}")
                        s1 = c2.number_input("S1", 0, key=f"r{current_r}_c{c_data['c']}_m{m_idx}_1")
                        s2 = c3.number_input("S2", 0, key=f"r{current_r}_c{c_data['c']}_m{m_idx}_2")
                        c4.text(f"{mm['t2'][0]} & {mm['t2'][1]}")
                        all_results.append(
                            {
                                "court": c_data["c"],
                                "t1_p1": mm["t1"][0],
                                "t1_p2": mm["t1"][1],
                                "t2_p1": mm["t2"][0],
                                "t2_p2": mm["t2"][1],
                                "s1": int(s1),
                                "s2": int(s2),
                            }
                        )

                if st.form_submit_button("Submit Round & Calculate Movement"):
                    valid_matches = []
                    for r in all_results:
                        if r["s1"] > 0 or r["s2"] > 0:
                            valid_matches.append(
                                {
                                    "t1_p1": r["t1_p1"],
                                    "t1_p2": r["t1_p2"],
                                    "t2_p1": r["t2_p1"],
                                    "t2_p2": r["t2_p2"],
                                    "s1": r["s1"],
                                    "s2": r["s2"],
                                    "date": str(datetime.now()),
                                    "league": st.session_state.saved_ladder_lg,
                                    "match_type": "Live Match",
                                    "week_tag": st.session_state.saved_ladder_wk,
                                    "is_popup": False,
                                }
                            )

                    if valid_matches:
                        process_matches(valid_matches, name_to_id, df_players_all, df_leagues, df_meta)
                        st.success("Matches Saved to Database!")

                        # movement calculations
                        round_stats = {}
                        all_names = st.session_state.ladder_live_roster["name"].unique()
                        for n in all_names:
                            round_stats[n] = {"w": 0, "diff": 0, "pts": 0}

                        for r in valid_matches:
                            win = r["s1"] > r["s2"]
                            diff = abs(r["s1"] - r["s2"])

                            for p in [r["t1_p1"], r["t1_p2"]]:
                                round_stats[p]["pts"] += r["s1"]
                                round_stats[p]["diff"] += diff if win else -diff
                                if win:
                                    round_stats[p]["w"] += 1

                            for p in [r["t2_p1"], r["t2_p2"]]:
                                round_stats[p]["pts"] += r["s2"]
                                round_stats[p]["diff"] += -diff if win else diff
                                if not win:
                                    round_stats[p]["w"] += 1

                        df_roster = st.session_state.ladder_live_roster.copy()
                        df_roster["Round Wins"] = df_roster["name"].map(lambda x: round_stats.get(x, {}).get("w", 0))
                        df_roster["Round Diff"] = df_roster["name"].map(lambda x: round_stats.get(x, {}).get("diff", 0))
                        df_roster["Round Pts"] = df_roster["name"].map(lambda x: round_stats.get(x, {}).get("pts", 0))

                        df_roster = df_roster.sort_values(by=["court", "Round Wins", "Round Diff", "Round Pts"], ascending=[True, False, False, False])
                        df_roster["Proposed Court"] = df_roster["court"]

                        max_court = len(st.session_state.ladder_court_sizes)
                        for c_num in sorted(df_roster["court"].unique()):
                            court_group = df_roster[df_roster["court"] == c_num]
                            if len(court_group) == 0:
                                continue
                            top_player = court_group.iloc[0]["name"]
                            btm_player = court_group.iloc[-1]["name"]
                            if c_num > 1:
                                df_roster.loc[df_roster["name"] == top_player, "Proposed Court"] = c_num - 1
                            if c_num < max_court:
                                df_roster.loc[df_roster["name"] == btm_player, "Proposed Court"] = c_num + 1

                        st.session_state.ladder_movement_preview = df_roster
                        st.session_state.ladder_state = "CONFIRM_MOVEMENT"
                        if "current_schedule" in st.session_state:
                            del st.session_state.current_schedule
                        st.rerun()

        # ---- 5) CONFIRM MOVEMENT ----
        if st.session_state.ladder_state == "CONFIRM_MOVEMENT":
            st.markdown("#### Round Results & Movement")

            preview_df = st.session_state.ladder_movement_preview.sort_values("court")
            for c_num in sorted(preview_df["court"].unique()):
                st.markdown(f"<div class='court-header'>Court {c_num} Results</div>", unsafe_allow_html=True)
                c_players = preview_df[preview_df["court"] == c_num]

                for _, p in c_players.iterrows():
                    move_icon = "➖"
                    move_class = "move-stay"
                    if int(p["Proposed Court"]) < int(p["court"]):
                        move_icon = f"🟢 ⬆️ To Ct {int(p['Proposed Court'])}"
                        move_class = "move-up"
                    elif int(p["Proposed Court"]) > int(p["court"]):
                        move_icon = f"🔴 ⬇️ To Ct {int(p['Proposed Court'])}"
                        move_class = "move-down"

                    col1, col2, col3 = st.columns([2, 2, 2])
                    col1.markdown(f"**{p['name']}**")
                    col2.markdown(f"W: {int(p['Round Wins'])} | Diff: {int(p['Round Diff'])}")
                    col3.markdown(f"<span class='{move_class}'>{move_icon}</span>", unsafe_allow_html=True)

                st.divider()

            st.markdown("#### 🛠️ Manual Override")
            st.info("If the arrows look right, click 'Start Next Round'. Otherwise, edit 'New Ct' below.")

            preview_df = st.session_state.ladder_movement_preview.copy()

            # Only show what we need to edit
            edit_view = preview_df[["name", "rating", "court", "Proposed Court"]].copy()

            editor_df = st.data_editor(
                edit_view,
                column_config={
                    "court": st.column_config.NumberColumn("Old Ct", disabled=True),
                    "Proposed Court": st.column_config.NumberColumn("New Ct", min_value=1, max_value=10, step=1),
                },
                hide_index=True,
                use_container_width=True,
            )

            btn_label = "Start Next Round"
            if int(st.session_state.ladder_round_num) >= int(st.session_state.ladder_total_rounds):
                btn_label = "🏁 Finish League Night"

            if st.button(btn_label):
                if int(st.session_state.ladder_round_num) >= int(st.session_state.ladder_total_rounds):
                    st.balloons()
                    st.success("League Night Complete! All matches saved.")
                    time.sleep(1)
                    st.session_state.ladder_state = "SETUP"
                    st.rerun()
                else:
                    new_roster = editor_df.copy()
                    new_roster["court"] = new_roster["Proposed Court"].astype(int)

                    # keep rating column from prior roster (editor already has rating, but keep safe)
                    rating_map = dict(zip(st.session_state.ladder_live_roster["name"], st.session_state.ladder_live_roster["rating"]))
                    new_roster["rating"] = new_roster["name"].map(lambda x: float(rating_map.get(x, 1200.0)))

                    # sort + slots
                    new_roster = new_roster.sort_values(["court", "name"]).copy()
                    new_roster["slot"] = new_roster.groupby("court").cumcount() + 1

                    new_sizes = new_roster["court"].value_counts().sort_index().tolist()
                    st.session_state.ladder_court_sizes = new_sizes
                    st.session_state.ladder_live_roster = new_roster[["name", "rating", "court", "slot"]].copy()

                    st.session_state.ladder_round_num = int(st.session_state.ladder_round_num) + 1
                    st.session_state.ladder_state = "PLAY_ROUND"
                    if "current_schedule" in st.session_state:
                        del st.session_state.current_schedule
                    st.rerun()


    # ---------- TAB 2: SETTINGS ----------
    with tabs[1]:
        st.subheader("Settings")

        if df_meta is not None and not df_meta.empty:
            # show editable config (club-scoped)
            cols = [c for c in ["id", "league_name", "is_active", "min_games", "description", "k_factor"] if c in df_meta.columns]
            editor = st.data_editor(
                df_meta[cols],
                disabled=["id", "league_name"],
                hide_index=True,
                use_container_width=True,
            )

            if st.button("Save Config"):
                for _, r in editor.iterrows():
                    supabase.table("leagues_metadata").update(
                        {
                            "is_active": bool(r.get("is_active", True)),
                            "min_games": int(r.get("min_games", 0) or 0),
                            "k_factor": int(r.get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR),
                            "description": str(r.get("description", "") or ""),
                        }
                    ).eq("id", int(r["id"])).eq("club_id", CLUB_ID).execute()
                st.rerun()

        st.divider()
        st.subheader("🗑️ Danger Zone (Leagues)")
        if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
            del_lg = st.selectbox("Select League to Delete", [""] + sorted(df_meta["league_name"].dropna().tolist()))
            if del_lg:
                st.error(f"⚠️ Delete metadata row for '{del_lg}'. (Matches stay in history.)")
                if st.button(f"Permanently Delete '{del_lg}'", type="primary"):
                    supabase.table("leagues_metadata").delete().eq("club_id", CLUB_ID).eq("league_name", del_lg).execute()
                    st.success(f"Deleted {del_lg}!")
                    time.sleep(1)
                    st.rerun()

        st.divider()
        st.write("#### 🆕 Create New League")
        with st.form("new_league_form"):
            n = st.text_input("Name")
            k = st.number_input("K", value=int(DEFAULT_K_FACTOR))
            mg = st.number_input("Min Games", value=12)
            if st.form_submit_button("Create"):
                supabase.table("leagues_metadata").insert(
                    {"club_id": CLUB_ID, "league_name": n, "is_active": True, "min_games": int(mg), "k_factor": int(k)}
                ).execute()
                st.rerun()

# -------------------------
# PAGE: MATCH UPLOADER
# -------------------------
elif sel == "📝 Match Uploader":
    st.header("📝 Match Uploader (Quick/Pop-Up)")

    c1, c2, c3 = st.columns(3)
    ctx_type = c1.radio("Context", ["🏆 Official League", "🎉 Pop-Up"], horizontal=True)

    selected_league = ""
    is_popup = False

    if ctx_type == "🏆 Official League":
        if df_meta is not None and not df_meta.empty and "is_active" in df_meta.columns:
            opts = sorted(df_meta[df_meta["is_active"] == True]["league_name"].dropna().tolist())
            if not opts:
                opts = ["Default"]
        else:
            opts = ["Default"]
        selected_league = c2.selectbox("Select League", opts)
        match_type_db = "Live Match"
        is_popup = False
    else:
        selected_league = c2.text_input("Event Name", "Saturday Social")
        match_type_db = "PopUp"
        is_popup = True

    week_tag = c3.selectbox("Week / Session", [f"Week {i}" for i in range(1, 13)] + ["Playoffs", "Finals", "Event"])

    st.divider()
    entry_method = st.radio("Entry Method", ["📋 Manual / Batch", "🏟️ Single Round Robin"], horizontal=True)
    st.write("")

    player_list = sorted(df_players["name"].astype(str).tolist()) if df_players is not None and not df_players.empty else []

    # ---- Manual / Batch ----
    if entry_method == "📋 Manual / Batch":
        if "batch_df" not in st.session_state:
            st.session_state.batch_df = pd.DataFrame(
                [{"T1_P1": None, "T1_P2": None, "Score_1": 0, "Score_2": 0, "T2_P1": None, "T2_P2": None} for _ in range(5)]
            )

        edited_batch = st.data_editor(
            st.session_state.batch_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "T1_P1": st.column_config.SelectboxColumn("T1 P1", options=player_list),
                "T1_P2": st.column_config.SelectboxColumn("T1 P2", options=player_list),
                "T2_P1": st.column_config.SelectboxColumn("T2 P1", options=player_list),
                "T2_P2": st.column_config.SelectboxColumn("T2 P2", options=player_list),
            },
        )

        if st.button("Submit Batch"):
            valid_batch = []
            for _, row in edited_batch.iterrows():
                if row["T1_P1"] and row["T2_P1"] and (int(row["Score_1"]) + int(row["Score_2"]) > 0):
                    valid_batch.append(
                        {
                            "t1_p1": row["T1_P1"],
                            "t1_p2": row["T1_P2"],
                            "t2_p1": row["T2_P1"],
                            "t2_p2": row["T2_P2"],
                            "s1": int(row["Score_1"]),
                            "s2": int(row["Score_2"]),
                            "date": str(datetime.now()),
                            "league": selected_league,
                            "match_type": match_type_db,
                            "week_tag": week_tag,
                            "is_popup": is_popup,
                        }
                    )
            if valid_batch:
                process_matches(valid_batch, name_to_id, df_players_all, df_leagues, df_meta)
                st.success("✅ Processed!")
                time.sleep(1)
                st.rerun()

    # ---- Single Round Robin ----
    else:
        if "lc_courts" not in st.session_state:
            st.session_state.lc_courts = 1

        st.session_state.lc_courts = st.number_input("Courts", 1, 10, int(st.session_state.lc_courts))

        with st.form("setup_lc"):
            c_data = []
            for i in range(int(st.session_state.lc_courts)):
                cc1, cc2 = st.columns([1, 3])
                t = cc1.selectbox(f"Format C{i+1}", ["4-Player", "5-Player", "6-Player", "8-Player", "12-Player"], key=f"t_{i}")
                n = cc2.text_area(f"Players C{i+1}", height=70, key=f"n_{i}")
                c_data.append({"type": t, "names": n})

            st.markdown("---")
            custom_sched = st.text_area("Overrides: Paste Custom Schedule Here (e.g., '1 2 3 4')", help="Overrides the format selection above.")

            if st.form_submit_button("Generate"):
                st.session_state.lc_schedule = []
                st.session_state.active_lg = selected_league
                st.session_state.active_wk = week_tag
                st.session_state.active_is_popup = is_popup
                st.session_state.active_mt = match_type_db

                for idx, c in enumerate(c_data):
                    pl = [x.strip() for x in c["names"].replace("\n", ",").split(",") if x.strip()]
                    st.session_state.lc_schedule.append({"c": idx + 1, "m": get_match_schedule(c["type"], pl, custom_text=custom_sched)})
                st.rerun()

        if "lc_schedule" in st.session_state:
            with st.form("scores_lc"):
                all_res = []
                for c in st.session_state.lc_schedule:
                    st.markdown(f"**Court {c['c']}**")
                    for i, m in enumerate(c["m"]):
                        cc1, cc2, cc3, cc4 = st.columns([3, 1, 1, 3])
                        cc1.text(f"{m['t1'][0]}/{m['t1'][1]}")
                        s1 = cc2.number_input("S1", 0, key=f"s1_{c['c']}_{i}")
                        s2 = cc3.number_input("S2", 0, key=f"s2_{c['c']}_{i}")
                        cc4.text(f"{m['t2'][0]}/{m['t2'][1]}")

                        all_res.append(
                            {
                                "t1_p1": m["t1"][0],
                                "t1_p2": m["t1"][1],
                                "t2_p1": m["t2"][0],
                                "t2_p2": m["t2"][1],
                                "s1": int(s1),
                                "s2": int(s2),
                                "date": str(datetime.now()),
                                "league": st.session_state.active_lg,
                                "match_type": st.session_state.active_mt,
                                "week_tag": st.session_state.active_wk,
                                "is_popup": bool(st.session_state.active_is_popup),
                            }
                        )

                if st.form_submit_button("Submit"):
                    payload = [x for x in all_res if x["s1"] > 0 or x["s2"] > 0]
                    if payload:
                        process_matches(payload, name_to_id, df_players_all, df_leagues, df_meta)
                    st.success("✅ Done!")
                    del st.session_state.lc_schedule
                    time.sleep(1)
                    st.rerun()

# -------------------------
# PAGE: PLAYER EDITOR
# -------------------------
elif sel == "👥 Player Editor":
    st.header("👥 Player Management")

    with st.expander("➕ Add New Player", expanded=False):
        with st.form("add_p"):
            n = st.text_input("Name")
            r = st.number_input("Rating", 1.0, 7.0, 3.5, step=0.1)
            if st.form_submit_button("Add Player"):
                ok, err = safe_add_player(n, r)
                if not ok:
                    st.error(err)
                st.rerun()

    st.divider()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Current Active Roster")
        if df_players is None or df_players.empty:
            st.info("No players.")
        else:
            display_roster = df_players.sort_values("name")[["name", "rating", "wins", "losses", "matches_played"]].copy()
            display_roster["JUPR"] = display_roster["rating"].astype(float) / 400.0
            st.dataframe(
                display_roster[["name", "JUPR", "wins", "losses", "matches_played"]],
                use_container_width=True,
                hide_index=True,
                column_config={"JUPR": st.column_config.NumberColumn("JUPR", format="%.3f")},
            )

    with col2:
        st.subheader("Manage Player")
        all_names = sorted(df_players_all["name"].astype(str).tolist()) if df_players_all is not None and not df_players_all.empty else []
        p_edit = st.selectbox("Select Player to Edit/Deactivate", [""] + all_names)

        if p_edit:
            curr = df_players_all[df_players_all["name"] == p_edit].iloc[0]
            pid = int(curr["id"])

            with st.form("edit_form"):
                st.caption(f"Editing: {p_edit}")
                new_n = st.text_input("Name", value=str(curr["name"]))
                new_r = st.number_input("Rating", 1.0, 7.0, float(curr.get("rating", 1200.0)) / 400.0, step=0.01)
                active_flag = st.checkbox("Active", value=bool(curr.get("active", True)))

                if st.form_submit_button("Update Player"):
                    supabase.table("players").update(
                        {"name": new_n, "rating": float(new_r) * 400.0, "active": bool(active_flag)}
                    ).eq("id", pid).eq("club_id", CLUB_ID).execute()
                    st.success("Updated!")
                    time.sleep(1)
                    st.rerun()

            st.write("---")
            st.write("**Danger Zone**")
            if st.button("🗑️ Deactivate Player", type="primary"):
                supabase.table("players").update({"active": False}).eq("id", pid).eq("club_id", CLUB_ID).execute()
                st.success(f"Player {curr['name']} has been deactivated.")
                time.sleep(1)
                st.rerun()

# -------------------------
# PAGE: MATCH LOG
# -------------------------
elif sel == "📝 Match Log":
    st.header("📝 Match Log")

    if df_matches is None or df_matches.empty:
        st.info("No matches loaded.")
    else:
        filter_type = st.radio("Filter", ["All", "League", "Pop-Up"], horizontal=True)
        if filter_type == "League":
            view_df = df_matches[df_matches["match_type"] != "PopUp"]
        elif filter_type == "Pop-Up":
            view_df = df_matches[df_matches["match_type"] == "PopUp"]
        else:
            view_df = df_matches

        col1, _ = st.columns([1, 4])
        id_filter = col1.number_input("Jump to ID:", min_value=0, value=0)
        if id_filter > 0:
            view_df = view_df[view_df["id"] == int(id_filter)]

        st.write("### 🗑️ Bulk Delete (first 500 rows shown)")
        edit_cols = [c for c in ["id", "date", "league", "match_type", "elo_delta", "p1", "p2", "p3", "p4", "score_t1", "score_t2"] if c in view_df.columns]
        edit_df = view_df.head(500)[edit_cols].copy()
        edit_df.insert(0, "Delete", False)

        edited_log = st.data_editor(
            edit_df,
            column_config={"Delete": st.column_config.CheckboxColumn(default=False)},
            hide_index=True,
            use_container_width=True,
        )

        to_delete = edited_log[edited_log["Delete"] == True]
        if not to_delete.empty:
            if st.button(f"Delete {len(to_delete)} Matches"):
                supabase.table("matches").delete().in_("id", to_delete["id"].astype(int).tolist()).eq("club_id", CLUB_ID).execute()
                st.success("Deleted!")
                time.sleep(1)
                st.rerun()

# -------------------------
# PAGE: ADMIN TOOLS
# -------------------------
elif sel == "⚙️ Admin Tools":
    st.header("⚙️ Admin Tools")

    # ---------- System Health Check ----------
    st.subheader("🏥 System Health Check")

    if st.button("Run Diagnostics"):
        with st.status("Checking System...", expanded=True) as status:
            try:
                sample = supabase.table("matches").select("*").eq("club_id", CLUB_ID).limit(1).execute()
                if sample.data:
                    keys = sample.data[0].keys()
                    if "t1_p1_r" in keys and "t1_p1_r_end" in keys:
                        st.success("✅ Snapshot columns exist.")
                    else:
                        st.error("❌ Snapshot columns missing in Supabase (matches table).")

                else:
                    st.warning("⚠️ No matches found.")

                # FIXED: proper .is_ usage in supabase-py (None, not "null"), and correct select syntax
                null_snaps = (
                    supabase.table("matches")
                    .select("id,t1_p1_r")
                    .eq("club_id", CLUB_ID)
                    .is_("t1_p1_r", None)
                    .limit(5000)
                    .execute()
                )

                if null_snaps.data and len(null_snaps.data) > 0:
                    st.error(f"❌ Found {len(null_snaps.data)} matches with EMPTY snapshots. Run Recalculate below.")
                else:
                    st.success("✅ All matches have snapshot data (in sampled range).")

            except Exception as e:
                st.error(f"Error: {e}")

            status.update(label="Complete", state="complete")

    st.divider()

    # ---------- Recalculate / Replay ----------
    st.subheader("🔄 Recalculate / Replay History")

    league_opts = ["ALL (Full System Reset)"]
    if df_leagues is not None and not df_leagues.empty and "league_name" in df_leagues.columns:
        league_opts += sorted(df_leagues["league_name"].dropna().unique().tolist())

    target_reset = st.selectbox("League", league_opts)

    if st.button(f"⚠️ Replay History for: {target_reset}"):
        with st.spinner("Crunching..."):
            all_players = (
                supabase.table("players")
                .select("*")
                .eq("club_id", CLUB_ID)
                .execute()
                .data
            )
            all_matches = (
                supabase.table("matches")
                .select("*")
                .eq("club_id", CLUB_ID)
                .order("date", desc=False)
                .order("id", desc=False)
                .execute()
                .data
            )

            # build k map from metadata
            k_map = {}
            if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
                for _, r in df_meta.iterrows():
                    try:
                        k_map[str(r["league_name"])] = int(r.get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
                    except Exception:
                        pass

            def k_for(lg):
                return int(k_map.get(str(lg), DEFAULT_K_FACTOR))

            # initialize overall from starting_rating (fallback to rating)
            p_map = {}
            for p in all_players:
                base = p.get("starting_rating", None)
                if base is None:
                    base = p.get("rating", 1200.0)
                p_map[int(p["id"])] = {"r": float(base), "w": 0, "l": 0, "mp": 0}

            island_map = {}  # (pid,lg)-> stats
            matches_to_update = []

            for m in all_matches:
                if target_reset != "ALL (Full System Reset)" and str(m.get("league", "")).strip() != str(target_reset).strip():
                    continue

                p1, p2, p3, p4 = m.get("t1_p1"), m.get("t1_p2"), m.get("t2_p1"), m.get("t2_p2")
                s1, s2 = int(m.get("score_t1", 0) or 0), int(m.get("score_t2", 0) or 0)

                def gr(pid):
                    if pid is None:
                        return 1200.0
                    return float(p_map[int(pid)]["r"])

                sr1, sr2, sr3, sr4 = gr(p1), gr(p2), gr(p3), gr(p4)
                do1, do2 = calculate_hybrid_elo((sr1 + sr2) / 2, (sr3 + sr4) / 2, s1, s2, k_factor=DEFAULT_K_FACTOR)

                win = s1 > s2

                for pid, d, won_flag in [(p1, do1, win), (p2, do1, win), (p3, do2, not win), (p4, do2, not win)]:
                    if pid is None:
                        continue
                    pid = int(pid)
                    p_map[pid]["r"] += float(d)
                    p_map[pid]["mp"] += 1
                    if won_flag:
                        p_map[pid]["w"] += 1
                    else:
                        p_map[pid]["l"] += 1

                er1, er2, er3, er4 = gr(p1), gr(p2), gr(p3), gr(p4)

                # League replay (skip PopUp)
                if str(m.get("match_type", "")) != "PopUp":
                    lg = str(m.get("league", "")).strip()

                    def gir(pid, lg_name):
                        if pid is None:
                            return 1200.0
                        key = (int(pid), lg_name)
                        if key not in island_map:
                            island_map[key] = {"r": float(p_map[int(pid)]["r"]), "w": 0, "l": 0, "mp": 0}
                        return float(island_map[key]["r"])

                    ir1, ir2, ir3, ir4 = gir(p1, lg), gir(p2, lg), gir(p3, lg), gir(p4, lg)
                    di1, di2 = calculate_hybrid_elo((ir1 + ir2) / 2, (ir3 + ir4) / 2, s1, s2, k_factor=k_for(lg))

                    for pid, d, won_flag in [(p1, di1, win), (p2, di1, win), (p3, di2, not win), (p4, di2, not win)]:
                        if pid is None:
                            continue
                        key = (int(pid), lg)
                        island_map[key]["r"] += float(d)
                        island_map[key]["mp"] += 1
                        if won_flag:
                            island_map[key]["w"] += 1
                        else:
                            island_map[key]["l"] += 1

                stored_elo_delta = abs(do1) if win else abs(do2)
                matches_to_update.append(
                    {
                        "id": int(m["id"]),
                        "elo_delta": float(stored_elo_delta),
                        "t1_p1_r": float(sr1),
                        "t1_p2_r": float(sr2),
                        "t2_p1_r": float(sr3),
                        "t2_p2_r": float(sr4),
                        "t1_p1_r_end": float(er1),
                        "t1_p2_r_end": float(er2),
                        "t2_p1_r_end": float(er3),
                        "t2_p2_r_end": float(er4),
                    }
                )

            # update players (only on full reset)
            if target_reset == "ALL (Full System Reset)":
                for pid, s in p_map.items():
                    supabase.table("players").update(
                        {"rating": s["r"], "wins": s["w"], "losses": s["l"], "matches_played": s["mp"]}
                    ).eq("club_id", CLUB_ID).eq("id", int(pid)).execute()

            # rebuild league_ratings
            if target_reset != "ALL (Full System Reset)":
                supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).eq("league_name", str(target_reset)).execute()
            else:
                supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).execute()

            new_is = []
            for (pid, lg), s in island_map.items():
                if target_reset == "ALL (Full System Reset)" or str(lg).strip() == str(target_reset).strip():
                    # starting_rating: current overall base at time of league start is not reconstructible perfectly here,
                    # so set to player's starting_rating if present; otherwise overall starting point.
                    start_base = None
                    for p in all_players:
                        if int(p["id"]) == int(pid):
                            start_base = p.get("starting_rating", p.get("rating", 1200.0))
                            break
                    if start_base is None:
                        start_base = 1200.0

                    new_is.append(
                        {
                            "club_id": CLUB_ID,
                            "player_id": int(pid),
                            "league_name": str(lg),
                            "rating": float(s["r"]),
                            "wins": int(s["w"]),
                            "losses": int(s["l"]),
                            "matches_played": int(s["mp"]),
                            "starting_rating": float(start_base),
                        }
                    )

            if new_is:
                for i in range(0, len(new_is), 1000):
                    supabase.table("league_ratings").insert(new_is[i : i + 1000]).execute()

            # update match snapshots (chunked)
            bar = st.progress(0.0)
            total = max(1, len(matches_to_update))
            for i, u in enumerate(matches_to_update):
                supabase.table("matches").update(u).eq("club_id", CLUB_ID).eq("id", int(u["id"])).execute()
                bar.progress((i + 1) / total)

            st.success("Done!")
            time.sleep(1)
            st.rerun()

    st.divider()

    # ---------- Backfill starting_rating on league_ratings ----------
    st.subheader("🛠️ Database Migration: Backfill League Start Ratings")
    st.info("Run this ONCE after adding the 'starting_rating' column to league_ratings. It estimates each player's league starting rating by reversing their league match deltas.")

    if st.button("🚀 Run Backfill Migration"):
        with st.status("Backfilling starting ratings...", expanded=True) as status:
            l_ratings = (
                supabase.table("league_ratings")
                .select("*")
                .eq("club_id", CLUB_ID)
                .execute()
                .data
            )

            matches = (
                supabase.table("matches")
                .select("id,league,match_type,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2,elo_delta")
                .eq("club_id", CLUB_ID)
                .execute()
                .data
            )
            df_m = pd.DataFrame(matches)

            updates = []
            for lr in l_ratings:
                pid = int(lr["player_id"])
                lg = str(lr["league_name"]).strip()
                curr = float(lr["rating"])

                total_change = 0.0
                if not df_m.empty:
                    rel = df_m[df_m["league"].astype(str).str.strip() == lg]
                    rel = rel[rel["match_type"].astype(str) != "PopUp"]

                    for _, mm in rel.iterrows():
                        delta = float(mm.get("elo_delta", 0) or 0)
                        s1, s2 = int(mm.get("score_t1", 0) or 0), int(mm.get("score_t2", 0) or 0)

                        if int(mm.get("t1_p1", -1) or -1) == pid or int(mm.get("t1_p2", -1) or -1) == pid:
                            if s1 > s2:
                                total_change += abs(delta)
                            elif s2 > s1:
                                total_change -= abs(delta)

                        elif int(mm.get("t2_p1", -1) or -1) == pid or int(mm.get("t2_p2", -1) or -1) == pid:
                            if s2 > s1:
                                total_change += abs(delta)
                            elif s1 > s2:
                                total_change -= abs(delta)

                start_r = curr - total_change
                updates.append({"id": int(lr["id"]), "starting_rating": float(start_r)})

            st.write(f"Calculated starting ratings for {len(updates)} records. Saving...")

            for i in range(0, len(updates), 100):
                chunk = updates[i : i + 100]
                for item in chunk:
                    supabase.table("league_ratings").update({"starting_rating": item["starting_rating"]}).eq("club_id", CLUB_ID).eq("id", int(item["id"])).execute()

            status.update(label="Migration Complete!", state="complete")
            st.success("✅ Database updated. Leaderboards can now compute Gain accurately.")
            st.divider()

    # ---------- Reports ----------
    st.subheader("📊 Reports & Exports")

    report_league = st.selectbox(
        "Select League for Report",
        ["OVERALL"] + (sorted(df_meta["league_name"].dropna().unique().tolist()) if df_meta is not None and not df_meta.empty else []),
    )

    if st.button("Generate Report"):
        if report_league == "OVERALL":
            rep_df = df_players.copy()
            if "starting_rating" not in rep_df.columns:
                rep_df["starting_rating"] = rep_df.get("rating", 1200.0)
        else:
            rep_df = df_leagues[df_leagues["league_name"] == report_league].copy() if df_leagues is not None else pd.DataFrame()
            rep_df["name"] = rep_df["player_id"].map(id_to_name)
            if "starting_rating" not in rep_df.columns:
                rep_df["starting_rating"] = rep_df["rating"]

        if rep_df is None or rep_df.empty:
            st.error("No data found for this league.")
        else:
            rep_df["JUPR"] = rep_df["rating"].astype(float) / 400.0
            rep_df["Win %"] = (rep_df["wins"].astype(float) / rep_df["matches_played"].replace(0, 1).astype(float)) * 100.0
            rep_df["Gain"] = (rep_df["rating"].astype(float) - rep_df["starting_rating"].astype(float)) / 400.0

            st.markdown(f"### 📄 {report_league} - Top Performers")

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 👑 Highest Rating")
                st.dataframe(rep_df.sort_values("rating", ascending=False).head(5)[["name", "JUPR"]], hide_index=True, use_container_width=True)

                st.markdown("##### 🎯 Best Win % (Min 5 games)")
                mask = rep_df["matches_played"].astype(int) >= 5
                st.dataframe(rep_df[mask].sort_values("Win %", ascending=False).head(5)[["name", "Win %"]], hide_index=True, use_container_width=True)

            with c2:
                st.markdown("##### 🔥 Most Improved")
                disp = rep_df.copy()
                disp["Gain"] = disp["Gain"].map("{:+.3f}".format)
                st.dataframe(disp.sort_values("Gain", ascending=False).head(5)[["name", "Gain"]], hide_index=True, use_container_width=True)

                st.markdown("##### 🚜 Most Wins")
                st.dataframe(rep_df.sort_values("wins", ascending=False).head(5)[["name", "wins"]], hide_index=True, use_container_width=True)

            st.divider()
            st.write("#### 📥 Export Data")

            export_df = rep_df[["name", "JUPR", "wins", "losses", "matches_played", "Win %", "Gain"]].copy()
            csv = export_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label=f"Download {report_league} Standings (CSV)",
                data=csv,
                file_name=f"{report_league}_Report_{str(datetime.now().date())}.csv",
                mime="text/csv",
            )

# -------------------------
# PAGE: ADMIN GUIDE
# -------------------------
elif sel == "📘 Admin Guide":
    st.header("📘 Admin Guide")
    st.markdown(
        """
### 🏟️ League Manager (Live Event)
1) **Setup:** Select League + Week, set Total Rounds, paste names.  
2) **Seeding:** Configure courts and court sizes.  
3) **Play:** Enter scores. Matches save immediately (with snapshots).  
4) **Movement:** Review green/red movement. Override via "New Ct" if needed.

### 📝 Match Uploader
Use this for pop-up events, paper sheets, or quick manual entries.  
Pop-Ups do **not** affect league ratings.

### ⚙️ Admin Tools
- **Diagnostics** checks snapshot columns and finds null snapshot rows.
- **Replay History** rebuilds ratings from match history and rewrites snapshots.
"""
    )

