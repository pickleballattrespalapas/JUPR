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
from postgrest.exceptions import APIError
import hashlib
from jupr_court_board import court_board


def match_key(round_num: int, court_num: int, t1: list[int], t2: list[int], side: str) -> str:
    a = sorted([int(t1[0]), int(t1[1])])
    b = sorted([int(t2[0]), int(t2[1])])
    raw = f"r{round_num}|c{court_num}|{a[0]}-{a[1]}|vs|{b[0]}-{b[1]}|{side}"
    return "sc_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def sb_retry(fn, retries: int = 4, base_sleep: float = 0.6):
    """
    Retries transient Supabase/httpx failures.
    DOES NOT retry PostgREST APIErrors (those are usually permanent: RLS/constraints).
    """
    last = None
    for attempt in range(retries):
        try:
            return fn()
        except APIError as e:
            # APIError contains a dict payload; don't retry it—surface it.
            payload = e.args[0] if e.args else {}
            msg = payload.get("message", str(e))
            details = payload.get("details", "")
            hint = payload.get("hint", "")
            code = payload.get("code", "")

            st.error(f"Supabase APIError ({code}): {msg}")
            if details:
                st.code(details)
            if hint:
                st.info(hint)

            raise  # stop retrying; this won't fix itself with sleep
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
MIN_WIN_DELTA_ELO = 1.0          # winner must gain at least +1 Elo
CAP_LOSER_GAIN_ELO = 16.0        # ONLY cap: if loser would gain, cap the gain to this Elo

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
    "match_explorer": "🎯 Match Explorer",
    "players": "🔍 Player Search",
    "challenge_ladder": "🪜 Challenge Ladder",
    "faqs": "❓ FAQs",
}
# Public-safe pages (both for deep-links and for the public nav)
PUBLIC_NAV_KEYS = ("leaderboards", "match_explorer", "players", "challenge_ladder", "faqs")
PUBLIC_NAV_LABELS = [PAGE_MAP[k] for k in PUBLIC_NAV_KEYS]
PUBLIC_ALLOWED = set(PUBLIC_NAV_KEYS)
NAV_TO_PAGE_PUBLIC = {PAGE_MAP[k]: k for k in PUBLIC_NAV_KEYS}

# Apply deep-links ONLY once per session, otherwise it overrides sidebar clicks on every rerun
# Apply deep-links ONLY once per session
if "deep_link_applied" not in st.session_state:
    st.session_state.deep_link_applied = False

if PUBLIC_MODE:
    st.session_state.admin_logged_in = False

    # Hide sidebar + header (kiosk feel)
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none;} header{visibility:hidden;}</style>",
        unsafe_allow_html=True
    )

    # Apply deep-link page once (public-safe pages)
    if not st.session_state.deep_link_applied:
        if DEEP_PAGE in PUBLIC_ALLOWED and DEEP_PAGE in PAGE_MAP:
            st.session_state["main_nav"] = PAGE_MAP[DEEP_PAGE]
        else:
            st.session_state["main_nav"] = PAGE_MAP["leaderboards"]

        if DEEP_LEAGUE:
            st.session_state["preselect_league"] = DEEP_LEAGUE

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

def build_match_explorer_link(
    ctx: str,
    me: int,
    partner: int,
    opp1: int,
    opp2: int,
    sy: int,
    so: int,
    public: bool = False,
) -> str:
    """
    Builds a deep link to the Match Explorer prefilled for a specific perspective.
    Uses numeric IDs to avoid name encoding issues.

    If PUBLIC_BASE_URL is set, returns an absolute URL; otherwise returns a relative querystring.
    """
    try:
        base = str(st.secrets.get("PUBLIC_BASE_URL", "") or "").rstrip("/")
    except Exception:
        base = ""

    params = {
        "page": "match_explorer",
        "ctx": str(ctx),
        "me": str(int(me)),
        "partner": str(int(partner)),
        "opp1": str(int(opp1)),
        "opp2": str(int(opp2)),
        "sy": str(int(sy)),
        "so": str(int(so)),
    }
    if public:
        params["public"] = "1"

    q = urllib.parse.urlencode(params, quote_via=urllib.parse.quote_plus)
    return f"{base}/?{q}" if base else f"?{q}"


# --- NAV <-> URL SYNC HELPERS ---
NAV_TO_PAGE = {
    "🏆 Leaderboards": "leaderboards",
    "🎯 Match Explorer": "match_explorer",
    "🔍 Player Search": "players",
    "🪜 Challenge Ladder": "challenge_ladder",
    "❓ FAQs": "faqs",
    "🏟️ League Manager": "league_manager",
    "📝 Match Uploader": "match_uploader",
    "👥 Player Editor": "player_editor",
    "📝 Match Log": "match_log",
    "⚙️ Admin Tools": "admin_tools",
    "📘 Admin Guide": "admin_guide",    
    "🛠️ Challenge Ladder Admin": "challenge ladder admin",
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
def calculate_hybrid_elo(
    t1_avg,
    t2_avg,
    score_t1,
    score_t2,
    k_factor=32,
    min_win_delta=1.0,
    cap_loser_gain=16,
):
    """
    Returns (delta_for_team1_players, delta_for_team2_players) in ELO points (not JUPR).

    Policy:
      - Winner hard rule: winner delta must be > 0. If computed <= 0, set to +min_win_delta.
      - Loser may gain if they beat expectations (non-zero-sum behavior).
      - ONLY cap: if the loser delta is positive, cap it to cap_loser_gain.
    """
    # Normalize inputs
    s1 = int(score_t1 or 0)
    s2 = int(score_t2 or 0)

    # No movement on ties or empty scores
    total_points = s1 + s2
    if total_points <= 0 or s1 == s2:
        return 0.0, 0.0

    # Expected outcomes from ratings
    expected_t1 = 1 / (1 + 10 ** ((t2_avg - t1_avg) / 400))
    expected_t2 = 1 - expected_t1

    # Observed performance proxy from score share
    share_t1 = s1 / total_points
    share_t2 = 1.0 - share_t1

    # Base deltas (symmetric)
    d1 = float(k_factor) * 2.0 * (share_t1 - expected_t1)
    d2 = float(k_factor) * 2.0 * (share_t2 - expected_t2)  # == -d1

    # Apply winner floor + loser-positive cap only
    if s1 > s2:
        # Team 1 wins
        if d1 <= 0:
            d1 = float(min_win_delta)

        if cap_loser_gain is not None and d2 > 0:
            d2 = min(d2, float(cap_loser_gain))

        return float(d1), float(d2)

    else:
        # Team 2 wins
        if d2 <= 0:
            d2 = float(min_win_delta)

        if cap_loser_gain is not None and d1 > 0:
            d1 = min(d1, float(cap_loser_gain))

        return float(d1), float(d2)


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

    match_list rows may contain player ids (int) or names (str). Supported score keys:
      - s1/s2 (preferred; used by live ladder + uploader)
      - score_t1/score_t2 (legacy)
    """
    db_matches = []
    overall_updates = {}  # pid -> {"r","w","l","mp"}
    island_updates = {}   # (pid, league_name) -> {"r","start","w","l","mp"}

    skipped_incomplete = 0
    skipped_empty = 0

    def get_k(league_name: str) -> int:
        if df_meta is None or df_meta.empty:
            return int(DEFAULT_K_FACTOR)
        row = df_meta[df_meta["league_name"] == league_name]
        if not row.empty:
            try:
                return int(row.iloc[0].get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
            except Exception:
                return int(DEFAULT_K_FACTOR)
        return int(DEFAULT_K_FACTOR)

    def get_player_row(pid: int):
        row = df_players_all[df_players_all["id"] == pid]
        if row.empty:
            return None
        return row.iloc[0]

    def ensure_overall_entry(pid: int):
        pid = int(pid)
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

    def get_overall_r(pid: int) -> float:
        pid = int(pid)
        if pid in overall_updates:
            return float(overall_updates[pid]["r"])
        pr = get_player_row(pid)
        if pr is None:
            return 1200.0
        return float(pr.get("rating", 1200.0) or 1200.0)

    def get_island_r(pid: int, league_name: str) -> float:
        key = (int(pid), str(league_name))
        if key in island_updates:
            return float(island_updates[key]["r"])

        if df_leagues is not None and not df_leagues.empty:
            m = df_leagues[
                (df_leagues["player_id"] == int(pid)) &
                (df_leagues["league_name"] == str(league_name))
            ]
            if not m.empty:
                return float(m.iloc[0].get("rating", 1200.0) or 1200.0)

        return get_overall_r(int(pid))

    def ensure_island_entry(pid: int, league_name: str):
        key = (int(pid), str(league_name))
        if key in island_updates:
            return
        start = float(get_island_r(int(pid), str(league_name)))
        island_updates[key] = {"r": start, "start": start, "w": 0, "l": 0, "mp": 0}

    def as_pid(x):
        """Accept int IDs OR numeric strings OR exact names. Returns int player_id or None."""
        if x is None:
            return None
        try:
            if isinstance(x, int):
                return int(x)
        except Exception:
            pass

        s = str(x).strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)

        return name_to_id.get(s)

    def apply_updates(pid: int, d_ov: float, d_isl: float, outcome, is_popup: bool, league_name: str) -> float:
        """
        outcome:
          - True  => this player won
          - False => this player lost
          - None  => tie/unknown (no W/L change)
        """
        pid = int(pid)
        ensure_overall_entry(pid)

        overall_updates[pid]["r"] += float(d_ov)
        overall_updates[pid]["mp"] += 1
        if outcome is True:
            overall_updates[pid]["w"] += 1
        elif outcome is False:
            overall_updates[pid]["l"] += 1

        if not bool(is_popup):
            ensure_island_entry(pid, league_name)
            key = (pid, league_name)
            island_updates[key]["r"] += float(d_isl)
            island_updates[key]["mp"] += 1
            if outcome is True:
                island_updates[key]["w"] += 1
            elif outcome is False:
                island_updates[key]["l"] += 1

        return float(overall_updates[pid]["r"])

    # -------------------------
    # Main match loop
    # -------------------------
    for m in match_list:
        p1 = as_pid(m.get("t1_p1"))
        p2 = as_pid(m.get("t1_p2"))
        p3 = as_pid(m.get("t2_p1"))
        p4 = as_pid(m.get("t2_p2"))

        # Require full doubles row
        if any(pid is None for pid in (p1, p2, p3, p4)):
            skipped_incomplete += 1
            continue

        p1, p2, p3, p4 = int(p1), int(p2), int(p3), int(p4)

        # score keys (support both)
        s1 = int(m.get("s1", m.get("score_t1", 0) or 0) or 0)
        s2 = int(m.get("s2", m.get("score_t2", 0) or 0) or 0)
        if (s1 + s2) <= 0:
            skipped_empty += 1
            continue

        league_name = str(m.get("league", "") or "").strip()
        week_tag = str(m.get("week_tag", "") or "")
        match_type = str(m.get("match_type", "") or "")
        is_popup = bool(m.get("is_popup", False)) or (match_type == "PopUp")

        # date normalize
        dt_val = m.get("date", None)
        try:
            if dt_val is None:
                dt_val = datetime.now().isoformat()
            elif hasattr(dt_val, "isoformat"):
                dt_val = dt_val.isoformat()
            else:
                dt_val = str(dt_val)
        except Exception:
            dt_val = str(dt_val) if dt_val is not None else str(datetime.now())

        # snapshots start (overall)
        ro1, ro2, ro3, ro4 = get_overall_r(p1), get_overall_r(p2), get_overall_r(p3), get_overall_r(p4)

        # overall deltas
        do1, do2 = calculate_hybrid_elo(
            (ro1 + ro2) / 2.0,
            (ro3 + ro4) / 2.0,
            s1,
            s2,
            k_factor=int(DEFAULT_K_FACTOR),
            min_win_delta=float(MIN_WIN_DELTA_ELO),
            cap_loser_gain=float(CAP_LOSER_GAIN_ELO),
        )

        # league deltas
        di1, di2 = 0.0, 0.0
        if not is_popup:
            k_val = get_k(league_name)
            ri1, ri2, ri3, ri4 = (
                get_island_r(p1, league_name),
                get_island_r(p2, league_name),
                get_island_r(p3, league_name),
                get_island_r(p4, league_name),
            )
            di1, di2 = calculate_hybrid_elo(
                (ri1 + ri2) / 2.0,
                (ri3 + ri4) / 2.0,
                s1,
                s2,
                k_factor=int(k_val),
                min_win_delta=float(MIN_WIN_DELTA_ELO),
                cap_loser_gain=float(CAP_LOSER_GAIN_ELO),
            )

        # outcome (tie-safe)
        if s1 == s2:
            t1_outcome = None
            t2_outcome = None
        else:
            t1_outcome = (s1 > s2)
            t2_outcome = (s2 > s1)

        # snapshots end (overall)
        end_r1 = apply_updates(p1, do1, di1, t1_outcome, is_popup, league_name)
        end_r2 = apply_updates(p2, do1, di1, t1_outcome, is_popup, league_name)
        end_r3 = apply_updates(p3, do2, di2, t2_outcome, is_popup, league_name)
        end_r4 = apply_updates(p4, do2, di2, t2_outcome, is_popup, league_name)

        stored_elo_delta = abs(do1) if (t1_outcome is True) else abs(do2)

        db_matches.append(
            {
                "club_id": CLUB_ID,
                "date": dt_val,
                "league": league_name,
                "t1_p1": p1,
                "t1_p2": p2,
                "t2_p1": p3,
                "t2_p2": p4,
                "score_t1": s1,
                "score_t2": s2,
                "elo_delta": float(stored_elo_delta),
                "match_type": match_type,
                "week_tag": week_tag,
                # snapshots (overall)
                "t1_p1_r": float(ro1),
                "t1_p2_r": float(ro2),
                "t2_p1_r": float(ro3),
                "t2_p2_r": float(ro4),
                "t1_p1_r_end": float(end_r1),
                "t1_p2_r_end": float(end_r2),
                "t2_p1_r_end": float(end_r3),
                "t2_p2_r_end": float(end_r4),
            }
        )

    # -------------------------
    # Write match rows
    # -------------------------
    if db_matches:
        CHUNK_M = 300
        for i in range(0, len(db_matches), CHUNK_M):
            chunk = db_matches[i : i + CHUNK_M]
            sb_retry(lambda chunk=chunk: supabase.table("matches").insert(chunk).execute())

    # -------------------------
    # Update overall player rows
    # -------------------------
    def update_player_row(row):
        pid = int(row["id"])
        payload = {
            "rating": float(row["rating"]),
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
            "matches_played": int(row["matches_played"]),
        }
        res = supabase.table("players").update(payload).eq("club_id", CLUB_ID).eq("id", pid).execute()
        if not res.data:
            payload_ins = {"club_id": CLUB_ID, "id": pid, **payload}
            supabase.table("players").insert(payload_ins).execute()

    for pid, stats in overall_updates.items():
        row = {
            "id": int(pid),
            "rating": float(stats["r"]),
            "wins": int(stats["w"]),
            "losses": int(stats["l"]),
            "matches_played": int(stats["mp"]),
        }
        sb_retry(lambda row=row: update_player_row(row))

    # -------------------------
    # Update league ratings
    # -------------------------
    if island_updates:
        for (pid, league_name), stats in island_updates.items():
            payload = {
                "club_id": CLUB_ID,
                "player_id": int(pid),
                "league_name": str(league_name),
                "rating": float(stats["r"]),
                "wins": int(stats["w"]),
                "losses": int(stats["l"]),
                "matches_played": int(stats["mp"]),
            }

            existing = sb_retry(lambda pid=pid, league_name=league_name: (
                supabase.table("league_ratings")
                .select("id,wins,losses,matches_played,starting_rating")
                .eq("club_id", CLUB_ID)
                .eq("player_id", int(pid))
                .eq("league_name", str(league_name))
                .limit(1)
                .execute()
            ))

            if existing.data:
                cur = existing.data[0]
                payload["wins"] += int(cur.get("wins", 0) or 0)
                payload["losses"] += int(cur.get("losses", 0) or 0)
                payload["matches_played"] += int(cur.get("matches_played", 0) or 0)

                if cur.get("starting_rating") is not None:
                    payload["starting_rating"] = float(cur["starting_rating"])
                else:
                    payload["starting_rating"] = float(stats.get("start", 1200.0))

                sb_retry(lambda payload=payload, rid=int(cur["id"]): (
                    supabase.table("league_ratings").update(payload).eq("id", rid).execute()
                ))
            else:
                payload["starting_rating"] = float(stats.get("start", 1200.0))
                sb_retry(lambda payload=payload: supabase.table("league_ratings").insert(payload).execute())

    return {
        "inserted": len(db_matches),
        "skipped_incomplete": int(skipped_incomplete),
        "skipped_empty": int(skipped_empty),
    }

import math

def to_int_or_neg1(x):
    """Convert to int if possible; return -1 for None/NaN/blank/bad values."""
    try:
        if x is None:
            return -1
        if isinstance(x, float) and math.isnan(x):
            return -1
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return -1
        return int(float(s))  # handles "12.0"
    except Exception:
        return -1

def canonical_dup_key(row, club_id: str):
    """
    Canonical key that matches duplicates even if:
    - players inside a team are swapped
    - team1/team2 are swapped (scores swapped too)
    """
    a1 = to_int_or_neg1(row.get("t1_p1"))
    a2 = to_int_or_neg1(row.get("t1_p2"))
    b1 = to_int_or_neg1(row.get("t2_p1"))
    b2 = to_int_or_neg1(row.get("t2_p2"))

    teamA = sorted([a1, a2])
    teamB = sorted([b1, b2])

    s1 = to_int_or_neg1(row.get("score_t1"))
    s2 = to_int_or_neg1(row.get("score_t2"))

    # normalize ordering across teams; if swapped, swap scores too
    if tuple(teamB) < tuple(teamA):
        teamA, teamB = teamB, teamA
        s1, s2 = s2, s1

    league = str(row.get("league", "") or "").strip()
    week = str(row.get("week_tag", "") or "").strip()
    mtype = str(row.get("match_type", "") or "").strip()

    return f"{club_id}|{league}|{week}|{mtype}|{teamA[0]}-{teamA[1]}|{teamB[0]}-{teamB[1]}|{s1}-{s2}"

def ensure_league_row(player_id: int, league_name: str, base_rating_elo: float = 1200.0):
    """
    Create league_ratings row if missing.
    """
    existing = (
        supabase.table("league_ratings")
        .select("id")
        .eq("club_id", CLUB_ID)
        .eq("player_id", int(player_id))
        .eq("league_name", str(league_name))
        .limit(1)
        .execute()
    )
    if existing.data:
        return existing.data[0]["id"]

    ins = (
        supabase.table("league_ratings")
        .insert({
            "club_id": CLUB_ID,
            "player_id": int(player_id),
            "league_name": str(league_name),
            "rating": float(base_rating_elo),
            "starting_rating": float(base_rating_elo),
            "wins": 0,
            "losses": 0,
            "matches_played": 0,
        })
        .execute()
    )
    # supabase returns inserted row(s) depending on config; safest: re-fetch
    existing2 = (
        supabase.table("league_ratings")
        .select("id")
        .eq("club_id", CLUB_ID)
        .eq("player_id", int(player_id))
        .eq("league_name", str(league_name))
        .limit(1)
        .execute()
    )
    return existing2.data[0]["id"] if existing2.data else None

def ladder_roster_active_df(df_roster: pd.DataFrame, id_to_name: dict) -> pd.DataFrame:
    """
    Returns a safe active roster DF with required columns:
      player_id, rank, is_active, name
    If roster is empty, returns an empty DF WITH columns (so downstream code won't KeyError).
    """
    cols = ["player_id", "rank", "is_active", "notes"]
    out = pd.DataFrame(columns=cols + ["name"])

    if df_roster is None or not isinstance(df_roster, pd.DataFrame) or df_roster.empty:
        return out

    if "player_id" not in df_roster.columns:
        return out

    tmp = df_roster.copy()

    if "is_active" in tmp.columns:
        tmp = tmp[tmp["is_active"] == True].copy()

    # Ensure rank exists (for sorting)
    if "rank" not in tmp.columns:
        tmp["rank"] = 999999

    tmp["name"] = tmp["player_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
    tmp = tmp.sort_values("rank", ascending=True)

    # Keep consistent columns
    keep = [c for c in cols if c in tmp.columns] + ["name"]
    return tmp[keep].copy()

# =========================
# CHALLENGE LADDER MODULE
# =========================
from datetime import timedelta, timezone

LADDER_OPEN_STATUSES = {
    "PENDING_ACCEPTANCE",
    "ACCEPTED_SCHEDULING",
    "IN_PROGRESS",
    "AWAITING_VERIFICATION",
    "OVERDUE_PLAY",
}
LADDER_FINAL_STATUSES = {"COMPLETED", "FORFEITED"}

def dt_utc_now():
    return datetime.now(timezone.utc)

def month_key_utc(dt: datetime) -> str:
    d = dt.astimezone(timezone.utc)
    return f"{d.year:04d}-{d.month:02d}"

def ladder_nm(pid: int, id_to_name: dict[int, str]) -> str:
    try:
        return str(id_to_name.get(int(pid), f"#{int(pid)}"))
    except Exception:
        return "—"

def ladder_fetch_settings():
    resp = sb_retry(lambda: (
        supabase.table("ladder_settings")
        .select("*")
        .eq("club_id", CLUB_ID)
        .limit(1)
        .execute()
    ))
    if resp.data:
        return resp.data[0]
    # Ensure exists
    sb_retry(lambda: supabase.table("ladder_settings").insert({"club_id": CLUB_ID}).execute())
    resp2 = sb_retry(lambda: (
        supabase.table("ladder_settings")
        .select("*")
        .eq("club_id", CLUB_ID)
        .limit(1)
        .execute()
    ))
    return resp2.data[0] if resp2.data else {
        "challenge_range": 3,
        "accept_window_hours": 48,
        "play_window_days": 7,
        "cooldown_hours": 72,
        "protected_hours": 72,
        "pass_hold_hours": 72,
    }

def ladder_load_core():
    # Roster
    roster = sb_retry(lambda: (
        supabase.table("ladder_roster")
        .select("id,club_id,player_id,tier_id,rank,is_active,joined_at,left_at,notes,updated_at")
        .eq("club_id", CLUB_ID)
        .order("rank", desc=False)
        .execute()
    ))
    df_roster = pd.DataFrame(roster.data)

    # Flags
    flags = sb_retry(lambda: (
        supabase.table("ladder_player_flags")
        .select(
            "club_id,player_id,"
            "vacation_until,reinstate_required,reinstate_notes,"
            "tier_move_flag,tier_move_dest_tier,tier_move_count,tier_move_triggered_at,tier_move_last_eval_at,"
            "updated_at"
        )
        .eq("club_id", CLUB_ID)
        .execute()
    ))
    df_flags = pd.DataFrame(flags.data)

    # Challenges (load enough history for status + “active challenges” view)
    ch = sb_retry(lambda: (
        supabase.table("ladder_challenges")
        .select("*")
        .eq("club_id", CLUB_ID)
        .order("created_at", desc=True)
        .limit(5000)
        .execute()
    ))
    df_ch = pd.DataFrame(ch.data)

    # Pass usage (for pass hold window + monthly check)
    pu = sb_retry(lambda: (
        supabase.table("ladder_pass_usage")
        .select("*")
        .eq("club_id", CLUB_ID)
        .order("used_at", desc=True)
        .limit(2000)
        .execute()
    ))
    df_pass = pd.DataFrame(pu.data)

    return df_roster, df_flags, df_ch, df_pass

def ladder_next_rank(club_id: str) -> int:
    max_rank_resp = sb_retry(lambda: (
        supabase.table("ladder_roster")
        .select("rank")
        .eq("club_id", club_id)
        .order("rank", desc=True)
        .limit(1)
        .execute()
    ))
    return (int(max_rank_resp.data[0]["rank"]) + 1) if max_rank_resp.data else 1


def ladder_set_roster_active(club_id: str, pid: int, make_active: bool, mode: str = "append", notes: str | None = None):
    """
    mode:
      - "append": when reactivating, place at bottom (rank = max+1)
      - "restore": when reactivating, keep existing rank (if it exists)
    """
    pid = int(pid)
    now_iso = dt_utc_now().isoformat()

    # Fetch existing row (if any)
    existing = sb_retry(lambda: (
        supabase.table("ladder_roster")
        .select("id,rank,is_active,joined_at,left_at,notes")
        .eq("club_id", club_id)
        .eq("player_id", pid)
        .limit(1)
        .execute()
    ))

    if not existing.data:
        # If no roster row exists, create one only when activating
        if not make_active:
            return False, "Player is not in ladder roster."
        ins = {
            "club_id": club_id,
            "player_id": pid,
            "rank": ladder_next_rank(club_id),
            "is_active": True,
            "joined_at": now_iso,
            "left_at": None,
            "notes": (notes.strip() if notes else None),
        }
        sb_retry(lambda: supabase.table("ladder_roster").insert(ins).execute())
        ladder_audit("roster_activate_insert", "ladder_roster", f"{club_id}:{pid}", None, ins)
        return True, "Activated (inserted) at bottom."

    row = existing.data[0]
    before = dict(row)

    if make_active:
        upd = {
            "is_active": True,
            "left_at": None,
            "joined_at": now_iso,
        }
        if notes is not None:
            upd["notes"] = notes.strip() or None

        if mode == "append":
            upd["rank"] = ladder_next_rank(club_id)
        # mode == "restore": keep existing rank as-is

        sb_retry(lambda: (
            supabase.table("ladder_roster")
            .update(upd)
            .eq("club_id", club_id)
            .eq("player_id", pid)
            .execute()
        ))
        ladder_audit("roster_reactivate" if before.get("is_active") == False else "roster_activate", "ladder_roster", f"{club_id}:{pid}", before, {**before, **upd})
        return True, "Activated."

    else:
        # Deactivate
        upd = {
            "is_active": False,
            "left_at": now_iso,
        }
        if notes is not None:
            upd["notes"] = notes.strip() or None

        sb_retry(lambda: (
            supabase.table("ladder_roster")
            .update(upd)
            .eq("club_id", club_id)
            .eq("player_id", pid)
            .execute()
        ))
        ladder_audit("roster_deactivate", "ladder_roster", f"{club_id}:{pid}", before, {**before, **upd})
        return True, "Deactivated."


def ladder_parse_dt(x):
    if x is None or str(x).strip() == "":
        return None
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        return None

def ladder_compute_status_map(df_roster, df_flags, df_ch, df_pass, settings, id_to_name):
    now = dt_utc_now()

    accept_h = int(settings.get("accept_window_hours", 48) or 48)
    play_d = int(settings.get("play_window_days", 7) or 7)
    cooldown_h = int(settings.get("cooldown_hours", 72) or 72)
    protected_h = int(settings.get("protected_hours", 72) or 72)
    passhold_h = int(settings.get("pass_hold_hours", 72) or 72)

    # Normalize datetime columns
    if df_flags is not None and not df_flags.empty:
        df_flags = df_flags.copy()
        df_flags["vacation_until_dt"] = df_flags["vacation_until"].apply(ladder_parse_dt)
    else:
        df_flags = pd.DataFrame(columns=["player_id", "vacation_until_dt", "reinstate_required", "reinstate_notes"])

    if df_ch is not None and not df_ch.empty:
        df_ch = df_ch.copy()
        df_ch["created_at_dt"] = df_ch["created_at"].apply(ladder_parse_dt)
        df_ch["accept_by_dt"] = df_ch["accept_by"].apply(ladder_parse_dt)
        df_ch["accepted_at_dt"] = df_ch["accepted_at"].apply(ladder_parse_dt)
        df_ch["play_by_dt"] = df_ch["play_by"].apply(ladder_parse_dt)
        df_ch["completed_at_dt"] = df_ch["completed_at"].apply(ladder_parse_dt)
    else:
        df_ch = pd.DataFrame()

    if df_pass is not None and not df_pass.empty:
        df_pass = df_pass.copy()
        df_pass["used_at_dt"] = df_pass["used_at"].apply(ladder_parse_dt)
    else:
        df_pass = pd.DataFrame()

    # Map: player -> flags
    flags_map = {}
    for _, r in df_flags.iterrows():
        pid = int(r.get("player_id"))
        flags_map[pid] = {
            "vacation_until": r.get("vacation_until_dt"),
            "reinstate_required": bool(r.get("reinstate_required", False)),
            "reinstate_notes": str(r.get("reinstate_notes", "") or ""),
        }

    # Map: player -> open challenge row (for Locked context)
    open_map = {}
    if not df_ch.empty:
        open_df = df_ch[df_ch["status"].isin(list(LADDER_OPEN_STATUSES))].copy()
        # For context, keep the most recent open challenge per player
        for _, r in open_df.iterrows():
            for pid in (r.get("challenger_id"), r.get("defender_id")):
                if pid is None:
                    continue
                pid = int(pid)
                if pid not in open_map:
                    open_map[pid] = r.to_dict()

    # Map: player -> last finalized challenge row (for Protected/Cooldown)
    final_map = {}
    if not df_ch.empty:
        fin_df = df_ch[df_ch["status"].isin(list(LADDER_FINAL_STATUSES))].copy()
        fin_df = fin_df.sort_values("completed_at_dt", ascending=False, na_position="last")
        for _, r in fin_df.iterrows():
            for pid in (r.get("challenger_id"), r.get("defender_id")):
                if pid is None:
                    continue
                pid = int(pid)
                if pid not in final_map:
                    final_map[pid] = r.to_dict()

    # Map: player -> last pass used
    pass_map = {}
    if not df_pass.empty:
        for _, r in df_pass.sort_values("used_at_dt", ascending=False).iterrows():
            pid = r.get("player_id")
            if pid is None:
                continue
            pid = int(pid)
            if pid not in pass_map:
                pass_map[pid] = r.to_dict()

    # Compute status for each roster player
    status_map = {}
    if df_roster is None or df_roster.empty:
        return status_map

    for _, rr in df_roster.iterrows():
        if not bool(rr.get("is_active", True)):
            continue

        pid = int(rr["player_id"])
        f = flags_map.get(pid, {})
        vacation_until = f.get("vacation_until")
        reinstate_required = bool(f.get("reinstate_required", False))

        # 1) Reinstate Required
        if reinstate_required:
            status_map[pid] = {"status": "Reinstate Required", "until": None, "detail": f.get("reinstate_notes", "")}
            continue

        # 2) Vacation
        if vacation_until is not None and vacation_until.to_pydatetime() >= now:
            status_map[pid] = {"status": "Vacation", "until": vacation_until, "detail": "Admin-only hold"}
            continue

        # 3) Pass Hold
        p_last = pass_map.get(pid)
        if p_last:
            used_at = p_last.get("used_at_dt")
            if used_at is not None:
                until = used_at.to_pydatetime() + timedelta(hours=passhold_h)
                if until >= now:
                    status_map[pid] = {"status": "Pass Hold", "until": pd.to_datetime(until, utc=True), "detail": "72h after Pass Used"}
                    continue

        # 4) Locked (any open challenge)
        oc = open_map.get(pid)
        if oc:
            opp = None
            ch_id = oc.get("id")
            ch_status = str(oc.get("status", "") or "")
            if int(oc.get("challenger_id")) == pid:
                opp = int(oc.get("defender_id"))
                role = "Challenger"
            else:
                opp = int(oc.get("challenger_id"))
                role = "Defender"

            # Deadline detail
            accept_by = oc.get("accept_by_dt")
            play_by = oc.get("play_by_dt")

            if ch_status == "PENDING_ACCEPTANCE" and accept_by is not None:
                detail = f"{role} vs {ladder_nm(opp, id_to_name)} • Accept by {accept_by.strftime('%Y-%m-%d %H:%M UTC')}"
            elif play_by is not None:
                detail = f"{role} vs {ladder_nm(opp, id_to_name)} • Play by {play_by.strftime('%Y-%m-%d %H:%M UTC')}"
            else:
                detail = f"{role} vs {ladder_nm(opp, id_to_name)}"

            status_map[pid] = {"status": "Locked", "until": None, "detail": detail, "challenge_id": ch_id}
            continue

        # 5/6) Protected or Cooldown based on last finalized
        last_fin = final_map.get(pid)
        if last_fin:
            completed_at = last_fin.get("completed_at_dt")
            winner_id = last_fin.get("winner_id")
            if completed_at is not None and winner_id is not None:
                completed_dt = completed_at.to_pydatetime()
                if int(winner_id) == pid:
                    until = completed_dt + timedelta(hours=protected_h)
                    if until >= now:
                        status_map[pid] = {"status": "Protected", "until": pd.to_datetime(until, utc=True), "detail": "72h after win"}
                        continue
                else:
                    until = completed_dt + timedelta(hours=cooldown_h)
                    if until >= now:
                        status_map[pid] = {"status": "Cooldown", "until": pd.to_datetime(until, utc=True), "detail": "72h after loss"}
                        continue

        # 7) Ready
        status_map[pid] = {"status": "Ready to Defend", "until": None, "detail": ""}

    return status_map

def match_end_elo_for_pid(m: dict, pid: int) -> float | None:
    try:
        pid = int(pid)
    except Exception:
        return None

    # End snapshots (overall)
    if int(m.get("t1_p1") or -1) == pid:
        return m.get("t1_p1_r_end")
    if int(m.get("t1_p2") or -1) == pid:
        return m.get("t1_p2_r_end")
    if int(m.get("t2_p1") or -1) == pid:
        return m.get("t2_p1_r_end")
    if int(m.get("t2_p2") or -1) == pid:
        return m.get("t2_p2_r_end")
    return None

def compute_out_of_tier_streak(
    pid: int,
    joined_at_utc: datetime | None,
    current_tier_id: str,
    df_matches: pd.DataFrame,
) -> dict:
    """
    Returns:
      {
        "dest_tier": str|None,
        "count": int,
        "latest_match_at": datetime|None
      }
    Rule: streak counts consecutive matches where post-match rating tier == same dest tier != current tier.
    If any match returns to current tier, streak resets (break).
    """

    if df_matches is None or df_matches.empty:
        return {"dest_tier": None, "count": 0, "latest_match_at": None}

    pid = int(pid)
    cur = str(current_tier_id)

    m = df_matches.copy()

    # Ensure date parse
    if "date" in m.columns:
        m["date_dt"] = pd.to_datetime(m["date"], utc=True, errors="coerce")
    else:
        m["date_dt"] = pd.NaT

    # Filter to matches containing pid
    mask = (
        (m.get("t1_p1") == pid) |
        (m.get("t1_p2") == pid) |
        (m.get("t2_p1") == pid) |
        (m.get("t2_p2") == pid)
    )
    m = m[mask].copy()
    if m.empty:
        return {"dest_tier": None, "count": 0, "latest_match_at": None}

    # Only matches after joined_at
    if joined_at_utc is not None:
        m = m[m["date_dt"] >= pd.to_datetime(joined_at_utc, utc=True)].copy()
        if m.empty:
            return {"dest_tier": None, "count": 0, "latest_match_at": None}

    # Most recent first
    m = m.sort_values(["date_dt", "id"], ascending=[False, False])

    dest = None
    count = 0
    latest_dt = None

    for _, row in m.iterrows():
        latest_dt = latest_dt or (row["date_dt"].to_pydatetime() if pd.notna(row["date_dt"]) else None)

        end_elo = match_end_elo_for_pid(row.to_dict(), pid)
        if end_elo is None:
            continue

        end_jupr = float(end_elo) / 400.0
        t = tier_for_jupr(end_jupr)

        # If rating is back in current tier, streak ends immediately
        if t == cur:
            break

        # Out-of-tier:
        if dest is None:
            dest = t
            count = 1
        else:
            if t == dest:
                count += 1
            else:
                # jumped to a different out-of-tier destination; streak is not "consistent"
                break

        # No need to go past 10 for triggering
        if count >= 10:
            break

    return {"dest_tier": dest, "count": int(count), "latest_match_at": latest_dt}


def ladder_bucket_challenge(row: dict) -> str:
    now = dt_utc_now()
    status = str(row.get("status", "") or "")
    accept_by = ladder_parse_dt(row.get("accept_by"))
    play_by = ladder_parse_dt(row.get("play_by"))
    accepted_at = ladder_parse_dt(row.get("accepted_at"))

    if status == "PENDING_ACCEPTANCE":
        if accept_by is not None and accept_by.to_pydatetime() < now:
            return "Acceptance Overdue"
        return "Pending Acceptance"

    if status in ("ACCEPTED_SCHEDULING", "IN_PROGRESS", "AWAITING_VERIFICATION", "OVERDUE_PLAY"):
        if play_by is not None and play_by.to_pydatetime() < now:
            return "Play Overdue"
        return "Accepted / In Window"

    if status in ("COMPLETED", "FORFEITED"):
        return "Recently Completed"

    if status in ("CANCELED", "EXPIRED_ACCEPTANCE"):
        return "Closed (No Result)"

    return "Other"

def ladder_audit(action_type: str, entity_type: str, entity_id: str, before: dict | None, after: dict | None):
    actor = "admin" if st.session_state.get("admin_logged_in", False) else "system"
    payload = {
        "club_id": CLUB_ID,
        "actor": actor,
        "action_type": str(action_type),
        "entity_type": str(entity_type),
        "entity_id": str(entity_id),
        "before": before,
        "after": after,
    }
    try:
        sb_retry(lambda: supabase.table("ladder_audit_log").insert(payload).execute())
    except Exception:
        # audit should never block core operations
        pass

def ladder_compute_challenge_outcome(df_match_rows: pd.DataFrame):
    """
    Returns dict:
      winner_side: 'DEF' or 'CHAL'
      match_wins_def/chal
      games_def/chal
      points_def/chal
      point_diff_def (def - chal)
    Enforces tie-break: match wins -> games -> point diff -> defender holds
    """
    def parse_int(x):
        try:
            if x is None: return None
            return int(x)
        except Exception:
            return None

    def match_summary(row):
        games = []
        for i in (1,2,3):
            a = parse_int(row.get(f"g{i}_def"))
            b = parse_int(row.get(f"g{i}_chal"))
            if a is None or b is None:
                continue
            games.append((a,b))

        def_g = sum(1 for a,b in games if a > b)
        chal_g = sum(1 for a,b in games if b > a)

        def_pts = sum(a for a,b in games)
        chal_pts = sum(b for a,b in games)

        # winner by best-of-3
        if def_g > chal_g:
            win = "DEF"
        elif chal_g > def_g:
            win = "CHAL"
        else:
            win = "TIE"

        return {"win": win, "def_g": def_g, "chal_g": chal_g, "def_pts": def_pts, "chal_pts": chal_pts}

    if df_match_rows is None or df_match_rows.empty:
        return None

    ms = []
    for _, r in df_match_rows.sort_values("match_no").iterrows():
        ms.append(match_summary(r.to_dict()))

    # Require 2 matches present
    if len(ms) < 2:
        return None

    match_wins_def = sum(1 for x in ms if x["win"] == "DEF")
    match_wins_chal = sum(1 for x in ms if x["win"] == "CHAL")

    games_def = sum(x["def_g"] for x in ms)
    games_chal = sum(x["chal_g"] for x in ms)

    pts_def = sum(x["def_pts"] for x in ms)
    pts_chal = sum(x["chal_pts"] for x in ms)
    pdiff = pts_def - pts_chal

    # Tie-break
    if match_wins_def > match_wins_chal:
        winner_side = "DEF"
    elif match_wins_chal > match_wins_def:
        winner_side = "CHAL"
    else:
        if games_def > games_chal:
            winner_side = "DEF"
        elif games_chal > games_def:
            winner_side = "CHAL"
        else:
            if pdiff > 0:
                winner_side = "DEF"
            elif pdiff < 0:
                winner_side = "CHAL"
            else:
                winner_side = "DEF"  # defender holds

    return {
        "winner_side": winner_side,
        "match_wins_def": match_wins_def,
        "match_wins_chal": match_wins_chal,
        "games_def": games_def,
        "games_chal": games_chal,
        "points_def": pts_def,
        "points_chal": pts_chal,
        "point_diff_def": pdiff,
    }

# -------------------------
# TIER DEFINITIONS (Option A, range-labeled)
# -------------------------
TIER_ORDER = ["DEV", "INT", "ADV", "PREM"]

TIER_DEFS = {
    "DEV":  {"label": "Developing",    "min": None,  "max": 3.25, "range": "< 3.25"},
    "INT":  {"label": "Intermediate",  "min": 3.25,  "max": 3.75, "range": "3.25–3.75"},
    "ADV":  {"label": "Advanced",      "min": 3.75,  "max": 4.25, "range": "3.75–4.25"},
    "PREM": {"label": "Premier",       "min": 4.25,  "max": None, "range": "4.25+"},
}

def tier_for_jupr(jupr: float) -> str:
    try:
        x = float(jupr)
    except Exception:
        return "INT"
    if x < 3.25:
        return "DEV"
    if x < 3.75:
        return "INT"
    if x < 4.25:
        return "ADV"
    return "PREM"

def tier_title(tier_id: str) -> str:
    t = TIER_DEFS.get(str(tier_id), {"label": str(tier_id), "range": ""})
    rng = t.get("range", "")
    return f"{t.get('label','Tier')} — {rng}".strip(" —")

def tier_idx(tier_id: str) -> int:
    tid = str(tier_id)
    return TIER_ORDER.index(tid) if tid in TIER_ORDER else 999

def is_promotion(from_tier: str, to_tier: str) -> bool:
    return tier_idx(to_tier) > tier_idx(from_tier)

def is_demotion(from_tier: str, to_tier: str) -> bool:
    return tier_idx(to_tier) < tier_idx(from_tier)


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
if PUBLIC_MODE:
    # Public-only navigation (no admin pages; sidebar hidden via CSS above)
    nav_public = PUBLIC_NAV_LABELS

    sel = st.radio("Go to:", nav_public, horizontal=True, key="main_nav")

    try:
        st.query_params["public"] = "1"
        st.query_params["page"] = NAV_TO_PAGE_PUBLIC.get(sel, "leaderboards")
    except Exception:
        pass


else:
    nav = ["🏆 Leaderboards", "🎯 Match Explorer", "🔍 Player Search", "🪜 Challenge Ladder", "❓ FAQs"]
    if st.session_state.admin_logged_in:
        nav += ["──────────", "🛠️ Challenge Ladder Admin", "🏟️ League Manager", "📝 Match Uploader", "👥 Player Editor", "📝 Match Log", "⚙️ Admin Tools", "📘 Admin Guide"]

    sel = st.sidebar.radio("Go to:", nav, key="main_nav")

    # Reset Player Search selection when user navigates to that page
    if "last_nav" not in st.session_state:
        st.session_state.last_nav = None

    if st.session_state.last_nav != sel:
        if sel == "🔍 Player Search":
            st.session_state["player_search_name"] = ""
        st.session_state.last_nav = sel

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

    # Shareable link (hide on kiosk/public screens)
    if (not PUBLIC_MODE) and st.session_state.admin_logged_in:
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
    inactive_hidden = 0

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
        # Default from preloaded df_leagues
        display_df = pd.DataFrame()
        if df_leagues is not None and not df_leagues.empty:
            display_df = df_leagues[df_leagues["league_name"] == target_league].copy()

        # ---------------------------------------------------------
        # NEW: Filter out per-league inactive players for:
        #      - Top Performers
        #      - Standings
        #
        # This expects `league_ratings.is_active` to exist and be loaded into df_leagues.
        # If df_leagues wasn't loaded with is_active yet, we fall back to a direct fetch.
        # ---------------------------------------------------------
        if display_df is not None and not display_df.empty:
            if "is_active" in display_df.columns:
                inactive_hidden = int((display_df["is_active"] == False).sum())
                display_df = display_df[display_df["is_active"] == True].copy()
            else:
                # fallback fetch so feature still works even if df_leagues select didn't include is_active
                try:
                    lr_resp = (
                        supabase.table("league_ratings")
                        .select("player_id,league_name,rating,starting_rating,wins,losses,matches_played,is_active")
                        .eq("club_id", CLUB_ID)
                        .eq("league_name", target_league)
                        .execute()
                    )
                    tmp = pd.DataFrame(lr_resp.data) if lr_resp and lr_resp.data is not None else pd.DataFrame()

                    if not tmp.empty and "is_active" in tmp.columns:
                        inactive_hidden = int((tmp["is_active"] == False).sum())
                        display_df = tmp[tmp["is_active"] == True].copy()
                    else:
                        display_df = tmp

                except Exception:
                    # If the DB column doesn't exist yet, or select fails, do not break leaderboards.
                    if (not PUBLIC_MODE) and st.session_state.admin_logged_in:
                        st.warning(
                            "Per-league inactive filtering is not enabled yet. "
                            "Add `is_active` (boolean) to `public.league_ratings` and ensure your df_leagues load includes it."
                        )

        if display_df is not None and not display_df.empty:
            display_df["name"] = display_df["player_id"].map(id_to_name)
            if "starting_rating" not in display_df.columns:
                # fallback if not loaded yet
                display_df["starting_rating"] = display_df["rating"]

            # normalize required columns (prevents crashes if a column is missing)
            for c in ["wins", "losses", "matches_played", "rating"]:
                if c not in display_df.columns:
                    display_df[c] = 0

            if inactive_hidden > 0:
                st.caption(f"{inactive_hidden} inactive player(s) hidden from Standings/Top Performers for this league.")

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


elif sel == "🎯 Match Explorer":
    st.header("🎯 Match Explorer")
    st.caption("Preview win odds, expectation, and projected JUPR movement — from your perspective. Preview only; ratings do not change.")

    # -------- Helpers --------
    def win_label(p: float) -> str:
        # from YOUR team perspective
        if p >= 0.70:
            return "Heavy Favorite"
        if p >= 0.55:
            return "Favored"
        if p >= 0.45:
            return "Toss-up"
        if p >= 0.30:
            return "Underdog"
        return "Heavy Underdog"

    def normal_cdf(z: float) -> float:
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        
    def qp_int(key: str, default: int | None = None) -> int | None:
        v = qp_get(key, "")
        if not v:
            return default
        try:
            return int(v)
        except Exception:
            return default
    
    # Apply params whenever they change (so different Explain links work)
    sig = "|".join([
        qp_get("ctx",""),
        qp_get("me",""),
        qp_get("partner",""),
        qp_get("opp1",""),
        qp_get("opp2",""),
        qp_get("sy",""),
        qp_get("so",""),
    ])
    
    if st.session_state.get("mx_qp_sig", "") != sig:
        ctx_q = qp_get("ctx", "")
        if ctx_q:
            st.session_state["mx_ctx"] = ctx_q
    
        me_q = qp_int("me")
        partner_q = qp_int("partner")
        opp1_q = qp_int("opp1")
        opp2_q = qp_int("opp2")
    
        if me_q and int(me_q) in id_to_name:
            st.session_state["mx_me"] = id_to_name[int(me_q)]
        if partner_q and int(partner_q) in id_to_name:
            st.session_state["mx_partner"] = id_to_name[int(partner_q)]
        if opp1_q and int(opp1_q) in id_to_name:
            st.session_state["mx_opp1"] = id_to_name[int(opp1_q)]
        if opp2_q and int(opp2_q) in id_to_name:
            st.session_state["mx_opp2"] = id_to_name[int(opp2_q)]
    
        st.session_state["mx_sy"] = int(qp_int("sy", 11) or 11)
        st.session_state["mx_so"] = int(qp_int("so", 9) or 9)
    
        st.session_state["mx_qp_sig"] = sig

    # League options (default OVERALL; user may select league)
    if df_meta is not None and not df_meta.empty and "is_active" in df_meta.columns:
        active_meta = df_meta[df_meta["is_active"] == True].copy()
        league_opts = ["OVERALL"] + sorted(active_meta["league_name"].dropna().unique().tolist())
    else:
        league_opts = ["OVERALL"]

    # Preselect league from URL if provided (works for public deep-links)
    pre = st.session_state.get("preselect_league", "")
    default_idx = 0
    if pre and pre in league_opts:
        default_idx = league_opts.index(pre)

    ctx = st.selectbox("Rating context", league_opts, index=default_idx, key="mx_ctx")
    st.caption("If you select a league, calculations and the graph use league ratings only (overall ratings shown for reference).")

    # Active players
    if df_players is None or df_players.empty:
        st.info("No active players found.")
        st.stop()

    player_names = sorted(df_players["name"].astype(str).tolist())

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

    # -------- Player selection (doubles only; user-centered) --------
    st.subheader("Your matchup (doubles)")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        me_name = st.selectbox("I am", [""] + player_names, index=0, key="mx_me")
    with c2:
        partner_pool = [n for n in player_names if n and n != me_name]
        partner_name = st.selectbox("My partner", [""] + partner_pool, index=0, key="mx_partner")
    with c3:
        opp_pool_1 = [n for n in player_names if n and n not in (me_name, partner_name)]
        opp1_name = st.selectbox("Opponent 1", [""] + opp_pool_1, index=0, key="mx_opp1")
    with c4:
        opp_pool_2 = [n for n in player_names if n and n not in (me_name, partner_name, opp1_name)]
        opp2_name = st.selectbox("Opponent 2", [""] + opp_pool_2, index=0, key="mx_opp2")

    if not (me_name and partner_name and opp1_name and opp2_name):
        st.info("Select yourself, your partner, and both opponents.")
        st.stop()

    me_id = int(name_to_id.get(me_name))
    partner_id = int(name_to_id.get(partner_name))
    opp1_id = int(name_to_id.get(opp1_name))
    opp2_id = int(name_to_id.get(opp2_name))

    # Ratings for display (always show overall; show league too if selected)
    r_me_overall = get_overall_elo(me_id)
    r_partner_overall = get_overall_elo(partner_id)
    r_opp1_overall = get_overall_elo(opp1_id)
    r_opp2_overall = get_overall_elo(opp2_id)

    if ctx != "OVERALL":
        r_me_ctx = get_league_elo(me_id, ctx)
        r_partner_ctx = get_league_elo(partner_id, ctx)
        r_opp1_ctx = get_league_elo(opp1_id, ctx)
        r_opp2_ctx = get_league_elo(opp2_id, ctx)
    else:
        r_me_ctx, r_partner_ctx, r_opp1_ctx, r_opp2_ctx = r_me_overall, r_partner_overall, r_opp1_overall, r_opp2_overall

    # Team averages (context drives the engine)
    team_you_avg = (r_me_ctx + r_partner_ctx) / 2.0
    team_opp_avg = (r_opp1_ctx + r_opp2_ctx) / 2.0

    # Expectation in the SAME way your engine computes it
    expected_you = 1.0 / (1.0 + 10 ** ((team_opp_avg - team_you_avg) / 400.0))
    label = win_label(float(expected_you))
    k_val = get_k_for_context(ctx)

    # -------- Ratings table (professional transparency) --------
    rows = [
        {"Role": "You", "Player": me_name, "Overall JUPR": r_me_overall / 400.0, "League JUPR": (r_me_ctx / 400.0) if ctx != "OVERALL" else None},
        {"Role": "Partner", "Player": partner_name, "Overall JUPR": r_partner_overall / 400.0, "League JUPR": (r_partner_ctx / 400.0) if ctx != "OVERALL" else None},
        {"Role": "Opponent 1", "Player": opp1_name, "Overall JUPR": r_opp1_overall / 400.0, "League JUPR": (r_opp1_ctx / 400.0) if ctx != "OVERALL" else None},
        {"Role": "Opponent 2", "Player": opp2_name, "Overall JUPR": r_opp2_overall / 400.0, "League JUPR": (r_opp2_ctx / 400.0) if ctx != "OVERALL" else None},
    ]
    df_view = pd.DataFrame(rows)

    show_cols = ["Role", "Player", "Overall JUPR"]
    if ctx != "OVERALL":
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

    # -------- Expectation header (centerpiece) --------
    h1, h2 = st.columns([2, 1])
    with h1:
        st.markdown(f"## Your Expected Win Rate = **{expected_you*100:.0f}%**")
        st.caption(f"{label} • Context: {ctx}")
        if ctx != "OVERALL":
            st.info("Graph + projected movement below are computed using LEAGUE ratings only. Overall ratings above are for reference.")
    with h2:
        st.metric("Opponents win %", f"{(1.0-expected_you)*100:.0f}%")

    st.divider()

    # -------- Score input (default to 11, but unconstrained) --------
    st.subheader("Hypothetical Score")
    
    # Ensure sane defaults exist once per session (deep-links will override via query params)
    if "mx_sy" not in st.session_state:
        st.session_state["mx_sy"] = 11
    if "mx_so" not in st.session_state:
        st.session_state["mx_so"] = 9
    
    scol1, scol2 = st.columns(2)
    with scol1:
        s_you = st.number_input(
            "Your points",
            min_value=0,
            max_value=99,
            step=1,
            key="mx_sy",
            value=int(st.session_state.get("mx_sy", 11)),
        )
    with scol2:
        s_opp = st.number_input(
            "Opp points",
            min_value=0,
            max_value=99,
            step=1,
            key="mx_so",
            value=int(st.session_state.get("mx_so", 9)),
        )

    # Exact engine preview (team1 = your team)
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

    # -------- Summary tiles --------
    t1, t2, t3, t4 = st.columns(4)

    if beat_pp is not None:
        t1.metric("Beat expectation", f"{beat_pp:+.0f} pp")
        t1.caption(f"Your share {share_you*100:.1f}% vs expected {expected_you*100:.1f}%")
    else:
        t1.metric("Beat expectation", "—")
        t1.caption("No movement on ties / empty scores.")

    t2.metric("Your projected JUPR change", f"{d_you_jupr:+.4f}")
    t2.caption(f"Context: {ctx}")

    t3.metric("Partner projected JUPR change", f"{d_you_jupr:+.4f}")
    t3.caption("Same delta as you (team-based update).")

    t4.metric("Opponents projected JUPR change", f"{d_opp_jupr:+.4f}")
    t4.caption("Preview only — nothing is saved.")

    st.divider()

    def share_to_score11_label(share: float) -> str:
        # Convert a points-share value into an "equivalent scoreline" on a game-to-11 scale.
        # Left side: x–11 (loss path), middle: 11–11 (tie), right side: 11–y (win path)
        if share is None:
            return "—"
        if abs(share - 0.5) < 1e-12:
            return "11–11"
        if share < 0.5:
            my_pts = int(round(11.0 * share / (1.0 - share)))
            return f"{my_pts}–11"
        opp_pts = int(round(11.0 * (1.0 - share) / share))
        return f"11–{opp_pts}"

    # -------- Predictor curve (your perspective; selected context only) --------
    st.subheader("Rating Impact Predictor")
    
    # --- share for chart positioning (ties sit at the center) ---
    total_pts = int(s_you) + int(s_opp)
    share_chart = None
    if total_pts > 0:
        share_chart = float(s_you) / float(total_pts)
    
    # --- Curve function (mirrors your engine behavior but expressed in share space) ---
    def delta_you_from_share(share: float, expected: float, k: float) -> float:
        # Tie is explicitly zero in your engine
        if abs(share - 0.5) < 1e-12:
            return 0.0
    
        d = float(k) * 2.0 * (float(share) - float(expected))
    
        # winner floor + loser-positive cap, consistent with calculate_hybrid_elo policy
        if share > 0.5:  # win
            if d <= 0:
                d = float(MIN_WIN_DELTA_ELO)
        else:  # loss
            if d > 0:
                d = min(d, float(CAP_LOSER_GAIN_ELO))
    
        return d
    
    # --- Build curve points across share 0..1 ---
    xs, ys, score11s = [], [], []
    for i in range(0, 101):
        sh = i / 100.0
        xs.append(sh)
        ys.append(delta_you_from_share(sh, expected_you, k_val) / 400.0)
        score11s.append(share_to_score11_label(sh))
    
    curve_df = pd.DataFrame({"share": xs, "delta": ys, "score11": score11s})
    
    # --- Axis labeling: show scorelines, not percent ---
    # We keep a numeric axis (share) so ANY entered score can be plotted accurately,
    # but we relabel ticks to look like scorelines on a to-11 scale.
    tick_vals = [
        0.0,                 # 0–11
        3.0 / 14.0,          # 3–11
        6.0 / 17.0,          # 6–11
        9.0 / 20.0,          # 9–11
        0.5,                 # 11–11
        11.0 / 20.0,         # 11–9
        11.0 / 17.0,         # 11–6
        11.0 / 14.0,         # 11–3
        1.0,                 # 11–0
    ]
    
    label_expr = (
        "datum.value==0.5 ? '11–11' : "
        "(datum.value<0.5 ? "
        "(round(11*datum.value/(1-datum.value)) + '–11') : "
        "('11–' + round(11*(1-datum.value)/datum.value)))"
    )
    
    # --- Base curve ---
    base = (
        alt.Chart(curve_df)
        .mark_line()
        .encode(
            x=alt.X(
                "share:Q",
                title="Score (to 11 scale)",
                axis=alt.Axis(values=tick_vals, labelExpr=label_expr),
            ),
            y=alt.Y(
                "delta:Q",
                title=f"Projected JUPR change (you) — {ctx}",
                axis=alt.Axis(format="+.4f"),
            ),
            tooltip=[
                alt.Tooltip("score11:N", title="Score (to 11)"),
                alt.Tooltip("delta:Q", title="Δ JUPR", format="+.4f"),
            ],
        )
    )
    
    layers = [base]
    
    # --- Expectation marker (rule + tooltip) ---
    exp_df = pd.DataFrame(
        {
            "share": [float(expected_you)],
            "score11": [share_to_score11_label(float(expected_you))],
        }
    )
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
    
    # --- Selected result marker (dot) ---
    if share_chart is not None:
        sel_df = pd.DataFrame(
            {
                "share": [float(share_chart)],
                "delta": [float(d_you_jupr)],  # from calculate_hybrid_elo / 400
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

    def nm(pid: int) -> str:
        try:
            return str(id_to_name.get(int(pid), f"#{int(pid)}"))
        except Exception:
            return "—"

    def explain_link_for_player_match(match: dict, pid: int) -> str:
        """
        Option A: explain from the selected player's perspective.
        Uses OVERALL context so the explanation aligns with the JUPR Change shown in Player Search
        (which is based on overall snapshots).
        """
        try:
            pid = int(pid)
            t1 = [int(match.get("t1_p1")), int(match.get("t1_p2"))]
            t2 = [int(match.get("t2_p1")), int(match.get("t2_p2"))]
            s1 = int(match.get("score_t1", 0) or 0)
            s2 = int(match.get("score_t2", 0) or 0)
        except Exception:
            return ""
    
        if pid in t1:
            partner = t1[0] if t1[1] == pid else t1[1]
            opp1, opp2 = t2[0], t2[1]
            sy, so = s1, s2
        elif pid in t2:
            partner = t2[0] if t2[1] == pid else t2[1]
            opp1, opp2 = t1[0], t1[1]
            sy, so = s2, s1
        else:
            return ""
    
        return build_match_explorer_link(
            ctx="OVERALL",      # matches Player Search’s displayed JUPR Change (overall snapshots)
            me=pid,
            partner=partner,
            opp1=opp1,
            opp2=opp2,
            sy=sy,
            so=so,
            public=PUBLIC_MODE, # if you ever expose this page publicly, the explainer link still works cleanly
        )

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
        selected_name = st.selectbox(
            "Select a Player:",
            [""] + player_names,
            index=0,
            key="player_search_name",
        )
        
        if not selected_name:
            st.info("Start typing a name to search.")
            st.stop()


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
            
                on_team1 = (match.get("t1_p1") == p_id) or (match.get("t1_p2") == p_id)
                on_team2 = (match.get("t2_p1") == p_id) or (match.get("t2_p2") == p_id)
            
                # Score from the selected player's perspective (fixes “I won but score looks reversed”)
                my_pts = s1 if on_team1 else (s2 if on_team2 else 0)
                opp_pts = s2 if on_team1 else (s1 if on_team2 else 0)
                display_score = f"{int(my_pts)}-{int(opp_pts)}"
            
                # Result icon based on my perspective
                if s1 == s2:
                    result_icon = "➖"
                elif (on_team1 and s1 > s2) or (on_team2 and s2 > s1):
                    result_icon = "✅"
                elif (on_team1 and s1 < s2) or (on_team2 and s2 < s1):
                    result_icon = "❌"
                else:
                    result_icon = "➖"
            
                # Partner / opponents (names)
                partner_name = "—"
                opp_names = "—"
                try:
                    t1p1, t1p2 = int(match.get("t1_p1")), int(match.get("t1_p2"))
                    t2p1, t2p2 = int(match.get("t2_p1")), int(match.get("t2_p2"))
            
                    if on_team1:
                        partner_id = t1p1 if t1p2 == p_id else t1p2
                        partner_name = nm(partner_id)
                        opp_names = f"{nm(t2p1)} / {nm(t2p2)}"
                    elif on_team2:
                        partner_id = t2p1 if t2p2 == p_id else t2p2
                        partner_name = nm(partner_id)
                        opp_names = f"{nm(t1p1)} / {nm(t1p2)}"
                except Exception:
                    pass
            
                # Snapshot-based JUPR change (your existing logic)
                start_elo, end_elo = get_player_snap(match, p_id)
                if start_elo is not None and end_elo is not None:
                    start_elo = float(start_elo)
                    end_elo = float(end_elo)
                    jupr_change = (end_elo - start_elo) / 400.0
                    rating_after = end_elo / 400.0
                else:
                    # fallback for legacy rows without snapshots
                    raw_delta = float(match.get("elo_delta", 0) or 0)
                    my_team = 1 if on_team1 else 2
                    winner_team = 1 if s1 > s2 else 2 if s2 > s1 else 0
                    if winner_team == 0:
                        signed_elo = 0.0
                    elif winner_team == my_team:
                        signed_elo = abs(raw_delta)
                    else:
                        signed_elo = -abs(raw_delta)
            
                    jupr_change = signed_elo / 400.0
                    rating_after = None
            
                explain_url = explain_link_for_player_match(match, p_id)
            
                processed.append(
                    {
                        "Date": match.get("date"),
                        "Result": result_icon,
                        "Score": display_score,
                        "Partner": partner_name,
                        "Opponents": opp_names,
                        "League": str(match.get("league", "") or "").strip(),
                        "JUPR Change": float(jupr_change),
                        "Rating After Match": rating_after,
                        "Explain": explain_url,
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

                st.subheader("Match Log")

                table_df = display_df.copy()
                table_df["Date"] = table_df["Date"].dt.strftime("%Y-%m-%d")
                
                # Format numeric columns for display
                table_df["JUPR Change"] = table_df["JUPR Change"].map(lambda x: f"{float(x):+.4f}")
                table_df["Rating After Match"] = table_df["Rating After Match"].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
                
                cols = ["Date", "Result", "Score", "Partner", "Opponents", "League", "JUPR Change", "Rating After Match", "Explain"]
                
                st.dataframe(
                    table_df[cols],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Explain": st.column_config.LinkColumn("Explain", display_text="Explain"),
                    },
                )



elif sel == "❓ FAQs":
    st.header("❓ JUPR Rating FAQs")

    st.markdown(
        """
JUPR (Joe’s Unique Pickleball Ratings) is Tres Palapas’ **in-house rating system** used to create better matchups, seed events, and keep leveled play fair.

---

## What is a JUPR rating?
- JUPR is a **single player rating** displayed on a **1.000–7.000** scale.
- Ratings update based on **recorded match results** from JUPR-eligible events at Tres Palapas.
- The decimals help show *small* movement over time, but they are not meant to imply “perfect precision.”

---

## How do I get a JUPR rating?
You get a JUPR rating once you have **recorded matches** in the system from a JUPR-eligible event (examples below).
- If you haven’t played a recorded event yet, you may show as **Unrated / No JUPR**.
- Your first set of results can move your rating more quickly while the system “learns” your level.

---

## What matches count toward JUPR?
**Counts (JUPR-eligible):**
- Official JUPR Ladders and JUPR Round Robins
- Tournaments run through Tres Palapas with official score entry

**Does not count (not recorded / not JUPR-eligible):**
- Open Play
- Social Round Robins
- Drills and clinics

---

## What affects how much my rating moves?
JUPR is performance-based. Rating movement depends on:
- **Opponent strength** (beating stronger opponents moves you more)
- **Expected outcome** (results that surprise the system move you more)
- **Game score** (the score matters—not only win/loss)
- **Consistency over time** (repeated results matter more than one match)

---

## Can my rating go up after a loss?
Yes, it can happen.
If you **perform better than expected** (for example: a very close loss against a significantly higher-rated team), your rating may increase.

---

## Can my rating go down after a win?
No, winning is rewarded, not punished.
However, because scores matter, a win that is **far below expected performance** can result in minimal movement but will never result in a loss of rating. 

---

## How does JUPR work for doubles?
JUPR is an **individual rating**, but doubles results are used to update each player.
In doubles:
- the system evaluates the matchup based on **both teams** (each team’s strength is derived from the two players),
- then adjusts each player based on the outcome and score.

This means playing with different partners over time helps the system find your true level faster.

---

## What is the difference between “Overall” and “League” ratings?
- **Overall JUPR**: your rating across all JUPR-eligible matches at Tres Palapas.
- **League JUPR** (if shown): your rating **within a specific league** or series.

If you only play one league, your league rating and overall rating may look similar. If you play multiple formats/events, they can differ.

---

## What if a score was entered wrong?
If you believe there is a data-entry error (wrong score, wrong partner, wrong opponent):
- report it to the organizer as soon as possible.
Once corrected, the system will recompute the rating impact from the accurate result.

---

## How should I use my JUPR rating?
Use it to:
- join the right leveled sessions,
- seed ladders and tournaments fairly,
- track improvement over time,
- create competitive, enjoyable matches.

JUPR is designed to reflect **performance at Tres Palapas** based on recorded play.

---
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

def compress_courts(roster_df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-map court numbers to be contiguous 1..N (fixes gaps if a court becomes empty).
    """
    df = roster_df.copy()
    courts = sorted(df["court"].astype(int).unique().tolist())
    mapping = {old: i + 1 for i, old in enumerate(courts)}
    df["court"] = df["court"].astype(int).map(mapping)
    return normalize_slots(df)

def move_player_to_court(roster_df: pd.DataFrame, player: str, target_court: int, target_slot: int = 1) -> pd.DataFrame:
    """
    Move a player to a different court and insert them at target_slot within that court.
    Court sizes will change accordingly. Slots are normalized after.
    """
    df = roster_df.copy()
    if player not in df["name"].astype(str).tolist():
        return df

    target_court = int(target_court)
    target_slot = int(target_slot)

    # Remove player row
    row = df[df["name"] == player].iloc[0].copy()
    df = df[df["name"] != player].copy()

    # Ensure courts/slots are stable before insert
    df = normalize_slots(df)

    # Build target court name order, insert player
    target_names = df[df["court"] == target_court].sort_values("slot")["name"].tolist()
    target_slot = max(1, min(target_slot, len(target_names) + 1))
    target_names.insert(target_slot - 1, player)

    # Apply target court ordering
    for i, nm in enumerate(target_names, start=1):
        df.loc[(df["court"] == target_court) & (df["name"] == nm), "slot"] = i

       # Add player back on target court (if it wasn’t already in df)
    if player not in df["name"].tolist():
        df = pd.concat([df, pd.DataFrame([{
            "player_id": int(row.get("player_id")) if "player_id" in row else None,
            "name": player,
            "rating": float(row.get("rating", 1200.0)),
            "court": target_court,
            "slot": target_slot
        }])], ignore_index=True)


    # Normalize + compress (handles empty courts)
    df = normalize_slots(df)
    df = compress_courts(df)
    return df

def sync_ladder_court_sizes_from_roster(roster_df: pd.DataFrame):
    """
    Update ladder_court_sizes to match current roster distribution.
    Assumes courts are contiguous (use compress_courts first).
    """
    courts = sorted(roster_df["court"].astype(int).unique().tolist())
    sizes = []
    for c in courts:
        sizes.append(int((roster_df["court"].astype(int) == c).sum()))
    st.session_state.ladder_court_sizes = sizes

def roster_df_to_courts(roster_df: pd.DataFrame) -> list[dict]:
    df = roster_df.copy()

    # Normalize types
    df["court"] = df["court"].astype(int)
    df["player_id"] = df["player_id"].astype(int)

    # Ensure stable ordering within court
    if "slot" in df.columns:
        df["slot"] = df["slot"].astype(int)
        df = df.sort_values(["court", "slot"], ascending=[True, True])
    else:
        df = df.sort_values(["court", "rating"], ascending=[True, False])

    courts: list[dict] = []
    for c in sorted(df["court"].unique().tolist()):
        cdf = df[df["court"] == c]
        players = [
            {
                "player_id": str(int(r["player_id"])),  # draggableId must be string
                "name": str(r["name"]),
                "rating": float(r.get("rating", 1200.0)) / 400.0,  # display JUPR
            }
            for _, r in cdf.iterrows()
        ]
        courts.append({"court_id": f"Court {c}", "players": players})

    # Always include Bench
    courts.append({"court_id": "Bench", "players": []})
    return courts



def courts_to_roster_df(courts: list[dict], prev_roster_df: pd.DataFrame) -> pd.DataFrame:
    df_prev = prev_roster_df.copy()
    df_prev["player_id"] = df_prev["player_id"].astype(int)

    elo_map = dict(zip(df_prev["player_id"], df_prev["rating"]))
    name_map = dict(zip(df_prev["player_id"], df_prev["name"]))

    rows = []
    for c in courts:
        cid = str(c.get("court_id", ""))
        if cid == "Bench":
            continue

        m = re.findall(r"\d+", cid)
        if not m:
            continue
        cnum = int(m[0])

        players = c.get("players", []) or []
        for i, p in enumerate(players, start=1):
            pid = int(p["player_id"])
            rows.append(
                {
                    "player_id": pid,
                    "name": name_map.get(pid, str(p.get("name", pid))),
                    "rating": float(elo_map.get(pid, 1200.0)),
                    "court": int(cnum),
                    "slot": int(i),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return prev_roster_df

    out = out.sort_values(["court", "slot"], ascending=[True, True]).reset_index(drop=True)
    out = compress_courts(normalize_slots(out))
    return out

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
                            final_assignments.append({
                                "player_id": int(pl["id"]),
                                "name": pl["name"],
                                "rating": float(pl["rating"]),
                                "court": c_idx + 1
                            })

                        current_idx += size  # <-- THIS FIXES IT



                    final_roster = pd.DataFrame(final_assignments)

                    final_roster = final_roster.sort_values(["court", "rating"], ascending=[True, False]).copy()

                    final_roster["slot"] = final_roster.groupby("court").cumcount() + 1

                    st.session_state.ladder_live_roster = final_roster[["player_id","name","rating","court","slot"]].copy()
                    st.session_state.ladder_court_sizes = court_sizes
                    st.session_state.ladder_state = "CONFIRM_START"
                    st.rerun()


                # ---- 3.5) CONFIRM START ----
        if st.session_state.ladder_state == "CONFIRM_START":
            c_back, _ = st.columns([1, 5])
            if c_back.button("⬅️ Back (edit courts)"):
                # optional: clear preview so it forces a fresh preview next time
                st.session_state.pop("ladder_live_roster", None)
                st.session_state.ladder_state = "CONFIG_COURTS"
                st.rerun()

            st.markdown("#### Step 4: Court Board Preview (Drag & Drop)")
            st.caption("Use the Court Board to make final adjustments. Bench players will not be scheduled.")

            # Build roster/courts payload for the board
            roster_df = normalize_slots(st.session_state.ladder_live_roster.copy())
            roster_df = compress_courts(roster_df)
            courts_payload = roster_df_to_courts(roster_df)

            # Round-aware key (prevents board state carryover between rounds)
            round_num = int(st.session_state.get("ladder_round_num", 1))
            total_r = int(st.session_state.get("ladder_total_rounds", 1))

            result = court_board(courts_payload, key=f"court_board_confirm_start_r{round_num}")

            # Apply board changes back into ladder_live_roster
            if result and isinstance(result, dict) and "courts" in result:
                updated_courts = result["courts"]
                new_df = courts_to_roster_df(updated_courts, roster_df)

                if not new_df.equals(st.session_state.ladder_live_roster):
                    st.session_state.ladder_live_roster = new_df
                    sync_ladder_court_sizes_from_roster(st.session_state.ladder_live_roster)

                    st.session_state.pop("current_schedule", None)
                    st.session_state.pop("current_schedule_round", None)
                    st.rerun()

            # --- Validation gate (must always define can_start BEFORE using it) ---
            can_start = True
            df_check = st.session_state.ladder_live_roster.copy()
            court_counts = df_check.groupby("court").size().to_dict()

            problems = []
            for c, n in sorted(court_counts.items()):
                if int(n) < 4:
                    problems.append(f"Court {c} has {n} players (min 4).")

            warnings = []
            for c, n in sorted(court_counts.items()):
                if int(n) != 4:
                    warnings.append(f"Court {c} has {n} players (target 4).")

            if warnings:
                st.info(
                    "Court sizes don't need to be perfect to edit, but double-check before you start:\n\n- "
                    + "\n- ".join(warnings)
                )

            if problems:
                st.warning("Fix these before starting:\n\n- " + "\n- ".join(problems))
                can_start = False

            # Start button (DO NOT reset round_num here)
            start_label = "✅ Start Event (Round 1)" if round_num == 1 else f"✅ Start Round {round_num} / {total_r}"

            if st.button(start_label, disabled=not can_start, key=f"start_round_btn_{round_num}"):
                st.session_state.setdefault("ladder_round_num", 1)
                st.session_state.ladder_state = "PLAY_ROUND"
                st.session_state.pop("current_schedule", None)
                st.session_state.pop("current_schedule_round", None)
                st.rerun()



        def get_court_player_ids(court_df: pd.DataFrame) -> list[int]:
            """
            Safely return a list of player_ids for a court, regardless of small schema drift.
            Priority:
              1) court_df['player_id'] if present
              2) court_df['id'] if present (some parts of your code used 'id' earlier)
            Raises a clear error if neither exists.
            """
            if court_df is None or court_df.empty:
                return []
        
            if "player_id" in court_df.columns:
                return court_df["player_id"].apply(lambda x: int(float(x))).tolist()
        
            if "id" in court_df.columns:
                return court_df["id"].apply(lambda x: int(float(x))).tolist()
        
            raise KeyError(
                f"Roster court_df is missing player id column. Columns present: {list(court_df.columns)}"
            )



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
                roster_df = compress_courts(roster_df)  # ✅ keep courts contiguous
                names_now = roster_df["name"].tolist()

                # -------------------------
                # A) Swap players (keeps court sizes constant)
                # -------------------------
                cA, cB, cC = st.columns([2, 2, 1])
                a = cA.selectbox("Swap Player A", names_now, key=f"swap_a_r{current_r}")
                b = cB.selectbox("with Player B", names_now, key=f"swap_b_r{current_r}", index=1 if len(names_now) > 1 else 0)
                if cC.button("Swap", key=f"swap_btn_r{current_r}"):
                    st.session_state.ladder_live_roster = compress_courts(swap_players(roster_df, a, b))
                    sync_ladder_court_sizes_from_roster(st.session_state.ladder_live_roster)
                    if "current_schedule" in st.session_state:
                        del st.session_state.current_schedule
                    st.rerun()

                st.divider()

                # -------------------------
                # B) Reorder within court (does NOT change court sizes)
                # -------------------------
                c1, c2, c3 = st.columns([2, 2, 1])
                court_list = sorted(roster_df["court"].unique().tolist())
                chosen_court = c1.selectbox("Court to reorder", court_list, key=f"re_ct_r{current_r}")
                court_players = roster_df[roster_df["court"] == int(chosen_court)].sort_values("slot")["name"].tolist()

                p = c2.selectbox("Player", court_players, key=f"re_p_r{current_r}")
                new_pos = c3.number_input(
                    "New position",
                    min_value=1,
                    max_value=max(1, len(court_players)),
                    value=1,
                    step=1,
                    key=f"re_pos_r{current_r}",
                )

                if st.button("Apply reorder", key=f"re_btn_r{current_r}"):
                    st.session_state.ladder_live_roster = compress_courts(move_within_court(roster_df, p, int(new_pos)))
                    sync_ladder_court_sizes_from_roster(st.session_state.ladder_live_roster)
                    if "current_schedule" in st.session_state:
                        del st.session_state.current_schedule
                    st.rerun()

                st.divider()

                # -------------------------
                # C) Move player to a DIFFERENT court (changes court sizes ✅)
                # -------------------------
                st.markdown("#### 🔁 Move player to a different court (fix court sizes)")

                m1, m2, m3, m4 = st.columns([2, 1, 1, 1])
                mv_player = m1.selectbox("Player to move", names_now, key=f"mv_p_r{current_r}")

                # choose target court from existing courts
                target_court = m2.selectbox("To court", court_list, key=f"mv_ct_r{current_r}")

                # position inside target court
                target_names = roster_df[roster_df["court"] == int(target_court)].sort_values("slot")["name"].tolist()
                target_pos = m3.number_input(
                    "Insert pos",
                    min_value=1,
                    max_value=max(1, len(target_names) + 1),
                    value=1,
                    step=1,
                    key=f"mv_pos_r{current_r}",
                )

                if m4.button("Move", key=f"mv_btn_r{current_r}"):
                    new_df = move_player_to_court(roster_df, mv_player, int(target_court), int(target_pos))
                    st.session_state.ladder_live_roster = new_df
                    sync_ladder_court_sizes_from_roster(st.session_state.ladder_live_roster)
                    if "current_schedule" in st.session_state:
                        del st.session_state.current_schedule
                    st.rerun()


            current_r = int(st.session_state.ladder_round_num)

            if (
                "current_schedule" not in st.session_state
                or st.session_state.get("current_schedule_round") != current_r
            ):
                schedule = []
                for c_idx in range(len(st.session_state.ladder_court_sizes)):
                    c_num = c_idx + 1
                    court_df = st.session_state.ladder_live_roster[
                        st.session_state.ladder_live_roster["court"] == c_num
                    ].copy()
            
                    if "slot" in court_df.columns:
                        court_df = court_df.sort_values("slot")
            
                    players = get_court_player_ids(court_df)
                    fmt = f"{len(players)}-Player"
                    matches = get_match_schedule(fmt, players)
                    schedule.append({"c": c_num, "matches": matches})
            
                st.session_state.current_schedule = schedule
                st.session_state.current_schedule_round = current_r


            all_results = []
            with st.form("round_score_form"):
                for c_data in st.session_state.current_schedule:
                    st.markdown(f"<div class='court-header'>Court {c_data['c']}</div>", unsafe_allow_html=True)
                    for m_idx, mm in enumerate(c_data["matches"]):
                        c1, c2, c3, c4 = st.columns([3, 1, 1, 3])
                        label = mm.get("desc", f"Game {m_idx+1}")
                        def nm(pid: int) -> str:
                            return str(id_to_name.get(int(pid), f"#{pid}"))

                        k1 = match_key(current_r, c_data["c"], mm["t1"], mm["t2"], "s1")
                        k2 = match_key(current_r, c_data["c"], mm["t1"], mm["t2"], "s2")
                        s1 = c2.number_input("S1", min_value=0, max_value=99, value=0, step=1, key=k1)
                        s2 = c3.number_input("S2", min_value=0, max_value=99, value=0, step=1, key=k2)

                        
                        c1.text(f"{label}: {nm(mm['t1'][0])} & {nm(mm['t1'][1])}")
                        c4.text(f"{nm(mm['t2'][0])} & {nm(mm['t2'][1])}")

                        all_results.append({
                            "court": c_data["c"],
                            "t1_p1": int(mm["t1"][0]),
                            "t1_p2": int(mm["t1"][1]),
                            "t2_p1": int(mm["t2"][0]),
                            "t2_p2": int(mm["t2"][1]),
                            "s1": int(s1),
                            "s2": int(s2),
                        })


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
                        try:
                            res = process_matches(valid_matches, name_to_id, df_players_all, df_leagues, df_meta)
                            st.success(f"Matches saved ({res['inserted']}). Skipped incomplete: {res['skipped_incomplete']}.")

                        except Exception as e:
                            st.error("Failed to save matches. See details below.")
                            st.exception(e)
                    else:
                        st.warning("No scores entered (all matches 0–0), so nothing was saved.")


                    # movement calculations
                    round_stats = {}
                    for pid in st.session_state.ladder_live_roster["player_id"].astype(int).unique():
                        round_stats[int(pid)] = {"w": 0, "diff": 0, "pts": 0}
                    
                    for r in valid_matches:
                        win_team1 = r["s1"] > r["s2"]
                        diff = abs(r["s1"] - r["s2"])
                    
                        for pid in [r["t1_p1"], r["t1_p2"]]:
                            pid = int(pid)
                            round_stats[pid]["pts"] += int(r["s1"])
                            round_stats[pid]["diff"] += diff if win_team1 else -diff
                            if win_team1:
                                round_stats[pid]["w"] += 1
                    
                        for pid in [r["t2_p1"], r["t2_p2"]]:
                            pid = int(pid)
                            round_stats[pid]["pts"] += int(r["s2"])
                            round_stats[pid]["diff"] += -diff if win_team1 else diff
                            if not win_team1:
                                round_stats[pid]["w"] += 1
                    
                    df_roster = st.session_state.ladder_live_roster.copy()
                    df_roster["Round Wins"] = df_roster["player_id"].astype(int).map(lambda pid: round_stats.get(int(pid), {}).get("w", 0))
                    df_roster["Round Diff"] = df_roster["player_id"].astype(int).map(lambda pid: round_stats.get(int(pid), {}).get("diff", 0))
                    df_roster["Round Pts"]  = df_roster["player_id"].astype(int).map(lambda pid: round_stats.get(int(pid), {}).get("pts", 0))


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

            # ✅ Include player_id so we never lose it between rounds
            edit_view = preview_df[["player_id", "name", "rating", "court", "Proposed Court"]].copy()
            
            editor_df = st.data_editor(
                edit_view,
                column_config={
                    "player_id": st.column_config.NumberColumn("ID", disabled=True),
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
                
                    # Clear event state
                    for k in [
                        "ladder_round_num",
                        "ladder_live_roster",
                        "ladder_court_sizes",
                        "ladder_movement_preview",
                        "current_schedule",
                        "current_schedule_round",
                    ]:
                        st.session_state.pop(k, None)
                
                    st.session_state.ladder_state = "SETUP"
                    st.rerun()

                else:
                    new_roster = editor_df.copy()
                    new_roster["court"] = new_roster["Proposed Court"].astype(int)

                    # keep rating column from prior roster (editor already has rating, but keep safe)
                    rating_map = dict(zip(st.session_state.ladder_live_roster["name"], st.session_state.ladder_live_roster["rating"]))
                    new_roster["rating"] = new_roster["name"].map(lambda x: float(rating_map.get(x, 1200.0)))

                    # sort + slots
                    # ✅ Preserve “top-to-bottom” feel on each court for next round:
                    # sort by rating desc (or keep existing slot if you prefer)
                    new_roster = new_roster.sort_values(["court", "rating"], ascending=[True, False]).copy()
                    new_roster["slot"] = new_roster.groupby("court").cumcount() + 1


                    new_sizes = new_roster["court"].value_counts().sort_index().tolist()
                    st.session_state.ladder_court_sizes = new_sizes
                    # Preserve player_id into next round
                    pid_map = dict(zip(
                        st.session_state.ladder_live_roster["name"],
                        st.session_state.ladder_live_roster["player_id"]
                    ))
                    
                    new_roster["player_id"] = new_roster["name"].map(lambda x: int(pid_map.get(x)))
                    
                    # Keep the canonical column set
                    st.session_state.ladder_live_roster = new_roster[["player_id", "name", "rating", "court", "slot"]].copy()


                    st.session_state.ladder_round_num = int(st.session_state.ladder_round_num) + 1
                    st.session_state.ladder_state = "CONFIRM_START"
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

# =========================
# PUBLIC: CHALLENGE LADDER
# =========================
if sel == "🪜 Challenge Ladder":
    st.header("🪜 Challenge Ladder")

    settings = ladder_fetch_settings()
    df_roster, df_flags, df_ch, df_pass = ladder_load_core()

    tab_ladder, tab_active, tab_info = st.tabs(["🪜 Ladder", "⚔️ Active Challenges", "📘 Info"])

    # -------------------------
    # TAB 1: LADDER
    # -------------------------
    with tab_ladder:
        if df_roster is None or df_roster.empty:
            st.info("Ladder roster not initialized yet.")
        else:
            # Compute status map once (works across tiers)
            status_map = ladder_compute_status_map(df_roster, df_flags, df_ch, df_pass, settings, id_to_name)
    
            # Tier sub-tabs
            t_tabs = st.tabs([tier_title(tid) for tid in TIER_ORDER])

        for i, tid in enumerate(TIER_ORDER):
            with t_tabs[i]:
                sub = df_roster[(df_roster["is_active"] == True) & (df_roster["tier_id"] == tid)].copy()

                if sub.empty:
                    st.info("No players in this tier.")
                    continue

                sub["name"] = sub["player_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
                sub["status"] = sub["player_id"].apply(lambda pid: status_map.get(int(pid), {}).get("status", "Ready to Defend"))
                sub["detail"] = sub["player_id"].apply(lambda pid: status_map.get(int(pid), {}).get("detail", ""))

                # Tier-local search
                q = st.text_input(f"Search ({tier_title(tid)})", value="", key=f"challenge_ladder_search_{tid}")
                if q.strip():
                    sub = sub[sub["name"].str.contains(q.strip(), case=False, na=False)].copy()

                sub = sub.sort_values("rank", ascending=True).copy()

                def rank_badge(r):
                    r = int(r)
                    if r == 1: return "🥇 1"
                    if r == 2: return "🥈 2"
                    if r == 3: return "🥉 3"
                    return str(r)

                sub["Rank"] = sub["rank"].astype(int).apply(rank_badge)

                st.dataframe(sub[["Rank", "name", "status", "detail"]], use_container_width=True, hide_index=True)


    # -------------------------
    # TAB 2: ACTIVE CHALLENGES
    # -------------------------
    with tab_active:
        if df_ch is None or df_ch.empty:
            st.info("No challenges yet.")
        else:
            df = df_ch.copy()
            df["challenger_name"] = df["challenger_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
            df["defender_name"] = df["defender_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
            df["bucket"] = df.apply(lambda r: ladder_bucket_challenge(r.to_dict()), axis=1)

            tab_names = [
                "Pending Acceptance",
                "Accepted / In Window",
                "Acceptance Overdue",
                "Play Overdue",
                "Recently Completed",
            ]
            tabs = st.tabs(tab_names)

            for i, tname in enumerate(tab_names):
                with tabs[i]:
                    view = df[df["bucket"] == tname].copy()
                    if view.empty:
                        st.info("No items.")
                        continue

                    view["created_at"] = pd.to_datetime(view["created_at"], utc=True, errors="coerce")
                    view = view.sort_values("created_at", ascending=False)

                    show = view[
                        ["id", "status", "challenger_name", "defender_name", "created_at", "accept_by", "play_by", "winner_id"]
                    ].copy()

                    # Optional: show winner name instead of raw ID
                    def winner_name(x):
                        if x is None or (isinstance(x, float) and pd.isna(x)):
                            return ""
                        try:
                            return ladder_nm(int(x), id_to_name)
                        except Exception:
                            return ""

                    show["winner"] = show["winner_id"].apply(winner_name)
                    show = show.drop(columns=["winner_id"])

                    st.dataframe(show, use_container_width=True, hide_index=True)

    # -------------------------
    # TAB 3: INFO (NEW)
    # -------------------------
    with tab_info:
        st.subheader("📘 Challenge Ladder — Quick Rules")

        st.markdown(
            """
**The Challenge Ladder is an ongoing, challenge-anytime ranking system.**  
Players move up by challenging and defeating players ranked above them.

---

1. **One active challenge at a time**  
   You may only be involved in one challenge at a time—either as the challenger or defender.

2. **Who you can challenge**  
   You may challenge a player ranked **above you**, up to **7 spots higher**, provided both players are eligible.

3. **Eligibility & status rules**  
   You cannot initiate or receive challenges if you are **Locked**, on **Vacation**, or **Reinstate Required**.  
   - **Cooldown**: you may be challenged, but cannot initiate.  
   - **Protected**: you may initiate, but cannot be challenged.

4. **Challenges must be officially recorded**  
   A challenge is only official once recorded by staff in the **Pro Shop Challenge Ledger**.  
   (For Top-20, token placement must also be logged by staff.)

5. **48-hour acceptance window**  
   The defending player has **48 hours** to accept a challenge once it is recorded.

6. **Monthly Pass (1 per month)**  
   You may decline one challenge per calendar month without losing your rank, as long as the pass is used within the 48-hour acceptance window.

7. **No response = forfeit**  
   If a challenge is not accepted within 48 hours and no Monthly Pass is used, the defender forfeits and positions are awarded accordingly.

8. **7-day play window**  
   Once accepted, the match must be completed within **7 days**. Failure to do so may result in an admin-determined forfeit.

9. **Match format: Swing Partner Swap**  
   - Two doubles matches are played.  
   - Each ranked player keeps the same opponent but swaps partners between matches.  
   - Swing partners never move on the ladder.

10. **How the winner is decided**  
   - Win both matches = win the challenge.  
   - Split matches = total games won.  
   - Still tied = total point differential.  
   - Exact tie favors the **defender**.

11. **Ladder movement**  
   - If the challenger wins, the two ranked players **swap ranks**.  
   - If the defender wins, **no ranks change**.

12. **Post-match timers**  
   - Challenger enters **Cooldown (72 hours)**.  
   - Defender enters **Protected (72 hours)**.

13. **Vacation is admin-controlled**  
   Vacation status is set by the Ladder Admin and typically requires **48 hours’ notice** when possible.

14. **Reinstatement required after vacation**  
   Returning players must complete a reinstatement match before resuming normal ladder activity.

15. **Disputes & enforcement**  
   The Ladder Admin resolves disputes and enforces rules using the **Pro Shop Challenge Ledger** as the source of truth.
""")



elif sel == "🛠️ Challenge Ladder Admin":
    st.header("🛠️ Challenge Ladder Admin (Challenge Ladder)")

    if not st.session_state.admin_logged_in:
        st.error("Admin login required.")
        st.stop()

    settings = ladder_fetch_settings()
    df_roster, df_flags, df_ch, df_pass = ladder_load_core()

    tabs = st.tabs(["📊 Dashboard", "🧾 Intake", "🗂 Challenge Detail", "👥 Roster", "⬆️⬇️ Tier Movement", "🏖 Overrides", "📜 Audit"])

    # -------------------------
    # TAB 1: DASHBOARD
    # -------------------------
    with tabs[0]:
        st.subheader("📊 Ops Dashboard")

        if df_ch is None or df_ch.empty:
            st.info("No challenges yet.")
        else:
            df = df_ch.copy()
            df["bucket"] = df.apply(lambda r: ladder_bucket_challenge(r.to_dict()), axis=1)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Pending Acceptance", int((df["bucket"] == "Pending Acceptance").sum()))
            c2.metric("Acceptance Overdue", int((df["bucket"] == "Acceptance Overdue").sum()))
            c3.metric("Play Overdue", int((df["bucket"] == "Play Overdue").sum()))
            c4.metric("Accepted / In Window", int((df["bucket"] == "Accepted / In Window").sum()))

            st.divider()
            needs = df[df["bucket"].isin(["Acceptance Overdue", "Play Overdue"])].copy()
            if needs.empty:
                st.success("No overdue items.")
            else:
                needs["challenger"] = needs["challenger_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
                needs["defender"] = needs["defender_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
                st.dataframe(needs[["id","bucket","status","challenger","defender","accept_by","play_by","created_at"]], use_container_width=True, hide_index=True)

    # -------------------------
    # TAB 2: INTAKE
    # -------------------------
    tier_pick = st.selectbox("Tier", TIER_ORDER, format_func=tier_title, key="ladder_intake_tier")

    with tabs[1]:
        st.subheader("🧾 Enter Challenge (from Pro Shop Ledger)")

        if df_roster is None or df_roster.empty:
            st.error("Roster not initialized yet. Go to the Roster tab to add players.")
        else:
            roster_active = df_roster[(df_roster["is_active"] == True) & (df_roster["tier_id"] == tier_pick)].copy()
            roster_active["name"] = roster_active["player_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
            roster_active = roster_active.sort_values("rank")
        
            name_to_pid = dict(zip(roster_active["name"], roster_active["player_id"]))
            pid_to_rank = dict(zip(roster_active["player_id"].astype(int), roster_active["rank"].astype(int)))

            # Compute current statuses for validation
            status_map = ladder_compute_status_map(df_roster, df_flags, df_ch, df_pass, settings, id_to_name)

            with st.form("ladder_intake_form"):
                challenger_name = st.selectbox("Challenger", [""] + roster_active["name"].tolist())
                defender_name = st.selectbox("Defender", [""] + roster_active["name"].tolist())
                ledger_ref = st.text_input("Ledger reference / notes (optional)", value="")
                override = st.checkbox("Admin override (bypass eligibility rules)", value=False)
                submitted = st.form_submit_button("Create Challenge")
            
            if submitted:
                if not challenger_name or not defender_name:
                    st.error("Select both Challenger and Defender.")
                    st.stop()
            
                chal_id = int(name_to_pid[challenger_name])
                def_id = int(name_to_pid[defender_name])
            
                if chal_id == def_id:
                    st.error("Challenger and Defender must be different.")
                    st.stop()
            
                chal_rank = int(pid_to_rank.get(chal_id, 999999))
                def_rank = int(pid_to_rank.get(def_id, 999999))
                challenge_range = int(settings.get("challenge_range", 3) or 3)
            
                errors = []
            
                # Must be upward challenge (defender above challenger)
                if def_rank >= chal_rank:
                    errors.append("Defender must be ranked ABOVE Challenger.")
            
                # Range
                if (chal_rank - def_rank) > challenge_range:
                    errors.append(f"Rank gap too large. Allowed: {challenge_range}. Gap: {chal_rank - def_rank}.")
            
                # Status eligibility
                chal_status = status_map.get(chal_id, {}).get("status", "Ready to Defend")
                def_status = status_map.get(def_id, {}).get("status", "Ready to Defend")
            
                if chal_status not in ("Ready to Defend",) and not override:
                    errors.append(f"Challenger is not eligible to initiate (status: {chal_status}).")
                if def_status not in ("Ready to Defend", "Cooldown") and not override:
                    errors.append(f"Defender is not eligible to be challenged (status: {def_status}).")
            
                if errors and not override:
                    st.error("Cannot create challenge:\n\n- " + "\n- ".join(errors))
                    st.stop()
            
                now = dt_utc_now()
                accept_by = now + timedelta(hours=int(settings.get("accept_window_hours", 48) or 48))
            
                payload = {
                    "club_id": CLUB_ID,
                    "challenger_id": chal_id,
                    "defender_id": def_id,
                    "challenger_rank_at_create": chal_rank,
                    "defender_rank_at_create": def_rank,
                    "status": "PENDING_ACCEPTANCE",
                    "created_by": "admin",
                    "ledger_ref": ledger_ref.strip() or None,
                    "accept_by": accept_by.isoformat(),
                    "tier_id": str(tier_pick),
                }
            
                try:
                    res = sb_retry(lambda: supabase.table("ladder_challenges").insert(payload).execute())
                    new_id = res.data[0]["id"] if res.data else None
                    ladder_audit("challenge_create", "ladder_challenges", str(new_id or ""), None, payload)
                    st.success(f"Challenge created. ID = {new_id}")
                    st.rerun()
                except Exception as e:
                    st.error("Failed to create challenge.")
                    st.exception(e)


    # -------------------------
    # TAB 3: CHALLENGE DETAIL
    # -------------------------
    with tabs[2]:
        st.subheader("🗂 Challenge Detail")
    
        # If no challenges yet, show message and DO NOT run the rest of this tab
        if df_ch is None or df_ch.empty:
            st.info("No challenges yet. Create one in the Intake tab.")
        else:
            df = df_ch.copy()
    
            # Build labels safely
            df["label"] = df.apply(
                lambda r: f"#{int(r['id'])} • {ladder_nm(int(r['challenger_id']), id_to_name)} vs {ladder_nm(int(r['defender_id']), id_to_name)} • {r.get('status','')}",
                axis=1,
            )
    
            pick = st.selectbox("Select challenge", df["label"].tolist(), index=0, key="ladder_admin_pick_challenge")
    
            # Guard: if something goes sideways, don't crash
            hit = df[df["label"] == pick]
            if hit.empty:
                st.warning("Selected challenge not found (refresh and try again).")
            else:
                ch_row = hit.iloc[0].to_dict()
    
                ch_id = int(ch_row["id"])
                chal_id = int(ch_row["challenger_id"])
                def_id = int(ch_row["defender_id"])
    
                st.write(f"**Challenge #{ch_id}**")
                st.write(f"- Challenger: **{ladder_nm(chal_id, id_to_name)}** (rank at create: {ch_row.get('challenger_rank_at_create')})")
                st.write(f"- Defender: **{ladder_nm(def_id, id_to_name)}** (rank at create: {ch_row.get('defender_rank_at_create')})")
                st.write(f"- Status: **{ch_row.get('status')}**")
                st.write(f"- Accept by: {ch_row.get('accept_by')}")
                st.write(f"- Play by: {ch_row.get('play_by')}")
    
                st.divider()
    
                # ---- Actions: Accept / Cancel / Forfeit / Pass ----
                c1, c2, c3, c4 = st.columns(4)
    
                if c1.button(
                    "✅ Mark Accepted",
                    disabled=(str(ch_row.get("status")) != "PENDING_ACCEPTANCE"),
                    key=f"accept_{ch_id}",
                ):
                    before = ch_row.copy()
                    now = dt_utc_now()
                    play_by = now + timedelta(days=int(settings.get("play_window_days", 7) or 7))
                    upd = {
                        "accepted_at": now.isoformat(),
                        "play_by": play_by.isoformat(),
                        "status": "ACCEPTED_SCHEDULING",
                    }
                    sb_retry(lambda: supabase.table("ladder_challenges").update(upd).eq("club_id", CLUB_ID).eq("id", ch_id).execute())
                    ladder_audit("challenge_accept", "ladder_challenges", str(ch_id), before, {**before, **upd})
                    st.success("Accepted.")
                    st.rerun()
    
                if c2.button("🗑 Cancel (Admin)", type="secondary", key=f"cancel_{ch_id}"):
                    before = ch_row.copy()
                    upd = {"status": "CANCELED", "resolution_notes": "Admin canceled", "completed_at": dt_utc_now().isoformat()}
                    sb_retry(lambda: supabase.table("ladder_challenges").update(upd).eq("club_id", CLUB_ID).eq("id", ch_id).execute())
                    ladder_audit("challenge_cancel", "ladder_challenges", str(ch_id), before, {**before, **upd})
                    st.success("Canceled.")
                    st.rerun()
    
                with c3:
                    forfeit_by = st.selectbox(
                        "Forfeit by",
                        ["", ladder_nm(chal_id, id_to_name), ladder_nm(def_id, id_to_name)],
                        key=f"ff_by_{ch_id}",
                    )
                if c3.button("🏳️ Record Forfeit", disabled=(forfeit_by == ""), key=f"ff_btn_{ch_id}"):
                    before = ch_row.copy()
                    fb = chal_id if forfeit_by == ladder_nm(chal_id, id_to_name) else def_id
                    winner = def_id if fb == chal_id else chal_id
                    upd = {
                        "status": "FORFEITED",
                        "forfeit_by": int(fb),
                        "forfeit_reason": "Forfeit (admin entry)",
                        "winner_id": int(winner),
                        "completed_at": dt_utc_now().isoformat(),
                    }
                    sb_retry(lambda: supabase.table("ladder_challenges").update(upd).eq("club_id", CLUB_ID).eq("id", ch_id).execute())
    
                    # If challenger wins, swap ranks
                    if int(winner) == chal_id:
                        sb_retry(lambda: supabase.rpc("ladder_swap_ranks", {"p_club_id": CLUB_ID, "p_player_a": chal_id, "p_player_b": def_id}).execute())
    
                    ladder_audit("challenge_forfeit", "ladder_challenges", str(ch_id), before, {**before, **upd})
                    st.success("Forfeit recorded.")
                    st.rerun()
    
                with c4:
                    pass_user = st.selectbox(
                        "Pass used by",
                        ["", ladder_nm(chal_id, id_to_name), ladder_nm(def_id, id_to_name)],
                        key=f"pass_by_{ch_id}",
                    )
                if c4.button("🎟 Record Pass Used", disabled=(pass_user == ""), key=f"pass_btn_{ch_id}"):
                    before = ch_row.copy()
                    pu_pid = chal_id if pass_user == ladder_nm(chal_id, id_to_name) else def_id
                    now = dt_utc_now()
                    mk = month_key_utc(now)
    
                    sb_retry(lambda: supabase.table("ladder_pass_usage").insert({
                        "club_id": CLUB_ID,
                        "player_id": int(pu_pid),
                        "month_key": mk,
                        "used_at": now.isoformat(),
                        "challenge_id": ch_id,
                    }).execute())
    
                    upd = {
                        "status": "CANCELED",
                        "pass_used_by": int(pu_pid),
                        "pass_used_at": now.isoformat(),
                        "resolution_notes": "Pass used",
                        "completed_at": now.isoformat(),
                    }
                    sb_retry(lambda: supabase.table("ladder_challenges").update(upd).eq("club_id", CLUB_ID).eq("id", ch_id).execute())
                    ladder_audit("challenge_pass_used", "ladder_challenges", str(ch_id), before, {**before, **upd})
                    st.success("Pass recorded (challenge closed).")
                    st.rerun()
    
                # The rest of your match-entry / finalize code can remain below this point,
                # but make sure it is also inside this same `else:` block so it only runs when ch_row exists.

    # -------------------------
    # TAB 4: ROSTER
    # -------------------------
    with tabs[3]:
        st.subheader("👥 Ladder Roster")

        # -------------------------
        # Add ONE player to bottom
        # -------------------------
            # Tier context for roster tools (everything below applies to this tier)
        tier_ctx = st.selectbox(
            "Tier to manage",
            TIER_ORDER,
            format_func=tier_title,
            key="ladder_roster_tier_ctx",
        )

        st.markdown("#### ➕ Add one player (appends to bottom)")
        st.caption("Existing players are appended to the bottom. New names will be created in Players (default rating) and appended to the bottom.")
        
        # Build selectable list from existing Players table
        all_player_names = []
        if df_players_all is not None and not df_players_all.empty and "name" in df_players_all.columns:
            all_player_names = sorted(df_players_all["name"].astype(str).tolist())
        
        with st.form("ladder_add_one_form"):
            existing_pick = st.selectbox("Pick an existing player", [""] + all_player_names, index=0)
            new_name = st.text_input("Or type a new player name", value="")
            new_rating = st.number_input("New player starting JUPR (only used if creating)", min_value=1.0, max_value=7.0, value=3.5, step=0.1)
            auto_assign = st.checkbox(
                "Auto-assign tier from current OVERALL JUPR",
                value=True,
                key="ladder_add_one_auto_tier",
            )
            
            manual_tier = st.selectbox(
                "Manual tier (used only if auto-assign is OFF)",
                TIER_ORDER,
                format_func=tier_title,
                index=TIER_ORDER.index(tier_ctx),
                disabled=auto_assign,
                key="ladder_add_one_manual_tier",
            )

            add_one = st.form_submit_button("Add to bottom")
        
        if add_one:
            nm = (new_name.strip() or existing_pick.strip())
            if not nm:
                st.error("Pick an existing player OR type a new name.")
                st.stop()
        
            # Ensure the player exists in Players table
            if nm not in name_to_id:
                ok, err = safe_add_player(nm, float(new_rating))
                if not ok:
                    st.error(f"Could not add player '{nm}': {err}")
                    st.stop()
        
                # Refresh mappings so name_to_id includes new player
                (
                    df_players_all,
                    df_players,
                    df_leagues,
                    df_matches,
                    df_meta,
                    name_to_id,
                    id_to_name,
                ) = load_data()
        
            pid = int(name_to_id.get(nm))
    if add_one:
        nm = (new_name.strip() or existing_pick.strip())
    if not nm:
        st.error("Pick an existing player OR type a new name.")
        st.stop()

    # Ensure the player exists in Players table
    if nm not in name_to_id:
        ok, err = safe_add_player(nm, float(new_rating))
        if not ok:
            st.error(f"Could not add player '{nm}': {err}")
            st.stop()

        # Refresh mappings so name_to_id includes new player
        (
            df_players_all,
            df_players,
            df_leagues,
            df_matches,
            df_meta,
            name_to_id,
            id_to_name,
        ) = load_data()

    pid = int(name_to_id.get(nm))

    # -------------------------
    # Tier assignment (THIS is where your snippet goes)
    # -------------------------
    auto_assign_val = bool(st.session_state.get("ladder_add_one_auto_tier", True))
    manual_tier_val = str(st.session_state.get("ladder_add_one_manual_tier", tier_ctx))

    if auto_assign_val:
        # compute from OVERALL rating in players table (ELO x400)
        p_row = df_players_all[df_players_all["id"] == pid]
        elo = float(p_row.iloc[0].get("rating", 1200.0) or 1200.0) if not p_row.empty else 1200.0
        jupr = elo / 400.0
        tier_for_player = tier_for_jupr(jupr)
    else:
        tier_for_player = manual_tier_val

    # -------------------------
    # Next rank within THAT tier (THIS is where your snippet goes)
    # -------------------------
    max_rank_resp = sb_retry(lambda: (
        supabase.table("ladder_roster")
        .select("rank")
        .eq("club_id", CLUB_ID)
        .eq("tier_id", tier_for_player)
        .eq("is_active", True)
        .order("rank", desc=True)
        .limit(1)
        .execute()
    ))
    next_rank = (int(max_rank_resp.data[0]["rank"]) + 1) if max_rank_resp.data else 1

    # If player already exists in ladder_roster, update/reactivate instead of inserting
    existing_row = sb_retry(lambda: (
        supabase.table("ladder_roster")
        .select("id,is_active,rank,tier_id")
        .eq("club_id", CLUB_ID)
        .eq("player_id", pid)
        .limit(1)
        .execute()
    ))

    now_iso = dt_utc_now().isoformat()

    if existing_row.data:
        row = existing_row.data[0]
        if bool(row.get("is_active", True)):
            st.info(
                f"'{nm}' is already ACTIVE on the ladder "
                f"({tier_title(row.get('tier_id'))}, rank {row.get('rank')})."
            )
        else:
            upd = {
                "is_active": True,
                "tier_id": tier_for_player,
                "rank": int(next_rank),
                "left_at": None,
                "joined_at": now_iso,
            }
            sb_retry(lambda: (
                supabase.table("ladder_roster")
                .update(upd)
                .eq("club_id", CLUB_ID)
                .eq("player_id", pid)
                .execute()
            ))
            ladder_audit("roster_reactivate_append", "ladder_roster", f"{CLUB_ID}:{pid}", row, upd)
            st.success(f"Reactivated '{nm}' into {tier_title(tier_for_player)} at rank {next_rank}.")
            st.rerun()
    else:
        ins = {
            "club_id": CLUB_ID,
            "player_id": pid,
            "tier_id": tier_for_player,
            "rank": int(next_rank),
            "is_active": True,
            "joined_at": now_iso,
            "left_at": None,
        }
        sb_retry(lambda: supabase.table("ladder_roster").insert(ins).execute())
        ladder_audit("roster_append", "ladder_roster", f"{CLUB_ID}:{pid}", None, ins)
        st.success(f"Added '{nm}' into {tier_title(tier_for_player)} at rank {next_rank}.")
        st.rerun()

        
            # If player already exists in ladder_roster, update/reactivate instead of inserting
        existing_row = sb_retry(lambda: (
            supabase.table("ladder_roster")
            .select("id,is_active,rank")
            .eq("club_id", CLUB_ID)
            .eq("player_id", pid)
            .limit(1)
            .execute()
        ))
        
        now_iso = dt_utc_now().isoformat()
    
        if existing_row.data:
            row = existing_row.data[0]
            if bool(row.get("is_active", True)):
                st.info(f"'{nm}' is already on the ladder roster (rank {row.get('rank')}).")
            else:
                upd = {
                    "is_active": True,
                    "rank": int(next_rank),
                    "left_at": None,
                    "joined_at": now_iso,
                }
                sb_retry(lambda: (
                    supabase.table("ladder_roster")
                    .update(upd)
                    .eq("club_id", CLUB_ID)
                    .eq("player_id", pid)
                    .execute()
                ))
                ladder_audit("roster_reactivate_append", "ladder_roster", f"{CLUB_ID}:{pid}", row, upd)
                st.success(f"Reactivated '{nm}' and appended to bottom at rank {next_rank}.")
                st.rerun()
        else:
            ins = {"club_id": CLUB_ID, "player_id": pid, "rank": int(next_rank), "is_active": True}
            sb_retry(lambda: supabase.table("ladder_roster").insert(ins).execute())
            ladder_audit("roster_append", "ladder_roster", f"{CLUB_ID}:{pid}", None, ins)
            st.success(f"Added '{nm}' to bottom at rank {next_rank}.")
            st.rerun()
        
        st.divider()

        # Initialize roster from ranked list
        st.markdown("#### Initialize / Replace Ladder (paste ranked list)")
        st.caption("Paste names top-to-bottom. This will REPLACE ladder_roster for this club.")

        raw = st.text_area("Ranked roster (top to bottom)", height=160, key="ladder_init_raw")
        if st.button("🚀 Replace Ladder Roster", type="primary"):
            names = [x.strip() for x in (raw or "").split("\n") if x.strip()]
            if not names:
                st.error("Paste at least one name.")
                st.stop()

            # Ensure players exist (add missing as active with 3.5 default)
            for nm in names:
                if nm not in name_to_id:
                    ok, err = safe_add_player(nm, 3.5)
                    if not ok:
                        st.error(f"Could not add {nm}: {err}")
                        st.stop()

            # Reload mappings
            (
                df_players_all,
                df_players,
                df_leagues,
                df_matches,
                df_meta,
                name_to_id,
                id_to_name,
            ) = load_data()

            # Replace roster
            now_iso = dt_utc_now().isoformat()

            # Soft-clear ONLY this tier (keeps history)
            sb_retry(lambda: (
                supabase.table("ladder_roster")
                .update({"is_active": False, "left_at": now_iso})
                .eq("club_id", CLUB_ID)
                .eq("tier_id", tier_ctx)
                .execute()
            ))


            rows = []
            for i, nm in enumerate(names, start=1):
                pid = int(name_to_id[nm])
                rows.append({
                    "club_id": CLUB_ID,
                    "player_id": pid,
                    "tier_id": tier_ctx,
                    "rank": i,
                    "is_active": True,
                    "joined_at": now_iso,
                    "left_at": None,
                })
            
            sb_retry(lambda: supabase.table("ladder_roster").upsert(rows, on_conflict="club_id,player_id").execute())
            ladder_audit("roster_replace_tier", "ladder_roster", f"{CLUB_ID}:{tier_ctx}", None, {"tier": tier_ctx, "count": len(rows)})
            st.success("Roster replaced.")
            st.rerun()

        st.divider()

        # Display roster
        df_roster, df_flags, df_ch, df_pass = ladder_load_core()

        if df_roster is None or df_roster.empty:
            st.info("No roster yet.")
            st.stop()
    
        df_roster = df_roster.copy()
        df_roster["player_id"] = df_roster["player_id"].astype(int)
        df_roster["name"] = df_roster["player_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
    
        # --- Management panel ---
        st.markdown("### Manage Ladder Player (activate/deactivate)")
    
        # Select by player_id (canonical)
        pid = st.selectbox(
            "Select player",
            options=df_roster.sort_values(["is_active", "rank"], ascending=[False, True])["player_id"].tolist(),
            format_func=lambda x: f"{ladder_nm(int(x), id_to_name)} (ID {int(x)})",
            key="ladder_roster_manage_pid",
        )
        pid = int(pid)
    
        row = df_roster[df_roster["player_id"] == pid].iloc[0].to_dict()
        is_active_now = bool(row.get("is_active", True))
        rank_now = row.get("rank", None)
    
        c1, c2, c3 = st.columns([2, 2, 3])
        c1.metric("Status", "Active" if is_active_now else "Inactive")
        c2.metric("Current rank", str(rank_now) if rank_now is not None else "—")
    
        notes_val = c3.text_input("Notes (optional)", value=str(row.get("notes", "") or ""), key="ladder_roster_notes")
    
        if is_active_now:
            st.warning("Deactivating removes the player from the ladder (no longer challengeable). History remains.")
            if st.button("Deactivate from Ladder", type="primary", key="btn_deactivate_ladder"):
                ok, msg = ladder_set_roster_active(CLUB_ID, pid, make_active=False, notes=notes_val)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        else:
            st.info("Reactivating puts the player back on the ladder.")
            mode = st.radio(
                "Reactivation placement",
                ["Append to bottom", "Restore previous rank"],
                horizontal=True,
                key="ladder_reactivate_mode",
            )
            mode_key = "append" if mode == "Append to bottom" else "restore"
    
            if st.button("Reactivate on Ladder", type="primary", key="btn_reactivate_ladder"):
                ok, msg = ladder_set_roster_active(CLUB_ID, pid, make_active=True, mode=mode_key, notes=notes_val)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
    
        st.divider()
    
        # --- Display tables ---
        active_df = df_roster[(df_roster["is_active"] == True) & (df_roster["tier_id"] == tier_ctx)].copy().sort_values("rank")
        inactive_df = df_roster[(df_roster["is_active"] == False) & (df_roster["tier_id"] == tier_ctx)].copy().sort_values("rank")
        show_cols = [c for c in ["rank", "name", "player_id", "left_at", "notes"] if c in inactive_df.columns]
        st.dataframe(inactive_df[show_cols], use_container_width=True, hide_index=True)


    with tabs[4]:
        st.subheader("⬆️⬇️ Tier Movement (Admin Review Queue)")
        st.caption("Triggers when a player has 10 consecutive rated games where their post-match rating is in a different tier than their current assigned tier.")
    
        # Safety: need roster + matches
        if df_roster is None or df_roster.empty:
            st.info("Roster required.")
            st.stop()
    
        # Compute status map so we can disable approvals if Locked
        status_map = ladder_compute_status_map(df_roster, df_flags, df_ch, df_pass, settings, id_to_name)
    
        active = df_roster[df_roster["is_active"] == True].copy()
        if active.empty:
            st.info("No active roster players.")
            st.stop()
    
        # joined_at parse
        active["joined_at_dt"] = pd.to_datetime(active.get("joined_at"), utc=True, errors="coerce")
        active["name"] = active["player_id"].apply(lambda x: ladder_nm(int(x), id_to_name))
    
        rows = []
        for _, rr in active.iterrows():
            pid = int(rr["player_id"])
            cur_tier = str(rr.get("tier_id") or "INT")
            joined_at = rr["joined_at_dt"].to_pydatetime() if pd.notna(rr["joined_at_dt"]) else None
    
            streak = compute_out_of_tier_streak(pid, joined_at, cur_tier, df_matches)  # uses global df_matches loaded at top
            dest = streak["dest_tier"]
            cnt = int(streak["count"] or 0)
    
            if dest and cnt >= 10 and dest != cur_tier:
                rows.append({
                    "player_id": pid,
                    "name": rr["name"],
                    "current_tier": cur_tier,
                    "dest_tier": dest,
                    "count": cnt,
                    "status": status_map.get(pid, {}).get("status", ""),
                })
    
        if not rows:
            st.success("No tier-move triggers at this time.")
            st.stop()
    
        qdf = pd.DataFrame(rows)
        qdf["Current Tier"] = qdf["current_tier"].apply(tier_title)
        qdf["Proposed Tier"] = qdf["dest_tier"].apply(tier_title)
        qdf = qdf.sort_values(["current_tier", "name"])
    
        st.dataframe(
            qdf[["name", "Current Tier", "Proposed Tier", "count", "status"]],
            use_container_width=True,
            hide_index=True,
        )
    
        st.divider()
        st.markdown("### Approve a Tier Move")
    
        pick_pid = st.selectbox(
            "Select flagged player",
            options=qdf["player_id"].tolist(),
            format_func=lambda x: f"{ladder_nm(int(x), id_to_name)} (ID {int(x)})",
            key="tier_move_pick_pid",
        )
        pick_pid = int(pick_pid)
    
        row = qdf[qdf["player_id"] == pick_pid].iloc[0].to_dict()
        cur_tier = str(row["current_tier"])
        dest_tier = str(row["dest_tier"])
        locked = (str(row.get("status","")) == "Locked")
    
        st.write(f"- Current: **{tier_title(cur_tier)}**")
        st.write(f"- Proposed: **{tier_title(dest_tier)}**")
        st.write(f"- Status: **{row.get('status','')}**")
    
        if locked:
            st.warning("Player is Locked (active challenge). Tier move should be approved only after the active challenge is finalized.")
    
        approve = st.button("✅ Approve Tier Move", disabled=locked, key="approve_tier_move_btn")
    
        if approve:
            # Determine placement rule
            promo = is_promotion(cur_tier, dest_tier)
            demo = is_demotion(cur_tier, dest_tier)
    
            placement = "bottom" if promo else "top" if demo else "bottom"
    
            # Load active rosters by tier
            all_roster = df_roster[df_roster["is_active"] == True].copy()
            old_df = all_roster[all_roster["tier_id"] == cur_tier].sort_values("rank")
            new_df = all_roster[all_roster["tier_id"] == dest_tier].sort_values("rank")
    
            old_pids = [int(x) for x in old_df["player_id"].tolist() if int(x) != pick_pid]
            new_pids = [int(x) for x in new_df["player_id"].tolist() if int(x) != pick_pid]
    
            if placement == "top":
                new_order = [pick_pid] + new_pids
            else:
                new_order = new_pids + [pick_pid]
    
            now_iso = dt_utc_now().isoformat()
    
            # 1) Update player's tier_id immediately
            sb_retry(lambda: (
                supabase.table("ladder_roster")
                .update({"tier_id": dest_tier})
                .eq("club_id", CLUB_ID)
                .eq("player_id", pick_pid)
                .execute()
            ))
    
            # 2) Resequence old tier ranks
            for i, pid in enumerate(old_pids, start=1):
                sb_retry(lambda pid=pid, i=i: (
                    supabase.table("ladder_roster")
                    .update({"rank": int(i)})
                    .eq("club_id", CLUB_ID)
                    .eq("player_id", int(pid))
                    .execute()
                ))
    
            # 3) Resequence destination tier ranks
            for i, pid in enumerate(new_order, start=1):
                sb_retry(lambda pid=pid, i=i: (
                    supabase.table("ladder_roster")
                    .update({"rank": int(i)})
                    .eq("club_id", CLUB_ID)
                    .eq("player_id", int(pid))
                    .execute()
                ))
    
            # 4) Clear tier-move flag fields
            sb_retry(lambda: supabase.table("ladder_player_flags").upsert({
                "club_id": CLUB_ID,
                "player_id": int(pick_pid),
                "tier_move_flag": False,
                "tier_move_dest_tier": None,
                "tier_move_count": 0,
                "tier_move_triggered_at": None,
                "tier_move_last_eval_at": now_iso,
            }, on_conflict="club_id,player_id").execute())
    
            ladder_audit(
                "tier_move_approved",
                "ladder_roster",
                f"{CLUB_ID}:{pick_pid}",
                {"tier_id": cur_tier},
                {"tier_id": dest_tier, "placement": placement},
            )
    
            st.success(f"Tier move approved. Placed at the {placement.upper()} of {tier_title(dest_tier)}.")
            st.rerun()

    
    # -------------------------
    # TAB 5: OVERRIDES
    # -------------------------
    with tabs[5]:
        st.subheader("🏖 Vacation / Reinstate Overrides")
    
        roster_active = ladder_roster_active_df(df_roster, id_to_name)
    
        if roster_active is None or roster_active.empty:
            st.info("Roster required. Add players in the Roster tab.")
            st.stop()
    
        # Ensure int ids
        roster_active = roster_active.copy()
        roster_active["player_id"] = roster_active["player_id"].apply(lambda x: int(float(x)) if x is not None else -1)
        roster_active = roster_active[roster_active["player_id"] > 0].copy()
    
        if roster_active.empty:
            st.info("No active players found in ladder roster.")
            st.stop()
    
        # Select by ID (canonical), display name (friendly)
        pid = st.selectbox(
            "Player",
            options=roster_active["player_id"].tolist(),
            format_func=lambda x: ladder_nm(int(x), id_to_name),
            key="ladder_override_pid",
        )
        pid = int(pid)
    
        # Load current flags (if any)
        cur = None
        if df_flags is not None and not df_flags.empty and "player_id" in df_flags.columns:
            hit = df_flags[df_flags["player_id"].astype(int) == pid]
            if not hit.empty:
                cur = hit.iloc[0].to_dict()
    
        vac_default = cur.get("vacation_until") if cur else None
        rein_default = bool(cur.get("reinstate_required", False)) if cur else False
        notes_default = str(cur.get("reinstate_notes", "") or "") if cur else ""
    
        vac = st.text_input("Vacation until (ISO, UTC) — leave blank to clear", value=str(vac_default or ""))
        rein = st.checkbox("Reinstate Required", value=rein_default)
        notes = st.text_area("Reinstate notes", value=notes_default, height=80)
    
        if st.button("💾 Save Overrides", key="save_overrides_btn"):
            before = cur
            payload = {
                "club_id": CLUB_ID,
                "player_id": pid,
                "vacation_until": (vac.strip() or None),
                "reinstate_required": bool(rein),
                "reinstate_notes": notes.strip() or None,
            }
            sb_retry(lambda: supabase.table("ladder_player_flags").upsert(payload, on_conflict="club_id,player_id").execute())
            ladder_audit("flags_save", "ladder_player_flags", f"{CLUB_ID}:{pid}", before, payload)
            st.success("Saved.")
            st.rerun()


    # -------------------------
    # TAB 6: AUDIT
    # -------------------------
    with tabs[6]:
        st.subheader("📜 Ladder Audit Log")
        resp = sb_retry(lambda: (
            supabase.table("ladder_audit_log")
            .select("*")
            .eq("club_id", CLUB_ID)
            .order("created_at", desc=True)
            .limit(500)
            .execute()
        ))
        df_a = pd.DataFrame(resp.data)
        if df_a.empty:
            st.info("No audit entries yet.")
        else:
            st.dataframe(
                df_a[["created_at","actor","action_type","entity_type","entity_id"]],
                use_container_width=True,
                hide_index=True
            )

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
            from datetime import datetime, timezone

            curr = df_players_all[df_players_all["name"] == p_edit].iloc[0]
            pid = int(curr["id"])

            with st.form("edit_form"):
                st.caption(f"Editing: {p_edit}")
                new_n = st.text_input("Name", value=str(curr["name"]))
                new_r = st.number_input("Rating", 1.0, 7.0, float(curr.get("rating", 1200.0)) / 400.0, step=0.01)
                active_flag = st.checkbox("Active (global player flag)", value=bool(curr.get("active", True)))

                if st.form_submit_button("Update Player"):
                    supabase.table("players").update(
                        {"name": new_n, "rating": float(new_r) * 400.0, "active": bool(active_flag)}
                    ).eq("id", pid).eq("club_id", CLUB_ID).execute()
                    st.success("Updated!")
                    time.sleep(1)
                    st.rerun()

            st.write("---")
            st.write("**Danger Zone**")
            if st.button("🗑️ Deactivate Player (global)", type="primary"):
                supabase.table("players").update({"active": False}).eq("id", pid).eq("club_id", CLUB_ID).execute()
                st.success(f"Player {curr['name']} has been deactivated.")
                time.sleep(1)
                st.rerun()

            st.divider()

            st.subheader("🏟️ League Ratings (Edit per League)")
            st.caption(
                "Per-league **Active** controls whether the player appears in **Standings** and **Top Performers** for that league. "
                "Other leaderboard pages can still include them if you do not filter by `is_active` there."
            )

            # --- Pull this player’s league rows (try to include per-league active fields if they exist) ---
            lr_has_active_cols = True
            try:
                lr_resp = (
                    supabase.table("league_ratings")
                    .select("id,league_name,rating,starting_rating,wins,losses,matches_played,is_active,inactive_at")
                    .eq("club_id", CLUB_ID)
                    .eq("player_id", int(pid))
                    .execute()
                )
                lr_df = pd.DataFrame(lr_resp.data)
            except Exception:
                lr_has_active_cols = False
                lr_resp = (
                    supabase.table("league_ratings")
                    .select("id,league_name,rating,starting_rating,wins,losses,matches_played")
                    .eq("club_id", CLUB_ID)
                    .eq("player_id", int(pid))
                    .execute()
                )
                lr_df = pd.DataFrame(lr_resp.data)

                st.warning(
                    "League-specific inactive controls are not enabled yet. "
                    "Add `is_active` (boolean) and `inactive_at` (timestamptz) columns to `public.league_ratings` in Supabase."
                )

            # League options (from metadata if available)
            if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
                league_opts = sorted(df_meta["league_name"].dropna().unique().tolist())
            else:
                league_opts = sorted(lr_df["league_name"].dropna().unique().tolist()) if not lr_df.empty else []

            cA, cB, cC = st.columns([2, 2, 2])

            with cA:
                add_league = st.selectbox("Add / ensure league row", [""] + league_opts, key=f"add_lg_{pid}")
            with cB:
                set_mode = st.selectbox(
                    "Quick set",
                    ["(none)", "Set league rating = overall rating", "Apply + / - adjustment"],
                    key=f"lg_set_mode_{pid}",
                )
            with cC:
                adj_val = st.number_input("Adj (JUPR)", value=0.00, step=0.01, key=f"lg_adj_{pid}")

            if st.button("➕ Ensure League Row"):
                if add_league:
                    base_elo = float(curr.get("rating", 1200.0) or 1200.0)
                    ensure_league_row(pid, add_league, base_rating_elo=base_elo)

                    # If per-league active columns exist, force this row to active (safe no-op if already active)
                    if lr_has_active_cols:
                        sb_retry(lambda lg=add_league: (
                            supabase.table("league_ratings")
                            .update({"is_active": True, "inactive_at": None})
                            .eq("club_id", CLUB_ID)
                            .eq("player_id", int(pid))
                            .eq("league_name", lg)
                            .execute()
                        ))

                    st.success("League row ready.")
                    time.sleep(0.5)
                    st.rerun()

            if lr_df.empty:
                st.info("No league ratings found for this player yet. Use 'Ensure League Row' to create one.")
            else:
                # Keep originals so we can preserve inactive_at when already inactive
                orig_is_active = {}
                orig_inactive_at = {}
                if lr_has_active_cols:
                    for _, rr in lr_df.iterrows():
                        rid0 = int(rr["id"])
                        orig_is_active[rid0] = bool(rr.get("is_active", True))
                        orig_inactive_at[rid0] = rr.get("inactive_at", None)

                # Convert to display
                lr_df = lr_df.copy()
                lr_df["JUPR"] = lr_df["rating"].astype(float) / 400.0
                lr_df["Start JUPR"] = lr_df["starting_rating"].astype(float) / 400.0

                # Optional quick actions (apply to ALL rows displayed)
                if set_mode == "Set league rating = overall rating":
                    base = float(curr.get("rating", 1200.0) or 1200.0) / 400.0
                    lr_df["JUPR"] = base
                elif set_mode == "Apply + / - adjustment":
                    lr_df["JUPR"] = lr_df["JUPR"].astype(float) + float(adj_val)

                # Columns shown in editor
                edit_cols = ["league_name"]
                if lr_has_active_cols:
                    # Ensure the columns exist in dataframe (defensive)
                    if "is_active" not in lr_df.columns:
                        lr_df["is_active"] = True
                    if "inactive_at" not in lr_df.columns:
                        lr_df["inactive_at"] = None
                    edit_cols += ["is_active", "inactive_at"]

                edit_cols += ["JUPR", "Start JUPR", "wins", "losses", "matches_played"]
                editable = lr_df[["id"] + edit_cols].copy()

                disabled_cols = ["id", "league_name"]
                if lr_has_active_cols:
                    disabled_cols.append("inactive_at")  # timestamp is managed automatically on save

                edited = st.data_editor(
                    editable,
                    hide_index=True,
                    use_container_width=True,
                    key=f"lr_editor_{pid}",
                    column_config={
                        "league_name": st.column_config.TextColumn("League", disabled=True),
                        "is_active": st.column_config.CheckboxColumn("Active"),
                        "inactive_at": st.column_config.TextColumn("Inactive At (auto)", disabled=True),
                        "JUPR": st.column_config.NumberColumn("League JUPR", min_value=1.0, max_value=7.0, step=0.01),
                        "Start JUPR": st.column_config.NumberColumn("Start JUPR", min_value=1.0, max_value=7.0, step=0.01),
                        "wins": st.column_config.NumberColumn("W", min_value=0, step=1),
                        "losses": st.column_config.NumberColumn("L", min_value=0, step=1),
                        "matches_played": st.column_config.NumberColumn("MP", min_value=0, step=1),
                    },
                    disabled=disabled_cols,
                )

                c1, c2 = st.columns([1, 3])

                if c1.button("💾 Save League Edits"):
                    now_iso = datetime.now(timezone.utc).isoformat()

                    for _, r in edited.iterrows():
                        rid = int(r["id"])

                        payload = {
                            "rating": float(r["JUPR"]) * 400.0,
                            "starting_rating": float(r["Start JUPR"]) * 400.0,
                            "wins": int(r["wins"]),
                            "losses": int(r["losses"]),
                            "matches_played": int(r["matches_played"]),
                        }

                        # Per-league active/inactive support
                        if lr_has_active_cols:
                            next_active = bool(r.get("is_active", True))
                            payload["is_active"] = next_active

                            # Manage inactive_at without triggers:
                            # - When activating: clear inactive_at
                            # - When deactivating: set inactive_at if it was previously empty; otherwise preserve existing
                            if next_active:
                                payload["inactive_at"] = None
                            else:
                                existing_ts = orig_inactive_at.get(rid, None)
                                payload["inactive_at"] = existing_ts if existing_ts else now_iso

                        sb_retry(lambda rid=rid, payload=payload: (
                            supabase.table("league_ratings")
                            .update(payload)
                            .eq("club_id", CLUB_ID)
                            .eq("id", rid)
                            .execute()
                        ))

                    st.success("Saved league ratings.")
                    time.sleep(0.5)
                    st.rerun()

                with c2:
                    st.caption(
                        "Heads up: manual edits change leaderboard/seeding going forward. "
                        "Past match snapshots won’t be rewritten unless you run a Replay."
                    )


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

        st.divider()

        # -------------------------
        # DUPLICATE SCANNER
        # -------------------------
        st.subheader("🔎 Find Duplicate Matches")
        st.caption("Detects duplicates even if teammates or teams are swapped (scores normalized too).")

        dup_scan_df = view_df.copy()

        if dup_scan_df.empty:
            st.info("No matches to scan.")
        else:
            dup_scan_df["dup_key"] = [
                canonical_dup_key(r, CLUB_ID) for _, r in dup_scan_df.iterrows()
            ]

            counts = dup_scan_df["dup_key"].value_counts()
            dup_keys = counts[counts > 1].index.tolist()

            if not dup_keys:
                st.success("✅ No duplicates found in the current view/filter.")
            else:
                st.error(f"⚠️ Found {len(dup_keys)} duplicate groups.")

                dup_only = dup_scan_df[dup_scan_df["dup_key"].isin(dup_keys)].copy()
                dup_only = dup_only.sort_values(["dup_key", "id"], ascending=[True, True])
                dup_only["dup_rank"] = dup_only.groupby("dup_key").cumcount() + 1
                dup_only["dup_count"] = dup_only.groupby("dup_key")["id"].transform("count")

                summary = (
                    dup_only.groupby("dup_key")
                    .agg(
                        dup_count=("id", "count"),
                        keep_id=("id", "min"),
                        delete_ids=("id", lambda x: ", ".join(map(str, sorted(x.tolist())[1:]))),
                        league=("league", "first"),
                        week_tag=("week_tag", "first"),
                        match_type=("match_type", "first"),
                    )
                    .reset_index(drop=True)
                    .sort_values(["league", "week_tag", "match_type"])
                )

                st.write("### Duplicate Groups (keep oldest, delete rest)")
                st.dataframe(summary, use_container_width=True, hide_index=True)

                st.write("### Duplicate Rows (detailed)")
                show_cols = [c for c in [
                    "id", "date", "league", "week_tag", "match_type",
                    "t1_p1", "t1_p2", "t2_p1", "t2_p2", "score_t1", "score_t2",
                    "dup_rank", "dup_count"
                ] if c in dup_only.columns]
                st.dataframe(dup_only[show_cols], use_container_width=True, hide_index=True)

                delete_mode = st.radio(
                    "Delete mode",
                    ["Delete duplicates (keep oldest in each group)", "I’ll delete manually"],
                    horizontal=True,
                )

                if delete_mode == "Delete duplicates (keep oldest in each group)":
                    ids_to_delete = dup_only[dup_only["dup_rank"] > 1]["id"].astype(int).tolist()
                    st.warning(
                        f"Ready to delete {len(ids_to_delete)} duplicated match rows "
                        f"(keeping the oldest copy per group)."
                    )

                    if st.button("🗑️ Delete duplicates now", type="primary"):
                        if ids_to_delete:
                            sb_retry(lambda: (
                                supabase.table("matches")
                                .delete()
                                .eq("club_id", CLUB_ID)
                                .in_("id", ids_to_delete)
                                .execute()
                            ))
                            st.success("Deleted duplicates. Now run ALL (Full System Reset) replay.")
                            time.sleep(1)
                            st.rerun()

        st.divider()

        # -------------------------
        # EXISTING BULK DELETE
        # -------------------------
        st.write("### 🗑️ Bulk Delete (first 500 rows shown)")
        edit_cols = [c for c in [
            "id", "date", "league", "match_type", "elo_delta",
            "p1", "p2", "p3", "p4", "score_t1", "score_t2"
        ] if c in view_df.columns]

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
                supabase.table("matches").delete().in_(
                    "id", to_delete["id"].astype(int).tolist()
                ).eq("club_id", CLUB_ID).execute()
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
    
            # -------------------------
            # Build K map from metadata
            # -------------------------
            k_map = {}
            if df_meta is not None and not df_meta.empty and "league_name" in df_meta.columns:
                for _, r in df_meta.iterrows():
                    try:
                        k_map[str(r["league_name"])] = int(r.get("k_factor", DEFAULT_K_FACTOR) or DEFAULT_K_FACTOR)
                    except Exception:
                        pass
    
            def k_for(lg: str) -> int:
                return int(k_map.get(str(lg), DEFAULT_K_FACTOR))
    
            # -------------------------
            # Initialize OVERALL rating map from starting_rating (fallback to rating)
            # -------------------------
            p_map = {}
            for p in all_players:
                base = p.get("starting_rating", None)
                if base is None:
                    base = p.get("rating", 1200.0)
                p_map[int(p["id"])] = {"r": float(base), "w": 0, "l": 0, "mp": 0}
    
            # League map (island ratings)
            island_map = {}          # (pid, league_name) -> {"r","w","l","mp"}
            matches_to_update = []   # snapshot rewrite payloads
    
            # Diagnostics (rating mass)
            mass_by_league = {}
            neg_mass_rows = 0
            skipped_incomplete = 0
    
            def gr(pid):
                """Get overall rating during replay."""
                if pid is None:
                    return 1200.0
                return float(p_map[int(pid)]["r"])
    
            def gir(pid, lg_name: str):
                """Get league rating during replay; lazily initialize if missing."""
                if pid is None:
                    return 1200.0
                key = (int(pid), lg_name)
                if key not in island_map:
                    # Initialize league rating at current overall at the moment of first appearance
                    island_map[key] = {"r": float(p_map[int(pid)]["r"]), "w": 0, "l": 0, "mp": 0}
                return float(island_map[key]["r"])
    
            # -------------------------
            # Replay loop
            # -------------------------
            for m in all_matches:
                league_name_raw = str(m.get("league", "") or "").strip()
    
                # Scope replay
                if target_reset != "ALL (Full System Reset)" and league_name_raw != str(target_reset).strip():
                    continue
    
                p1, p2 = m.get("t1_p1"), m.get("t1_p2")
                p3, p4 = m.get("t2_p1"), m.get("t2_p2")
                s1 = int(m.get("score_t1", 0) or 0)
                s2 = int(m.get("score_t2", 0) or 0)
    
                # -------------------------
                # Integrity gate: skip incomplete doubles rows
                # (Prevents phantom-1200 averages and 3-player deltas)
                # -------------------------
                if p1 is None or p2 is None or p3 is None or p4 is None:
                    skipped_incomplete += 1
                    continue
    
                # ---------- OVERALL snapshots (start) ----------
                sr1, sr2, sr3, sr4 = gr(p1), gr(p2), gr(p3), gr(p4)
    
                do1, do2 = calculate_hybrid_elo(
                    (sr1 + sr2) / 2,
                    (sr3 + sr4) / 2,
                    s1,
                    s2,
                    k_factor=DEFAULT_K_FACTOR,
                )
    
                # ---------- Rating mass diagnostic (overall deltas applied) ----------
                match_mass = do1 + do1 + do2 + do2
                mass_by_league[league_name_raw] = mass_by_league.get(league_name_raw, 0.0) + float(match_mass)
    
                if match_mass < -1e-6:
                    neg_mass_rows += 1
                    st.error(
                        f"NEGATIVE MASS match_id={m.get('id')} league={league_name_raw} "
                        f"pids={[p1,p2,p3,p4]} scores={s1}-{s2} do1={do1:.3f} do2={do2:.3f} mass={match_mass:.6f}"
                    )
    
                win = s1 > s2
    
                # ---------- Apply OVERALL deltas ----------
                for pid, d, won_flag in [
                    (p1, do1, win),
                    (p2, do1, win),
                    (p3, do2, not win),
                    (p4, do2, not win),
                ]:
                    pid = int(pid)
                    p_map[pid]["r"] += float(d)
                    p_map[pid]["mp"] += 1
                    if won_flag:
                        p_map[pid]["w"] += 1
                    else:
                        p_map[pid]["l"] += 1
    
                # ---------- OVERALL snapshots (end) ----------
                er1, er2, er3, er4 = gr(p1), gr(p2), gr(p3), gr(p4)
    
                # ---------- LEAGUE replay (skip PopUp) ----------
                if str(m.get("match_type", "")) != "PopUp":
                    lg = league_name_raw
    
                    ir1, ir2, ir3, ir4 = gir(p1, lg), gir(p2, lg), gir(p3, lg), gir(p4, lg)
    
                    di1, di2 = calculate_hybrid_elo(
                        (ir1 + ir2) / 2,
                        (ir3 + ir4) / 2,
                        s1,
                        s2,
                        k_factor=k_for(lg),
                    )
    
                    for pid, d, won_flag in [
                        (p1, di1, win),
                        (p2, di1, win),
                        (p3, di2, not win),
                        (p4, di2, not win),
                    ]:
                        key = (int(pid), lg)
                        island_map[key]["r"] += float(d)
                        island_map[key]["mp"] += 1
                        if won_flag:
                            island_map[key]["w"] += 1
                        else:
                            island_map[key]["l"] += 1
    
                # ---------- Store match snapshot rewrite payload ----------
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
    
            # -------------------------
            # Diagnostics summary
            # -------------------------
            st.write(f"Skipped incomplete doubles rows: {skipped_incomplete}")
            st.write(f"Negative-mass matches found: {neg_mass_rows}")
    
            if mass_by_league:
                summary = pd.DataFrame(
                    [{"league": k, "mass_elo": v, "mass_jupr": v / 400.0} for k, v in mass_by_league.items()]
                ).sort_values("mass_elo")
                st.dataframe(summary, use_container_width=True, hide_index=True)
    
            # -------------------------
            # Update players (only on full reset)
            # -------------------------
            if target_reset == "ALL (Full System Reset)":
                for pid, s in p_map.items():
                    supabase.table("players").update(
                        {"rating": s["r"], "wins": s["w"], "losses": s["l"], "matches_played": s["mp"]}
                    ).eq("club_id", CLUB_ID).eq("id", int(pid)).execute()
    
            # -------------------------
            # Rebuild league_ratings
            # -------------------------
            if target_reset != "ALL (Full System Reset)":
                supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).eq("league_name", str(target_reset)).execute()
            else:
                supabase.table("league_ratings").delete().eq("club_id", CLUB_ID).execute()
    
            new_is = []
            for (pid, lg), s in island_map.items():
                if target_reset == "ALL (Full System Reset)" or str(lg).strip() == str(target_reset).strip():
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
    
            # -------------------------
            # Update match snapshots (chunked)
            # -------------------------
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
                .select("id,player_id,league_name,rating,starting_rating")
                .eq("club_id", CLUB_ID)
                .execute()
                .data
            )

        # Pull league matches with snapshot columns (overall snapshots)
            matches = (
                supabase.table("matches")
                .select(
                    "id,date,league,match_type,"
                    "t1_p1,t1_p2,t2_p1,t2_p2,"
                    "t1_p1_r,t1_p2_r,t2_p1_r,t2_p2_r"
                )
                .eq("club_id", CLUB_ID)
                .execute()
                .data
            )

            df_m = pd.DataFrame(matches)
            if df_m.empty:
                st.error("No matches found. Cannot backfill.")
                status.update(label="Migration failed", state="error")
                st.stop()

            # Ensure sortable
            df_m["date"] = pd.to_datetime(df_m["date"], errors="coerce")
            df_m = df_m.dropna(subset=["date"])
            df_m["league"] = df_m["league"].astype(str).str.strip()
            df_m["match_type"] = df_m["match_type"].astype(str)

            def start_snap_for_player(row, pid: int):
                """Return the start snapshot rating for pid from a match row, else None."""
                try:
                    pid = int(pid)
                except Exception:
                    return None
    
                if int(row.get("t1_p1") or -1) == pid:
                    return row.get("t1_p1_r")
                if int(row.get("t1_p2") or -1) == pid:
                    return row.get("t1_p2_r")
                if int(row.get("t2_p1") or -1) == pid:
                    return row.get("t2_p1_r")
                if int(row.get("t2_p2") or -1) == pid:
                    return row.get("t2_p2_r")
                return None

            updates = []
            for lr in l_ratings:
                lr_id = int(lr["id"])
                pid = int(lr["player_id"])
                lg = str(lr["league_name"]).strip()

                 # Find earliest non-PopUp match in that league where this player appears
                rel = df_m[(df_m["league"] == lg) & (df_m["match_type"] != "PopUp")].copy()
                if rel.empty:
                    continue

                # filter to matches containing pid
                rel = rel[
                    (rel["t1_p1"] == pid) | (rel["t1_p2"] == pid) | (rel["t2_p1"] == pid) | (rel["t2_p2"] == pid)
                ].copy()

                if rel.empty:
                    continue

                rel = rel.sort_values(["date", "id"], ascending=[True, True])
                first = rel.iloc[0].to_dict()

                start_r = start_snap_for_player(first, pid)

                # If snapshot missing, fall back to current rating as last resort
                if start_r is None:
                    start_r = float(lr.get("rating", 1200.0) or 1200.0)

                updates.append({"id": lr_id, "starting_rating": float(start_r)})

            st.write(f"Calculated starting ratings for {len(updates)} records. Saving...")

            for i in range(0, len(updates), 100):
                chunk = updates[i : i + 100]
                for item in chunk:
                    supabase.table("league_ratings").update(
                        {"starting_rating": item["starting_rating"]}
                    ).eq("club_id", CLUB_ID).eq("id", int(item["id"])).execute()

            status.update(label="Migration Complete!", state="complete")
        st.success("✅ Database updated. Leaderboards can now compute Gain accurately.")


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

