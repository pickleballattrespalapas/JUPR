# streamlit_app.py
import streamlit as st
import pandas as pd

from jupr_app.data.client import make_supabase
from jupr_app.data.load import load_data
from jupr_app.ui.pages import (
    leaderboards,
    match_explorer,
    faqs,
    players,
    challenge_ladder,
    challenge_ladder_admin,
    match_uploader,
    # restored pages
    league_manager,
    match_log,
    player_editor,
    admin_tools,
    admin_guide,
)


from jupr_app.domain.match_processing import process_matches

from jupr_app.ui.context import AppContext
from jupr_app.ui.url import qp_get

# ---- CONFIG ----
CLUB_ID = "tres_palapas"

# ---- Page config ----
st.set_page_config(page_title="JUPR Leagues", layout="wide", page_icon="🌵")


# ---- Supabase client ----
@st.cache_resource
def get_supabase():
    return make_supabase(
        st.secrets["supabase"]["url"],
        st.secrets["supabase"]["key"],
    )


# ---- Load data ----
@st.cache_data(ttl=30)
def get_data(club_id: str):
    supabase = get_supabase()
    return load_data(supabase, club_id, match_limit=5000)


# ---- Auth / public mode ----
PUBLIC_MODE = qp_get("public", "0").lower() in ("1", "true", "yes", "y")

# Ensure session key exists BEFORE reading it
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

# Sidebar login (session-based)
if PUBLIC_MODE:
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none;} header{visibility:hidden;}</style>",
        unsafe_allow_html=True
    )
else:
    st.sidebar.title("JUPR Leagues 🌵")

    if not st.session_state.admin_logged_in:
        with st.sidebar.expander("🔒 Admin Login"):
            pwd = st.text_input("Password", type="password", key="admin_pwd")
            if st.button("Login", key="admin_login_btn"):
                if pwd == st.secrets["supabase"]["admin_password"]:
                    st.session_state.admin_logged_in = True
                    st.rerun()

                else:
                    st.error("Incorrect password.")
    else:
        st.sidebar.success("Logged In: Admin")
        if st.sidebar.button("Log Out", key="admin_logout_btn"):
            st.session_state.admin_logged_in = False
            st.rerun()

# Canonical admin flag for this run (never true in public mode)
admin_logged_in = (not PUBLIC_MODE) and bool(st.session_state.admin_logged_in)

# Allow pages to force-refresh cached data (e.g., when creating new players mid-flow)
if st.session_state.get("force_data_refresh", False):
    try:
        get_data.clear()
    except Exception:
        pass
    st.session_state["force_data_refresh"] = False

# ---- Build ctx ----
supabase = get_supabase()
(
    df_players_all,
    df_players_active,
    df_leagues,
    df_matches,
    df_meta,
    name_to_id,
    id_to_name,
) = get_data(CLUB_ID)

ctx = AppContext(
    supabase=supabase,
    club_id=CLUB_ID,
    df_players_all=df_players_all,
    df_players_active=df_players_active,
    df_leagues=df_leagues,
    df_matches=df_matches,
    df_meta=df_meta,
    name_to_id=name_to_id,
    id_to_name=id_to_name,
    public_mode=PUBLIC_MODE,
    admin_logged_in=admin_logged_in,
)



# ---- NAV (router) ----
PAGES = {
    "🏆 Leaderboards": leaderboards,
    "🎯 Match Explorer": match_explorer,
    "🔍 Player Search": players,
    "🪜 Challenge Ladder": challenge_ladder,

    # Admin-only
    "🏟️ League Manager": league_manager,
    "📝 Match Uploader": match_uploader,
    "📝 Match Log": match_log,
    "👥 Player Editor": player_editor,
    "⚙️ Admin Tools": admin_tools,
    "📘 Admin Guide": admin_guide,
    "🛠️ Challenge Ladder Admin": challenge_ladder_admin,

    "❓ FAQs": faqs,
}



# Query param page key -> UI label
PAGE_KEY_TO_LABEL = {
    "leaderboards": "🏆 Leaderboards",
    "match_explorer": "🎯 Match Explorer",
    "players": "🔍 Player Search",
    "challenge_ladder": "🪜 Challenge Ladder",
    "faqs": "❓ FAQs",

    # Admin-only deep links (will only apply if visible)
    "league_manager": "🏟️ League Manager",
    "match_uploader": "📝 Match Uploader",
    "match_log": "📝 Match Log",
    "player_editor": "👥 Player Editor",
    "admin_tools": "⚙️ Admin Tools",
    "admin_guide": "📘 Admin Guide",
    "challenge_ladder_admin": "🛠️ Challenge Ladder Admin",
}



# UI label -> query param page key
LABEL_TO_PAGE_KEY = {v: k for k, v in PAGE_KEY_TO_LABEL.items()}



labels = list(PAGES.keys())
ADMIN_ONLY_LABELS = {
    "🏟️ League Manager",
    "📝 Match Uploader",
    "📝 Match Log",
    "👥 Player Editor",
    "⚙️ Admin Tools",
    "📘 Admin Guide",
    "🛠️ Challenge Ladder Admin",
}

if not admin_logged_in:
    labels = [x for x in labels if x not in ADMIN_ONLY_LABELS]


# Deep link (applied once per session)
if "deep_link_applied" not in st.session_state:
    st.session_state.deep_link_applied = False

deep_page_key = qp_get("page", "").strip().lower()
deep_label = PAGE_KEY_TO_LABEL.get(deep_page_key, "")

# Apply deep link once, but only if it is actually visible/allowed
if (not st.session_state.deep_link_applied) and (deep_label in labels):
    st.session_state["main_nav"] = deep_label
    st.session_state.deep_link_applied = True

# Ensure main_nav always has a valid default
if "main_nav" not in st.session_state or st.session_state["main_nav"] not in labels:
    st.session_state["main_nav"] = labels[0]

# Sidebar / selection
if not PUBLIC_MODE:
    if admin_logged_in:
        if st.sidebar.button("🔄 Refresh data"):
            get_data.clear()
            st.rerun()

    sel = st.sidebar.radio("Go to:", labels, key="main_nav")
else:
    sel = st.session_state["main_nav"]

# Final guard (covers logout edge cases mid-run)
if sel not in labels:
    sel = labels[0]
    st.session_state["main_nav"] = sel

# Keep URL in sync with nav selection
try:
    st.query_params["page"] = LABEL_TO_PAGE_KEY.get(sel, "leaderboards")
    if PUBLIC_MODE:
        st.query_params["public"] = "1"
except Exception:
    pass

# Render page
PAGES[sel].render(ctx)

