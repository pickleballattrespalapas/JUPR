# jupr/streamlit_app.py
from __future__ import annotations

import traceback
from collections.abc import Mapping

import streamlit as st
import pandas as pd  # kept because your pages may rely on it

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
    league_manager,
    match_log,
    player_editor,
    admin_tools,
    admin_guide,
)

from jupr_app.ui.context import AppContext
from jupr_app.ui.url import qp_get


# -------------------------
# CONFIG
# -------------------------
CLUB_ID = "tres_palapas"


# -------------------------
# Secrets helpers (FIXED)
# -------------------------
def get_secret(path: list[str], default=None):
    """
    Safe nested secret getter that works with Streamlit's secrets object.
    Streamlit secrets behaves like a Mapping, not necessarily a dict.
    """
    cur = st.secrets
    try:
        for k in path:
            if not isinstance(cur, Mapping):
                return default
            cur = cur[k]
        return cur
    except Exception:
        return default


def require_secret(path: list[str]) -> str:
    v = get_secret(path, None)
    if v is None or (isinstance(v, str) and v.strip() == ""):
        raise KeyError("Missing secret: " + ".".join(path))
    return v


# -------------------------
# Supabase + data loading
# -------------------------
@st.cache_resource
def get_supabase():
    """
    Requires Streamlit secrets:

    [supabase]
    url = "https://....supabase.co"
    anon_key = "..."   # OR key = "..." (either is accepted)
    admin_password = "..."  # your chosen admin login password
    """
    url = get_secret(["supabase", "url"])
    key = get_secret(["supabase", "anon_key"]) or get_secret(["supabase", "key"])

    if not url or not key:
        st.error("Supabase secrets are missing or misnamed.")
        st.code(
            "[supabase]\n"
            'url = "https://YOUR_PROJECT_REF.supabase.co"\n'
            'anon_key = "YOUR_SUPABASE_ANON_KEY"  # or use key = "…"\n'
            'admin_password = "YOUR_ADMIN_PASSWORD"\n'
        )
        # Debug keys only (no values)
        try:
            st.write("Secrets keys:", list(st.secrets.keys()))
            st.write("Supabase keys:", list(st.secrets.get("supabase", {}).keys()))
        except Exception:
            pass
        st.stop()

    return make_supabase(url, key)


@st.cache_data(ttl=30)
def get_data(club_id: str):
    supabase = get_supabase()
    return load_data(supabase, club_id, match_limit=5000)


# -------------------------
# UI helpers
# -------------------------
def hide_sidebar_and_header_for_public():
    st.markdown(
        "<style>"
        "[data-testid='stSidebar']{display:none;}"
        "header{visibility:hidden;}"
        "</style>",
        unsafe_allow_html=True,
    )


def main():
    """
    Main Streamlit entrypoint. Keep this deterministic for reloads.
    """
    try:
        st.set_page_config(page_title="JUPR Leagues", layout="wide", page_icon="🌵")

        # ---- Public mode ----
        PUBLIC_MODE = qp_get("public", "0").lower() in ("1", "true", "yes", "y")

        # ---- Session defaults ----
        st.session_state.setdefault("admin_logged_in", False)
        st.session_state.setdefault("deep_link_applied", False)

        # ---- Sidebar / Auth ----
        if PUBLIC_MODE:
            hide_sidebar_and_header_for_public()
        else:
            st.sidebar.title("JUPR Leagues 🌵")

            if not st.session_state.admin_logged_in:
                with st.sidebar.expander("🔒 Admin Login"):
                    pwd = st.text_input("Password", type="password", key="admin_pwd")

                    if st.button("Login", key="admin_login_btn"):
                        expected = get_secret(["supabase", "admin_password"], "")
                        if not expected:
                            st.error("Admin password is not configured in secrets (supabase.admin_password).")
                        elif pwd == expected:
                            st.session_state.admin_logged_in = True
                            st.rerun()
                        else:
                            st.error("Incorrect password.")
            else:
                st.sidebar.success("Logged In: Admin")
                if st.sidebar.button("Log Out", key="admin_logout_btn"):
                    st.session_state.admin_logged_in = False
                    st.rerun()

        # Canonical admin flag (never true in public mode)
        admin_logged_in = (not PUBLIC_MODE) and bool(st.session_state.admin_logged_in)

        # Optional: allow pages to request a refresh of cached data
        if st.session_state.get("force_data_refresh", False):
            try:
                get_data.clear()
            except Exception:
                pass
            st.session_state["force_data_refresh"] = False

        # ---- Load data + ctx ----
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

        # ---- Router ----
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

        PAGE_KEY_TO_LABEL = {
            "leaderboards": "🏆 Leaderboards",
            "match_explorer": "🎯 Match Explorer",
            "players": "🔍 Player Search",
            "challenge_ladder": "🪜 Challenge Ladder",
            "faqs": "❓ FAQs",

            # Admin-only deep links
            "league_manager": "🏟️ League Manager",
            "match_uploader": "📝 Match Uploader",
            "match_log": "📝 Match Log",
            "player_editor": "👥 Player Editor",
            "admin_tools": "⚙️ Admin Tools",
            "admin_guide": "📘 Admin Guide",
            "challenge_ladder_admin": "🛠️ Challenge Ladder Admin",
        }
        LABEL_TO_PAGE_KEY = {v: k for k, v in PAGE_KEY_TO_LABEL.items()}

        ADMIN_ONLY_LABELS = {
            "🏟️ League Manager",
            "📝 Match Uploader",
            "📝 Match Log",
            "👥 Player Editor",
            "⚙️ Admin Tools",
            "📘 Admin Guide",
            "🛠️ Challenge Ladder Admin",
        }

        labels = list(PAGES.keys())
        if not admin_logged_in:
            labels = [x for x in labels if x not in ADMIN_ONLY_LABELS]

        # ---- Deep link (apply once, only if visible) ----
        deep_page_key = qp_get("page", "").strip().lower()
        deep_label = PAGE_KEY_TO_LABEL.get(deep_page_key, "")

        if (not st.session_state.deep_link_applied) and (deep_label in labels):
            st.session_state["main_nav"] = deep_label
            st.session_state.deep_link_applied = True

        # Ensure valid selection
        if "main_nav" not in st.session_state or st.session_state["main_nav"] not in labels:
            st.session_state["main_nav"] = labels[0]

        # Sidebar selection
        if not PUBLIC_MODE:
            if admin_logged_in and st.sidebar.button("🔄 Refresh data"):
                get_data.clear()
                st.rerun()

            sel = st.sidebar.radio("Go to:", labels, key="main_nav")
        else:
            sel = st.session_state["main_nav"]

        # Final guard
        if sel not in labels:
            sel = labels[0]
            st.session_state["main_nav"] = sel

        # Keep URL synced
        try:
            st.query_params["page"] = LABEL_TO_PAGE_KEY.get(sel, "leaderboards")
            if PUBLIC_MODE:
                st.query_params["public"] = "1"
            else:
                if "public" in st.query_params:
                    st.query_params.pop("public", None)
        except Exception:
            pass

        # Render page
        page_mod = PAGES[sel]
        render_fn = getattr(page_mod, "render", None)
        if not callable(render_fn):
            st.error(f"Page module for '{sel}' has no render(ctx) function.")
            st.stop()

        render_fn(ctx)

    except Exception:
        st.error("streamlit_app.main() crashed")
        st.code(traceback.format_exc())
        st.stop()


if __name__ == "__main__":
    main()



