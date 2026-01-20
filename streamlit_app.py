# jupr/streamlit_app.py
from __future__ import annotations

import traceback
from collections.abc import Mapping

import streamlit as st
import pandas as pd  # kept because pages may rely on it

from jupr_app.data.client import make_supabase
from jupr_app.data.load import load_data
from jupr_app.ui.context import AppContext
from jupr_app.ui.url import qp_get


# -------------------------
# CONFIG
# -------------------------
CLUB_ID = "tres_palapas"

# Public base URL used for share links + link buttons (Streamlit Cloud)
PUBLIC_BASE_URL = "https://8lkemld946rmtwwptk2gcs.streamlit.app"


# -------------------------
# Secrets helpers (SAFE)
# -------------------------
def get_secret(path: list[str], default=None):
    """
    Safe nested secret getter that works with Streamlit's secrets object.
    Never raises KeyError.
    """
    try:
        cur: object = st.secrets
    except Exception:
        return default

    for k in path:
        if not isinstance(cur, Mapping):
            return default
        if k not in cur:
            return default
        cur = cur[k]

    return cur


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
    url = get_secret(["supabase", "url"], "")
    key = get_secret(["supabase", "anon_key"], "") or get_secret(["supabase", "key"], "")

    if not url or not key:
        st.error("Supabase secrets are missing or misnamed.")
        st.code(
            "[supabase]\n"
            'url = "https://YOUR_PROJECT_REF.supabase.co"\n'
            'anon_key = "YOUR_SUPABASE_ANON_KEY"  # or use key = "…"\n'
            'admin_password = "YOUR_ADMIN_PASSWORD"\n'
        )

        # Debug keys only (no values) - Mapping-safe (no .get usage)
        try:
            st.write("Secrets keys:", list(st.secrets.keys()))
            sb = get_secret(["supabase"], default={})
            if isinstance(sb, Mapping):
                st.write("Supabase keys:", list(sb.keys()))
        except Exception:
            pass

        st.stop()

    return make_supabase(str(url), str(key))


@st.cache_data(ttl=30)
def get_data(club_id: str):
    supabase = get_supabase()
    return load_data(supabase, club_id, match_limit=5000)


# -------------------------
# UI helpers
# -------------------------
def hide_sidebar_and_header_for_public():
    # Hide sidebar + collapse control for public, keep app content full-width.
    st.markdown(
        "<style>"
        "section[data-testid='stSidebar']{display:none;}"
        "div[data-testid='collapsedControl']{display:none;}"
        "header{visibility:hidden;}"
        "</style>",
        unsafe_allow_html=True,
    )


def render_public_top_nav(*, labels_in_order: list[str], current_label: str) -> str:
    """
    Public mode top navigation (horizontal radio).
    Returns the selected label.
    """
    st.markdown("**Go to:**")

    try:
        idx = labels_in_order.index(current_label)
    except ValueError:
        idx = 0

    sel = st.radio(
        label="public_top_nav",
        options=labels_in_order,
        index=idx,
        horizontal=True,
        key="public_top_nav_radio",
        label_visibility="collapsed",
    )
    return sel


def main():
    """
    Main Streamlit entrypoint. Keep this deterministic for reloads.
    """
    try:
        st.set_page_config(
            page_title="JUPR Leagues",
            layout="wide",
            page_icon="🌵",
            initial_sidebar_state="collapsed",
        )

        # ---- Public mode ----
        PUBLIC_MODE = qp_get("public", "0").lower() in ("1", "true", "yes", "y")

        # Make base_url available to all pages (leaderboards uses this for share links)
        # Use session_state because ctx is a frozen-ish dataclass and you don't want to refactor it mid-stream.
        st.session_state["base_url"] = PUBLIC_BASE_URL

        # ---- Session defaults ----
        st.session_state.setdefault("admin_logged_in", False)
        st.session_state.setdefault("deep_link_applied", False)

        # ---- Sidebar / Auth ----
        if PUBLIC_MODE:
            hide_sidebar_and_header_for_public()
        else:
            st.sidebar.title("JUPR Leagues 🌵")

            if not bool(st.session_state.get("admin_logged_in", False)):
                with st.sidebar.expander("🔒 Admin Login"):
                    pwd = st.text_input("Password", type="password", key="admin_pwd")

                    if st.button("Login", key="admin_login_btn"):
                        expected = str(get_secret(["supabase", "admin_password"], "") or "")
                        if not expected:
                            st.error("Admin password is not configured in secrets (supabase.admin_password).")
                        elif pwd == expected:
                            st.session_state["admin_logged_in"] = True
                            st.rerun()
                        else:
                            st.error("Incorrect password.")
            else:
                st.sidebar.success("Logged In: Admin")
                if st.sidebar.button("Log Out", key="admin_logout_btn"):
                    st.session_state["admin_logged_in"] = False
                    st.rerun()

        # Canonical admin flag (never true in public mode)
        admin_logged_in = (not PUBLIC_MODE) and bool(st.session_state.get("admin_logged_in", False))

        # Optional: allow pages to request a refresh of cached data
        if bool(st.session_state.get("force_data_refresh", False)):
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

        # -------------------------
        # LAZY IMPORT PAGES (prevents import-time KeyError crashes)
        # -------------------------
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

        # ---- Router ----
        PAGES = {
            "🏆 Leaderboards": leaderboards,
            "🎯 Match Explorer": match_explorer,
            "🔍 Player Search": players,
            "🪜 Challenge Ladder": challenge_ladder,
            "❓ FAQs": faqs,

            # Admin-only
            "🏟️ League Manager": league_manager,
            "📝 Match Uploader": match_uploader,
            "📝 Match Log": match_log,
            "👥 Player Editor": player_editor,
            "⚙️ Admin Tools": admin_tools,
            "📘 Admin Guide": admin_guide,
            "🛠️ Challenge Ladder Admin": challenge_ladder_admin,
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

        # Visible labels based on auth
        all_labels = list(PAGES.keys())
        if not admin_logged_in:
            visible_labels = [x for x in all_labels if x not in ADMIN_ONLY_LABELS]
        else:
            visible_labels = all_labels

        # Public nav order (old UX)
        PUBLIC_NAV_KEYS = ["leaderboards", "match_explorer", "players", "challenge_ladder", "faqs"]
        public_labels_in_order = [PAGE_KEY_TO_LABEL[k] for k in PUBLIC_NAV_KEYS if PAGE_KEY_TO_LABEL.get(k)]

        # -------------------------
        # Deep link resolution
        # -------------------------
        deep_page_key = qp_get("page", "").strip().lower()
        deep_label = PAGE_KEY_TO_LABEL.get(deep_page_key, "")

        if PUBLIC_MODE:
            # Block admin-only deep links in public mode
            if deep_label in ADMIN_ONLY_LABELS:
                deep_label = ""

            current_label = deep_label if deep_label in public_labels_in_order else public_labels_in_order[0]
            sel = render_public_top_nav(
                labels_in_order=public_labels_in_order,
                current_label=current_label,
            )

        else:
            # Apply deep link once (only if that page is visible)
            if (not bool(st.session_state.get("deep_link_applied", False))) and (deep_label in visible_labels):
                st.session_state["main_nav"] = deep_label
                st.session_state["deep_link_applied"] = True

            # Ensure valid selection
            if "main_nav" not in st.session_state or st.session_state["main_nav"] not in visible_labels:
                st.session_state["main_nav"] = visible_labels[0]

            if admin_logged_in and st.sidebar.button("🔄 Refresh data"):
                get_data.clear()
                st.rerun()

            sel = st.sidebar.radio("Go to:", visible_labels, key="main_nav")

            if sel not in visible_labels:
                sel = visible_labels[0]
                st.session_state["main_nav"] = sel

        # -------------------------
        # Keep URL synced (canonical deep links)
        # -------------------------
        try:
            st.query_params["page"] = LABEL_TO_PAGE_KEY.get(sel, "leaderboards")
            if PUBLIC_MODE:
                st.query_params["public"] = "1"
            else:
                if "public" in st.query_params:
                    st.query_params.pop("public", None)
        except Exception:
            pass

        # -------------------------
        # Render page
        # -------------------------
        page_mod = PAGES.get(sel)
        if page_mod is None:
            st.error(f"Unknown page selection: {sel}")
            st.stop()

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


