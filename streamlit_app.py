# jupr/streamlit_app.py
from __future__ import annotations

import os
import time
import hmac
import hashlib
import traceback
import re

import streamlit as st
import pandas as pd  # noqa: F401  # kept because pages may rely on it

# -------------------------
# CONFIG
# -------------------------
CLUB_ID = "tres_palapas"
ADMIN_SESSION_TTL_SECONDS = 60 * 60

# Local/dev fallback for share links + link buttons.
LOCAL_PUBLIC_BASE_URL_DEFAULT = "http://localhost:8501"


def _get_admin_password() -> str:
    import os

    return os.environ.get("SUPABASE_ADMIN_PASSWORD", "")


def _get_session_secret() -> str:
    import os

    return os.environ.get("SUPABASE_ADMIN_SESSION_SECRET", "")


def _sign(exp: int, secret: str) -> str:
    msg = str(exp).encode()
    return hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


def _create_admin_session():
    secret = _get_session_secret()
    if not secret:
        st.error("SUPABASE_ADMIN_SESSION_SECRET missing.")
        st.stop()

    exp = int(time.time()) + ADMIN_SESSION_TTL_SECONDS
    token = _sign(exp, secret)

    st.session_state["admin_session"] = {
        "exp": exp,
        "token": token,
    }


def _clear_admin_session():
    st.session_state.pop("admin_session", None)


def _validate_admin_session() -> bool:
    secret = _get_session_secret()
    if not secret:
        return False

    data = st.session_state.get("admin_session")
    if not isinstance(data, dict):
        return False

    exp = data.get("exp")
    token = data.get("token")

    if not exp or not token:
        return False

    if int(exp) < int(time.time()):
        _clear_admin_session()
        return False

    expected = _sign(int(exp), secret)

    if not hmac.compare_digest(token, expected):
        _clear_admin_session()
        return False

    return True


# -------------------------
# Supabase + data loading
# -------------------------
@st.cache_resource
def get_supabase():
    """
    Supabase bootstrap using environment variables (Fly-compatible).

    Required ENV variables:
        SUPABASE_URL
        SUPABASE_ANON_KEY
    """
    import os

    from jupr_app.data.client import make_supabase

    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_ANON_KEY", "")

    if not url or not key:
        st.error("Supabase environment variables are missing.")
        st.code(
            "Required environment variables:\n"
            "SUPABASE_URL\n"
            "SUPABASE_ANON_KEY\n"
        )
        st.stop()

    if not url.startswith("https://"):
        st.error("SUPABASE_URL appears invalid.")
        st.stop()

    return make_supabase(url, key)


@st.cache_data(ttl=30)
def get_data(club_id: str):
    from jupr_app.data.load import load_data

    supabase = get_supabase()
    return load_data(supabase, club_id, match_limit=5000)


# -------------------------
# UI helpers
# -------------------------
def hide_sidebar_and_header_for_public():
    # Intentionally a no-op: public mode relies on Streamlit config
    # (initial_sidebar_state="collapsed") and avoids raw HTML/CSS injection.
    return None


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


def render_admin_sidebar_nav(*, current_label: str, admin_logged_in: bool) -> str:
    taxonomy = [
        ("Command Center", ["🧭 Command Center"]),
        ("Competitions", ["🪜 Challenge Ladder", "🏆 Tournaments", "🏆 Tournament Manager", "🏆 Division Manager", "💰 Moneyball"]),
        ("Results", ["🧾 Record Match", "📊 League Results", "🎯 Match Explorer", "📝 Match Log", "🗞️ Weekly Recap", "🗞️ Weekly Recap Admin"]),
        ("Players", ["🔍 Player Search", "👥 Player Editor", "🖨️ League Night Printout"]),
        ("Recognition", ["🏆 Leaderboards", "📼 Badge Codex", "🧪 Badge Debug"]),
        ("Administration", ["🏟️ League Manager", "🛠️ Challenge Ladder Admin", "⚙️ Admin Tools", "📘 Admin Guide", "🎨 Theme QA", "❓ FAQs"]),
    ]

    selected = current_label
    for section_name, labels in taxonomy:
        available = []
        for label in labels:
            if label == "🗞️ Weekly Recap Admin" and not admin_logged_in:
                continue
            if label in st.session_state.get("_visible_labels", []):
                available.append(label)
        if not available:
            continue

        st.sidebar.markdown(f"### {section_name}")
        for label in available:
            prefix = "👉 " if label == selected else ""
            if st.sidebar.button(f"{prefix}{label}", key=f"nav_btn_{label}", use_container_width=True):
                selected = label
                st.session_state["main_nav"] = label
        st.sidebar.markdown(" ")

    return selected


def main():
    """
    Main Streamlit entrypoint. Keep this deterministic for reloads.
    """
    try:
        from jupr_app.ui.context import AppContext
        from jupr_app.ui.theme_clean import apply_clean_theme
        from jupr_app.ui.url import qp_get

        st.set_page_config(
            page_title="JUPR Leagues",
            layout="wide",
            page_icon="🌵",
            initial_sidebar_state="collapsed",
        )
        apply_clean_theme(accent_hex="#2F6FED")  # pick your accent once (can later be club-specific)

        # ---- Public mode ----
        PUBLIC_MODE = qp_get("public", "0").lower() in ("1", "true", "yes", "y")

        # Make base_url available to all pages (leaderboards uses this for share links)
        # Use session_state because ctx is a frozen-ish dataclass and you don't want to refactor it mid-stream.
        # Fly env vars required in production/staging: PUBLIC_BASE_URL, SUPABASE_URL,
        # SUPABASE_ANON_KEY, SUPABASE_ADMIN_PASSWORD,
        # and SUPABASE_ADMIN_SESSION_SECRET.
        base_url = os.getenv("PUBLIC_BASE_URL", LOCAL_PUBLIC_BASE_URL_DEFAULT)
        st.session_state["base_url"] = str(base_url)

        # ---- Session defaults ----
        st.session_state.setdefault("deep_link_applied", False)
        # ---- Sidebar / Auth ----
        if PUBLIC_MODE:
            hide_sidebar_and_header_for_public()
        else:
            st.sidebar.title("JUPR Leagues 🌵")

            if not _validate_admin_session():
                with st.sidebar.expander("🔒 Admin Login"):
                    pwd = st.text_input("Password", type="password")

                    if st.button("Login"):
                        expected = _get_admin_password()
                        if not expected:
                            st.error("Admin password not configured.")
                        elif pwd != expected:
                            st.error("Incorrect password.")
                        else:
                            _create_admin_session()
                            st.success("Logged in.")
                            st.rerun()
            else:
                st.sidebar.success("Logged In: Admin")
                if st.sidebar.button("Log Out"):
                    _clear_admin_session()
                    st.rerun()

        # Canonical admin flag (never true in public mode)
        admin_logged_in = (not PUBLIC_MODE) and _validate_admin_session()

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
            df_badges,
            df_player_badges,
            name_to_id,
            id_to_name,
            schema_degraded,
            schema_degraded_reason,
        ) = get_data(CLUB_ID)
        test_clubs = supabase.table("clubs").select("*").execute()
        st.write("CLUBS RAW:", test_clubs.data)

        # 🔎 DEBUG BLOCK — add this
        st.write("PLAYERS ALL COUNT:", len(df_players_all) if df_players_all is not None else "None")
        st.write("MATCHES COUNT:", len(df_matches) if df_matches is not None else "None")
        st.write("LEAGUES COUNT:", len(df_leagues) if df_leagues is not None else "None")
        
        ctx = AppContext(
            supabase=supabase,
            club_id=CLUB_ID,
            df_players_all=df_players_all,
            df_players_active=df_players_active,
            df_leagues=df_leagues,
            df_matches=df_matches,
            df_meta=df_meta,
            df_badges=df_badges,
            df_player_badges=df_player_badges,
            name_to_id=name_to_id,
            id_to_name=id_to_name,
            public_mode=PUBLIC_MODE,
            admin_logged_in=admin_logged_in,
            schema_degraded=schema_degraded,
            schema_degraded_reason=schema_degraded_reason,
        )

        if admin_logged_in and schema_degraded:
            st.warning(
                "Badge schema is behind the app code. Apply migrations/20260625_badge_recompute_runs.sql and "
                "migrations/20260630_player_badges_revocation.sql to restore awarded_by/rule_version/"
                "eval_run_id/revoked_* support. "
                f"Details: {schema_degraded_reason}"
            )

        if df_player_badges is not None and df_player_badges.empty:
            from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval

            player_ids = []
            if df_players_all is not None and not df_players_all.empty and "id" in df_players_all.columns:
                player_ids = df_players_all["id"].dropna().astype(int).tolist()
            enqueue_badge_eval(
                supabase,
                club_id=CLUB_ID,
                event_type="match_recorded",
                player_ids=player_ids,
                match_id=f"initial_load:{CLUB_ID}",
                payload={"initial_load": True},
            )

        # -------------------------
        # LAZY IMPORT PAGES (prevents import-time KeyError crashes)
        # -------------------------
        from jupr_app.ui.pages import (
            leaderboards,
            league_results,
            league_printout,
            match_explorer,
            faqs,
            players,
            badge_codex,
            badge_debug,
            challenge_ladder,
            challenge_ladder_admin,
            league_manager,
            match_log,
            player_editor,
            admin_tools,
            admin_guide,
            moneyball,
            theme_gallery,
            tournaments,
            tournament_manager,
            division_manager,
            tournament_public,
            weekly_recap,
            weekly_recap_admin,
            admin_dashboard,
            record_match,
        )

        # ---- Router ----
        PAGES = {
            "🧭 Command Center": admin_dashboard,
            "🏆 Leaderboards": leaderboards,
            "📊 League Results": league_results,
            "🖨️ League Night Printout": league_printout,
            "🎯 Match Explorer": match_explorer,
            "🔍 Player Search": players,
            "📼 Badge Codex": badge_codex,
            "🧪 Badge Debug": badge_debug,
            "🪜 Challenge Ladder": challenge_ladder,
            "❓ FAQs": faqs,

            # Admin-only
            "🏟️ League Manager": league_manager,
            "📝 Match Uploader": record_match,
            "🧾 Record Match": record_match,
            "📝 Match Log": match_log,
            "👥 Player Editor": player_editor,
            "⚙️ Admin Tools": admin_tools,
            "📘 Admin Guide": admin_guide,
            "🛠️ Challenge Ladder Admin": challenge_ladder_admin,
            "💰 Moneyball": moneyball,
            "🎨 Theme QA": theme_gallery,
            "🏆 Tournaments": tournaments,
            "🏆 Tournament Manager": tournament_manager,
            "🏆 Division Manager": division_manager,
            "🏆 Tournament Bracket": tournament_public,
            "🗞️ Weekly Recap": weekly_recap,

            # Admin-only
            "🗞️ Weekly Recap Admin": weekly_recap_admin,
        }

        PAGE_KEY_TO_LABEL = {
            "command_center": "🧭 Command Center",
            "admin": "🧭 Command Center",
            "leaderboards": "🏆 Leaderboards",
            "league_results": "📊 League Results",
            "league_printout": "🖨️ League Night Printout",
            "match_explorer": "🎯 Match Explorer",
            "players": "🔍 Player Search",
            "badge_codex": "📼 Badge Codex",
            "badge_debug": "🧪 Badge Debug",
            "challenge_ladder": "🪜 Challenge Ladder",
            "faqs": "❓ FAQs",

            # Admin-only deep links
            "league_manager": "🏟️ League Manager",
            "match_uploader": "🧾 Record Match",
            "record_match": "🧾 Record Match",
            "match_log": "📝 Match Log",
            "player_editor": "👥 Player Editor",
            "admin_tools": "⚙️ Admin Tools",
            "admin_guide": "📘 Admin Guide",
            "challenge_ladder_admin": "🛠️ Challenge Ladder Admin",
            "legacy_match_entry": "🧾 Record Match",
            "moneyball": "💰 Moneyball",
            "theme_qa": "🎨 Theme QA",
            "tournaments": "🏆 Tournaments",
            "tournament_manager": "🏆 Tournament Manager",
            "tournament_divisions": "🏆 Tournament Manager",
            "division_manager": "🏆 Division Manager",
            "tournament_public": "🏆 Tournament Bracket",
            "weekly_recap": "🗞️ Weekly Recap",

            # Admin-only deep links
            "weekly_recap_admin": "🗞️ Weekly Recap Admin",
        }
        LABEL_TO_PAGE_KEY = {v: k for k, v in PAGE_KEY_TO_LABEL.items()}

        ADMIN_ONLY_LABELS = {
            "🧭 Command Center",
            "🖨️ League Night Printout",
            "🏟️ League Manager",
            "📝 Match Uploader",
            "🧾 Record Match",
            "📝 Match Log",
            "👥 Player Editor",
            "⚙️ Admin Tools",
            "📘 Admin Guide",
            "🛠️ Challenge Ladder Admin",
            "💰 Moneyball",
            "🎨 Theme QA",
            "🏆 Tournaments",
            "🏆 Tournament Manager",
            "🏆 Division Manager",
            "🧪 Badge Debug",
            "🗞️ Weekly Recap Admin",
        }

        # Visible labels based on auth
        all_labels = list(PAGES.keys())
        if not admin_logged_in:
            visible_labels = [x for x in all_labels if x not in ADMIN_ONLY_LABELS]
        else:
            visible_labels = all_labels
        st.session_state["_visible_labels"] = visible_labels

        # Public nav order (old UX)
        PUBLIC_NAV_KEYS = [
            "leaderboards",
            "league_results",
            "tournament_public",
            "weekly_recap",
            "match_explorer",
            "players",
            "badge_codex",
            "challenge_ladder",
            "faqs",
        ]
        public_labels_in_order = [PAGE_KEY_TO_LABEL[k] for k in PUBLIC_NAV_KEYS if PAGE_KEY_TO_LABEL.get(k)]

        # -------------------------
        # Deep link resolution
        # -------------------------
        deep_page_key = qp_get("page", "").strip().lower()
        deep_label = PAGE_KEY_TO_LABEL.get(deep_page_key, "")

        deep_route = qp_get("route", "").strip().strip("/")
        tournament_match = re.fullmatch(r"tournament/([^/]+)", deep_route)
        route_match = re.fullmatch(r"tournament/([^/]+)/division/([^/]+)", deep_route)
        if tournament_match:
            if PUBLIC_MODE:
                st.query_params["tournament_id"] = tournament_match.group(1)
            deep_label = "🏆 Tournament Bracket"
        if route_match:
            if PUBLIC_MODE:
                st.query_params["tournament_id"] = route_match.group(1)
                st.query_params["division_id"] = route_match.group(2)
            deep_label = "🏆 Tournament Bracket" if PUBLIC_MODE else "🏆 Division Manager"

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
            if "main_nav" not in st.session_state:
                if admin_logged_in:
                    st.session_state["main_nav"] = "🧭 Command Center"
                else:
                    st.session_state["main_nav"] = visible_labels[0]

            if st.session_state["main_nav"] not in visible_labels:
                st.session_state["main_nav"] = visible_labels[0]

            prev_admin_logged_in = bool(st.session_state.get("prev_admin_logged_in", False))
            if admin_logged_in and not prev_admin_logged_in:
                st.session_state["main_nav"] = "🧭 Command Center"
            st.session_state["prev_admin_logged_in"] = admin_logged_in

            if admin_logged_in and st.sidebar.button("🔄 Refresh data"):
                get_data.clear()
                try:
                    from jupr_app.domain.gamification.requirements import clear_requirements_cache

                    clear_requirements_cache()
                except Exception:
                    pass
                st.rerun()

            sel = render_admin_sidebar_nav(
                current_label=st.session_state.get("main_nav", "🧭 Command Center"),
                admin_logged_in=admin_logged_in,
            )

            if sel not in visible_labels:
                sel = visible_labels[0]
                st.session_state["main_nav"] = sel

        # -------------------------
        # Keep URL synced ONLY in public mode
        # -------------------------
        if PUBLIC_MODE:
            try:
                st.query_params["page"] = LABEL_TO_PAGE_KEY.get(sel, "leaderboards")
                st.query_params["public"] = "1"
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
