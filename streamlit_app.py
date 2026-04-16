# jupr/streamlit_app.py
from __future__ import annotations

import traceback
from collections.abc import Mapping

import pandas as pd  # kept because pages may rely on it
import streamlit as st

from jupr_app.data.client import make_supabase
from jupr_app.data.load import load_data
from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue
from jupr_app.ui.admin_auth import (
    ADMIN_SESSION_TTL_SECONDS,
    clear_admin_session,
    create_admin_session,
    flush_admin_cookie_action,
    restore_admin_session,
    validate_admin_session,
)
from jupr_app.ui.context import AppContext
from jupr_app.ui.page_registry import (
    ADMIN_ONLY_LABELS,
    LABEL_TO_PAGE_KEY,
    PAGE_KEY_TO_LABEL,
    PUBLIC_NAV_KEYS,
    labels_for_keys,
)
from jupr_app.ui.public_nav import render_public_top_nav
from jupr_app.ui.theme_clean import apply_clean_theme
from jupr_app.ui.url import qp_get


# -------------------------
# CONFIG
# -------------------------
CLUB_ID = "tres_palapas"

# Public base URL used for share links + link buttons (Streamlit Cloud)
PUBLIC_BASE_URL = "https://juprtrespalapas.streamlit.app"


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
# Admin session helpers
# -------------------------
def _get_admin_session_secret() -> str:
    return str(
        get_secret(["supabase", "admin_session_secret"], "")
        or get_secret(["admin_session_secret"], "")
        or ""
    )


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
    admin_session_secret = "..."  # used to sign short-lived admin sessions
    """
    url = st.secrets.get("SUPABASE_URL") or get_secret(["supabase", "url"])

    key = (
        st.secrets.get("SUPABASE_ANON_KEY")
        or st.secrets.get("SUPABASE_SERVICE_ROLE_KEY")
        or get_secret(["supabase", "anon_key"])
        or get_secret(["supabase", "service_role_key"])
    )

    if not url or not key:
        st.error("Supabase credentials missing.")
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
    # Hide sidebar + collapse control for public, keep app content full-width.
    st.markdown(
        "<style>"
        "section[data-testid='stSidebar']{display:none;}"
        "div[data-testid='collapsedControl']{display:none;}"
        "header{visibility:hidden;}"
        "</style>",
        unsafe_allow_html=True,
    )


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
        apply_clean_theme(
            accent_hex="#2F6FED"
        )  # pick your accent once (can later be club-specific)
        st.markdown(
            "<!-- JUPR_THEME_ACTIVE_2026_01_22 -->",  # TODO: remove after deployment verification
            unsafe_allow_html=True,
        )

        # ---- Public mode ----
        PUBLIC_MODE = qp_get("public", "0").lower() in ("1", "true", "yes", "y")

        # Make base_url available to all pages (leaderboards uses this for share links)
        # Use session_state because ctx is a frozen-ish dataclass and you don't want to refactor it mid-stream.
        st.session_state["base_url"] = PUBLIC_BASE_URL

        # ---- Session defaults ----
        st.session_state.setdefault("deep_link_applied", False)

        admin_session_secret = _get_admin_session_secret()
        restore_admin_session(secret=admin_session_secret)

        # ---- Sidebar / Auth ----
        if PUBLIC_MODE:
            hide_sidebar_and_header_for_public()
        else:
            st.sidebar.title("JUPR Leagues 🌵")

            if not validate_admin_session(secret=admin_session_secret):
                with st.sidebar.expander("🔒 Admin Login"):
                    pwd = st.text_input("Password", type="password", key="admin_pwd")

                    if st.button("Login", key="admin_login_btn"):
                        expected = str(
                            get_secret(["supabase", "admin_password"], "") or ""
                        )
                        if not expected:
                            st.error(
                                "Admin password is not configured in secrets (supabase.admin_password)."
                            )
                        elif pwd == expected:
                            create_admin_session(
                                secret=admin_session_secret,
                                ttl_seconds=ADMIN_SESSION_TTL_SECONDS,
                            )
                        else:
                            st.error("Incorrect password.")
            else:
                st.sidebar.success("Logged In: Admin")
                if st.sidebar.button("Log Out", key="admin_logout_btn"):
                    clear_admin_session()

        if flush_admin_cookie_action():
            st.info("Syncing admin session...")
            st.stop()

        # Canonical admin flag (never true in public mode)
        admin_logged_in = (not PUBLIC_MODE) and validate_admin_session(
            secret=admin_session_secret
        )

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
            player_ids = []
            if (
                df_players_all is not None
                and not df_players_all.empty
                and "id" in df_players_all.columns
            ):
                player_ids = df_players_all["id"].dropna().astype(int).tolist()
            queued = enqueue_badge_eval(
                supabase,
                club_id=CLUB_ID,
                event_type="match_recorded",
                player_ids=player_ids,
                match_id=f"initial_load:{CLUB_ID}",
                payload={"initial_load": True},
            )
            if queued:
                process_badge_eval_queue(supabase, max_jobs=2, time_budget_seconds=2)

        # -------------------------
        # LAZY IMPORT PAGES (prevents import-time KeyError crashes)
        # -------------------------
        from jupr_app.ui.pages import (
            admin_guide,
            admin_tools,
            badge_codex,
            badge_debug,
            challenge_ladder,
            challenge_ladder_admin,
            faqs,
            jupr_live,
            jupr_live_admin,
            leaderboards,
            league_manager,
            league_printout,
            league_results,
            match_explorer,
            match_log,
            match_uploader,
            moneyball,
            player_editor,
            players,
            theme_gallery,
            top_players_printable,
            tournament_manager,
            tournament_partner_board,
            tournament_registration,
            tournaments,
            weekly_recap,
            weekly_recap_admin,
        )

        # ---- Router ----
        PAGES = {
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
            "📝 Match Uploader": match_uploader,
            "📝 Match Log": match_log,
            "👥 Player Editor": player_editor,
            "⚙️ Admin Tools": admin_tools,
            "📘 Admin Guide": admin_guide,
            "🛠️ Challenge Ladder Admin": challenge_ladder_admin,
            "💰 Moneyball": moneyball,
            "🔴 JUPR Live": jupr_live,
            "🔴 JUPR Live Admin": jupr_live_admin,
            "🎨 Theme QA": theme_gallery,
            "🏆 Tournaments": tournaments,
            "🏆 Tournament Manager": tournament_manager,
            "📝 Tournament Registration": tournament_registration,
            "🤝 Partner Board": tournament_partner_board,
            "🗞️ Weekly Recap": weekly_recap,
            "🧾 Top Active Players PDF": top_players_printable,
            # Admin-only
            "🗞️ Weekly Recap Admin": weekly_recap_admin,
        }

        # Visible labels based on auth
        all_labels = list(PAGES.keys())
        if not admin_logged_in:
            visible_labels = [x for x in all_labels if x not in ADMIN_ONLY_LABELS]
        else:
            visible_labels = all_labels

        public_labels_in_order = labels_for_keys(PUBLIC_NAV_KEYS)

        # -------------------------
        # Deep link resolution
        # -------------------------
        deep_page_key = qp_get("page", "").strip().lower()
        deep_label = PAGE_KEY_TO_LABEL.get(deep_page_key, "")

        if PUBLIC_MODE:
            # Block admin-only deep links in public mode
            if deep_label in ADMIN_ONLY_LABELS:
                deep_label = ""

            current_label = (
                deep_label
                if deep_label in public_labels_in_order
                else public_labels_in_order[0]
            )
            sel = render_public_top_nav(
                labels_in_order=public_labels_in_order,
                current_label=current_label,
            )

        else:
            # Apply deep link once (only if that page is visible)
            if (not bool(st.session_state.get("deep_link_applied", False))) and (
                deep_label in visible_labels
            ):
                st.session_state["main_nav"] = deep_label
                st.session_state["deep_link_applied"] = True

            # Ensure valid selection
            if (
                "main_nav" not in st.session_state
                or st.session_state["main_nav"] not in visible_labels
            ):
                st.session_state["main_nav"] = visible_labels[0]

            if admin_logged_in and st.sidebar.button("🔄 Refresh data"):
                get_data.clear()
                try:
                    from jupr_app.domain.gamification.requirements import (
                        clear_requirements_cache,
                    )

                    clear_requirements_cache()
                except Exception:
                    pass
                st.rerun()

            sel = st.sidebar.radio("Go to:", visible_labels, key="main_nav")

            if sel not in visible_labels:
                sel = visible_labels[0]
                st.session_state["main_nav"] = sel

        # -------------------------
        # Keep URL synced (canonical deep links)
        # -------------------------
        try:
            target_page = LABEL_TO_PAGE_KEY.get(sel, "leaderboards")
            current_page = qp_get("page", "").strip()
            if current_page != target_page:
                st.query_params["page"] = target_page

            current_public = qp_get("public", "").strip().lower()
            if PUBLIC_MODE:
                if current_public != "1":
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
