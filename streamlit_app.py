# jupr/streamlit_app.py
from __future__ import annotations

import os
import traceback
from collections.abc import Mapping
import logging

import pandas as pd  # kept because pages may rely on it
import streamlit as st

from jupr_app.data.client import make_supabase
from jupr_app.data.load import load_data
from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue
from jupr_app.ui.admin_auth import (
    AdminAuthConfigError,
    AdminAuthError,
    bootstrap_admin_auth,
    get_current_admin_user,
    is_allowed_admin_email,
    is_recovery_flow_query,
    load_admin_allowlist,
    login_admin,
    logout_admin,
    maybe_restore_admin_login_from_browser,
    render_admin_browser_session_bridge,
    send_password_reset_email,
)
from jupr_app.ui.context import AppContext
from jupr_app.ui.page_registry import (
    ADMIN_ONLY_LABELS,
    HIDDEN_PAGE_LABELS,
    LABEL_TO_PAGE_KEY,
    PAGE_KEY_TO_LABEL,
    PUBLIC_NAV_KEYS,
    labels_for_keys,
)
from jupr_app.ui.public_nav import render_public_top_nav
from jupr_app.ui.theme_clean import apply_clean_theme
from jupr_app.ui.url import qp_get

logger = logging.getLogger(__name__)


def _debug_exceptions_enabled() -> bool:
    qp_debug = str(qp_get("debug", "")).strip().lower()
    if qp_debug in {"1", "true", "yes", "y", "on"}:
        return True

    env_debug = str(os.getenv("JUPR_DEBUG", "")).strip().lower()
    if env_debug in {"1", "true", "yes", "y", "on", "dev", "debug"}:
        return True

    env_name = str(os.getenv("ENV", "")).strip().lower()
    return env_name in {"dev", "development", "local"}


# -------------------------
# CONFIG
# -------------------------
CLUB_ID = "tres_palapas"

# Public base URL used for share links + link buttons (Streamlit Cloud)
PUBLIC_BASE_URL = "https://juprtrespalapas.streamlit.app"
RESET_PASSWORD_REDIRECT_URL = f"{PUBLIC_BASE_URL}/?page=reset_password&public=1"


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
        debug_exceptions = _debug_exceptions_enabled()

        # Make base_url available to all pages (leaderboards uses this for share links)
        # Use session_state because ctx is a frozen-ish dataclass and you don't want to refactor it mid-stream.
        st.session_state["base_url"] = PUBLIC_BASE_URL

        # ---- Session defaults ----
        st.session_state.setdefault("_last_rendered_nav", None)

        # Admin auth uses Supabase email/password + allowlist and can restore
        # a previously persisted browser token pair after refresh.
        bootstrap_admin_auth()
        render_admin_browser_session_bridge()
        maybe_restore_admin_login_from_browser()

        admin_allowlist = load_admin_allowlist()
        auth_config_error: str | None = None

        # ---- Sidebar / Auth ----
        if PUBLIC_MODE:
            hide_sidebar_and_header_for_public()
        else:
            st.sidebar.title("JUPR Leagues 🌵")

            user = get_current_admin_user()
            user_email = ""
            if user is not None:
                user_email = str(getattr(user, "email", "") or "").strip().lower()

            authenticated = bool(user and user_email)
            authorized = authenticated and is_allowed_admin_email(user_email, admin_allowlist)

            if authenticated and not authorized:
                logout_admin()
                user = None
                user_email = ""
                authenticated = False
                st.sidebar.error("Authenticated but not authorized for admin access.")

            if not authenticated:
                with st.sidebar.expander("🔒 Admin Login"):
                    show_forgot_password = st.session_state.get("show_forgot_password", False)

                    with st.form("admin_login_form", clear_on_submit=False):
                        email = st.text_input("Email", key="admin_email")
                        password = st.text_input("Password", type="password", key="admin_pwd")
                        login_submitted = st.form_submit_button("Login")

                    if login_submitted:
                        if not admin_allowlist:
                            auth_config_error = "Supabase auth is not configured. Set SUPABASE_URL, SUPABASE_ANON_KEY, and admin.allowed_emails."
                        else:
                            try:
                                result = login_admin(email=email, password=password)
                                login_user = result.get("user")
                                login_email = str(getattr(login_user, "email", "") or "").strip().lower()
                                if not is_allowed_admin_email(login_email, admin_allowlist):
                                    logout_admin()
                                    st.sidebar.error("Authenticated but not authorized for admin access.")
                                else:
                                    st.rerun()
                            except AdminAuthConfigError as exc:
                                auth_config_error = str(exc)
                            except AdminAuthError as exc:
                                st.sidebar.error(str(exc))

                    if st.button("Forgot password?", key="admin_forgot_password_btn"):
                        st.session_state["show_forgot_password"] = not show_forgot_password
                        st.rerun()

                    if st.session_state.get("show_forgot_password", False):
                        with st.form("admin_reset_password_form", clear_on_submit=False):
                            reset_email = st.text_input("Email for reset link", key="admin_reset_email")
                            send_reset_submitted = st.form_submit_button("Send reset email")

                        if send_reset_submitted:
                            try:
                                send_password_reset_email(
                                    reset_email,
                                    redirect_to=RESET_PASSWORD_REDIRECT_URL,
                                )
                            except AdminAuthConfigError as exc:
                                auth_config_error = str(exc)
                            except AdminAuthError as exc:
                                st.sidebar.error(str(exc))
                            else:
                                st.sidebar.success(
                                    "If that email exists, a reset link has been sent."
                                )

                if auth_config_error:
                    st.sidebar.error(auth_config_error)
            else:
                st.sidebar.success(f"Logged In: {user_email}")
                if st.sidebar.button("Log Out", key="admin_logout_btn"):
                    logout_admin()
                    st.rerun()

        # Canonical admin flag (never true in public mode)
        current_admin = get_current_admin_user()
        current_admin_email = str(getattr(current_admin, "email", "") or "").strip().lower()
        authenticated_and_allowlisted = bool(current_admin and is_allowed_admin_email(current_admin_email, admin_allowlist))
        admin_logged_in = (not PUBLIC_MODE) and authenticated_and_allowlisted

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
            enqueue_result = enqueue_badge_eval(
                supabase,
                club_id=CLUB_ID,
                event_type="match_recorded",
                player_ids=player_ids,
                match_id=f"initial_load:{CLUB_ID}",
                payload={"initial_load": True},
            )
            if enqueue_result.get("queued"):
                process_badge_eval_queue(supabase, max_jobs=2, time_budget_seconds=2)

        # -------------------------
        # LAZY IMPORT PAGES (prevents import-time KeyError crashes)
        # -------------------------
        from jupr_app.ui.pages import (
            admin_guide,
            admin_tools,
            badge_codex,
            badge_debug,
            badge_audit,
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
            match_canonical_audit,
            match_log,
            match_uploader,
            moneyball,
            player_updates_admin,
            player_editor,
            players,
            reset_password,
            theme_gallery,
            top_players_printable,
            tournament_manager,
            tournament_partner_board,
            tournament_registration,
            tournament_live,
            tournament_ops,
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
            "🧾 Badge Audit": badge_audit,
            "🧩 Match Canonical Audit": match_canonical_audit,
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
            "🛠️ Tournament Setup": tournament_manager,
            "📋 Tournament Operations": tournament_ops,
            "🔴 Tournament Live": tournament_live,
            "📝 Tournament Registration": tournament_registration,
            "🤝 Partner Board": tournament_partner_board,
            "🗞️ Weekly Recap": weekly_recap,
            "🧾 Top Active Players PDF": top_players_printable,
            # Hidden deep-link page
            "🔐 Reset Password": reset_password,
            "📬 Verified Updates Request": players,
            # Admin-only
            "🗞️ Weekly Recap Admin": weekly_recap_admin,
            "📬 Player Updates Admin": player_updates_admin,
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
        incoming_page_param = qp_get("page", "").strip().lower()
        deep_page_key = incoming_page_param
        if not deep_page_key and is_recovery_flow_query():
            deep_page_key = "reset_password"

        deep_label = PAGE_KEY_TO_LABEL.get(deep_page_key, "")

        if PUBLIC_MODE:
            # Block admin-only deep links in public mode
            if deep_label in ADMIN_ONLY_LABELS:
                logger.warning(
                    "Public route request blocked for admin-only page. page=%s",
                    incoming_page_param,
                )
                deep_label = ""

            if deep_label in HIDDEN_PAGE_LABELS:
                sel = deep_label
            else:
                if incoming_page_param and not deep_label:
                    logger.warning(
                        "Public route fallback to default page due to unknown page key. page=%s",
                        incoming_page_param,
                    )
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
            visible_labels = [x for x in visible_labels if x not in HIDDEN_PAGE_LABELS]

            valid_admin_deep_label = ""
            if deep_label in HIDDEN_PAGE_LABELS and is_recovery_flow_query():
                valid_admin_deep_label = deep_label
            elif deep_label in visible_labels:
                valid_admin_deep_label = deep_label

            last_seen_page_param = st.session_state.get("_last_seen_page_param")
            if incoming_page_param != last_seen_page_param:
                st.session_state["_last_seen_page_param"] = incoming_page_param
                if valid_admin_deep_label:
                    st.session_state["main_nav"] = valid_admin_deep_label

            current_nav = st.session_state.get("main_nav")

            if (
                "main_nav" not in st.session_state
                or (current_nav not in visible_labels and current_nav not in HIDDEN_PAGE_LABELS)
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

            if st.session_state.get("main_nav") in HIDDEN_PAGE_LABELS:
                sel = st.session_state["main_nav"]
            else:
                sel = st.sidebar.radio("Go to:", visible_labels, key="main_nav")

            if sel not in visible_labels and sel not in HIDDEN_PAGE_LABELS:
                sel = visible_labels[0]
                st.session_state["main_nav"] = sel

        # -------------------------
        # Keep URL synced (canonical deep links)
        # -------------------------
        try:
            target_page = LABEL_TO_PAGE_KEY.get(sel, "leaderboards")
            current_page = qp_get("page", "").strip()
            last_rendered_nav = st.session_state.get("_last_rendered_nav")
            nav_changed = sel != last_rendered_nav

            if nav_changed and current_page != target_page:
                st.query_params["page"] = target_page

            current_public = qp_get("public", "").strip().lower()
            if PUBLIC_MODE:
                if current_public != "1":
                    st.query_params["public"] = "1"
            else:
                if "public" in st.query_params:
                    st.query_params.pop("public", None)

            st.session_state["_last_rendered_nav"] = sel
        except Exception:
            logger.exception("Failed to sync canonical query params for page selection.")
            if debug_exceptions:
                st.warning("Failed to sync URL query params for the selected page.")

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

        try:
            render_fn(ctx)
        except Exception as exc:
            target_page_key = LABEL_TO_PAGE_KEY.get(sel, "")
            logger.exception(
                "Page render failed. sel=%s target_page_key=%s public_mode=%s query=%s",
                sel,
                target_page_key,
                PUBLIC_MODE,
                dict(st.query_params),
            )
            st.error("This page failed to render.")
            if debug_exceptions:
                st.exception(exc)
            else:
                st.caption("Append ?debug=1 to the URL to view exception details in development.")
            st.stop()

    except Exception:
        logger.exception("streamlit_app.main() crashed before page render.")
        st.error("streamlit_app.main() crashed")
        st.code(traceback.format_exc())
        st.stop()


if __name__ == "__main__":
    main()
