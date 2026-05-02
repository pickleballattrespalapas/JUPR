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
from jupr_app.domain.admin.roles import resolve_admin_role
from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue
from jupr_app.ui.admin_auth import (
    bootstrap_admin_auth,
    get_current_admin_user,
    is_allowed_admin_email,
    is_recovery_flow_query,
    load_admin_allowlist,
    logout_admin,
    maybe_restore_admin_login_from_browser,
    render_admin_browser_session_bridge,
)
from jupr_app.ui.context import AppContext
from jupr_app.ui.page_registry import (
    ADMIN_ONLY_LABELS,
    ADMIN_ONLY_PAGE_KEYS,
    HIDDEN_PAGE_LABELS,
    LABEL_TO_PAGE_KEY,
    PAGE_KEY_TO_LABEL,
    PUBLIC_NAV_KEYS,
    labels_for_keys,
)
from jupr_app.ui.branding import CLUB_ID, PUBLIC_BASE_URL_FALLBACK
from jupr_app.ui.public_nav import render_public_app_header, render_public_footer
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




def get_public_base_url() -> str:
    """Resolve public base URL from env/secrets with stable fallback."""
    env_value = str(os.getenv("JUPR_PUBLIC_BASE_URL", "") or "").strip().rstrip("/")
    if env_value:
        return env_value

    secret_candidates = (
        get_secret(["JUPR_PUBLIC_BASE_URL"], ""),
        get_secret(["PUBLIC_BASE_URL"], ""),
        get_secret(["public", "base_url"], ""),
    )
    for candidate in secret_candidates:
        secret_value = str(candidate or "").strip().rstrip("/")
        if secret_value:
            return secret_value

    return PUBLIC_BASE_URL_FALLBACK


PUBLIC_BASE_URL = get_public_base_url()


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
        'header[data-testid="stHeader"]{visibility:hidden;}'
        "</style>",
        unsafe_allow_html=True,
    )


def _query_param_value(key: str) -> str:
    value = st.query_params.get(key, "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "")


def _set_query_params_idempotent(
    *,
    updates: dict[str, str] | None = None,
    removals: set[str] | None = None,
) -> bool:
    changed = False

    for key, value in (updates or {}).items():
        value_text = str(value)
        if _query_param_value(key) != value_text:
            st.query_params[key] = value_text
            changed = True

    for key in removals or set():
        if key in st.query_params:
            st.query_params.pop(key, None)
            changed = True

    return changed


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

        def _is_truthy(value: str | None) -> bool:
            return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}

        # ---- Route intent ----
        public_requested = _is_truthy(qp_get("public", "0"))
        admin_requested = _is_truthy(qp_get("admin", "0"))
        incoming_page_param = qp_get("page", "").strip().lower()
        logout_requested = _is_truthy(qp_get("logout", "0"))
        admin_login_requested = incoming_page_param == "admin_login"
        requested_admin_page = incoming_page_param in ADMIN_ONLY_PAGE_KEYS
        recovery_flow = is_recovery_flow_query()
        admin_entry_requested = (
            admin_requested or admin_login_requested or requested_admin_page or recovery_flow
        )
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

        admin_allowlist = load_admin_allowlist()
        if admin_entry_requested:
            maybe_restore_admin_login_from_browser()

        current_admin = get_current_admin_user()
        current_admin_email = str(getattr(current_admin, "email", "") or "").strip().lower()
        authenticated = bool(current_admin and current_admin_email)
        authorized = authenticated and is_allowed_admin_email(current_admin_email, admin_allowlist)

        if authenticated and not authorized:
            logout_admin()
            current_admin = None
            current_admin_email = ""
            authenticated = False
            authorized = False

        admin_authenticated = authenticated and authorized
        st.session_state["jupr_admin_authenticated"] = bool(admin_authenticated)

        if logout_requested:
            if authenticated:
                logout_admin()
                st.session_state.pop("post_login_admin_page_key", None)
            params_changed = _set_query_params_idempotent(
                updates={"public": "1", "page": "home"},
                removals={
                    "admin",
                    "next",
                    "logout",
                    "jupr_admin_access_token",
                    "jupr_admin_refresh_token",
                    "jupr_admin_restore_from_storage",
                },
            )
            if params_changed:
                st.rerun()

        if admin_login_requested and admin_authenticated:
            post_login_target = str(
                st.session_state.get("post_login_admin_page_key", "") or ""
            ).strip().lower()
            if post_login_target not in ADMIN_ONLY_PAGE_KEYS:
                post_login_target = str(qp_get("next", "") or "").strip().lower()
            if post_login_target not in ADMIN_ONLY_PAGE_KEYS:
                post_login_target = "league_manager"

            st.session_state.pop("post_login_admin_page_key", None)
            params_changed = _set_query_params_idempotent(
                updates={
                    "admin": "1",
                    "page": post_login_target,
                },
                removals={
                    "public",
                    "next",
                    "jupr_admin_access_token",
                    "jupr_admin_refresh_token",
                    "jupr_admin_restore_from_storage",
                },
            )
            if params_changed:
                st.rerun()

        PUBLIC_MODE = not admin_entry_requested
        if public_requested and not admin_entry_requested:
            PUBLIC_MODE = True
        unauthenticated_admin_page_request = requested_admin_page and not admin_authenticated
        if unauthenticated_admin_page_request:
            st.session_state["post_login_admin_page_key"] = incoming_page_param
            params_changed = _set_query_params_idempotent(
                updates={"admin": "1", "page": "admin_login"},
                removals={"public"},
            )
            if params_changed:
                st.rerun()

        admin_logged_in = admin_authenticated and (not PUBLIC_MODE) and (not admin_login_requested)
        st.session_state["jupr_public_mode"] = bool(PUBLIC_MODE)
        st.session_state["jupr_admin_entry_active"] = bool(admin_entry_requested)
        st.session_state["admin_allowlist"] = admin_allowlist
        st.session_state["admin_role"] = "read_only"
        st.session_state["admin_role_source"] = "not_authenticated"

        # ---- Sidebar / Auth ----
        if PUBLIC_MODE or admin_login_requested:
            hide_sidebar_and_header_for_public()
        else:
            st.sidebar.title("JUPR Leagues 🌵")
            if authenticated:
                st.sidebar.success(f"Logged In: {current_admin_email}")
                if st.sidebar.button("Log Out", key="admin_logout_btn"):
                    logout_admin()
                    st.session_state.pop("post_login_admin_page_key", None)
                    _set_query_params_idempotent(
                        updates={
                            "public": "1",
                            "page": "home",
                        },
                        removals={
                            "admin",
                            "next",
                            "jupr_admin_access_token",
                            "jupr_admin_refresh_token",
                            "jupr_admin_restore_from_storage",
                        },
                    )
                    st.rerun()

        # Optional: allow pages to request a refresh of cached data
        if bool(st.session_state.get("force_data_refresh", False)):
            try:
                get_data.clear()
            except Exception:
                pass
            st.session_state["force_data_refresh"] = False

        # ---- Load data + ctx ----
        supabase = get_supabase()

        if admin_authenticated:
            role_resolution = resolve_admin_role(
                supabase=supabase,
                email=current_admin_email,
                user_id=str(getattr(current_admin, "id", "") or "").strip() or None,
                allowlist=admin_allowlist,
            )
            st.session_state["admin_role"] = role_resolution.role
            st.session_state["admin_role_source"] = role_resolution.source
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
            admin_login,
            admin_tools,
            badge_codex,
            badge_debug,
            badge_audit,
            challenge_ladder,
            challenge_ladder_admin,
            contact_support,
            data_corrections,
            email_preferences,
            faqs,
            home,
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
            player_updates_subscribe,
            player_editor,
            players,
            privacy_policy,
            profile_privacy,
            rating_rules,
            reset_password,
            terms_of_use,
            theme_gallery,
            top_players_printable,
            tournament_manager,
            tournament_partner_board,
            tournament_roster,
            tournament_registration,
            tournament_live,
            tournament_ops,
            tournaments,
            weekly_recap,
            weekly_recap_admin,
        )

        # ---- Router ----
        PAGES = {
            "Home": home,
            "🏆 Leaderboards": leaderboards,
            "📊 League Results": league_results,
            "🖨️ League Night Printout": league_printout,
            "🎯 Match Explorer": match_explorer,
            "🔍 Player Search": players,
            "Rating Rules": rating_rules,
            "📼 Badge Codex": badge_codex,
            "🧪 Badge Debug": badge_debug,
            "🧾 Badge Audit": badge_audit,
            "🧩 Match Canonical Audit": match_canonical_audit,
            "🪜 Challenge Ladder": challenge_ladder,
            "❓ FAQs": faqs,
            "Privacy Policy": privacy_policy,
            "Terms of Use": terms_of_use,
            "Contact Support": contact_support,
            "Data Corrections": data_corrections,
            "Email Preferences": email_preferences,
            "Profile Privacy": profile_privacy,
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
            "📋 Tournament Roster": tournament_roster,
            "🤝 Partner Board": tournament_partner_board,
            "🗞️ Weekly Recap": weekly_recap,
            "🧾 Top Active Players PDF": top_players_printable,
            # Hidden deep-link page
            "🔐 Admin Login": admin_login,
            "🔐 Reset Password": reset_password,
            "📬 Verified Updates Request": player_updates_subscribe,
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
        deep_page_key = incoming_page_param
        if not deep_page_key and recovery_flow:
            deep_page_key = "reset_password"
        elif PUBLIC_MODE and not deep_page_key:
            deep_page_key = "home"

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
                sel = render_public_app_header(
                    labels_in_order=public_labels_in_order,
                    current_label=current_label,
                )

        else:
            visible_labels = [x for x in visible_labels if x not in HIDDEN_PAGE_LABELS]

            valid_admin_deep_label = ""
            if deep_page_key == "admin_login" and not authenticated:
                valid_admin_deep_label = deep_label
            elif deep_label in HIDDEN_PAGE_LABELS and is_recovery_flow_query():
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
                if admin_logged_in:
                    sel = st.sidebar.radio("Go to:", visible_labels, key="main_nav")
                else:
                    sel = st.session_state.get("main_nav", visible_labels[0])

            if sel not in visible_labels and sel not in HIDDEN_PAGE_LABELS:
                sel = visible_labels[0]
                st.session_state["main_nav"] = sel

        # -------------------------
        # Keep URL synced (canonical deep links)
        # -------------------------
        try:
            target_page = LABEL_TO_PAGE_KEY.get(sel, "home")
            current_page = qp_get("page", "").strip().lower()
            canonical_updates: dict[str, str] = {}
            canonical_removals: set[str] = set()

            if PUBLIC_MODE:
                canonical_removals.update(
                    {
                        "admin",
                        "next",
                        "jupr_admin_access_token",
                        "jupr_admin_refresh_token",
                        "jupr_admin_restore_from_storage",
                    }
                )
                if target_page == "home":
                    canonical_removals.update({"public", "page"})
                else:
                    canonical_removals.add("public")
                    canonical_updates["page"] = target_page
            else:
                canonical_updates["admin"] = "1"
                canonical_removals.add("public")
                if target_page:
                    canonical_updates["page"] = target_page

            params_changed = _set_query_params_idempotent(
                updates=canonical_updates,
                removals=canonical_removals,
            )

            st.session_state["_last_rendered_nav"] = sel

            if params_changed and current_page != (target_page or ""):
                logger.info(
                    "Canonical route sync updated page query param: requested=%s resolved=%s public_mode=%s",
                    current_page,
                    target_page,
                    PUBLIC_MODE,
                )
                st.rerun()
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

        target_page_key = LABEL_TO_PAGE_KEY.get(sel, "")
        if target_page_key in ADMIN_ONLY_PAGE_KEYS and not admin_logged_in:
            st.info("Admin login required to access this page.")
            st.stop()

        try:
            render_fn(ctx)
            if PUBLIC_MODE:
                render_public_footer(current_label=sel)
        except Exception as exc:
            requested_page_key = qp_get("page", "").strip().lower()
            route_info = {
                "selected_label": sel,
                "target_page_key": target_page_key,
                "requested_page_key": requested_page_key,
                "public_mode": PUBLIC_MODE,
            }
            st.session_state["last_page_render_error"] = route_info
            logger.exception(
                "Page render failed. sel=%s target_page_key=%s public_mode=%s query=%s",
                sel,
                target_page_key,
                PUBLIC_MODE,
                dict(st.query_params),
            )
            st.error("This page failed to render, and navigation has been paused on this route.")
            st.caption(
                "If you reached this view from a deep link, keep the same URL and retry after fixing the underlying error."
            )
            st.caption(
                f"Route context: requested page '{requested_page_key or '(none)'}', resolved page '{target_page_key or '(none)'}'."
            )
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
