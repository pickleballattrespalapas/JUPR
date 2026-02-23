# jupr/streamlit_app.py
from __future__ import annotations

import os
import traceback
import re
from typing import Any

import jwt
from supabase import create_client

import streamlit as st
import pandas as pd  # noqa: F401  # kept because pages may rely on it

from jupr_app.ui.streamlit_exceptions import rethrow_if_streamlit_control

LOCAL_PUBLIC_BASE_URL_DEFAULT = "http://localhost:8501"


@st.cache_resource
def get_supabase_service():
    url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

    if not url or not service_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

    return create_client(url, service_key)


@st.cache_resource
def get_supabase_auth():
    url = os.getenv("SUPABASE_URL")
    anon_key = os.getenv("SUPABASE_ANON_KEY")

    if not url or not anon_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_ANON_KEY")

    return create_client(url, anon_key)


def assert_schema_health(supabase):
    from jupr_app.data.schema_preflight import ensure_badge_schema_preflight

    ensure_badge_schema_preflight(supabase)


@st.cache_data(ttl=30)
def get_data(club_id: str):
    from jupr_app.data.load import load_data

    supabase = get_supabase_service()
    return load_data(supabase, club_id, match_limit=5000)


def hide_sidebar_and_header_for_public():
    return None


def render_public_top_nav(*, labels_in_order: list[str], current_label: str) -> str:
    st.markdown("**Go to:**")
    try:
        idx = labels_in_order.index(current_label)
    except ValueError:
        idx = 0

    return st.radio(
        label="public_top_nav",
        options=labels_in_order,
        index=idx,
        horizontal=True,
        key="public_top_nav_radio",
        label_visibility="collapsed",
    )


def render_admin_sidebar_nav(*, current_label: str, admin_logged_in: bool) -> str:
    def _nav_to(target: str):
        st.session_state["_nav_pending"] = target

    taxonomy = [
        ("Command Center", ["🧭 Command Center"]),
        ("Competitions", ["🪜 Challenge Ladder", "🏆 Tournaments", "🏆 Tournament Manager", "🏆 Division Manager", "💰 Moneyball"]),
        ("Results", ["🧾 Record Match", "📊 League Results", "🎯 Match Explorer", "📝 Match Log", "🗞️ Weekly Recap", "🗞️ Weekly Recap Admin"]),
        ("Players", ["🔍 Player Search", "👥 Player Editor", "🖨️ League Night Printout"]),
        ("Recognition", ["🏆 Leaderboards", "📼 Badge Codex", "🧪 Badge Debug"]),
        ("Administration", ["🏟️ League Manager", "🗂️ Session Console", "🛠️ Challenge Ladder Admin", "⚙️ Admin Tools", "📘 Admin Guide", "🎨 Theme QA", "❓ FAQs"]),
    ]

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
            prefix = "👉 " if label == current_label else ""
            st.sidebar.button(
                f"{prefix}{label}",
                key=f"nav_btn_{label}",
                use_container_width=True,
                on_click=_nav_to,
                args=(label,),
            )
        st.sidebar.markdown(" ")

    return current_label


def resolve_auth_session(supabase):
    try:
        session_obj = supabase.auth.get_session()
        session = getattr(session_obj, "session", session_obj)
        if session:
            return session
    except Exception:
        pass

    params = st.query_params

    access_token = params.get("access_token")
    refresh_token = params.get("refresh_token")

    if access_token and refresh_token:
        try:
            supabase.auth.set_session(
                {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                }
            )
            session_obj = supabase.auth.get_session()
            session = getattr(session_obj, "session", session_obj)
            if session:
                st.query_params.clear()
                return session
        except Exception:
            pass

    return None


def resolve_entry_mode(session):
    if st.session_state.get("entry_mode"):
        return st.session_state["entry_mode"]
    if session:
        return "auth"
    return "gateway"


def get_secret(path: list[str], default: Any = None):
    current: Any = st.secrets
    try:
        for key in path:
            current = current[key]
        return current
    except Exception:
        return default


def _init_session():
    defaults = {
        "_nav_target": "home",
        "_nav_pending": None,
        "_ui_event": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _process_pending_nav():
    pending = st.session_state.get("_nav_pending")
    if pending:
        st.session_state["_nav_target"] = pending
        st.session_state["_nav_pending"] = None


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def resolve_tenant(session):
    if session:
        token = session.access_token
        payload = jwt.decode(token, options={"verify_signature": False})
        club_id = payload.get("user_metadata", {}).get("club_id")
        if not club_id:
            raise RuntimeError("Authenticated user missing club_id claim.")

        st.session_state["jwt_payload"] = {
            "sub": payload.get("sub"),
            "email": payload.get("email"),
            "club_id": club_id,
        }
        return club_id

    st.session_state.pop("jwt_payload", None)
    return os.getenv("DEFAULT_PUBLIC_CLUB_ID", "tres_palapas")


def main():
    try:
        from jupr_app.ui.context import AppContext
        from jupr_app.ui.theme_clean import apply_clean_theme

        _init_session()
        st.session_state.setdefault("entry_mode", None)
        if "auth" not in st.session_state:
            st.session_state["auth"] = {
                "mode": "public",
                "supabase_session": None,
                "admin_override": False,
            }

        st.set_page_config(
            page_title="JUPR Leagues",
            layout="wide",
            page_icon="🌵",
            initial_sidebar_state="collapsed",
        )
        apply_clean_theme(accent_hex="#2F6FED")
        st.write("DEBUG — session id:", id(st.session_state))
        base_url = os.getenv("PUBLIC_BASE_URL", LOCAL_PUBLIC_BASE_URL_DEFAULT)
        st.session_state["base_url"] = str(base_url)

        supabase_service = get_supabase_service()
        supabase_auth = get_supabase_auth()
        assert_schema_health(supabase_service)

        # Attempt token rehydration before resolving session
        if (
            "sb_access_token" in st.session_state
            and "sb_refresh_token" in st.session_state
        ):
            try:
                supabase_auth.auth.set_session(
                    {
                        "access_token": st.session_state["sb_access_token"],
                        "refresh_token": st.session_state["sb_refresh_token"],
                    }
                )
            except Exception:
                # If token invalid or expired, clear them
                st.session_state.pop("sb_access_token", None)
                st.session_state.pop("sb_refresh_token", None)

        session = resolve_auth_session(supabase_auth)
        st.write("DEBUG — resolve_auth_session:", bool(session))
        if session is not None:
            st.session_state["auth"]["mode"] = "user"
            st.session_state["auth"]["supabase_session"] = session
        flow_type = st.query_params.get("type")
        if flow_type == "recovery":
            st.session_state["recovery_mode"] = True

        if flow_type == "signup":
            st.success("Email confirmed successfully.")
            return

        entry_mode = resolve_entry_mode(session)
        st.session_state["entry_mode"] = entry_mode

        if entry_mode == "gateway":
            st.markdown("# JUPR Leagues 🌵")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔐 Login", use_container_width=True):
                    if st.session_state.get("entry_mode") != "login":
                        st.session_state["entry_mode"] = "login"
                        _rerun()
            with col2:
                if st.button("🌎 Public Pages", use_container_width=True):
                    changed = st.session_state.get("entry_mode") != "public" or st.session_state["auth"].get("mode") != "public"
                    st.session_state["entry_mode"] = "public"
                    st.session_state["auth"]["mode"] = "public"
                    if changed:
                        _rerun()
            return

        if entry_mode == "login" and not session:
            st.markdown("## Login")
            with st.form("login_form"):
                email = st.text_input("Email", key="sb_email")
                password = st.text_input("Password", type="password", key="sb_password")
                submitted = st.form_submit_button("Login")

            if submitted:
                try:
                    auth_response = supabase_auth.auth.sign_in_with_password(
                        {
                            "email": email.strip(),
                            "password": password,
                        }
                    )
                    session_obj = getattr(auth_response, "session", auth_response)

                    # Persist raw tokens
                    st.session_state["sb_access_token"] = session_obj.access_token
                    st.session_state["sb_refresh_token"] = session_obj.refresh_token

                    # Update layered auth state
                    st.session_state["auth"]["mode"] = "user"
                    st.session_state["auth"]["supabase_session"] = session_obj
                    st.session_state["entry_mode"] = "auth"
                    _rerun()
                except BaseException as exc:
                    rethrow_if_streamlit_control(exc)
                    st.error(f"Login exception: {exc}")

            if st.button("Send Magic Link", use_container_width=True):
                if not email:
                    st.error("Email required.")
                else:
                    try:
                        supabase_auth.auth.sign_in_with_otp({"email": email})
                        st.success("Magic link sent.")
                    except Exception:
                        st.error("Unable to send magic link.")
            return

        session = st.session_state["auth"].get("supabase_session")
        if session:
            try:
                current_obj = supabase_auth.auth.get_session()
                current_session = getattr(current_obj, "session", current_obj)
                if current_session:
                    st.session_state["auth"]["supabase_session"] = current_session
                    session = current_session
            except Exception:
                pass

        if st.session_state.get("recovery_mode"):
            st.subheader("Set New Password")
            new_password = st.text_input("New Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")

            if st.button("Update Password"):
                if not new_password or new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    try:
                        supabase_auth.auth.update_user({"password": new_password})
                        st.success("Password updated successfully.")
                        if st.session_state.pop("recovery_mode", None) is not None:
                            _rerun()
                    except Exception:
                        st.error("Password update failed.")
            return

        try:
            club_id = resolve_tenant(session)
        except Exception:
            st.error("Invalid authentication token.")
            st.stop()

        auth = st.session_state["auth"]
        PUBLIC_MODE = auth["mode"] == "public"
        admin_logged_in = (
            auth["mode"] == "user"
            and auth["supabase_session"] is not None
            and auth["admin_override"]
        )
        st.session_state["admin_logged_in"] = admin_logged_in

        def elevate_admin():
            expected = os.getenv("ADMIN_PASSWORD", "").strip()
            pwd = str(st.session_state.get("admin_pwd", "")).strip()
        
            if not expected:
                st.error("Admin password not configured.")
                return
        
            if pwd == expected:
                st.session_state["auth"]["admin_override"] = True
            else:
                st.error("Incorrect admin password.")

        def logout():
            try:
                supabase_auth.auth.sign_out()
            except Exception:
                pass
            st.session_state["auth"] = {
                "mode": "public",
                "supabase_session": None,
                "admin_override": False,
            }
            st.session_state.pop("sb_access_token", None)
            st.session_state.pop("sb_refresh_token", None)
            st.session_state["entry_mode"] = "gateway"

        if st.session_state["auth"]["mode"] != "public":
            with st.sidebar.expander("🔒 Admin Access"):
                st.text_input("Admin Password", type="password", key="admin_pwd")
                st.button("Elevate", on_click=elevate_admin)

        def _clear_app_caches():
            get_data.clear()
            try:
                from jupr_app.domain.gamification.requirements import clear_requirements_cache

                clear_requirements_cache()
            except Exception:
                pass

        st.sidebar.button("Logout", on_click=logout)

        if st.session_state.pop("_force_data_reload", False) or st.session_state.pop("force_data_refresh", False):
            _clear_app_caches()

        if (not PUBLIC_MODE) and admin_logged_in:
            if st.sidebar.button("🔄 Refresh data", key="refresh_data_btn"):
                _clear_app_caches()
                st.session_state["force_data_refresh"] = True
                return

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
        ) = get_data(club_id)

        ctx = AppContext(
            supabase=supabase_service,
            club_id=club_id,
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

        event = st.session_state.get("_ui_event")
        if event:
            if event.get("type") == "replay":
                from jupr_app.domain.replay_history import replay_history

                replay_history(
                    supabase=ctx.supabase,
                    club_id=str(ctx.club_id),
                    df_meta=ctx.df_meta,
                    target_reset=event.get("target"),
                )
                st.session_state["force_data_refresh"] = True
            st.session_state["_ui_event"] = None

        if admin_logged_in and schema_degraded:
            st.warning(f"Schema preflight degraded mode enabled. {schema_degraded_reason}")

        if df_player_badges is not None and df_player_badges.empty:
            from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval

            player_ids = []
            if df_players_all is not None and not df_players_all.empty and "id" in df_players_all.columns:
                player_ids = df_players_all["id"].dropna().astype(int).tolist()
            enqueue_badge_eval(
                supabase_service,
                club_id=club_id,
                event_type="match_recorded",
                player_ids=player_ids,
                match_id=f"initial_load:{club_id}",
                payload={"initial_load": True},
            )

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
            session_console,
        )

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
            "🗞️ Weekly Recap Admin": weekly_recap_admin,
            "🗂️ Session Console": session_console,
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
            "weekly_recap_admin": "🗞️ Weekly Recap Admin",
            "session_console": "🗂️ Session Console",
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
            "🗂️ Session Console",
        }

        all_labels = list(PAGES.keys())
        visible_labels = all_labels if admin_logged_in else [x for x in all_labels if x not in ADMIN_ONLY_LABELS]
        st.session_state["_visible_labels"] = visible_labels

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

        # Respect explicit in-app navigation clicks.
        # Sidebar buttons set `_nav_pending` via callback before the script reruns.
        if not st.session_state.get("_nav_pending"):
            deep_label = ""
            deep_route = st.query_params.get("route", "").strip().strip("/")
        
            tournament_match = re.fullmatch(r"tournament/([^/]+)", deep_route)
            route_match = re.fullmatch(r"tournament/([^/]+)/division/([^/]+)", deep_route)
            session_match = re.fullmatch(r"sessions/([^/]+)", deep_route)
            session_court_match = re.fullmatch(r"sessions/([^/]+)/rounds/(\d+)/courts/([^/]+)", deep_route)
        
            if tournament_match:
                if PUBLIC_MODE:
                    st.query_params["tournament_id"] = tournament_match.group(1)
                deep_label = "🏆 Tournament Bracket"
        
            if route_match:
                if PUBLIC_MODE:
                    st.query_params["tournament_id"] = route_match.group(1)
                    st.query_params["division_id"] = route_match.group(2)
                deep_label = "🏆 Tournament Bracket" if PUBLIC_MODE else "🏆 Division Manager"

            if session_match or session_court_match:
                deep_label = "🗂️ Session Console"
        
            if PUBLIC_MODE and deep_label in ADMIN_ONLY_LABELS:
                deep_label = ""
        
            if deep_label:
                # Only keep URL 'page=' syncing for public mode (your existing intent)
                if PUBLIC_MODE:
                    st.query_params["page"] = LABEL_TO_PAGE_KEY.get(deep_label, "leaderboards")
                st.session_state["_nav_pending"] = deep_label
        
            elif PUBLIC_MODE:
                # In admin mode, do NOT let ?page=... force navigation every rerun.
                deep_page_key = st.query_params.get("page", "").strip().lower()
                deep_page_label = PAGE_KEY_TO_LABEL.get(deep_page_key)
                if deep_page_label in visible_labels:
                    st.session_state["_nav_pending"] = deep_page_label

        if st.session_state["_nav_target"] == "home":
            if st.session_state["entry_mode"] == "auth" and "🧭 Command Center" in visible_labels:
                st.session_state["_nav_pending"] = "🧭 Command Center"
            elif st.session_state["entry_mode"] == "public" and public_labels_in_order:
                st.session_state["_nav_pending"] = public_labels_in_order[0]
            elif visible_labels:
                st.session_state["_nav_pending"] = visible_labels[0]

        _process_pending_nav()

        if st.session_state.get("_nav_target") == "login":
            if st.session_state.get("entry_mode") != "login":
                st.session_state["entry_mode"] = "login"
                _rerun()
            return

        current_page = st.session_state["_nav_target"]
        if current_page not in visible_labels:
            current_page = visible_labels[0]
            st.session_state["_nav_target"] = current_page

        if current_page in ADMIN_ONLY_LABELS and not admin_logged_in:
            if st.session_state.get("_nav_pending") != "login":
                st.session_state["_nav_pending"] = "login"
                _rerun()
            return

        if PUBLIC_MODE:
            selected = render_public_top_nav(labels_in_order=public_labels_in_order, current_label=current_page)
            if selected != current_page and st.session_state.get("_nav_pending") != selected:
                st.session_state["_nav_pending"] = selected
                _rerun()
        else:
            render_admin_sidebar_nav(current_label=current_page, admin_logged_in=admin_logged_in)

        if PUBLIC_MODE:
            try:
                page_key = LABEL_TO_PAGE_KEY.get(current_page, "leaderboards")
                if st.query_params.get("page") != page_key:
                    st.query_params["page"] = page_key
            except Exception:
                pass

        page_mod = PAGES.get(current_page)
        if page_mod is None:
            st.error(f"Unknown page selection: {current_page}")
            st.stop()

        render_fn = getattr(page_mod, "render", None)
        if not callable(render_fn):
            st.error(f"Page module for '{current_page}' has no render(ctx) function.")
            st.stop()

        render_fn(ctx)

    except BaseException as exc:
        rethrow_if_streamlit_control(exc)
        st.error("streamlit_app.main() crashed")
        st.code(traceback.format_exc())
        st.stop()


if __name__ == "__main__":
    main()
