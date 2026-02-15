# jupr/streamlit_app.py
from __future__ import annotations

import hashlib
import hmac
import os
import time
import traceback
import re
from collections.abc import Mapping

import streamlit as st
import pandas as pd  # noqa: F401  # kept because pages may rely on it


# -------------------------
# CONFIG
# -------------------------
CLUB_ID = "tres_palapas"

# Local/dev fallback for share links + link buttons.
LOCAL_PUBLIC_BASE_URL_DEFAULT = "http://localhost:8501"


# -------------------------
# Secrets helpers (SAFE)
# -------------------------
def get_secret(path: list[str], default=None):
    """
    Nested secret getter.

    Priority:
    1) Streamlit secrets (st.secrets)
    2) Environment variables (Fly / Docker / etc.)

    Path ["supabase","url"] → SUPABASE__URL
    """

    # --- 1) Try Streamlit secrets ---
    try:
        cur = st.secrets
        for k in path:
            if not isinstance(cur, Mapping) or k not in cur:
                raise KeyError
            cur = cur[k]
        return cur
    except Exception:
        pass

    # --- 2) Fallback to environment variable ---
    env_key = "__".join([p.upper() for p in path])
    return os.environ.get(env_key, default)


def _get_config_value(path: list[str], default=None):
    return get_secret(path, default)


# -------------------------
# Admin session helpers
# -------------------------
ADMIN_SESSION_TTL_SECONDS = 60 * 60  # 1 hour
ADMIN_MAX_FAILED_ATTEMPTS = 5
ADMIN_LOCKOUT_SECONDS = 60


def _get_admin_session_secret() -> str:
    return os.getenv("SUPABASE__ADMIN_SESSION_SECRET", "")


def _get_admin_password() -> str:
    return os.getenv("SUPABASE__ADMIN_PASSWORD", "")


def _sign_admin_session(expires_at: int, secret: str) -> str:
    msg = f"{expires_at}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _create_admin_session() -> None:
    secret = _get_admin_session_secret()
    if not secret:
        st.error("Admin session secret is missing.")
        st.stop()

    expires_at = int(time.time()) + int(ADMIN_SESSION_TTL_SECONDS)
    token = _sign_admin_session(expires_at, secret)
    st.session_state["admin_session"] = {"exp": expires_at, "token": token}


def _clear_admin_session() -> None:
    st.session_state.pop("admin_session", None)


def _validate_admin_session() -> bool:
    secret = _get_admin_session_secret()
    if not secret:
        _clear_admin_session()
        return False

    data = st.session_state.get("admin_session")
    if not isinstance(data, dict):
        return False

    try:
        expires_at = int(data.get("exp", 0))
    except Exception:
        _clear_admin_session()
        return False

    if expires_at <= int(time.time()):
        _clear_admin_session()
        return False

    token = str(data.get("token", ""))
    expected = _sign_admin_session(expires_at, secret)
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
    Requires Supabase configuration.
    """
    from jupr_app.data.client import make_supabase

    url = _get_config_value(["supabase", "url"], "")
    key = _get_config_value(["supabase", "anon_key"], "") or _get_config_value(["supabase", "key"], "")

    if not url or not key:
        st.error("Supabase secrets are missing or misnamed.")
        st.code(
            "[supabase]\n"
            'url = "https://YOUR_PROJECT_REF.supabase.co"\n'
            'anon_key = "YOUR_SUPABASE_ANON_KEY"  # or use key = "…"\n'
        )

        try:
            st.write("Secrets keys:", list(st.secrets.keys()))
            sb = _get_config_value(["supabase"], default={})
            if isinstance(sb, Mapping):
                st.write("Supabase keys:", list(sb.keys()))
        except Exception:
            pass

        st.stop()

    return make_supabase(str(url), str(key))


@st.cache_data(ttl=30)
def get_data(club_id: str):
    from jupr_app.data.load import load_data

    supabase = get_supabase()
    return load_data(supabase, club_id, match_limit=5000)


# -------------------------
# UI helpers
# -------------------------
def hide_sidebar_and_header_for_public():
    return None


def render_public_top_nav(*, labels_in_order: list[str], current_label: str) -> str:
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


# -------------------------
# MAIN
# -------------------------
def main():
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
        apply_clean_theme(accent_hex="#2F6FED")

        PUBLIC_MODE = qp_get("public", "0").lower() in ("1", "true", "yes", "y")

        base_url = _get_config_value(["public_base_url"], LOCAL_PUBLIC_BASE_URL_DEFAULT)
        st.session_state["base_url"] = str(base_url)

        st.session_state.setdefault("deep_link_applied", False)
        st.session_state.setdefault("admin_failed_attempts", 0)
        st.session_state.setdefault("admin_lock_until", 0)

        if PUBLIC_MODE:
            hide_sidebar_and_header_for_public()
        else:
            st.sidebar.title("JUPR Leagues 🌵")

            if not _validate_admin_session():
                with st.sidebar.expander("🔒 Admin Login"):
                    pwd = st.text_input("Password", type="password", key="admin_pwd")

                    if st.button("Login", key="admin_login_btn"):
                        now = int(time.time())
                        lock_until = int(st.session_state.get("admin_lock_until", 0) or 0)
                        if now < lock_until:
                            wait_seconds = lock_until - now
                            st.error(f"Too many failed attempts. Try again in {wait_seconds} seconds.")
                        else:
                            expected = _get_admin_password()
                            if not expected:
                                st.error("Admin password is not configured.")
                            elif pwd == expected:
                                st.session_state["admin_failed_attempts"] = 0
                                st.session_state["admin_lock_until"] = 0
                                _create_admin_session()
                                st.rerun()
                            else:
                                failed_attempts = int(st.session_state.get("admin_failed_attempts", 0) or 0) + 1
                                st.session_state["admin_failed_attempts"] = failed_attempts
                                if failed_attempts >= ADMIN_MAX_FAILED_ATTEMPTS:
                                    st.session_state["admin_lock_until"] = now + ADMIN_LOCKOUT_SECONDS
                                    st.session_state["admin_failed_attempts"] = 0
                                    st.error(
                                        f"Too many failed attempts. Try again in {ADMIN_LOCKOUT_SECONDS} seconds."
                                    )
                                else:
                                    remaining = ADMIN_MAX_FAILED_ATTEMPTS - failed_attempts
                                    st.error(
                                        "Incorrect password. "
                                        f"{remaining} attempt{'s' if remaining != 1 else ''} remaining before lockout."
                                    )
            else:
                st.sidebar.success("Logged In: Admin")
                if st.sidebar.button("Log Out", key="admin_logout_btn"):
                    _clear_admin_session()
                    st.rerun()

        admin_logged_in = (not PUBLIC_MODE) and _validate_admin_session()

        supabase = get_supabase()
        get_data(CLUB_ID)

    except Exception:
        st.error("streamlit_app.main() crashed")
        st.code(traceback.format_exc())
        st.stop()


if __name__ == "__main__":
    main()
