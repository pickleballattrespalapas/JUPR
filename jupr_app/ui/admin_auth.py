from __future__ import annotations

import os
from collections.abc import Mapping

import streamlit as st
from supabase import create_client


_AUTH_USER_KEY = "admin_auth_user"
_AUTH_SESSION_KEY = "admin_auth_session"


class AdminAuthError(RuntimeError):
    """Raised for clean, operator-facing admin auth failures."""


class AdminAuthConfigError(AdminAuthError):
    """Raised when required auth config is missing."""


def _get_secret(path: list[str], default=None):
    """Safe nested secret getter that never raises KeyError."""
    try:
        cur: object = st.secrets
    except Exception:
        return default

    for key in path:
        if not isinstance(cur, Mapping):
            return default
        if key not in cur:
            return default
        cur = cur[key]

    return cur


def _normalize_email(value: str | None) -> str:
    return str(value or "").strip().lower()


def load_admin_allowlist() -> set[str]:
    """
    Load allowlisted admin emails from either:
    - [admin].allowed_emails = ["a@x.com", "b@y.com"]
    - ADMIN_ALLOWED_EMAILS = "a@x.com,b@y.com"
    """
    allowed = _get_secret(["admin", "allowed_emails"], None)
    if allowed is None:
        allowed = st.secrets.get("ADMIN_ALLOWED_EMAILS")
    if allowed is None:
        allowed = os.getenv("ADMIN_ALLOWED_EMAILS")

    normalized: set[str] = set()
    if isinstance(allowed, (list, tuple, set)):
        for raw in allowed:
            email = _normalize_email(str(raw))
            if email:
                normalized.add(email)
    elif isinstance(allowed, str):
        for raw in allowed.split(","):
            email = _normalize_email(raw)
            if email:
                normalized.add(email)

    return normalized


def _get_supabase_auth_config() -> tuple[str, str]:
    url = str(
        st.secrets.get("SUPABASE_URL")
        or _get_secret(["supabase", "url"], "")
        or os.getenv("SUPABASE_URL", "")
    ).strip()
    anon_key = str(
        st.secrets.get("SUPABASE_ANON_KEY")
        or _get_secret(["supabase", "anon_key"], "")
        or os.getenv("SUPABASE_ANON_KEY", "")
    ).strip()

    if not url or not anon_key:
        raise AdminAuthConfigError(
            "Supabase auth is not configured. Set SUPABASE_URL, SUPABASE_ANON_KEY, and admin.allowed_emails."
        )

    return url, anon_key


def make_supabase_auth_client():
    """Build a dedicated auth client using Supabase URL + anon key only."""
    url, anon_key = _get_supabase_auth_config()
    return create_client(url, anon_key)


def bootstrap_admin_auth() -> None:
    """
    Initialize auth keys for this Streamlit session.

    This app no longer uses shared admin passwords or cookie/HMAC session bridges.
    Auth state is local to the current Streamlit session; browser refresh may require
    logging in again. Durable browser auth would require Streamlit-native OIDC or a
    dedicated frontend shell.
    """
    st.session_state.setdefault(_AUTH_USER_KEY, None)
    st.session_state.setdefault(_AUTH_SESSION_KEY, None)


def get_current_admin_user():
    return st.session_state.get(_AUTH_USER_KEY)


def is_allowed_admin_email(email: str, allowlist: set[str]) -> bool:
    return _normalize_email(email) in allowlist


def login_admin(email: str, password: str) -> dict:
    bootstrap_admin_auth()

    clean_email = _normalize_email(email)
    if not clean_email or not str(password or ""):
        raise AdminAuthError("Enter both email and password.")

    client = make_supabase_auth_client()

    try:
        response = client.auth.sign_in_with_password(
            {"email": clean_email, "password": password}
        )
    except AdminAuthConfigError:
        raise
    except Exception:
        raise AdminAuthError("Login failed. Check your email/password and try again.")

    user = getattr(response, "user", None)
    session = getattr(response, "session", None)
    if not user or not session:
        raise AdminAuthError("Login failed. Check your email/password and try again.")

    st.session_state[_AUTH_USER_KEY] = user
    st.session_state[_AUTH_SESSION_KEY] = session
    return {"user": user, "session": session}


def logout_admin() -> None:
    """Clear only local Streamlit auth state for the current session."""
    st.session_state.pop(_AUTH_USER_KEY, None)
    st.session_state.pop(_AUTH_SESSION_KEY, None)
