from __future__ import annotations

import logging
import os
from collections.abc import Mapping

import streamlit as st
from supabase import create_client


logger = logging.getLogger(__name__)

_AUTH_USER_KEY = "admin_auth_user"
_AUTH_SESSION_KEY = "admin_auth_session"
_RECOVERY_SESSION_KEY = "admin_recovery_session"


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


def _query_param_text(key: str) -> str:
    """Read a query param as text while tolerating list-like values."""
    value = st.query_params.get(key, "")
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def _extract_session(obj):
    if obj is None:
        return None

    nested = getattr(obj, "session", None)
    if nested is not None:
        return nested

    access_token = getattr(obj, "access_token", None)
    refresh_token = getattr(obj, "refresh_token", None)
    if access_token and refresh_token:
        return obj

    return None


def _stash_recovery_session(session_obj) -> None:
    session = _extract_session(session_obj)
    if session is None:
        return

    access_token = getattr(session, "access_token", None)
    refresh_token = getattr(session, "refresh_token", None)
    if access_token and refresh_token:
        st.session_state[_RECOVERY_SESSION_KEY] = {
            "access_token": str(access_token),
            "refresh_token": str(refresh_token),
        }


def _get_stashed_recovery_session() -> dict[str, str] | None:
    raw = st.session_state.get(_RECOVERY_SESSION_KEY)
    if not isinstance(raw, dict):
        return None

    access_token = str(raw.get("access_token") or "").strip()
    refresh_token = str(raw.get("refresh_token") or "").strip()
    if not access_token or not refresh_token:
        return None

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
    }


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


def send_password_reset_email(email: str, *, redirect_to: str) -> None:
    """
    Send a password reset email via Supabase Auth without leaking account existence.
    Supports multiple Supabase client method/signature variants.
    """
    clean_email = _normalize_email(email)
    if not clean_email:
        raise AdminAuthError("Enter your email address.")

    client = make_supabase_auth_client()
    auth = getattr(client, "auth", None)
    if auth is None:
        logger.error("Supabase auth client missing .auth while sending password reset email")
        raise AdminAuthError("Unable to send reset email right now. Please try again.")

    attempts = []

    if hasattr(auth, "reset_password_email"):
        attempts.append(
            (
                "reset_password_email positional options",
                lambda: auth.reset_password_email(
                    clean_email,
                    {"redirect_to": redirect_to},
                ),
            )
        )
        attempts.append(
            (
                "reset_password_email keyword options",
                lambda: auth.reset_password_email(
                    clean_email,
                    options={"redirect_to": redirect_to},
                ),
            )
        )

    if hasattr(auth, "reset_password_for_email"):
        attempts.append(
            (
                "reset_password_for_email positional options",
                lambda: auth.reset_password_for_email(
                    clean_email,
                    {"redirect_to": redirect_to},
                ),
            )
        )
        attempts.append(
            (
                "reset_password_for_email keyword options",
                lambda: auth.reset_password_for_email(
                    clean_email,
                    options={"redirect_to": redirect_to},
                ),
            )
        )

    last_exc: Exception | None = None

    for label, attempt in attempts:
        try:
            attempt()
            return
        except (TypeError, AttributeError) as exc:
            last_exc = exc
            logger.warning("Supabase password reset attempt failed for %s: %s", label, exc)
            continue
        except Exception as exc:
            last_exc = exc
            logger.warning("Supabase password reset attempt failed for %s: %s", label, exc)
            continue

    if last_exc is not None:
        logger.error(
            "Supabase password reset email failed after trying all supported client variants: %s",
            last_exc,
            exc_info=(type(last_exc), last_exc, last_exc.__traceback__),
        )
    else:
        logger.error(
            "Supabase password reset email failed because no supported reset method exists on the auth client"
        )

    raise AdminAuthError("Unable to send reset email right now. Please try again.")


def get_recovery_query_params() -> dict[str, str]:
    """
    Collect Supabase recovery-related params from URL query params.
    Supports both token pair and PKCE code flows.
    """
    fields = (
        "access_token",
        "refresh_token",
        "type",
        "code",
        "token_hash",
        "error",
        "error_code",
        "error_description",
    )
    params: dict[str, str] = {}
    for key in fields:
        value = _query_param_text(key)
        if value:
            params[key] = value
    return params


def is_recovery_flow_query(params: dict[str, str] | None = None) -> bool:
    """Return True when query params look like a Supabase password recovery callback."""
    query = params or get_recovery_query_params()
    flow_type = str(query.get("type", "")).strip().lower()
    if flow_type == "recovery":
        return True

    return any(
        key in query for key in ("access_token", "refresh_token", "code", "token_hash")
    )


def establish_recovery_session(client, params: dict[str, str] | None = None) -> bool:
    """
    Exchange callback params into an auth session for password update.
    Returns True when a usable auth session is present.
    """
    query = params or get_recovery_query_params()

    def _response_has_session(response) -> bool:
        session = _extract_session(response)
        if session is not None:
            _stash_recovery_session(session)
            return True
        return False

    def _has_usable_session() -> bool:
        try:
            existing = client.auth.get_session()
        except Exception:
            return False
        session = _extract_session(existing)
        if session is not None:
            _stash_recovery_session(session)
            return True
        return False

    try:
        if _has_usable_session():
            return True
    except Exception:
        pass

    if query.get("error") or query.get("error_code"):
        return False

    access_token = query.get("access_token", "")
    refresh_token = query.get("refresh_token", "")
    if access_token and refresh_token:
        try:
            response = client.auth.set_session(access_token, refresh_token)
        except Exception as exc:
            logger.warning("Recovery set_session failed: %s", exc)
            return False
        return _response_has_session(response)

    token_hash = query.get("token_hash", "")
    otp_type = str(query.get("type", "")).strip().lower()
    if token_hash and otp_type:
        try:
            response = client.auth.verify_otp(
                {
                    "token_hash": token_hash,
                    "type": otp_type,
                }
            )
        except Exception as exc:
            logger.warning("Recovery verify_otp failed: %s", exc)
            return False
        return _response_has_session(response)

    auth_code = query.get("code", "")
    if auth_code:
        try:
            response = client.auth.exchange_code_for_session({"auth_code": auth_code})
        except Exception as exc:
            logger.warning("Recovery exchange_code_for_session failed: %s", exc)
            return False
        return _response_has_session(response)

    return False


def update_recovered_user_password(client, new_password: str) -> None:
    """Update password for the currently recovered/authenticated user session."""
    if not str(new_password or ""):
        raise AdminAuthError("Enter a new password.")

    recovery_session = _get_stashed_recovery_session()
    if recovery_session:
        try:
            client.auth.set_session(
                recovery_session["access_token"],
                recovery_session["refresh_token"],
            )
        except Exception as exc:
            logger.warning("Re-attaching recovery session before password update failed: %s", exc)

    last_exc: Exception | None = None

    try:
        response = client.auth.update_user({"password": new_password})
        if getattr(response, "user", None) is not None:
            return
    except TypeError as exc:
        last_exc = exc
    except Exception as exc:
        last_exc = exc

    try:
        response = client.auth.update_user(attributes={"password": new_password})
        if getattr(response, "user", None) is not None:
            return
    except Exception as exc:
        last_exc = exc

    logger.error(
        "Supabase password update failed during recovery flow: %s",
        last_exc,
        exc_info=(type(last_exc), last_exc, last_exc.__traceback__) if last_exc else None,
    )
    raise AdminAuthError(
        "Unable to update password. This password reset link may be invalid or expired. Request a new reset email."
    )


def clear_local_admin_auth_state() -> None:
    """Clear local Streamlit auth state after password reset success."""
    logout_admin()
    st.session_state.pop(_RECOVERY_SESSION_KEY, None)


def logout_admin() -> None:
    """Clear only local Streamlit auth state for the current session."""
    st.session_state.pop(_AUTH_USER_KEY, None)
    st.session_state.pop(_AUTH_SESSION_KEY, None)
