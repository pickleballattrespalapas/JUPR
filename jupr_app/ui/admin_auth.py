from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from datetime import datetime, timezone

import streamlit as st
import streamlit.components.v1 as components
from supabase import create_client

from jupr_app.ui.public_links import redact_query_params


logger = logging.getLogger(__name__)

_AUTH_USER_KEY = "admin_auth_user"
_AUTH_SESSION_KEY = "admin_auth_session"
_RECOVERY_SESSION_KEY = "admin_recovery_session"
_BROWSER_ACCESS_TOKEN_KEY = "jupr_admin_access_token"
_BROWSER_REFRESH_TOKEN_KEY = "jupr_admin_refresh_token"
_BROWSER_RESTORE_FLAG_KEY = "jupr_admin_restore_from_storage"
_BROWSER_SYNC_PAYLOAD_KEY = "_admin_browser_sync_payload"
_BROWSER_CLEAR_PENDING_KEY = "_admin_browser_clear_pending"
_BROWSER_RESTORE_INFLIGHT_SESSION_KEY = "jupr_admin_restore_inflight_at"
_ADMIN_RESTORE_FAILED_THIS_RUN_KEY = "_admin_restore_failed_this_run"
_AUTH_DEBUG_EVENTS_KEY = "jupr_auth_debug_events"
_AUTH_DEBUG_EVENTS_MAX = 50


class AdminAuthError(RuntimeError):
    """Raised for clean, operator-facing admin auth failures."""


class AdminAuthConfigError(AdminAuthError):
    """Raised when required auth config is missing."""


def _safe_route_snapshot() -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for key in st.query_params.keys():
        value = st.query_params.get(key, "")
        if isinstance(value, list):
            value = value[0] if value else ""
        text_value = str(value or "").strip()
        if text_value:
            snapshot[str(key)] = text_value
    return redact_query_params(snapshot)


def _sanitize_debug_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lowered = text.lower()
    sensitive_terms = (
        "access_token",
        "refresh_token",
        "password",
        "auth code",
        "auth_code",
        "token hash",
        "token_hash",
        "supabase",
        "anon_key",
        "service_role",
    )
    if any(term in lowered for term in sensitive_terms):
        return "[REDACTED]"
    return text


def _append_auth_debug_event(event_type: str, *, success: bool, reason: str = "") -> None:
    events = list(st.session_state.get(_AUTH_DEBUG_EVENTS_KEY, []))
    events.append(
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": str(event_type or "").strip() or "unknown",
            "success": bool(success),
            "reason": _sanitize_debug_text(reason),
            "route_query_params": _safe_route_snapshot(),
        }
    )
    st.session_state[_AUTH_DEBUG_EVENTS_KEY] = events[-_AUTH_DEBUG_EVENTS_MAX:]


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


def _safe_auth_method_names(auth) -> dict[str, bool]:
    """Return availability of auth methods used by recovery + reset flows."""
    return {
        "reset_password_email": hasattr(auth, "reset_password_email"),
        "reset_password_for_email": hasattr(auth, "reset_password_for_email"),
    }


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


def _set_auth_session(client, access_token: str, refresh_token: str):
    """Set auth session across Supabase client signature variants."""
    auth = getattr(client, "auth", None)
    if auth is None:
        raise AttributeError("Supabase client is missing auth")

    attempts = (
        ("set_session(access_token, refresh_token)", lambda: auth.set_session(access_token, refresh_token)),
        (
            "set_session({access_token, refresh_token})",
            lambda: auth.set_session(
                {
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                }
            ),
        ),
        (
            "set_session(session={access_token, refresh_token})",
            lambda: auth.set_session(
                session={
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                }
            ),
        ),
    )

    last_exc: Exception | None = None
    for idx, (label, attempt) in enumerate(attempts):
        try:
            return attempt()
        except TypeError as exc:
            # Only TypeError from early attempts should trigger compatibility fallbacks.
            # Invalid refresh token failures can occasionally surface as TypeError wrappers.
            if _is_invalid_refresh_token_error(exc):
                logger.warning("Supabase auth %s failed with invalid refresh token: %r", label, exc)
                raise
            last_exc = exc
            logger.warning("Supabase auth %s failed: %r", label, exc)
            if idx == len(attempts) - 1:
                raise
        except Exception as exc:
            # Preserve real auth failures (AuthApiError, invalid refresh token, etc.).
            logger.warning("Supabase auth %s failed: %r", label, exc)
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unable to set auth session")


def _is_invalid_refresh_token_error(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    if not text:
        return False
    needles = (
        "invalid refresh token",
        "already used",
        "refresh token",
    )
    return any(needle in text for needle in needles)


def _exchange_auth_code_for_session(client, auth_code: str):
    """Exchange auth code for session across Supabase client signature variants."""
    auth = getattr(client, "auth", None)
    if auth is None:
        raise AttributeError("Supabase client is missing auth")

    attempts = (
        (
            "exchange_code_for_session({auth_code})",
            lambda: auth.exchange_code_for_session({"auth_code": auth_code}),
        ),
        (
            "exchange_code_for_session(auth_code)",
            lambda: auth.exchange_code_for_session(auth_code),
        ),
        (
            "exchange_code_for_session(code=auth_code)",
            lambda: auth.exchange_code_for_session(code=auth_code),
        ),
    )

    last_exc: Exception | None = None
    for label, attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            last_exc = exc
            logger.warning("Supabase auth %s failed: %r", label, exc)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unable to exchange auth code for session")


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

    Auth state starts local to this Streamlit session but may be rehydrated from
    browser localStorage using the lightweight JS bridge below.
    """
    st.session_state.setdefault(_AUTH_USER_KEY, None)
    st.session_state.setdefault(_AUTH_SESSION_KEY, None)


def get_current_admin_user():
    return st.session_state.get(_AUTH_USER_KEY)


def persist_admin_browser_session(access_token: str, refresh_token: str) -> None:
    """Queue browser token persistence for render-time JS bridge execution."""
    access = str(access_token or "").strip()
    refresh = str(refresh_token or "").strip()
    if not access or not refresh:
        return

    st.session_state[_BROWSER_SYNC_PAYLOAD_KEY] = {
        "access_token": access,
        "refresh_token": refresh,
        "action": "persist",
    }
    st.session_state.pop(_BROWSER_CLEAR_PENDING_KEY, None)
    logger.info("Pending browser token persist queued")


def render_admin_browser_session_bridge() -> None:
    """Render browser storage bridge once per pending persist/clear action."""
    payload = st.session_state.get(_BROWSER_SYNC_PAYLOAD_KEY)
    clear_pending = bool(st.session_state.get(_BROWSER_CLEAR_PENDING_KEY))

    persist_access = ""
    persist_refresh = ""
    persist_requested = False
    if isinstance(payload, dict) and str(payload.get("action", "")).strip() == "persist":
        persist_access = str(payload.get("access_token", "")).strip()
        persist_refresh = str(payload.get("refresh_token", "")).strip()
        persist_requested = bool(persist_access and persist_refresh)

    if not persist_requested and not clear_pending:
        return

    components.html(
        f"""
        <script>
        try {{
          const appWindow = window.parent || window;
          const persistRequested = {str(persist_requested).lower()};
          const clearPending = {str(clear_pending).lower()};

          if (persistRequested) {{
            appWindow.localStorage.setItem("{_BROWSER_ACCESS_TOKEN_KEY}", {persist_access!r});
            appWindow.localStorage.setItem("{_BROWSER_REFRESH_TOKEN_KEY}", {persist_refresh!r});
            appWindow.sessionStorage.removeItem("{_BROWSER_RESTORE_INFLIGHT_SESSION_KEY}");
          }}

          if (clearPending) {{
            appWindow.localStorage.removeItem("{_BROWSER_ACCESS_TOKEN_KEY}");
            appWindow.localStorage.removeItem("{_BROWSER_REFRESH_TOKEN_KEY}");
            appWindow.sessionStorage.removeItem("{_BROWSER_RESTORE_INFLIGHT_SESSION_KEY}");
          }}
        }} catch (e) {{}}
        </script>
        """,
        height=0,
    )

    if persist_requested:
        logger.info("Browser token persist bridge rendered")
        st.session_state.pop(_BROWSER_SYNC_PAYLOAD_KEY, None)

    if clear_pending:
        logger.info("Browser token clear bridge rendered")
        st.session_state.pop(_BROWSER_CLEAR_PENDING_KEY, None)


def _clear_sensitive_query_params() -> None:
    sensitive = {
        "jupr_admin_access_token",
        "jupr_admin_refresh_token",
        _BROWSER_RESTORE_FLAG_KEY,
    }
    current = dict(st.query_params)
    sanitized = {k: v for k, v in current.items() if k not in sensitive}
    st.query_params.clear()
    st.query_params.update(sanitized)


def restore_admin_browser_session() -> dict[str, str] | None:
    """
    Restore browser-persisted tokens via a one-time URL handshake.
    Tokens are removed from URL immediately after they are read.
    """
    access = _query_param_text("jupr_admin_access_token")
    refresh = _query_param_text("jupr_admin_refresh_token")
    restore_flag = _query_param_text(_BROWSER_RESTORE_FLAG_KEY)
    if bool(st.session_state.get(_ADMIN_RESTORE_FAILED_THIS_RUN_KEY, False)):
        logger.info("Admin restore handshake skipped: restore already failed in this run")
        return None

    existing_user = get_current_admin_user()
    if existing_user is not None:
        existing_email = str(getattr(existing_user, "email", "") or "").strip().lower()
        if existing_email and is_allowed_admin_email(existing_email, load_admin_allowlist()):
            logger.info("Admin restore handshake skipped: already authenticated admin user in session")
            return None

    if access and refresh and restore_flag == "1":
        logger.info(
            "Admin restore handshake consumed (full_reload=True): access+refresh tokens present in query params"
        )
        _clear_sensitive_query_params()
        components.html(
            f"""
            <script>
            try {{
              const appWindow = window.parent || window;
              appWindow.sessionStorage.removeItem("{_BROWSER_RESTORE_INFLIGHT_SESSION_KEY}");
            }} catch (e) {{}}
            </script>
            """,
            height=0,
        )
        return {"access_token": access, "refresh_token": refresh}

    if restore_flag == "1":
        logger.info("Admin restore skipped: handshake flag present but token query params missing")
        return None

    components.html(
        f"""
        <script>
        try {{
          const appWindow = window.parent || window;
          const access = appWindow.localStorage.getItem("{_BROWSER_ACCESS_TOKEN_KEY}") || "";
          const refresh = appWindow.localStorage.getItem("{_BROWSER_REFRESH_TOKEN_KEY}") || "";
          const appUrl = new URL(appWindow.location.href);
          const params = appUrl.searchParams;
          const hasHandshake = params.get("{_BROWSER_RESTORE_FLAG_KEY}") === "1";
          const inflightRaw = appWindow.sessionStorage.getItem("{_BROWSER_RESTORE_INFLIGHT_SESSION_KEY}") || "";
          const inflightAt = Number(inflightRaw || "0");
          const now = Date.now();
          const inflightActive = Number.isFinite(inflightAt) && inflightAt > 0 && (now - inflightAt) < 15000;
          if (hasHandshake) {{
            appWindow.sessionStorage.removeItem("{_BROWSER_RESTORE_INFLIGHT_SESSION_KEY}");
          }}
          if (access && refresh && !hasHandshake && !inflightActive) {{
            appWindow.sessionStorage.setItem("{_BROWSER_RESTORE_INFLIGHT_SESSION_KEY}", String(now));
            params.set("jupr_admin_access_token", access);
            params.set("jupr_admin_refresh_token", refresh);
            params.set("{_BROWSER_RESTORE_FLAG_KEY}", "1");
            appWindow.location.replace(appUrl.toString());
          }}
        }} catch (e) {{}}
        </script>
        """,
        height=0,
    )
    return None


def clear_admin_browser_session() -> None:
    st.session_state[_BROWSER_CLEAR_PENDING_KEY] = True
    st.session_state.pop(_BROWSER_SYNC_PAYLOAD_KEY, None)
    _append_auth_debug_event(
        "clear_browser_tokens_queued",
        success=True,
        reason="Queued browser token clear for logout/session cleanup.",
    )
    logger.info("Browser token clear queued")


def render_admin_browser_session_clear_now() -> None:
    """Immediately clear browser auth token state for this tab."""
    components.html(
        f"""
        <script>
        try {{
          const appWindow = window.parent || window;
          appWindow.localStorage.removeItem("{_BROWSER_ACCESS_TOKEN_KEY}");
          appWindow.localStorage.removeItem("{_BROWSER_REFRESH_TOKEN_KEY}");
          appWindow.sessionStorage.removeItem("{_BROWSER_RESTORE_INFLIGHT_SESSION_KEY}");
        }} catch (e) {{}}
        </script>
        """,
        height=0,
    )
    st.session_state.pop(_BROWSER_SYNC_PAYLOAD_KEY, None)
    st.session_state.pop(_BROWSER_CLEAR_PENDING_KEY, None)
    _append_auth_debug_event(
        "clear_browser_tokens_now",
        success=True,
        reason="Cleared browser auth tokens immediately in current tab.",
    )
    logger.info("Browser token clear rendered immediately")


def maybe_restore_admin_login_from_browser() -> bool:
    """Try to restore allowlisted admin auth from browser storage when needed."""
    bootstrap_admin_auth()

    existing_user = get_current_admin_user()
    if existing_user is not None:
        existing_email = str(getattr(existing_user, "email", "") or "").strip().lower()
        logger.info("Browser restore skipped: user already present in session state")
        _append_auth_debug_event(
            "restore_skipped_already_authenticated",
            success=True,
            reason="Existing admin user already present in this Streamlit session.",
        )
        return bool(
            existing_email and is_allowed_admin_email(existing_email, load_admin_allowlist())
        )
    if bool(st.session_state.get(_ADMIN_RESTORE_FAILED_THIS_RUN_KEY, False)):
        logger.info("Browser restore skipped: this run already recorded a restore failure")
        _append_auth_debug_event(
            "restore_skipped_after_failure",
            success=False,
            reason="Restore already failed earlier in this app run.",
        )
        return False

    logger.info("Browser token restore attempt started")
    _append_auth_debug_event(
        "restore_attempt_started",
        success=True,
        reason="Attempting to restore admin auth from browser storage handshake.",
    )
    stored_tokens = restore_admin_browser_session()
    if not stored_tokens:
        logger.info("Browser token restore skipped: no handshake tokens available yet")
        _append_auth_debug_event(
            "restore_skipped_no_handshake_tokens",
            success=False,
            reason="No restore handshake tokens available in query params yet.",
        )
        return False

    try:
        client = make_supabase_auth_client()
        response = _set_auth_session(
            client,
            stored_tokens["access_token"],
            stored_tokens["refresh_token"],
        )

        session = _extract_session(response)
        if session is None:
            existing = client.auth.get_session()
            session = _extract_session(existing)

        user = getattr(response, "user", None)
        if user is None:
            user_resp = client.auth.get_user()
            user = getattr(user_resp, "user", None)

        user_email = str(getattr(user, "email", "") or "").strip().lower()
        if not session or not user or not is_allowed_admin_email(
            user_email, load_admin_allowlist()
        ):
            clear_local_admin_auth_state()
            logger.info("Browser token restore failed: invalid session/user or disallowed email")
            _append_auth_debug_event(
                "restore_failed",
                success=False,
                reason="Invalid restore session/user or user not in admin allowlist.",
            )
            return False

        st.session_state[_AUTH_USER_KEY] = user
        st.session_state[_AUTH_SESSION_KEY] = session
        access_token = str(getattr(session, "access_token", "") or "").strip()
        refresh_token = str(getattr(session, "refresh_token", "") or "").strip()
        if access_token and refresh_token:
            persist_admin_browser_session(access_token, refresh_token)
        st.session_state.pop(_ADMIN_RESTORE_FAILED_THIS_RUN_KEY, None)
        logger.info("Browser token restore succeeded")
        _append_auth_debug_event(
            "restore_succeeded",
            success=True,
            reason="Browser token restore completed for allowlisted admin user.",
        )
        return True
    except Exception as exc:
        logger.warning("Browser token restore failed: %r", exc)
        _append_auth_debug_event(
            "restore_failed",
            success=False,
            reason=f"Exception during restore: {exc}",
        )
        clear_local_admin_auth_state()
        render_admin_browser_session_clear_now()
        _clear_sensitive_query_params()
        st.session_state[_ADMIN_RESTORE_FAILED_THIS_RUN_KEY] = True
        return False


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
    # A failed automatic restore should not remain visible after a successful manual login.
    st.session_state.pop(_ADMIN_RESTORE_FAILED_THIS_RUN_KEY, None)
    access_token = str(getattr(session, "access_token", "") or "").strip()
    refresh_token = str(getattr(session, "refresh_token", "") or "").strip()
    if access_token and refresh_token:
        persist_admin_browser_session(access_token, refresh_token)
    _append_auth_debug_event(
        "login_succeeded",
        success=True,
        reason="Manual admin login succeeded.",
    )
    return {"user": user, "session": session}


def send_password_reset_email(email: str, *, redirect_to: str) -> None:
    """
    Send a password reset email via Supabase Auth without leaking account existence.
    Supports multiple Supabase client method/signature variants.
    """
    clean_email = _normalize_email(email)
    if not clean_email:
        raise AdminAuthError("Enter your email address.")

    logger.info("Entering forgot-password flow for admin sidebar")
    logger.info(
        "Sending reset email attempt started for normalized email with redirect_to=%s",
        redirect_to,
    )

    client = make_supabase_auth_client()
    auth = getattr(client, "auth", None)
    if auth is None:
        logger.error("Supabase auth client missing .auth while sending password reset email")
        raise AdminAuthError("Unable to send reset email right now. Please try again.")

    method_flags = _safe_auth_method_names(auth)
    logger.info(
        "Forgot-password auth method availability: %s (redirect_to=%s)",
        method_flags,
        redirect_to,
    )

    attempts = []

    if method_flags["reset_password_email"]:
        attempts.extend(
            [
                (
                    "reset_password_email(clean_email, {'redirect_to': redirect_to})",
                    lambda: auth.reset_password_email(
                        clean_email,
                        {"redirect_to": redirect_to},
                    ),
                    True,
                ),
                (
                    "reset_password_email(clean_email, options={'redirect_to': redirect_to})",
                    lambda: auth.reset_password_email(
                        clean_email,
                        options={"redirect_to": redirect_to},
                    ),
                    True,
                ),
                (
                    "reset_password_email(clean_email)",
                    lambda: auth.reset_password_email(clean_email),
                    False,
                ),
            ]
        )

    if method_flags["reset_password_for_email"]:
        attempts.extend(
            [
                (
                    "reset_password_for_email(clean_email, {'redirect_to': redirect_to})",
                    lambda: auth.reset_password_for_email(
                        clean_email,
                        {"redirect_to": redirect_to},
                    ),
                    True,
                ),
                (
                    "reset_password_for_email(clean_email, options={'redirect_to': redirect_to})",
                    lambda: auth.reset_password_for_email(
                        clean_email,
                        options={"redirect_to": redirect_to},
                    ),
                    True,
                ),
                (
                    "reset_password_for_email(clean_email)",
                    lambda: auth.reset_password_for_email(clean_email),
                    False,
                ),
            ]
        )

    ordered_attempts = []
    desired_order = [
        "reset_password_email(clean_email, {'redirect_to': redirect_to})",
        "reset_password_email(clean_email, options={'redirect_to': redirect_to})",
        "reset_password_for_email(clean_email, {'redirect_to': redirect_to})",
        "reset_password_for_email(clean_email, options={'redirect_to': redirect_to})",
        "reset_password_email(clean_email)",
        "reset_password_for_email(clean_email)",
    ]
    for signature in desired_order:
        for attempt in attempts:
            if attempt[0] == signature:
                ordered_attempts.append(attempt)
                break

    last_exc: Exception | None = None
    primary_auth_exc: Exception | None = None

    for signature, attempt, uses_redirect in ordered_attempts:
        logger.info(
            "Supabase password reset attempt started: %s (redirect_to=%s)",
            signature,
            redirect_to,
        )
        try:
            attempt()
            if not uses_redirect:
                logger.warning(
                    "Supabase password reset succeeded only without redirect_to; redirect URL is likely invalid or unsupported. redirect_to=%s",
                    redirect_to,
                )
            return
        except Exception as exc:
            last_exc = exc
            if type(exc).__name__ == "AuthApiError" and primary_auth_exc is None:
                primary_auth_exc = exc
            logger.warning(
                "Supabase password reset attempt failed: %s | redirect_to=%s | exc=%r",
                signature,
                redirect_to,
                exc,
            )
            continue

    preferred_exc = primary_auth_exc or last_exc

    if preferred_exc is not None:
        logger.error(
            "Supabase password reset email failed after all variants. redirect_to=%s methods=%s last_exc=%r",
            redirect_to,
            method_flags,
            preferred_exc,
            exc_info=(type(preferred_exc), preferred_exc, preferred_exc.__traceback__),
        )
    else:
        logger.error(
            "Supabase password reset email failed because no supported reset method exists on the auth client. redirect_to=%s methods=%s",
            redirect_to,
            method_flags,
        )

    if preferred_exc is not None and type(preferred_exc).__name__ == "AuthApiError":
        if "email rate limit exceeded" in str(preferred_exc).lower():
            raise AdminAuthError(
                "Too many reset emails have been requested. Please wait a few minutes and try again."
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
            logger.info("Recovery session established from existing auth session")
            return True
    except Exception:
        pass

    recovery_session = _get_stashed_recovery_session()
    if recovery_session:
        logger.info("Attempting to restore recovery session from stashed session state")
        try:
            response = _set_auth_session(
                client,
                recovery_session["access_token"],
                recovery_session["refresh_token"],
            )
        except Exception as exc:
            logger.warning("Recovery stashed session rehydrate failed: %r", exc)
        else:
            if _response_has_session(response) and _has_usable_session():
                logger.info("Recovery session established from stashed session")
                return True

    if query.get("error") or query.get("error_code"):
        return False

    access_token = query.get("access_token", "")
    refresh_token = query.get("refresh_token", "")
    if access_token and refresh_token:
        try:
            response = _set_auth_session(client, access_token, refresh_token)
        except Exception as exc:
            logger.warning("Recovery set_session failed: %r", exc)
            return False
        if _response_has_session(response) and _has_usable_session():
            logger.info("Recovery session established from access/refresh query params")
            return True
        return False

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
            logger.warning("Recovery verify_otp failed: %r", exc)
            return False
        if _response_has_session(response) and _has_usable_session():
            logger.info("Recovery session established from token_hash verify_otp")
            return True
        return False

    auth_code = query.get("code", "")
    if auth_code:
        try:
            response = _exchange_auth_code_for_session(client, auth_code)
        except Exception as exc:
            logger.warning("Recovery exchange_code_for_session failed: %r", exc)
            return False
        if _response_has_session(response) and _has_usable_session():
            logger.info("Recovery session established from auth code exchange")
            return True
        return False

    return False


def update_recovered_user_password(client, new_password: str) -> None:
    """Update password for the currently recovered/authenticated user session."""
    if not str(new_password or ""):
        raise AdminAuthError("Enter a new password.")

    recovery_session = _get_stashed_recovery_session()
    if recovery_session:
        try:
            _set_auth_session(
                client,
                recovery_session["access_token"],
                recovery_session["refresh_token"],
            )
        except Exception as exc:
            logger.warning("Re-attaching recovery session before password update failed: %r", exc)

    last_exc: Exception | None = None
    primary_auth_exc: Exception | None = None

    logger.info("update_user attempt started (variant: dict positional)")
    try:
        response = client.auth.update_user({"password": new_password})
        if getattr(response, "user", None) is not None:
            logger.info("update_user success (variant: dict positional)")
            return
    except Exception as exc:
        logger.warning("update_user failed (variant: dict positional): %r", exc)
        last_exc = exc
        if type(exc).__name__ == "AuthApiError" and primary_auth_exc is None:
            primary_auth_exc = exc

    logger.info("update_user attempt started (variant: attributes kwarg dict)")
    try:
        response = client.auth.update_user(attributes={"password": new_password})
        if getattr(response, "user", None) is not None:
            logger.info("update_user success (variant: attributes kwarg dict)")
            return
    except Exception as exc:
        logger.warning("update_user failed (variant: attributes kwarg dict): %r", exc)
        last_exc = exc
        if type(exc).__name__ == "AuthApiError" and primary_auth_exc is None:
            primary_auth_exc = exc

    logger.info("update_user attempt started (variant: UserAttributes positional)")
    try:
        try:
            from gotrue.types import UserAttributes
        except Exception as import_exc_types:
            logger.warning(
                "UserAttributes import from gotrue.types failed (variant: UserAttributes positional): %r",
                import_exc_types,
            )
            try:
                from gotrue import UserAttributes
            except Exception as import_exc:
                logger.warning(
                    "Skipping UserAttributes positional variant because gotrue import failed: %r",
                    import_exc,
                )
                if primary_auth_exc is None:
                    last_exc = import_exc
                raise

        response = client.auth.update_user(UserAttributes(password=new_password))
        if getattr(response, "user", None) is not None:
            logger.info("update_user success (variant: UserAttributes positional)")
            return
    except Exception as exc:
        logger.warning("update_user failed (variant: UserAttributes positional): %r", exc)
        if isinstance(exc, ModuleNotFoundError) and "gotrue" in str(exc).lower():
            if primary_auth_exc is None:
                last_exc = exc
        else:
            last_exc = exc
            if type(exc).__name__ == "AuthApiError" and primary_auth_exc is None:
                primary_auth_exc = exc

    logger.info("update_user attempt started (variant: attributes kwarg UserAttributes)")
    try:
        try:
            from gotrue.types import UserAttributes
        except Exception as import_exc_types:
            logger.warning(
                "UserAttributes import from gotrue.types failed (variant: attributes kwarg UserAttributes): %r",
                import_exc_types,
            )
            try:
                from gotrue import UserAttributes
            except Exception as import_exc:
                logger.warning(
                    "Skipping attributes kwarg UserAttributes variant because gotrue import failed: %r",
                    import_exc,
                )
                if primary_auth_exc is None:
                    last_exc = import_exc
                raise

        response = client.auth.update_user(
            attributes=UserAttributes(password=new_password)
        )
        if getattr(response, "user", None) is not None:
            logger.info("update_user success (variant: attributes kwarg UserAttributes)")
            return
    except Exception as exc:
        logger.warning("update_user failed (variant: attributes kwarg UserAttributes): %r", exc)
        if isinstance(exc, ModuleNotFoundError) and "gotrue" in str(exc).lower():
            if primary_auth_exc is None:
                last_exc = exc
        else:
            last_exc = exc
            if type(exc).__name__ == "AuthApiError" and primary_auth_exc is None:
                primary_auth_exc = exc

    preferred_exc = primary_auth_exc or last_exc

    logger.error(
        "Supabase password update failed during recovery flow: %r",
        preferred_exc,
        exc_info=(
            (type(preferred_exc), preferred_exc, preferred_exc.__traceback__)
            if preferred_exc
            else None
        ),
    )
    if preferred_exc is not None and "new password should be different from the old password" in str(
        preferred_exc
    ).lower():
        raise AdminAuthError("New password must be different from your current password.")

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
    clear_admin_browser_session()
