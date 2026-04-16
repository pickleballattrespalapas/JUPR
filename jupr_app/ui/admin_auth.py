from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.parse

import streamlit as st
import streamlit.components.v1 as components


ADMIN_SESSION_TTL_SECONDS = 60 * 60  # 1 hour
ADMIN_SESSION_COOKIE_KEY = "jupr_admin_session"
PENDING_COOKIE_ACTION_KEY = "_jupr_admin_cookie_action"


def _sign_admin_session(expires_at: int, secret: str) -> str:
    msg = f"{expires_at}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _read_request_cookies() -> dict[str, str]:
    try:
        cookies = getattr(st.context, "cookies", None)
    except Exception:
        cookies = None

    if not cookies:
        return {}

    try:
        return {str(k): str(v) for k, v in dict(cookies).items()}
    except Exception:
        return {}


def _get_cookie_value() -> str | None:
    raw = _read_request_cookies().get(ADMIN_SESSION_COOKIE_KEY)
    if not raw:
        return None
    try:
        return urllib.parse.unquote(raw)
    except Exception:
        return raw


def _queue_set_cookie(data: dict, ttl_seconds: int) -> None:
    st.session_state[PENDING_COOKIE_ACTION_KEY] = {
        "mode": "set",
        "value": json.dumps(data, separators=(",", ":")),
        "ttl_seconds": max(60, int(ttl_seconds)),
        "nonce": str(int(time.time() * 1000)),
    }


def _queue_delete_cookie() -> None:
    st.session_state[PENDING_COOKIE_ACTION_KEY] = {
        "mode": "delete",
        "nonce": str(int(time.time() * 1000)),
    }


def create_admin_session(*, secret: str, ttl_seconds: int = ADMIN_SESSION_TTL_SECONDS) -> None:
    if not secret:
        st.error(
            "Admin session secret is missing. Set supabase.admin_session_secret in secrets."
        )
        st.stop()

    expires_at = int(time.time()) + int(ttl_seconds)
    token = _sign_admin_session(expires_at, secret)
    data = {"exp": expires_at, "token": token}
    st.session_state["admin_session"] = data
    _queue_set_cookie(data, ttl_seconds)


def clear_admin_session() -> None:
    st.session_state.pop("admin_session", None)
    _queue_delete_cookie()


def validate_admin_session(*, secret: str) -> bool:
    if not secret:
        st.session_state.pop("admin_session", None)
        return False

    data = st.session_state.get("admin_session")
    if not isinstance(data, dict):
        return False

    try:
        expires_at = int(data.get("exp", 0))
    except Exception:
        st.session_state.pop("admin_session", None)
        return False

    if expires_at <= int(time.time()):
        st.session_state.pop("admin_session", None)
        return False

    token = str(data.get("token", ""))
    expected = _sign_admin_session(expires_at, secret)
    if not hmac.compare_digest(token, expected):
        st.session_state.pop("admin_session", None)
        return False

    return True


def restore_admin_session(*, secret: str) -> None:
    current = st.session_state.get("admin_session")
    if isinstance(current, dict) and validate_admin_session(secret=secret):
        return

    raw = _get_cookie_value()
    if not raw:
        return

    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        st.session_state.pop("admin_session", None)
        return

    if not isinstance(data, dict):
        st.session_state.pop("admin_session", None)
        return

    st.session_state["admin_session"] = data
    if not validate_admin_session(secret=secret):
        st.session_state.pop("admin_session", None)


def flush_admin_cookie_action() -> bool:
    action = st.session_state.pop(PENDING_COOKIE_ACTION_KEY, None)
    if not isinstance(action, dict):
        return False

    mode = str(action.get("mode") or "").strip().lower()
    if mode not in {"set", "delete"}:
        return False

    if mode == "set":
        raw_value = str(action.get("value") or "")
        cookie_value = urllib.parse.quote(raw_value, safe="")
        ttl_seconds = max(60, int(action.get("ttl_seconds") or ADMIN_SESSION_TTL_SECONDS))
        cookie_stmt = (
            f'document.cookie = "{ADMIN_SESSION_COOKIE_KEY}={cookie_value}; '
            f'path=/; max-age={ttl_seconds}; SameSite=Lax";'
        )
    else:
        cookie_stmt = (
            f'document.cookie = "{ADMIN_SESSION_COOKIE_KEY}=; '
            f'path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; SameSite=Lax";'
        )

    html = f"""
    <script>
    (function() {{
        try {{
            {cookie_stmt}
        }} catch (e) {{}}
        setTimeout(function() {{
            window.location.reload();
        }}, 60);
    }})();
    </script>
    """
    components.html(html, height=0, width=0)
    return True
