from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timedelta, timezone

import streamlit as st

try:
    import extra_streamlit_components as stx
except Exception:  # pragma: no cover
    stx = None


ADMIN_SESSION_TTL_SECONDS = 60 * 60  # 1 hour
ADMIN_SESSION_COOKIE_KEY = "jupr_admin_session"
COOKIE_MANAGER_SESSION_KEY = "_jupr_cookie_manager"


def get_cookie_manager():
    """
    Do not cache this with st.cache_*.
    CookieManager is a Streamlit component/widget, and caching it triggers
    CachedWidgetWarning on cache misses.
    """
    if stx is None:
        return None

    mgr = st.session_state.get(COOKIE_MANAGER_SESSION_KEY)
    if mgr is not None:
        return mgr

    try:
        mgr = stx.CookieManager(key="jupr_admin_cookie_manager")
    except TypeError:
        mgr = stx.CookieManager()

    st.session_state[COOKIE_MANAGER_SESSION_KEY] = mgr
    return mgr


def _sign_admin_session(expires_at: int, secret: str) -> str:
    msg = f"{expires_at}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _get_cookie_value() -> str | None:
    mgr = get_cookie_manager()
    if mgr is None:
        return None
    try:
        return mgr.get(ADMIN_SESSION_COOKIE_KEY)
    except TypeError:
        try:
            return mgr.get(cookie=ADMIN_SESSION_COOKIE_KEY)
        except Exception:
            return None
    except Exception:
        return None


def _set_cookie_value(data: dict, ttl_seconds: int) -> None:
    mgr = get_cookie_manager()
    if mgr is None:
        return
    try:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=max(60, int(ttl_seconds)))
        try:
            mgr.set(ADMIN_SESSION_COOKIE_KEY, json.dumps(data), expires_at=expires_at)
        except TypeError:
            mgr.set(cookie=ADMIN_SESSION_COOKIE_KEY, val=json.dumps(data), expires_at=expires_at)
    except Exception:
        pass


def _delete_cookie_value() -> None:
    mgr = get_cookie_manager()
    if mgr is None:
        return
    try:
        try:
            mgr.delete(ADMIN_SESSION_COOKIE_KEY)
        except TypeError:
            mgr.delete(cookie=ADMIN_SESSION_COOKIE_KEY)
    except Exception:
        pass


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
    _set_cookie_value(data, ttl_seconds)


def clear_admin_session() -> None:
    st.session_state.pop("admin_session", None)
    _delete_cookie_value()


def validate_admin_session(*, secret: str) -> bool:
    if not secret:
        clear_admin_session()
        return False

    data = st.session_state.get("admin_session")
    if not isinstance(data, dict):
        return False

    try:
        expires_at = int(data.get("exp", 0))
    except Exception:
        clear_admin_session()
        return False

    if expires_at <= int(time.time()):
        clear_admin_session()
        return False

    token = str(data.get("token", ""))
    expected = _sign_admin_session(expires_at, secret)
    if not hmac.compare_digest(token, expected):
        clear_admin_session()
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
        clear_admin_session()
        return

    if not isinstance(data, dict):
        clear_admin_session()
        return

    st.session_state["admin_session"] = data
    if not validate_admin_session(secret=secret):
        clear_admin_session()
