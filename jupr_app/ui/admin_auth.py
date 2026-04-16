
from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone

import streamlit as st

ADMIN_SESSION_TTL_SECONDS = 60 * 60  # 1 hour
COOKIE_NAME = "jupr_admin_session"
COOKIE_MANAGER_KEY = "jupr_admin_cookie_manager"


def _get_cookie_manager():
    try:
        import extra_streamlit_components as stx
    except Exception:
        return None

    try:
        return stx.CookieManager(key=COOKIE_MANAGER_KEY)
    except Exception:
        try:
            return stx.CookieManager()
        except Exception:
            return None


def _cookie_get(name: str) -> str | None:
    manager = _get_cookie_manager()
    if manager is None:
        return None

    try:
        value = manager.get(name)
    except TypeError:
        try:
            value = manager.get(cookie=name)
        except Exception:
            return None
    except Exception:
        return None

    if value in (None, "", "null", "None"):
        return None
    return str(value)


def _cookie_set(name: str, value: str, *, expires_at_epoch: int) -> None:
    manager = _get_cookie_manager()
    if manager is None:
        return

    expires_at = datetime.fromtimestamp(int(expires_at_epoch), tz=timezone.utc)
    try:
        manager.set(name, value, expires_at=expires_at, key=f"{name}_set")
        return
    except TypeError:
        pass
    except Exception:
        return

    try:
        manager.set(cookie=name, val=value, expires_at=expires_at, key=f"{name}_set")
    except Exception:
        return


def _cookie_delete(name: str) -> None:
    manager = _get_cookie_manager()
    if manager is None:
        return

    try:
        manager.delete(name, key=f"{name}_delete")
        return
    except TypeError:
        pass
    except Exception:
        return

    try:
        manager.delete(cookie=name, key=f"{name}_delete")
    except Exception:
        return


def _sign_admin_session(expires_at: int, secret: str) -> str:
    msg = f"{expires_at}".encode("utf-8")
    return hmac.new(secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()


def _session_payload(expires_at: int, token: str) -> dict[str, int | str]:
    return {"exp": int(expires_at), "token": str(token)}


def _normalize_session_payload(value: object) -> dict[str, int | str] | None:
    if not isinstance(value, dict):
        return None
    try:
        expires_at = int(value.get("exp", 0))
    except Exception:
        return None
    token = str(value.get("token", "") or "")
    if expires_at <= 0 or not token:
        return None
    return _session_payload(expires_at, token)


def _read_cookie_session() -> dict[str, int | str] | None:
    raw = _cookie_get(COOKIE_NAME)
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except Exception:
        return None
    return _normalize_session_payload(parsed)


def _write_cookie_session(data: dict[str, int | str]) -> None:
    normalized = _normalize_session_payload(data)
    if not normalized:
        return
    _cookie_set(
        COOKIE_NAME,
        json.dumps(normalized, separators=(",", ":")),
        expires_at_epoch=int(normalized["exp"]),
    )


def create_admin_session(*, secret: str, ttl_seconds: int = ADMIN_SESSION_TTL_SECONDS) -> None:
    if not secret:
        st.error(
            "Admin session secret is missing. Set supabase.admin_session_secret in secrets."
        )
        st.stop()

    expires_at = int(time.time()) + int(ttl_seconds)
    token = _sign_admin_session(expires_at, secret)
    payload = _session_payload(expires_at, token)
    st.session_state["admin_session"] = payload
    _write_cookie_session(payload)


def clear_admin_session() -> None:
    st.session_state.pop("admin_session", None)
    _cookie_delete(COOKIE_NAME)


def restore_admin_session(*, secret: str) -> bool:
    if "admin_session" in st.session_state:
        return validate_admin_session(secret=secret)

    if not secret:
        clear_admin_session()
        return False

    payload = _read_cookie_session()
    normalized = _normalize_session_payload(payload)
    if not normalized:
        return False

    st.session_state["admin_session"] = normalized
    return validate_admin_session(secret=secret)


def validate_admin_session(*, secret: str) -> bool:
    if not secret:
        clear_admin_session()
        return False

    data = _normalize_session_payload(st.session_state.get("admin_session"))
    if not data:
        cookie_payload = _read_cookie_session()
        data = _normalize_session_payload(cookie_payload)
        if not data:
            return False
        st.session_state["admin_session"] = data

    expires_at = int(data["exp"])
    if expires_at <= int(time.time()):
        clear_admin_session()
        return False

    token = str(data["token"])
    expected = _sign_admin_session(expires_at, secret)
    if not hmac.compare_digest(token, expected):
        clear_admin_session()
        return False

    _write_cookie_session(data)
    return True
