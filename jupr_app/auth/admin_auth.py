from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

import streamlit as st

SESSION_COOKIE_NAME = "jupr_admin_session"
SESSION_TTL_SECONDS = 60 * 60  # 1 hour


def _get_secret(key: str) -> str:
    try:
        if key in st.secrets:
            return str(st.secrets[key])
        if "supabase" in st.secrets and key in st.secrets["supabase"]:
            return str(st.secrets["supabase"][key])
    except Exception:
        pass

    return os.getenv(key.upper(), "")


def _get_admin_password() -> str:
    return _get_secret("admin_password")


def _get_session_secret() -> str:
    return _get_secret("admin_session_secret")


def _sign(payload: str, secret: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def create_admin_session():
    secret = _get_session_secret()
    if not secret:
        raise RuntimeError("ADMIN_SESSION_SECRET missing.")

    expires_at = int(time.time()) + SESSION_TTL_SECONDS
    payload = json.dumps({"exp": expires_at})
    sig = _sign(payload, secret)

    token = json.dumps({"payload": payload, "sig": sig})

    st.experimental_set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=SESSION_TTL_SECONDS,
        secure=True,
        httponly=True,
        samesite="Strict",
    )


def clear_admin_session():
    st.experimental_set_cookie(SESSION_COOKIE_NAME, "", max_age=0)


def validate_admin_session() -> bool:
    secret = _get_session_secret()
    if not secret:
        return False

    token = st.experimental_get_cookie(SESSION_COOKIE_NAME)
    if not token:
        return False

    try:
        data = json.loads(token)
        payload = data["payload"]
        sig = data["sig"]

        expected_sig = _sign(payload, secret)
        if not hmac.compare_digest(sig, expected_sig):
            return False

        payload_data = json.loads(payload)
        if int(payload_data["exp"]) < int(time.time()):
            return False

        st.session_state["is_admin"] = True
        st.session_state["admin_email"] = "admin"
        return True

    except Exception:
        return False


def attempt_login(password: str) -> tuple[bool, str]:
    expected = _get_admin_password()
    if not expected:
        return False, "Admin password not configured."

    if password != expected:
        return False, "Incorrect password."

    create_admin_session()
    return True, "Success"
