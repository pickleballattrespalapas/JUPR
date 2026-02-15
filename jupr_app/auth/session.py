import base64
import hashlib
import hmac
import json
import os
import time

import streamlit as st

SESSION_TTL_SECONDS = 60 * 60 * 24 * 14  # 14 days


def _get_secret():
    try:
        return st.secrets["supabase"]["admin_session_secret"]
    except Exception:
        return os.getenv("SUPABASE__ADMIN_SESSION_SECRET", "")


def _sign(data: str) -> str:
    return hmac.new(_get_secret().encode(), data.encode(), hashlib.sha256).hexdigest()


def create_session_token(admin_email: str) -> str:
    payload = {"email": admin_email, "exp": int(time.time()) + SESSION_TTL_SECONDS}

    raw = json.dumps(payload, separators=(",", ":"))
    signature = _sign(raw)

    token = base64.urlsafe_b64encode(f"{raw}|{signature}".encode()).decode()

    return token


def verify_session_token(token: str):
    try:
        decoded = base64.urlsafe_b64decode(token.encode()).decode()
        raw, signature = decoded.rsplit("|", 1)

        expected_sig = _sign(raw)

        if not hmac.compare_digest(signature, expected_sig):
            return None

        payload = json.loads(raw)

        if payload["exp"] < time.time():
            return None

        return payload

    except Exception:
        return None
