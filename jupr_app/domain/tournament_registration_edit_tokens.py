from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from jupr_app.config import get_registration_edit_token_secret

TOKEN_VERSION = "v1"


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + padding).encode("ascii"))


def _normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def _email_hash(email: str) -> str:
    return hashlib.sha256(_normalize_email(email).encode("utf-8")).hexdigest()


def _secret_bytes(secret: str | None) -> bytes:
    value = secret if secret is not None else get_registration_edit_token_secret()
    if not str(value or "").strip():
        raise ValueError("Registration edit token secret is required.")
    return str(value).encode("utf-8")


def _sign(payload_b64: str, secret: str | None) -> str:
    digest = hmac.new(_secret_bytes(secret), payload_b64.encode("ascii"), hashlib.sha256).digest()
    return _b64url_encode(digest)


def build_registration_edit_token(
    *,
    tournament_id: str,
    registration_id: str,
    email: str,
    expires_in_seconds: int = 48 * 60 * 60,
    now: int | None = None,
    secret: str | None = None,
) -> str:
    issued_at = int(time.time() if now is None else now)
    payload: dict[str, Any] = {
        "version": TOKEN_VERSION,
        "tournament_id": str(tournament_id),
        "registration_id": str(registration_id),
        "exp": issued_at + int(expires_in_seconds),
        "email_hash": _email_hash(email),
    }
    payload_b64 = _b64url_encode(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    return f"{payload_b64}.{_sign(payload_b64, secret)}"


def verify_registration_edit_token(
    token: str,
    *,
    expected_tournament_id: str | None = None,
    expected_registration_id: str | None = None,
    expected_email: str | None = None,
    now: int | None = None,
    secret: str | None = None,
) -> dict[str, str]:
    try:
        payload_b64, signature = str(token or "").split(".", 1)
    except ValueError as exc:
        raise ValueError("Invalid registration edit link.") from exc
    expected_signature = _sign(payload_b64, secret)
    if not hmac.compare_digest(signature, expected_signature):
        raise ValueError("Invalid registration edit link.")
    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid registration edit link.") from exc
    if payload.get("version") != TOKEN_VERSION:
        raise ValueError("Invalid registration edit link.")
    current = int(time.time() if now is None else now)
    if int(payload.get("exp") or 0) < current:
        raise ValueError("Registration edit link has expired.")
    if expected_tournament_id is not None and str(payload.get("tournament_id")) != str(expected_tournament_id):
        raise ValueError("Registration edit link is for a different tournament.")
    if expected_registration_id is not None and str(payload.get("registration_id")) != str(expected_registration_id):
        raise ValueError("Registration edit link is for a different registration.")
    if expected_email is not None and str(payload.get("email_hash")) != _email_hash(expected_email):
        raise ValueError("Registration edit link is for a different email.")
    return {
        "version": str(payload.get("version")),
        "tournament_id": str(payload.get("tournament_id")),
        "registration_id": str(payload.get("registration_id")),
        "exp": str(payload.get("exp")),
        "email_hash": str(payload.get("email_hash")),
    }


def registration_edit_email_hash(email: str) -> str:
    return _email_hash(email)
