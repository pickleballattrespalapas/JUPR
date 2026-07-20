from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


TOKEN_VERSION = "v1"
TOKEN_PURPOSE = "tournament_registration_confirmation"
DEFAULT_LIFETIME_SECONDS = 30 * 24 * 60 * 60


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    padding = "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode((text + padding).encode("ascii"))


def _email_hash(email: str) -> str:
    normalized = str(email or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _secret_bytes(secret: str | None) -> bytes:
    if secret is None:
        from jupr_app.config import get_registration_confirmation_token_secret

        secret = get_registration_confirmation_token_secret()
    value = str(secret or "").strip()
    if not value:
        raise ValueError("Registration confirmation token secret is required.")
    return value.encode("utf-8")


def _signature(payload_b64: str, secret: str | None) -> str:
    message = f"{TOKEN_PURPOSE}.{payload_b64}".encode("ascii")
    digest = hmac.new(_secret_bytes(secret), message, hashlib.sha256).digest()
    return _b64url_encode(digest)


def build_registration_confirmation_token(
    *,
    tournament_id: str,
    registration_id: str,
    email: str,
    expires_in_seconds: int = DEFAULT_LIFETIME_SECONDS,
    now: int | None = None,
    secret: str | None = None,
) -> str:
    issued_at = int(time.time() if now is None else now)
    payload: dict[str, Any] = {
        "version": TOKEN_VERSION,
        "purpose": TOKEN_PURPOSE,
        "tournament_id": str(tournament_id),
        "registration_id": str(registration_id),
        "email_hash": _email_hash(email),
        "iat": issued_at,
        "exp": issued_at + int(expires_in_seconds),
    }
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return f"{payload_b64}.{_signature(payload_b64, secret)}"


def verify_registration_confirmation_token(
    token: str,
    *,
    expected_tournament_id: str | None = None,
    expected_registration_id: str | None = None,
    expected_email: str | None = None,
    now: int | None = None,
    secret: str | None = None,
) -> dict[str, str]:
    try:
        payload_b64, supplied_signature = str(token or "").split(".", 1)
    except ValueError as exc:
        raise ValueError("Invalid registration confirmation link.") from exc

    if not hmac.compare_digest(supplied_signature, _signature(payload_b64, secret)):
        raise ValueError("Invalid registration confirmation link.")
    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid registration confirmation link.") from exc

    if payload.get("version") != TOKEN_VERSION or payload.get("purpose") != TOKEN_PURPOSE:
        raise ValueError("Invalid registration confirmation link.")
    current = int(time.time() if now is None else now)
    if int(payload.get("exp") or 0) < current:
        raise ValueError("Registration confirmation link has expired.")
    if expected_tournament_id is not None and str(payload.get("tournament_id")) != str(expected_tournament_id):
        raise ValueError("Invalid registration confirmation link.")
    if expected_registration_id is not None and str(payload.get("registration_id")) != str(expected_registration_id):
        raise ValueError("Invalid registration confirmation link.")
    if expected_email is not None and str(payload.get("email_hash")) != _email_hash(expected_email):
        raise ValueError("Invalid registration confirmation link.")

    return {
        "version": str(payload.get("version")),
        "purpose": str(payload.get("purpose")),
        "tournament_id": str(payload.get("tournament_id")),
        "registration_id": str(payload.get("registration_id")),
        "email_hash": str(payload.get("email_hash")),
        "iat": str(payload.get("iat")),
        "exp": str(payload.get("exp")),
    }
