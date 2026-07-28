from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

TOKEN_VERSION = "v1"
SECRET_ENV = "JUPR_TOURNAMENT_TEAM_INVITATION_SECRET"
FALLBACK_SECRET_ENV = "JUPR_REGISTRATION_EDIT_SECRET"
MIN_SECRET_BYTES = 32


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode((value + "=" * (-len(value) % 4)).encode("ascii"))


def tournament_team_invitation_email_hash(email: str) -> str:
    return hashlib.sha256(str(email or "").strip().lower().encode("utf-8")).hexdigest()


def tournament_team_invitation_token_hash(token: str) -> str:
    return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()


def _secret(secret: str | None) -> bytes:
    if secret is not None:
        value = str(secret)
        source = SECRET_ENV
    else:
        dedicated = os.getenv(SECRET_ENV, "")
        if dedicated:
            value = dedicated
            source = SECRET_ENV
        else:
            value = os.getenv(FALLBACK_SECRET_ENV, "")
            source = FALLBACK_SECRET_ENV
    raw = value.encode("utf-8")
    if len(raw) < MIN_SECRET_BYTES:
        raise ValueError(
            f"{source} must be a dedicated secret of at least "
            f"{MIN_SECRET_BYTES} bytes."
        )
    return raw


def _sign(payload: str, secret: str | None) -> str:
    return _encode(hmac.new(_secret(secret), payload.encode("ascii"), hashlib.sha256).digest())


def build_tournament_team_invitation_token(
    *,
    tournament_id: str,
    team_id: str,
    member_id: str,
    invited_email: str,
    invitation_version: int,
    expires_in_seconds: int = 7 * 24 * 60 * 60,
    now: int | None = None,
    secret: str | None = None,
) -> str:
    issued_at = int(time.time() if now is None else now)
    payload: dict[str, Any] = {
        "version": TOKEN_VERSION,
        "tournament_id": str(tournament_id),
        "team_id": str(team_id),
        "member_id": str(member_id),
        "invitation_version": int(invitation_version),
        "email_hash": tournament_team_invitation_email_hash(invited_email),
        "exp": issued_at + int(expires_in_seconds),
    }
    encoded = _encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    return f"{encoded}.{_sign(encoded, secret)}"


def verify_tournament_team_invitation_token(
    token: str,
    *,
    expected_tournament_id: str | None = None,
    expected_team_id: str | None = None,
    expected_member_id: str | None = None,
    expected_invited_email: str | None = None,
    expected_invitation_version: int | None = None,
    now: int | None = None,
    secret: str | None = None,
) -> dict[str, Any]:
    try:
        encoded, signature = str(token or "").split(".", 1)
    except ValueError as exc:
        raise ValueError("Invalid team invitation.") from exc
    if not hmac.compare_digest(signature, _sign(encoded, secret)):
        raise ValueError("Invalid team invitation.")
    try:
        payload = json.loads(_decode(encoded).decode("utf-8"))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid team invitation.") from exc
    if payload.get("version") != TOKEN_VERSION:
        raise ValueError("Invalid team invitation.")
    if int(payload.get("exp") or 0) < int(time.time() if now is None else now):
        raise ValueError("Team invitation has expired.")
    comparisons = (
        ("tournament_id", expected_tournament_id, "different tournament"),
        ("team_id", expected_team_id, "does not match this team"),
        ("member_id", expected_member_id, "does not match this invitation"),
    )
    for key, expected, message in comparisons:
        if expected is not None and str(payload.get(key)) != str(expected):
            raise ValueError(f"Team invitation is for a {message}.")
    if (
        expected_invited_email is not None
        and str(payload.get("email_hash"))
        != tournament_team_invitation_email_hash(expected_invited_email)
    ):
        raise ValueError("Team invitation is for a different email.")
    if (
        expected_invitation_version is not None
        and int(payload.get("invitation_version") or 0)
        != int(expected_invitation_version)
    ):
        raise ValueError("Team invitation was replaced by a newer invitation.")
    return {
        "version": str(payload["version"]),
        "tournament_id": str(payload["tournament_id"]),
        "team_id": str(payload["team_id"]),
        "member_id": str(payload["member_id"]),
        "invitation_version": int(payload["invitation_version"]),
        "email_hash": str(payload["email_hash"]),
        "exp": int(payload["exp"]),
    }


build_team_invitation_token = build_tournament_team_invitation_token
verify_team_invitation_token = verify_tournament_team_invitation_token
team_invitation_email_hash = tournament_team_invitation_email_hash
team_invitation_token_hash = tournament_team_invitation_token_hash
