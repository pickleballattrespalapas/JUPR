#!/usr/bin/env python3
"""Prepare and end refreshable, candidate-bound parity staging Auth sessions.

The workflow uses Supabase Admin only to identify the one already-bound staging
operator and generate a magic-link token without sending email.  It never
creates or deletes an Auth user and never mutates an admin role assignment or
business table. Generating and verifying the link and ending its refreshable
session intentionally change staging Auth token, session, and audit state.
Supabase logout ends the refreshable session but does not guarantee immediate
rejection of its access JWT; the workflow constrains that JWT to at most one
hour and treats it as potentially usable until its ``exp`` claim. Sensitive
response bodies and credential values are excluded from errors and evidence.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID

EXPECTED_SUPABASE_ORIGIN = "https://sijpxjxvdtrehmqvirfi.supabase.co"
EXPECTED_SUPABASE_PROJECT_REF = "sijpxjxvdtrehmqvirfi"
EXPECTED_SUPABASE_ISSUER = f"{EXPECTED_SUPABASE_ORIGIN}/auth/v1"
EXPECTED_AUTH_AUDIENCE = "authenticated"
EXPECTED_API_ORIGIN = "https://juprleagues-api-staging.fly.dev"
EXPECTED_CLUB_ID = "tres_palapas"
EXPECTED_RECAP_ID = "98000000-0000-4000-8000-000000000002"
EXPECTED_RECAP_WEEK_START = "2099-01-05"
EXPECTED_TOURNAMENT_ID = "93000000-0000-4000-8000-000000000002"
EXPECTED_DRAW_ID = "94000000-0000-4000-8000-000000000001"
ELIGIBLE_ROLES = frozenset({"club_owner", "super_admin"})
SUPPORTED_MODES = (
    "public-intake-auth",
    "admin-read-export",
    "match-rating-writes",
)
REQUIRED_PERMISSIONS = frozenset(
    {
        "manage_matches",
        "manage_subscriptions",
        "manage_tournaments",
    }
)
JWT_RE = re.compile(
    r"^[A-Za-z0-9_-]{1,4096}\.[A-Za-z0-9_-]{1,4096}\.[A-Za-z0-9_-]{1,4096}$"
)
TOKEN_HASH_RE = re.compile(r"^[A-Za-z0-9_-]{32,1024}$")
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,189}$")
POSITIVE_INTEGER_RE = re.compile(r"^[1-9][0-9]{0,18}$")
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_ACCESS_JWT_LIFETIME_SECONDS = 60 * 60
ACCESS_JWT_CLOCK_SKEW_SECONDS = 60
PREPARATION_REPORT_NAME = "parity-session-preparation.json"
CLEANUP_REPORT_NAME = "parity-session-cleanup.json"

Transport = Callable[[Request], tuple[int, bytes]]


class SessionPreparationError(RuntimeError):
    """A sanitized, operator-safe parity-session operation failure."""


class _RefuseRedirects(HTTPRedirectHandler):
    """Never forward staging credentials through an HTTP redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


@dataclass(frozen=True)
class _AdminAssignment:
    assignment_id: str
    user_id: str
    email: str
    role: str


_STAGING_OPENER = build_opener(_RefuseRedirects())


def _required_env(env: Mapping[str, str], name: str) -> str:
    value = str(env.get(name) or "").strip()
    if not value:
        raise SessionPreparationError(f"Required environment value {name} is missing.")
    if "\r" in value or "\n" in value:
        raise SessionPreparationError(f"Required environment value {name} is invalid.")
    return value


def _default_transport(request: Request) -> tuple[int, bytes]:
    with _STAGING_OPENER.open(request, timeout=20) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        status = int(getattr(response, "status", response.getcode()))
    return status, body


def _request_json_value(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any] | None = None,
    transport: Transport,
    operation: str,
) -> Any:
    body = None
    request_headers = dict(headers)
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=request_headers, method=method)

    try:
        status, raw = transport(request)
    except HTTPError as exc:
        raise SessionPreparationError(
            f"{operation} failed with HTTP {int(exc.code)}."
        ) from None
    except (URLError, OSError):
        raise SessionPreparationError(f"{operation} could not reach staging.") from None
    except Exception:  # noqa: BLE001 - keep transport failures sanitized
        raise SessionPreparationError(f"{operation} failed unexpectedly.") from None

    if not 200 <= int(status) < 300:
        raise SessionPreparationError(f"{operation} failed with HTTP {int(status)}.")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise SessionPreparationError(f"{operation} returned an oversized response.")
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SessionPreparationError(
            f"{operation} returned an invalid response."
        ) from None


def _request_json(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any] | None = None,
    transport: Transport,
    operation: str,
) -> dict[str, Any]:
    decoded = _request_json_value(
        method=method,
        url=url,
        headers=headers,
        payload=payload,
        transport=transport,
        operation=operation,
    )
    if not isinstance(decoded, dict):
        raise SessionPreparationError(f"{operation} returned an invalid response.")
    return decoded


def _request_logout(
    *,
    url: str,
    headers: Mapping[str, str],
    transport: Transport,
) -> str:
    operation = "Supabase staging session cleanup"
    request = Request(url, headers=dict(headers), method="POST")
    try:
        status, raw = transport(request)
    except HTTPError as exc:
        status = int(exc.code)
        try:
            raw = exc.read(MAX_RESPONSE_BYTES + 1)
        except (OSError, AttributeError):
            raw = b""
    except (URLError, OSError):
        raise SessionPreparationError(f"{operation} could not reach staging.") from None
    except Exception:  # noqa: BLE001 - keep transport failures sanitized
        raise SessionPreparationError(f"{operation} failed unexpectedly.") from None

    if len(raw) > MAX_RESPONSE_BYTES:
        raise SessionPreparationError(f"{operation} returned an oversized response.")
    if 200 <= int(status) < 300:
        return "ended"

    error_code = ""
    if int(status) in {401, 403}:
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = None
        if isinstance(payload, Mapping):
            # GoTrue response versions vary: some expose the symbolic value as
            # ``error_code`` with numeric ``code``; newer versioned responses
            # may expose it as string ``code``. Accept both fail-closed shapes.
            raw_error_code = payload.get("error_code")
            if not isinstance(raw_error_code, str):
                legacy_code = payload.get("code")
                raw_error_code = legacy_code if isinstance(legacy_code, str) else ""
            error_code = raw_error_code.strip().lower()
    if error_code in {"session_expired", "session_not_found"}:
        return "already_inactive"
    raise SessionPreparationError(f"{operation} failed with HTTP {int(status)}.")


def _service_headers(service_role_key: str) -> dict[str, str]:
    return {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Accept": "application/json",
    }


def _anon_headers(anon_key: str) -> dict[str, str]:
    return {
        "apikey": anon_key,
        "Authorization": f"Bearer {anon_key}",
        "Accept": "application/json",
    }


def _validated_uuid(value: object, *, operation: str) -> str:
    raw = str(value or "").strip().lower()
    try:
        parsed = UUID(raw)
    except (ValueError, AttributeError):
        raise SessionPreparationError(
            f"{operation} returned an invalid response."
        ) from None
    if str(parsed) != raw:
        raise SessionPreparationError(f"{operation} returned an invalid response.")
    return raw


def _validated_email(value: object, *, operation: str) -> str:
    raw = str(value or "")
    normalized = raw.strip().casefold()
    if (
        raw != normalized
        or len(normalized) > 254
        or EMAIL_RE.fullmatch(normalized) is None
    ):
        raise SessionPreparationError(f"{operation} returned an invalid response.")
    return normalized


def _validate_role_assignment(payload: object) -> _AdminAssignment:
    operation = "Staging admin assignment lookup"
    if not isinstance(payload, list) or len(payload) != 1:
        raise SessionPreparationError(
            "Staging must have exactly one eligible bound admin assignment."
        )
    row = payload[0]
    if not isinstance(row, Mapping):
        raise SessionPreparationError(f"{operation} returned an invalid response.")

    assignment_id = str(row.get("id") or "").strip()
    role = str(row.get("role") or "").strip().lower()
    if (
        POSITIVE_INTEGER_RE.fullmatch(assignment_id) is None
        or str(row.get("club_id") or "").strip() != EXPECTED_CLUB_ID
        or role not in ELIGIBLE_ROLES
    ):
        raise SessionPreparationError(f"{operation} returned an invalid response.")
    return _AdminAssignment(
        assignment_id=assignment_id,
        user_id=_validated_uuid(row.get("user_id"), operation=operation),
        email=_validated_email(row.get("email"), operation=operation),
        role=role,
    )


def _validate_auth_admin_user(
    payload: Mapping[str, Any],
    assignment: _AdminAssignment,
) -> None:
    operation = "Supabase staging admin identity lookup"
    returned_user_id = _validated_uuid(payload.get("id"), operation=operation)
    returned_email = _validated_email(payload.get("email"), operation=operation)
    if (
        returned_user_id != assignment.user_id
        or returned_email != assignment.email
        or not payload.get("email_confirmed_at")
        or payload.get("deleted_at")
        or payload.get("is_anonymous") is True
    ):
        raise SessionPreparationError(
            "Supabase staging admin identity did not match the bound assignment."
        )


def _validate_generate_link(
    payload: Mapping[str, Any],
    assignment: _AdminAssignment,
) -> tuple[str, str]:
    operation = "Supabase staging session link generation"
    token_hash = str(payload.get("hashed_token") or "").strip()
    verification_type = str(payload.get("verification_type") or "").strip().lower()
    if TOKEN_HASH_RE.fullmatch(token_hash) is None or verification_type not in {
        "email",
        "magiclink",
    }:
        raise SessionPreparationError(f"{operation} returned an invalid response.")
    # The raw GoTrue endpoint embeds the Auth user in GenerateLinkResponse, so
    # identity fields are top-level. The JavaScript SDK reshapes this response
    # into ``{ user, properties }``; this dependency-free workflow intentionally
    # validates the raw endpoint contract instead.
    returned_user_id = _validated_uuid(payload.get("id"), operation=operation)
    returned_email = _validated_email(payload.get("email"), operation=operation)
    if returned_user_id != assignment.user_id or returned_email != assignment.email:
        raise SessionPreparationError(
            "Supabase staging session link returned an unexpected user identity."
        )
    return token_hash, verification_type


def _validate_access_token(
    payload: Mapping[str, Any],
    assignment: _AdminAssignment,
) -> str:
    token = _candidate_access_token(payload)
    if token is None:
        raise SessionPreparationError(
            "Supabase authentication returned an invalid access token."
        )
    if payload.get("token_type") != "bearer":
        raise SessionPreparationError(
            "Supabase authentication returned an invalid token type."
        )
    user = payload.get("user")
    if not isinstance(user, Mapping):
        raise SessionPreparationError(
            "Supabase authentication returned an unexpected user identity."
        )
    returned_user_id = _validated_uuid(
        user.get("id"),
        operation="Supabase staging session verification",
    )
    returned_email = _validated_email(
        user.get("email"),
        operation="Supabase staging session verification",
    )
    if returned_user_id != assignment.user_id or returned_email != assignment.email:
        raise SessionPreparationError(
            "Supabase authentication returned an unexpected user identity."
        )

    expires_in = payload.get("expires_in")
    if (
        type(expires_in) is not int
        or not 0 < expires_in <= MAX_ACCESS_JWT_LIFETIME_SECONDS
    ):
        raise SessionPreparationError(
            "Supabase authentication returned an invalid access-token lifetime."
        )

    claims = _decode_unverified_jwt_claims(token)
    session_id = claims.get("session_id")
    try:
        parsed_session_id = UUID(session_id) if isinstance(session_id, str) else None
    except ValueError:
        parsed_session_id = None
    audience = claims.get("aud")
    audience_matches = audience == EXPECTED_AUTH_AUDIENCE or (
        isinstance(audience, list)
        and bool(audience)
        and all(isinstance(item, str) for item in audience)
        and EXPECTED_AUTH_AUDIENCE in audience
    )
    if (
        claims.get("sub") != assignment.user_id
        or claims.get("iss") != EXPECTED_SUPABASE_ISSUER
        or not audience_matches
        or claims.get("email") != assignment.email
        or parsed_session_id is None
        or str(parsed_session_id) != session_id
    ):
        raise SessionPreparationError(
            "Supabase authentication returned invalid access-token claims."
        )

    issued_at = claims.get("iat")
    expires_at = claims.get("exp")
    response_expires_at = payload.get("expires_at")
    now = int(time.time())
    if type(issued_at) is not int or type(expires_at) is not int:
        raise SessionPreparationError(
            "Supabase authentication returned an invalid access-token lifetime."
        )
    if "expires_at" in payload and (
        type(response_expires_at) is not int or response_expires_at != expires_at
    ):
        raise SessionPreparationError(
            "Supabase authentication returned an invalid access-token lifetime."
        )
    claimed_lifetime = expires_at - issued_at
    if (
        not 0 < claimed_lifetime <= MAX_ACCESS_JWT_LIFETIME_SECONDS
        or issued_at > now + ACCESS_JWT_CLOCK_SKEW_SECONDS
        or expires_at <= now
        or abs(expires_in - claimed_lifetime) > ACCESS_JWT_CLOCK_SKEW_SECONDS
    ):
        raise SessionPreparationError(
            "Supabase authentication returned an invalid access-token lifetime."
        )
    return token


def _decode_unverified_jwt_claims(token: str) -> Mapping[str, Any]:
    """Decode bounded JWT claims for preflight; authenticity is checked by FastAPI."""

    payload_segment = token.split(".", 2)[1]
    padding = "=" * (-len(payload_segment) % 4)
    try:
        decoded = base64.b64decode(
            (payload_segment + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        claims = json.loads(decoded.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError, binascii.Error):
        raise SessionPreparationError(
            "Supabase authentication returned invalid access-token claims."
        ) from None
    if not isinstance(claims, Mapping):
        raise SessionPreparationError(
            "Supabase authentication returned invalid access-token claims."
        )
    return claims


def _candidate_access_token(payload: Mapping[str, Any]) -> str | None:
    """Return only a syntactically cleanup-eligible JWT candidate."""

    token = str(payload.get("access_token") or "").strip()
    if not 100 <= len(token) <= 8192 or JWT_RE.fullmatch(token) is None:
        return None
    return token


def _validate_capabilities(
    payload: Mapping[str, Any],
    assignment: _AdminAssignment,
) -> None:
    if payload.get("authorized") is not True:
        raise SessionPreparationError("Staging admin capability validation was denied.")
    user = payload.get("user")
    returned_email = (
        str(user.get("email") or "").strip().casefold()
        if isinstance(user, Mapping)
        else ""
    )
    if returned_email != assignment.email:
        raise SessionPreparationError(
            "Staging admin capability validation returned an unexpected identity."
        )
    if str(payload.get("requested_club_id") or "").strip() != EXPECTED_CLUB_ID:
        raise SessionPreparationError(
            "Staging admin capability validation returned an unexpected club."
        )

    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or len(assignments) != 1:
        raise SessionPreparationError(
            "Staging admin capability validation returned unexpected assignments."
        )
    capability_assignment = assignments[0]
    if not isinstance(capability_assignment, Mapping):
        raise SessionPreparationError(
            "Staging admin capability validation returned unexpected assignments."
        )
    if (
        str(capability_assignment.get("club_id") or "").strip() != EXPECTED_CLUB_ID
        or str(capability_assignment.get("role") or "").strip().lower()
        != assignment.role
    ):
        raise SessionPreparationError(
            "Staging admin capability validation returned an unexpected assignment."
        )
    raw_permissions = capability_assignment.get("permissions")
    if not isinstance(raw_permissions, list):
        raise SessionPreparationError(
            "Staging admin capability validation returned invalid permissions."
        )
    permissions = {
        str(permission).strip()
        for permission in raw_permissions
        if str(permission).strip()
    }
    missing = sorted(REQUIRED_PERMISSIONS - permissions)
    if missing:
        raise SessionPreparationError(
            "Staging admin capability validation is missing required permissions: "
            + ", ".join(missing)
            + "."
        )


def _validate_recap_fixture(payload: Mapping[str, Any]) -> None:
    if payload.get("ok") is not True or payload.get("mode") != "weekly_recap_list":
        raise SessionPreparationError("The staging recap fixture list is unavailable.")
    recaps = payload.get("recaps")
    if not isinstance(recaps, list):
        raise SessionPreparationError("The staging recap fixture list is invalid.")
    matching = [
        recap
        for recap in recaps
        if isinstance(recap, Mapping)
        and str(recap.get("id") or "").strip() == EXPECTED_RECAP_ID
    ]
    if len(matching) != 1:
        raise SessionPreparationError("The exact staging recap fixture is unavailable.")
    recap = matching[0]
    if (
        str(recap.get("week_start") or "").strip() != EXPECTED_RECAP_WEEK_START
        or str(recap.get("status") or "").strip().lower() != "draft"
    ):
        raise SessionPreparationError("The exact staging recap fixture is invalid.")


def _validate_tournament_list(payload: Mapping[str, Any]) -> None:
    if payload.get("ok") is not True or payload.get("mode") != "tournament_admin_list":
        raise SessionPreparationError(
            "The staging tournament fixture list is unavailable."
        )
    tournaments = payload.get("tournaments")
    if not isinstance(tournaments, list):
        raise SessionPreparationError("The staging tournament fixture list is invalid.")
    matching = [
        tournament
        for tournament in tournaments
        if isinstance(tournament, Mapping)
        and str(tournament.get("id") or "").strip() == EXPECTED_TOURNAMENT_ID
    ]
    if len(matching) != 1:
        raise SessionPreparationError(
            "The exact staging tournament fixture is unavailable."
        )


def _validate_tournament_snapshot(payload: Mapping[str, Any]) -> None:
    tournament = payload.get("tournament")
    draws = payload.get("draws")
    if (
        payload.get("ok") is not True
        or payload.get("mode") != "tournament_ops_snapshot"
        or not isinstance(tournament, Mapping)
        or str(tournament.get("id") or "").strip() != EXPECTED_TOURNAMENT_ID
        or str(payload.get("draw_id") or "").strip() != EXPECTED_DRAW_ID
        or payload.get("state_ready") is not True
        or not isinstance(draws, list)
    ):
        raise SessionPreparationError(
            "The exact staging tournament snapshot is invalid."
        )
    matching = [
        draw
        for draw in draws
        if isinstance(draw, Mapping)
        and str(draw.get("id") or "").strip() == EXPECTED_DRAW_ID
        and str(draw.get("tournament_id") or "").strip() == EXPECTED_TOURNAMENT_ID
    ]
    if len(matching) != 1:
        raise SessionPreparationError("The exact staging draw fixture is unavailable.")


def _append_github_env(path: Path, values: Mapping[str, str]) -> None:
    for name, value in values.items():
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
            raise SessionPreparationError(
                "An internal GitHub environment name is invalid."
            )
        if not value or "\r" in value or "\n" in value:
            raise SessionPreparationError(
                f"Prepared GitHub environment value {name} is invalid."
            )
    try:
        with path.open("a", encoding="utf-8") as env_file:
            for name, value in values.items():
                env_file.write(f"{name}={value}\n")
    except OSError:
        raise SessionPreparationError(
            "Could not append the prepared values to GITHUB_ENV."
        ) from None


def _clear_github_env(path: Path) -> None:
    names = (
        "STAGING_ADMIN_BEARER_TOKEN",
        "JUPR_STAGING_ADMIN_ACCESS_TOKEN",
        "STAGING_ADMIN_EMAIL",
        "JUPR_STAGING_ADMIN_EMAIL",
        "JUPR_REAL_AUTH_EXPECTED_ROLE",
    )
    try:
        with path.open("a", encoding="utf-8") as env_file:
            for name in names:
                env_file.write(f"{name}=\n")
    except OSError:
        raise SessionPreparationError(
            "Could not clear the prepared values from GITHUB_ENV."
        ) from None


def _write_report(
    report_dir: Path | None,
    filename: str,
    payload: Mapping[str, Any],
) -> None:
    if report_dir is None:
        return
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / filename
        report_path.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        raise SessionPreparationError(
            "Could not write the sanitized parity session evidence report."
        ) from None


def _preparation_report(
    mode: str,
    *,
    status: str,
    prepared: Mapping[str, str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "operation": "prepare_parity_staging_session",
        "mode": mode,
        "status": status,
        "supabase_project_ref": EXPECTED_SUPABASE_PROJECT_REF,
        "api_origin": EXPECTED_API_ORIGIN,
        "club_id": EXPECTED_CLUB_ID,
    }
    if prepared is not None:
        email = prepared["STAGING_ADMIN_EMAIL"]
        report["identity"] = {
            "email_sha256": hashlib.sha256(email.encode("utf-8")).hexdigest(),
            "role": prepared["JUPR_REAL_AUTH_EXPECTED_ROLE"],
        }
        report["validated_fixture_sets"] = (
            ["weekly_recap", "tournament", "tournament_draw"]
            if mode == "admin-read-export"
            else []
        )
    if error is not None:
        report["error"] = error
    return report


def _lookup_assignment(
    *,
    supabase_origin: str,
    service_role_key: str,
    transport: Transport,
) -> _AdminAssignment:
    filters = urlencode(
        {
            "select": "id,club_id,email,role,user_id",
            "club_id": f"eq.{EXPECTED_CLUB_ID}",
            "role": "in.(club_owner,super_admin)",
            "user_id": "not.is.null",
            "order": "id.asc",
            "limit": "2",
        }
    )
    payload = _request_json_value(
        method="GET",
        url=f"{supabase_origin}/rest/v1/admin_role_assignments?{filters}",
        headers=_service_headers(service_role_key),
        transport=transport,
        operation="Staging admin assignment lookup",
    )
    return _validate_role_assignment(payload)


def _prepare_staging_session(
    mode: str,
    *,
    env: Mapping[str, str],
    transport: Transport,
) -> dict[str, str]:
    if mode not in SUPPORTED_MODES:
        raise SessionPreparationError(f"Unsupported parity evidence mode: {mode}.")
    supabase_origin = _required_env(env, "STAGING_SUPABASE_URL").rstrip("/")
    if supabase_origin != EXPECTED_SUPABASE_ORIGIN:
        raise SessionPreparationError(
            "STAGING_SUPABASE_URL does not identify the permitted staging project."
        )
    api_origin = _required_env(env, "STAGING_API_BASE_URL").rstrip("/")
    if api_origin != EXPECTED_API_ORIGIN:
        raise SessionPreparationError(
            "STAGING_API_BASE_URL does not identify the permitted staging API."
        )
    anon_key = _required_env(env, "STAGING_SUPABASE_ANON_KEY")
    service_role_key = _required_env(env, "STAGING_SUPABASE_SERVICE_ROLE_KEY")
    github_env_path = Path(_required_env(env, "GITHUB_ENV"))

    assignment = _lookup_assignment(
        supabase_origin=supabase_origin,
        service_role_key=service_role_key,
        transport=transport,
    )

    admin_user = _request_json(
        method="GET",
        url=f"{supabase_origin}/auth/v1/admin/users/{assignment.user_id}",
        headers=_service_headers(service_role_key),
        transport=transport,
        operation="Supabase staging admin identity lookup",
    )
    _validate_auth_admin_user(admin_user, assignment)

    generated_link = _request_json(
        method="POST",
        url=f"{supabase_origin}/auth/v1/admin/generate_link",
        headers=_service_headers(service_role_key),
        payload={"type": "magiclink", "email": assignment.email},
        transport=transport,
        operation="Supabase staging session link generation",
    )
    token_hash, verification_type = _validate_generate_link(
        generated_link,
        assignment,
    )

    auth_payload = _request_json(
        method="POST",
        url=f"{supabase_origin}/auth/v1/verify",
        headers=_anon_headers(anon_key),
        payload={"token_hash": token_hash, "type": verification_type},
        transport=transport,
        operation="Supabase staging session verification",
    )
    candidate_token = _candidate_access_token(auth_payload)
    if candidate_token is not None:
        # Export and mask any syntactically usable bearer before validating its
        # returned identity and claims. If validation rejects it, cleanup can
        # still end the corresponding refresh session. The access JWT itself may
        # remain usable until its exp claim.
        print(f"::add-mask::{candidate_token}")
        try:
            _append_github_env(
                github_env_path,
                {"STAGING_ADMIN_BEARER_TOKEN": candidate_token},
            )
        except SessionPreparationError as export_error:
            try:
                _request_logout(
                    url=f"{supabase_origin}/auth/v1/logout?scope=local",
                    headers={
                        "apikey": anon_key,
                        "Authorization": f"Bearer {candidate_token}",
                        "Accept": "application/json",
                    },
                    transport=transport,
                )
            except SessionPreparationError:
                raise SessionPreparationError(
                    f"{export_error} Immediate session cleanup also failed."
                ) from None
            raise

    try:
        token = _validate_access_token(auth_payload, assignment)
    except SessionPreparationError as validation_error:
        if candidate_token is not None:
            try:
                _request_logout(
                    url=f"{supabase_origin}/auth/v1/logout?scope=local",
                    headers={
                        "apikey": anon_key,
                        "Authorization": f"Bearer {candidate_token}",
                        "Accept": "application/json",
                    },
                    transport=transport,
                )
                _clear_github_env(github_env_path)
            except SessionPreparationError:
                # The masked token remains in GITHUB_ENV so the workflow's
                # mandatory cleanup step can retry before artifact retention.
                raise SessionPreparationError(
                    f"{validation_error} Immediate session cleanup also failed."
                ) from None
        raise

    prepared = {
        "STAGING_ADMIN_BEARER_TOKEN": token,
        "JUPR_STAGING_ADMIN_ACCESS_TOKEN": token,
        "STAGING_ADMIN_EMAIL": assignment.email,
        "JUPR_STAGING_ADMIN_EMAIL": assignment.email,
        "JUPR_REAL_AUTH_EXPECTED_ROLE": assignment.role,
    }

    # Export the cleanup-eligible bearer before downstream API/fixture checks.
    # If one fails, the workflow's always-running cleanup can still end the
    # refresh session; the access JWT may remain usable until its exp claim.
    print(f"::add-mask::{assignment.email}")
    try:
        _append_github_env(
            github_env_path,
            {
                name: value
                for name, value in prepared.items()
                if name != "STAGING_ADMIN_BEARER_TOKEN"
            },
        )
    except SessionPreparationError as export_error:
        try:
            _request_logout(
                url=f"{supabase_origin}/auth/v1/logout?scope=local",
                headers={
                    "apikey": anon_key,
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                },
                transport=transport,
            )
        except SessionPreparationError:
            raise SessionPreparationError(
                f"{export_error} Immediate session cleanup also failed."
            ) from None
        raise

    bearer_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }

    capabilities = _request_json(
        method="GET",
        url=(
            f"{api_origin}/admin/auth/capabilities?"
            + urlencode({"club_id": EXPECTED_CLUB_ID})
        ),
        headers=bearer_headers,
        transport=transport,
        operation="Staging admin capability validation",
    )
    _validate_capabilities(capabilities, assignment)

    if mode == "admin-read-export":
        recap_list = _request_json(
            method="GET",
            url=(
                f"{api_origin}/admin/clubs/{EXPECTED_CLUB_ID}"
                "/weekly-recap/recaps?limit=200"
            ),
            headers=bearer_headers,
            transport=transport,
            operation="Staging recap fixture validation",
        )
        _validate_recap_fixture(recap_list)

        tournament_list = _request_json(
            method="GET",
            url=(
                f"{api_origin}/admin/clubs/{EXPECTED_CLUB_ID}"
                "/tournaments/admin/ops/tournaments"
            ),
            headers=bearer_headers,
            transport=transport,
            operation="Staging tournament fixture validation",
        )
        _validate_tournament_list(tournament_list)

        snapshot = _request_json(
            method="GET",
            url=(
                f"{api_origin}/admin/clubs/{EXPECTED_CLUB_ID}"
                f"/tournaments/admin/tournaments/{EXPECTED_TOURNAMENT_ID}/ops?"
                + urlencode({"draw_id": EXPECTED_DRAW_ID})
            ),
            headers=bearer_headers,
            transport=transport,
            operation="Staging tournament draw validation",
        )
        _validate_tournament_snapshot(snapshot)
        prepared.update(
            {
                "JUPR_COMMUNICATIONS_DRAFT_WEEK_START": EXPECTED_RECAP_WEEK_START,
                "JUPR_TOURNAMENT_OPS_TOURNAMENT_ID": EXPECTED_TOURNAMENT_ID,
                "JUPR_TOURNAMENT_OPS_DRAW_ID": EXPECTED_DRAW_ID,
            }
        )
        _append_github_env(
            github_env_path,
            {
                "JUPR_COMMUNICATIONS_DRAFT_WEEK_START": EXPECTED_RECAP_WEEK_START,
                "JUPR_TOURNAMENT_OPS_TOURNAMENT_ID": EXPECTED_TOURNAMENT_ID,
                "JUPR_TOURNAMENT_OPS_DRAW_ID": EXPECTED_DRAW_ID,
            },
        )
    return prepared


def prepare_staging_session(
    mode: str,
    *,
    env: Mapping[str, str] | None = None,
    transport: Transport = _default_transport,
    report_dir: Path | None = None,
) -> dict[str, str]:
    """Mint and validate a bounded session without user/role/business DML."""

    values = os.environ if env is None else env
    try:
        prepared = _prepare_staging_session(mode, env=values, transport=transport)
    except SessionPreparationError as exc:
        _write_report(
            report_dir,
            PREPARATION_REPORT_NAME,
            _preparation_report(mode, status="failed", error=str(exc)),
        )
        raise
    _write_report(
        report_dir,
        PREPARATION_REPORT_NAME,
        _preparation_report(mode, status="passed", prepared=prepared),
    )
    return prepared


def _cleanup_staging_session(
    *,
    env: Mapping[str, str],
    transport: Transport,
) -> dict[str, str]:
    supabase_origin = _required_env(env, "STAGING_SUPABASE_URL").rstrip("/")
    if supabase_origin != EXPECTED_SUPABASE_ORIGIN:
        raise SessionPreparationError(
            "STAGING_SUPABASE_URL does not identify the permitted staging project."
        )
    github_env_path = Path(_required_env(env, "GITHUB_ENV"))
    token = str(env.get("STAGING_ADMIN_BEARER_TOKEN") or "").strip()
    if not token:
        _clear_github_env(github_env_path)
        return {
            "refresh_session_status": "not_started",
            "access_jwt_status": "not_exported",
        }
    if not 100 <= len(token) <= 8192 or JWT_RE.fullmatch(token) is None:
        try:
            raise SessionPreparationError(
                "STAGING_ADMIN_BEARER_TOKEN is not a valid staging session token."
            )
        finally:
            _clear_github_env(github_env_path)

    try:
        anon_key = _required_env(env, "STAGING_SUPABASE_ANON_KEY")
        refresh_session_status = _request_logout(
            url=f"{supabase_origin}/auth/v1/logout?scope=local",
            headers={
                "apikey": anon_key,
                "Authorization": f"Bearer {token}",
                "Accept": "application/json",
            },
            transport=transport,
        )
    finally:
        _clear_github_env(github_env_path)
    return {
        "refresh_session_status": refresh_session_status,
        "access_jwt_status": "may_remain_valid_until_exp_claim",
    }


def cleanup_staging_session(
    *,
    env: Mapping[str, str] | None = None,
    transport: Transport = _default_transport,
    report_dir: Path | None = None,
) -> dict[str, str]:
    """End refreshability and clear exports; the access JWT expires by ``exp``."""

    values = os.environ if env is None else env
    try:
        cleaned = _cleanup_staging_session(env=values, transport=transport)
    except SessionPreparationError as exc:
        _write_report(
            report_dir,
            CLEANUP_REPORT_NAME,
            {
                "schema_version": 2,
                "operation": "cleanup_parity_staging_session",
                "status": "failed",
                "supabase_project_ref": EXPECTED_SUPABASE_PROJECT_REF,
                "error": str(exc),
            },
        )
        raise
    _write_report(
        report_dir,
        CLEANUP_REPORT_NAME,
        {
            "schema_version": 2,
            "operation": "cleanup_parity_staging_session",
            "status": "passed",
            "supabase_project_ref": EXPECTED_SUPABASE_PROJECT_REF,
            **cleaned,
        },
    )
    return cleaned


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a bounded authenticated staging parity session or end its "
            "refreshable Supabase session."
        )
    )
    parser.add_argument("mode", choices=(*SUPPORTED_MODES, "cleanup"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        help="Write a credential-free JSON evidence report to this directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "cleanup":
            cleaned = cleanup_staging_session(report_dir=args.report_dir)
        else:
            prepare_staging_session(
                args.mode,
                report_dir=args.report_dir,
            )
    except SessionPreparationError as exc:
        print(f"Parity staging session operation failed: {exc}", file=sys.stderr)
        return 1

    if args.mode == "cleanup":
        print(
            "Parity staging refresh-session cleanup completed: "
            f"{cleaned['refresh_session_status']}; access JWT "
            f"{cleaned['access_jwt_status'].replace('_', ' ')}."
        )
    else:
        fixture_count = 3 if args.mode == "admin-read-export" else 0
        print(
            f"Prepared authenticated {args.mode} staging session "
            f"with {fixture_count} validated fixture set(s)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
