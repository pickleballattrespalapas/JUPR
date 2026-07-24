#!/usr/bin/env python3
"""Prepare short-lived, candidate-bound credentials and fixtures for parity evidence.

The workflow deliberately mints a fresh Supabase access token instead of storing
one as a long-lived GitHub secret.  Sensitive response bodies are never included
in errors or normal output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener


EXPECTED_SUPABASE_ORIGIN = "https://sijpxjxvdtrehmqvirfi.supabase.co"
EXPECTED_API_ORIGIN = "https://juprleagues-api-staging.fly.dev"
EXPECTED_CLUB_ID = "tres_palapas"
EXPECTED_RECAP_ID = "98000000-0000-4000-8000-000000000002"
EXPECTED_RECAP_WEEK_START = "2099-01-05"
EXPECTED_TOURNAMENT_ID = "93000000-0000-4000-8000-000000000002"
EXPECTED_DRAW_ID = "94000000-0000-4000-8000-000000000001"
SUPPORTED_MODES = ("admin-read-export", "match-rating-writes")
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
MAX_RESPONSE_BYTES = 2 * 1024 * 1024

Transport = Callable[[Request], tuple[int, bytes]]


class SessionPreparationError(RuntimeError):
    """A sanitized, operator-safe parity-session preparation failure."""


class _RefuseRedirects(HTTPRedirectHandler):
    """Never forward staging credentials through an HTTP redirect."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


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


def _request_json(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any] | None = None,
    transport: Transport,
    operation: str,
) -> dict[str, Any]:
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
    except Exception:
        raise SessionPreparationError(f"{operation} failed unexpectedly.") from None

    if not 200 <= int(status) < 300:
        raise SessionPreparationError(f"{operation} failed with HTTP {int(status)}.")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise SessionPreparationError(f"{operation} returned an oversized response.")
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise SessionPreparationError(f"{operation} returned an invalid response.") from None
    if not isinstance(decoded, dict):
        raise SessionPreparationError(f"{operation} returned an invalid response.")
    return decoded


def _validate_access_token(payload: Mapping[str, Any], expected_email: str) -> str:
    token = str(payload.get("access_token") or "").strip()
    if not 100 <= len(token) <= 8192 or JWT_RE.fullmatch(token) is None:
        raise SessionPreparationError("Supabase authentication returned an invalid access token.")
    token_type = str(payload.get("token_type") or "").strip().lower()
    if token_type and token_type != "bearer":
        raise SessionPreparationError("Supabase authentication returned an invalid token type.")
    user = payload.get("user")
    returned_email = (
        str(user.get("email") or "").strip()
        if isinstance(user, Mapping)
        else ""
    )
    if not returned_email or returned_email.casefold() != expected_email.casefold():
        raise SessionPreparationError(
            "Supabase authentication returned an unexpected user identity."
        )
    return token


def _validate_capabilities(payload: Mapping[str, Any], expected_email: str) -> None:
    if payload.get("authorized") is not True:
        raise SessionPreparationError("Staging admin capability validation was denied.")
    user = payload.get("user")
    returned_email = (
        str(user.get("email") or "").strip()
        if isinstance(user, Mapping)
        else ""
    )
    if not returned_email or returned_email.casefold() != expected_email.casefold():
        raise SessionPreparationError(
            "Staging admin capability validation returned an unexpected identity."
        )
    requested_club_id = str(payload.get("requested_club_id") or "").strip()
    if requested_club_id != EXPECTED_CLUB_ID:
        raise SessionPreparationError(
            "Staging admin capability validation returned an unexpected club."
        )

    assignments = payload.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        raise SessionPreparationError(
            "Staging admin capability validation returned no assignments."
        )
    permissions: set[str] = set()
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            continue
        if str(assignment.get("club_id") or "").strip() != EXPECTED_CLUB_ID:
            continue
        raw_permissions = assignment.get("permissions")
        if isinstance(raw_permissions, list):
            permissions.update(
                str(permission).strip()
                for permission in raw_permissions
                if str(permission).strip()
            )
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
        raise SessionPreparationError("The staging tournament fixture list is unavailable.")
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
        raise SessionPreparationError("The exact staging tournament fixture is unavailable.")


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
        raise SessionPreparationError("The exact staging tournament snapshot is invalid.")
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
            raise SessionPreparationError("An internal GitHub environment name is invalid.")
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


def prepare_staging_session(
    mode: str,
    *,
    env: Mapping[str, str] | None = None,
    transport: Transport = _default_transport,
) -> dict[str, str]:
    """Mint a token, validate staging identity, and append safe workflow values."""

    if mode not in SUPPORTED_MODES:
        raise SessionPreparationError(f"Unsupported parity evidence mode: {mode}.")
    values = os.environ if env is None else env
    supabase_origin = _required_env(values, "STAGING_SUPABASE_URL").rstrip("/")
    if supabase_origin != EXPECTED_SUPABASE_ORIGIN:
        raise SessionPreparationError(
            "STAGING_SUPABASE_URL does not identify the permitted staging project."
        )
    api_origin = _required_env(values, "STAGING_API_BASE_URL").rstrip("/")
    if api_origin != EXPECTED_API_ORIGIN:
        raise SessionPreparationError(
            "STAGING_API_BASE_URL does not identify the permitted staging API."
        )
    anon_key = _required_env(values, "STAGING_SUPABASE_ANON_KEY")
    admin_email = _required_env(values, "STAGING_ADMIN_EMAIL")
    admin_password = _required_env(values, "STAGING_ADMIN_PASSWORD")
    github_env_path = Path(_required_env(values, "GITHUB_ENV"))

    auth_payload = _request_json(
        method="POST",
        url=f"{supabase_origin}/auth/v1/token?grant_type=password",
        headers={
            "apikey": anon_key,
            "Authorization": f"Bearer {anon_key}",
            "Accept": "application/json",
        },
        payload={"email": admin_email, "password": admin_password},
        transport=transport,
        operation="Supabase staging authentication",
    )
    token = _validate_access_token(auth_payload, admin_email)
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
    _validate_capabilities(capabilities, admin_email)

    prepared = {
        "STAGING_ADMIN_BEARER_TOKEN": token,
        "JUPR_STAGING_ADMIN_ACCESS_TOKEN": token,
    }
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

    # GitHub interprets this workflow command before subsequent log output.
    print(f"::add-mask::{token}")
    _append_github_env(github_env_path, prepared)
    return prepared


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare a short-lived authenticated staging parity session."
    )
    parser.add_argument("mode", choices=SUPPORTED_MODES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        prepared = prepare_staging_session(args.mode)
    except SessionPreparationError as exc:
        print(f"Parity staging session preparation failed: {exc}", file=sys.stderr)
        return 1
    fixture_count = max(0, len(prepared) - 2)
    print(
        f"Prepared authenticated {args.mode} staging session "
        f"with {fixture_count} validated fixture value(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
