#!/usr/bin/env python3
"""Prepare and remove an isolated Match Log exclusion/recovery fixture.

The fixture is deliberately created in a unique, inactive staging-only club.
The helper runs only in the candidate-bound GitHub workflow and uses the
Supabase service role only in server-side preparation and cleanup steps. The
Playwright step receives sanitized fixture identifiers through ``GITHUB_ENV``;
it never receives the service-role credential.

A manifest is persisted before the first mutation. Cleanup requires exact
ownership markers and terminal exclusion, replay, and badge-reconciliation
state. If recovery is ambiguous, cleanup refuses to delete anything and leaves
only the isolated fixture for investigation. Completed operation, replay,
badge-progress, worker, and audit evidence may remain after core fixture rows
are removed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener
from uuid import UUID, uuid4

EXPECTED_SUPABASE_ORIGIN = "https://sijpxjxvdtrehmqvirfi.supabase.co"
EXPECTED_SUPABASE_PROJECT_REF = "sijpxjxvdtrehmqvirfi"
EXPECTED_REPOSITORY = "pickleballattrespalapas/JUPR"
EXPECTED_SOURCE_CLUB_ID = "tres_palapas"
ALLOWED_GITHUB_REFS = frozenset(
    {"refs/heads/staging", "refs/heads/rollback-feb8"}
)
EXPECTED_AUTH_ISSUER = f"{EXPECTED_SUPABASE_ORIGIN}/auth/v1"
FIXTURE_CONTRACT = "jupr:parity-match-exclusion-recovery-fixture:v1"
FIXTURE_SOURCE = "staging_parity_match_exclusion_recovery"
FIXTURE_CLUB_PREFIX = "jupr_parity_mex_"
FIXTURE_SLUG_PREFIX = "jupr-parity-mex-"
FIXTURE_NAME_PREFIX = "JUPR parity match exclusion"
MATCH_CONTEXT_TYPE = "staging_match_exclusion"
MANIFEST_NAME = "match-exclusion-recovery-fixture.json"
CLEANUP_REPORT_NAME = "match-exclusion-recovery-fixture-cleanup.json"
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
CANDIDATE_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
JWT_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,4096}$")
DELETE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
TERMINAL_OPERATION_STATUS = "succeeded"
TERMINAL_REPLAY_STATUS = "succeeded"
TERMINAL_BADGE_PROGRESS_STATUS = "succeeded"
PLAYER_BASELINE = {
    "rating": 1200.0,
    "wins": 0,
    "losses": 0,
    "matches_played": 0,
    "last_game_at": None,
}

Transport = Callable[[Request], tuple[int, bytes]]
UuidFactory = Callable[[], UUID]


class FixtureError(RuntimeError):
    """A sanitized, operator-safe fixture preparation or cleanup failure."""


class _RefuseRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[override]
        return None


_STAGING_OPENER = build_opener(_RefuseRedirects())


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _utc_datetime(value: Any, *, label: str) -> datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        raise FixtureError(f"{label} is not a valid timestamp.") from None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _required_env(env: Mapping[str, str], name: str) -> str:
    value = str(env.get(name) or "").strip()
    if not value:
        raise FixtureError(f"Required environment value {name} is missing.")
    if "\r" in value or "\n" in value:
        raise FixtureError(f"Required environment value {name} is invalid.")
    return value


def _validate_staging_environment(
    env: Mapping[str, str],
) -> tuple[str, str, str]:
    origin = _required_env(env, "STAGING_SUPABASE_URL").rstrip("/")
    project_ref = _required_env(env, "STAGING_SUPABASE_PROJECT_REF")
    repository = _required_env(env, "GITHUB_REPOSITORY")
    github_ref = _required_env(env, "GITHUB_REF")
    candidate_sha = _required_env(env, "CANDIDATE_SHA")
    if origin != EXPECTED_SUPABASE_ORIGIN:
        raise FixtureError("Refusing a non-allowlisted Supabase origin.")
    parsed = urlparse(origin)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed.hostname != f"{EXPECTED_SUPABASE_PROJECT_REF}.supabase.co"
    ):
        raise FixtureError("Refusing an invalid Supabase staging origin.")
    if project_ref != EXPECTED_SUPABASE_PROJECT_REF:
        raise FixtureError("Refusing a non-allowlisted Supabase project.")
    if repository != EXPECTED_REPOSITORY:
        raise FixtureError("Refusing a fixture run outside the allowlisted repository.")
    if github_ref not in ALLOWED_GITHUB_REFS:
        raise FixtureError("Refusing a fixture run from a non-allowlisted Git ref.")
    if CANDIDATE_SHA_RE.fullmatch(candidate_sha) is None:
        raise FixtureError("CANDIDATE_SHA must be an exact lowercase commit SHA.")
    return (
        origin,
        _required_env(env, "STAGING_SUPABASE_SERVICE_ROLE_KEY"),
        candidate_sha,
    )


def _decode_jwt_identity(env: Mapping[str, str]) -> tuple[str, str]:
    token = _required_env(env, "STAGING_ADMIN_BEARER_TOKEN")
    email = _required_env(env, "STAGING_ADMIN_EMAIL").lower()
    parts = token.split(".")
    if len(parts) != 3 or any(JWT_SEGMENT_RE.fullmatch(part) is None for part in parts):
        raise FixtureError("The prepared staging bearer token is invalid.")
    try:
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        raise FixtureError("The prepared staging bearer token is invalid.") from None
    if not isinstance(claims, dict):
        raise FixtureError("The prepared staging bearer token is invalid.")
    user_id = str(claims.get("sub") or "").strip().lower()
    claim_email = str(claims.get("email") or "").strip().lower()
    try:
        parsed_user_id = UUID(user_id)
    except (ValueError, AttributeError):
        raise FixtureError("The prepared staging bearer identity is invalid.") from None
    audiences = claims.get("aud")
    audience_values = (
        {str(value) for value in audiences}
        if isinstance(audiences, list)
        else {str(audiences or "")}
    )
    if (
        str(parsed_user_id) != user_id
        or claim_email != email
        or str(claims.get("iss") or "").rstrip("/") != EXPECTED_AUTH_ISSUER
        or "authenticated" not in audience_values
    ):
        raise FixtureError(
            "The prepared staging bearer identity does not match the expected Auth boundary."
        )
    return email, user_id


def _default_transport(request: Request) -> tuple[int, bytes]:
    with _STAGING_OPENER.open(request, timeout=20) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        status = int(getattr(response, "status", response.getcode()))
    return status, body


def _service_headers(
    service_role_key: str,
    *,
    return_representation: bool,
) -> dict[str, str]:
    headers = {
        "apikey": service_role_key,
        "Authorization": f"Bearer {service_role_key}",
        "Accept": "application/json",
    }
    if return_representation:
        headers["Prefer"] = "return=representation"
    return headers


def _request_rows(
    *,
    origin: str,
    service_role_key: str,
    table: str,
    method: str,
    query: Sequence[tuple[str, str]] = (),
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    operation: str,
    transport: Transport,
) -> list[dict[str, Any]]:
    url = f"{origin}/rest/v1/{table}"
    if query:
        url = f"{url}?{urlencode(list(query), safe='(),.*')}"
    body = None
    headers = _service_headers(
        service_role_key,
        return_representation=method in {"POST", "DELETE"},
    )
    if payload is not None:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=body, headers=headers, method=method)
    try:
        status, raw = transport(request)
    except HTTPError as exc:
        raise FixtureError(f"{operation} failed with HTTP {int(exc.code)}.") from None
    except (URLError, OSError):
        raise FixtureError(f"{operation} could not reach staging.") from None
    except FixtureError:
        raise
    except Exception:  # noqa: BLE001 - external bodies and credentials stay hidden
        raise FixtureError(f"{operation} failed unexpectedly.") from None
    if not 200 <= int(status) < 300:
        raise FixtureError(f"{operation} failed with HTTP {int(status)}.")
    if len(raw) > MAX_RESPONSE_BYTES:
        raise FixtureError(f"{operation} returned an oversized response.")
    if not raw:
        return []
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise FixtureError(f"{operation} returned an invalid response.") from None
    if not isinstance(decoded, list) or any(not isinstance(row, dict) for row in decoded):
        raise FixtureError(f"{operation} returned an invalid response.")
    return [dict(row) for row in decoded]


class _RestClient:
    def __init__(
        self,
        *,
        origin: str,
        service_role_key: str,
        transport: Transport,
    ) -> None:
        self.origin = origin
        self.service_role_key = service_role_key
        self.transport = transport

    def rows(
        self,
        table: str,
        method: str,
        *,
        query: Sequence[tuple[str, str]] = (),
        payload: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        operation: str,
    ) -> list[dict[str, Any]]:
        return _request_rows(
            origin=self.origin,
            service_role_key=self.service_role_key,
            table=table,
            method=method,
            query=query,
            payload=payload,
            operation=operation,
            transport=self.transport,
        )


def _select(
    client: _RestClient,
    table: str,
    *,
    filters: Sequence[tuple[str, str]],
    select: str = "*",
    operation: str,
) -> list[dict[str, Any]]:
    return client.rows(
        table,
        "GET",
        query=(("select", select), *filters),
        operation=operation,
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        raise FixtureError("Could not write sanitized fixture evidence.") from None


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise FixtureError("The match-exclusion fixture manifest is unavailable or invalid.") from None
    if not isinstance(payload, dict) or payload.get("contract") != FIXTURE_CONTRACT:
        raise FixtureError("The match-exclusion fixture manifest has an unexpected contract.")
    return payload


def _uuid_text(value: object, *, label: str) -> str:
    raw = str(value or "").strip().lower()
    try:
        parsed = UUID(raw)
    except (ValueError, AttributeError):
        raise FixtureError(f"The fixture manifest has an invalid {label}.") from None
    if str(parsed) != raw:
        raise FixtureError(f"The fixture manifest has an invalid {label}.")
    return raw


def _positive_int(value: object, *, label: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise FixtureError(f"The fixture manifest has an invalid {label}.") from None
    if parsed < 1 or str(parsed) != str(value):
        raise FixtureError(f"The fixture manifest has an invalid {label}.")
    return parsed


def _new_manifest(
    env: Mapping[str, str],
    *,
    candidate_sha: str,
    operator_email: str,
    uuid_factory: UuidFactory,
) -> dict[str, Any]:
    fixture_id = str(uuid_factory())
    suffix = fixture_id.replace("-", "")[:16]
    club_id = f"{FIXTURE_CLUB_PREFIX}{suffix}"
    slug = f"{FIXTURE_SLUG_PREFIX}{suffix}"
    run_id = str(env.get("GITHUB_RUN_ID") or "unknown").strip()[:40] or "unknown"
    run_attempt = str(env.get("GITHUB_RUN_ATTEMPT") or "1").strip()[:12] or "1"
    marker = f"{FIXTURE_CONTRACT}:{run_id}:{run_attempt}:{fixture_id}"
    player_names = [
        f"{FIXTURE_NAME_PREFIX} {suffix} P{index}" for index in range(1, 5)
    ]
    context_ids = {
        "duplicate_a": f"{club_id}:duplicate-a",
        "duplicate_b": f"{club_id}:duplicate-b",
        "distinct": f"{club_id}:distinct",
    }
    return {
        "contract": FIXTURE_CONTRACT,
        "status": "planned",
        "created_at": _utc_now(),
        "candidate_sha": candidate_sha,
        "github_repository": EXPECTED_REPOSITORY,
        "github_ref": str(env["GITHUB_REF"]),
        "github_run_id": run_id,
        "github_run_attempt": run_attempt,
        "supabase_project_ref": EXPECTED_SUPABASE_PROJECT_REF,
        "fixture_id": fixture_id,
        "club_id": club_id,
        "club_slug": slug,
        "marker": marker,
        "operator_email_sha256": hashlib.sha256(
            operator_email.encode("utf-8")
        ).hexdigest(),
        "player_names": player_names,
        "player_ids": [],
        "context_ids": context_ids,
        "match_ids": {},
        "match_row_versions": {},
        "temporary_role_assignment_id": None,
        "idempotency_keys": {
            "stale": str(uuid_factory()),
            "duplicate_cleanup": str(uuid_factory()),
            "direct_exclusion": str(uuid_factory()),
        },
    }


def _manifest_contract(
    manifest: Mapping[str, Any],
) -> tuple[str, str, str, list[str], dict[str, str], dict[str, str]]:
    if manifest.get("supabase_project_ref") != EXPECTED_SUPABASE_PROJECT_REF:
        raise FixtureError("The fixture manifest belongs to another Supabase project.")
    if manifest.get("github_repository") != EXPECTED_REPOSITORY:
        raise FixtureError("The fixture manifest belongs to another repository.")
    if manifest.get("github_ref") not in ALLOWED_GITHUB_REFS:
        raise FixtureError("The fixture manifest belongs to a non-allowlisted Git ref.")
    if CANDIDATE_SHA_RE.fullmatch(str(manifest.get("candidate_sha") or "")) is None:
        raise FixtureError("The fixture manifest is not bound to an exact candidate.")
    fixture_id = _uuid_text(manifest.get("fixture_id"), label="fixture ID")
    suffix = fixture_id.replace("-", "")[:16]
    club_id = str(manifest.get("club_id") or "")
    club_slug = str(manifest.get("club_slug") or "")
    marker = str(manifest.get("marker") or "")
    if (
        club_id != f"{FIXTURE_CLUB_PREFIX}{suffix}"
        or club_slug != f"{FIXTURE_SLUG_PREFIX}{suffix}"
        or not marker.endswith(f":{fixture_id}")
        or not marker.startswith(f"{FIXTURE_CONTRACT}:")
        or "\n" in marker
        or "\r" in marker
    ):
        raise FixtureError("The fixture manifest has an invalid ownership marker.")
    raw_names = manifest.get("player_names")
    expected_names = [
        f"{FIXTURE_NAME_PREFIX} {suffix} P{index}" for index in range(1, 5)
    ]
    if raw_names != expected_names:
        raise FixtureError("The fixture manifest has invalid player ownership markers.")
    raw_context_ids = manifest.get("context_ids")
    expected_context_ids = {
        "duplicate_a": f"{club_id}:duplicate-a",
        "duplicate_b": f"{club_id}:duplicate-b",
        "distinct": f"{club_id}:distinct",
    }
    if raw_context_ids != expected_context_ids:
        raise FixtureError("The fixture manifest has invalid match ownership markers.")
    raw_keys = manifest.get("idempotency_keys")
    if not isinstance(raw_keys, dict) or set(raw_keys) != {
        "stale",
        "duplicate_cleanup",
        "direct_exclusion",
    }:
        raise FixtureError("The fixture manifest has an invalid idempotency contract.")
    idempotency_keys = {
        name: _uuid_text(value, label=f"{name} idempotency key")
        for name, value in raw_keys.items()
    }
    if len(set(idempotency_keys.values())) != 3:
        raise FixtureError("The fixture manifest contains colliding request identities.")
    return (
        fixture_id,
        club_id,
        club_slug,
        expected_names,
        expected_context_ids,
        idempotency_keys,
    )


def _optional_manifest_ids(
    manifest: Mapping[str, Any],
) -> tuple[list[int], dict[str, int], dict[str, int], int | None]:
    raw_player_ids = manifest.get("player_ids")
    if not isinstance(raw_player_ids, list):
        raise FixtureError("The fixture manifest has invalid player IDs.")
    player_ids = [
        _positive_int(value, label="player ID") for value in raw_player_ids
    ]
    if len(player_ids) not in {0, 4} or len(set(player_ids)) != len(player_ids):
        raise FixtureError("The fixture manifest has invalid player IDs.")
    raw_match_ids = manifest.get("match_ids")
    raw_versions = manifest.get("match_row_versions")
    if not isinstance(raw_match_ids, dict) or not isinstance(raw_versions, dict):
        raise FixtureError("The fixture manifest has invalid match IDs.")
    allowed_names = {"duplicate_keep", "duplicate_target", "distinct"}
    match_id_names = set(raw_match_ids)
    if match_id_names and match_id_names != allowed_names:
        raise FixtureError("The fixture manifest has invalid match IDs.")
    if set(raw_versions) != set(raw_match_ids):
        raise FixtureError("The fixture manifest has invalid match versions.")
    match_ids = {
        name: _positive_int(value, label="match ID")
        for name, value in raw_match_ids.items()
    }
    row_versions = {
        name: _positive_int(value, label="match row version")
        for name, value in raw_versions.items()
    }
    if len(set(match_ids.values())) != len(match_ids):
        raise FixtureError("The fixture manifest contains duplicate match IDs.")
    role_id_value = manifest.get("temporary_role_assignment_id")
    role_id = (
        None
        if role_id_value is None
        else _positive_int(role_id_value, label="role assignment ID")
    )
    return player_ids, match_ids, row_versions, role_id


def _assert_operator_assignment(
    client: _RestClient,
    *,
    email: str,
    user_id: str,
) -> dict[str, Any]:
    rows = _select(
        client,
        "admin_role_assignments",
        filters=(
            ("club_id", f"eq.{EXPECTED_SOURCE_CLUB_ID}"),
            ("email", f"eq.{email}"),
            ("user_id", f"eq.{user_id}"),
        ),
        select="id,club_id,email,role,user_id",
        operation="Bound staging operator assignment lookup",
    )
    if len(rows) != 1:
        raise FixtureError(
            "Expected exactly one already-bound staging operator assignment."
        )
    row = rows[0]
    if (
        str(row.get("club_id") or "") != EXPECTED_SOURCE_CLUB_ID
        or str(row.get("email") or "").strip().lower() != email
        or str(row.get("user_id") or "").strip().lower() != user_id
        or str(row.get("role") or "").strip().lower()
        not in {"club_owner", "super_admin"}
    ):
        raise FixtureError("The staging operator assignment is not eligible.")
    _positive_int(row.get("id"), label="source assignment ID")
    return row


def _assert_exact_generated_rows(
    rows: list[dict[str, Any]],
    *,
    count: int,
    club_id: str,
    operation: str,
) -> None:
    ids = []
    for row in rows:
        ids.append(_positive_int(row.get("id"), label="generated row ID"))
        if str(row.get("club_id") or "") != club_id:
            raise FixtureError(f"{operation} returned a row outside the fixture club.")
    if len(rows) != count or len(set(ids)) != count:
        raise FixtureError(f"{operation} did not return the exact generated rows.")


def _player_payloads(
    *,
    club_id: str,
    player_names: list[str],
) -> list[dict[str, Any]]:
    return [
        {
            "club_id": club_id,
            "name": name,
            "rating": 1200.0,
            "starting_rating": 1200.0,
            "wins": 0,
            "losses": 0,
            "matches_played": 0,
            "active": True,
            "last_game_at": None,
            "inactive_at": None,
            "singles_rating": 1200.0,
            "singles_wins": 0,
            "singles_losses": 0,
            "singles_matches_played": 0,
            "singles_last_game_at": None,
            "singles_replay_baseline": dict(PLAYER_BASELINE),
        }
        for name in player_names
    ]


def _match_payloads(
    *,
    club_id: str,
    marker: str,
    context_ids: Mapping[str, str],
    player_ids: list[int],
) -> list[dict[str, Any]]:
    if len(player_ids) != 4:
        raise FixtureError("Four exact fixture players are required before match creation.")
    common = {
        "club_id": club_id,
        "date": "2099-01-05T18:00:00Z",
        "league": f"{FIXTURE_NAME_PREFIX} league",
        "week_tag": "Fixture",
        "match_type": "PopUp",
        "t1_p1": player_ids[0],
        "t1_p2": player_ids[1],
        "t2_p1": player_ids[2],
        "t2_p2": player_ids[3],
        "score_t1": 11,
        "score_t2": 7,
        "elo_delta": 8.0,
        "t1_p1_r": 1200.0,
        "t1_p2_r": 1200.0,
        "t2_p1_r": 1200.0,
        "t2_p2_r": 1200.0,
        "t1_p1_r_end": 1208.0,
        "t1_p2_r_end": 1208.0,
        "t2_p1_r_end": 1192.0,
        "t2_p2_r_end": 1192.0,
        "notes": marker,
        "context_type": MATCH_CONTEXT_TYPE,
        "rating_scope": "overall_only",
        "match_format": "doubles",
        "singles_replay_managed": False,
    }
    return [
        {**common, "context_id": context_ids["duplicate_a"]},
        {**common, "context_id": context_ids["duplicate_b"]},
        {
            **common,
            "date": "2099-01-05T18:05:00Z",
            "score_t2": 9,
            "context_id": context_ids["distinct"],
        },
    ]


def _fixture_rows(
    client: _RestClient,
    manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    _fixture_id, club_id, club_slug, _names, _contexts, _keys = (
        _manifest_contract(manifest)
    )
    rows: dict[str, list[dict[str, Any]]] = {}
    rows["clubs"] = _select(
        client,
        "clubs",
        filters=(("id", f"eq.{club_id}"),),
        operation="Fixture club readback",
    )
    rows["clubs_by_slug"] = _select(
        client,
        "clubs",
        filters=(("slug", f"eq.{club_slug}"),),
        select="id,slug",
        operation="Fixture club-slug readback",
    )
    rows["roles"] = _select(
        client,
        "admin_role_assignments",
        filters=(("club_id", f"eq.{club_id}"),),
        select="id,club_id,email,role,user_id",
        operation="Fixture role readback",
    )
    rows["players"] = _select(
        client,
        "players",
        filters=(("club_id", f"eq.{club_id}"),),
        operation="Fixture player readback",
    )
    rows["matches"] = _select(
        client,
        "matches",
        filters=(("club_id", f"eq.{club_id}"),),
        operation="Fixture match readback",
    )
    rows["league_ratings"] = _select(
        client,
        "league_ratings",
        filters=(("club_id", f"eq.{club_id}"),),
        operation="Fixture league-rating readback",
    )
    rows["player_badges"] = _select(
        client,
        "player_badges",
        filters=(("club_id", f"eq.{club_id}"),),
        operation="Fixture player-badge readback",
    )
    rows["operations"] = _select(
        client,
        "match_exclusion_operations",
        filters=(("club_id", f"eq.{club_id}"),),
        operation="Fixture exclusion-operation readback",
    )
    rows["replay_jobs"] = _select(
        client,
        "replay_jobs",
        filters=(("club_id", f"eq.{club_id}"),),
        operation="Fixture replay-job readback",
    )
    rows["badge_progress"] = _select(
        client,
        "match_exclusion_badge_progress",
        filters=(("club_id", f"eq.{club_id}"),),
        operation="Fixture badge-progress readback",
    )
    return rows


def _validate_owned_rows(
    rows: Mapping[str, list[dict[str, Any]]],
    manifest: Mapping[str, Any],
    *,
    operator_email: str,
    operator_user_id: str,
    require_complete: bool,
    require_terminal: bool,
) -> None:
    (
        _fixture_id,
        club_id,
        club_slug,
        player_names,
        context_ids,
        idempotency_keys,
    ) = _manifest_contract(manifest)
    player_ids, match_ids, row_versions, role_id = _optional_manifest_ids(manifest)
    marker = str(manifest["marker"])
    clubs = rows["clubs"]
    clubs_by_slug = rows["clubs_by_slug"]
    roles = rows["roles"]
    players = rows["players"]
    matches = rows["matches"]
    if len(clubs) > 1 or len(clubs_by_slug) > 1 or len(roles) > 1:
        raise FixtureError("The fixture scope contains unexpected duplicate rows.")
    if clubs:
        club = clubs[0]
        if (
            str(club.get("id") or "") != club_id
            or str(club.get("slug") or "") != club_slug
            or str(club.get("name") or "") != f"{FIXTURE_NAME_PREFIX} club"
            or str(club.get("tagline") or "") != marker
            or club.get("is_active") is not False
        ):
            raise FixtureError("The staging club does not match the ownership marker.")
    if clubs_by_slug and str(clubs_by_slug[0].get("id") or "") != club_id:
        raise FixtureError("The staging club slug belongs to another row.")
    if roles:
        role = roles[0]
        if (
            (role_id is not None and _positive_int(role.get("id"), label="role ID") != role_id)
            or str(role.get("club_id") or "") != club_id
            or str(role.get("email") or "").strip().lower() != operator_email
            or str(role.get("user_id") or "").strip().lower() != operator_user_id
            or str(role.get("role") or "").strip().lower() != "super_admin"
        ):
            raise FixtureError("The temporary admin role does not match the fixture owner.")
    expected_name_set = set(player_names)
    actual_names = {str(row.get("name") or "") for row in players}
    if actual_names - expected_name_set or len(actual_names) != len(players):
        raise FixtureError("The isolated club contains an unowned player row.")
    for player in players:
        if (
            str(player.get("club_id") or "") != club_id
            or player.get("singles_replay_baseline") != PLAYER_BASELINE
        ):
            raise FixtureError("A fixture player has an invalid ownership or replay baseline.")
    actual_player_ids = {
        _positive_int(row.get("id"), label="player ID") for row in players
    }
    if player_ids and not actual_player_ids.issubset(set(player_ids)):
        raise FixtureError("Fixture player IDs no longer match the manifest.")
    expected_context_set = set(context_ids.values())
    actual_contexts = {str(row.get("context_id") or "") for row in matches}
    if actual_contexts - expected_context_set or len(actual_contexts) != len(matches):
        raise FixtureError("The isolated club contains an unowned match row.")
    for match in matches:
        participant_ids = {
            _positive_int(match.get(column), label="match player ID")
            for column in ("t1_p1", "t1_p2", "t2_p1", "t2_p2")
        }
        if (
            str(match.get("club_id") or "") != club_id
            or str(match.get("context_type") or "") != MATCH_CONTEXT_TYPE
            or str(match.get("notes") or "") != marker
            or str(match.get("match_format") or "").lower() != "doubles"
            or str(match.get("rating_scope") or "").lower() == "unrated"
            or (player_ids and not participant_ids.issubset(set(player_ids)))
        ):
            raise FixtureError("A fixture match has an invalid ownership or rating contract.")
    actual_match_ids = {
        _positive_int(row.get("id"), label="match ID") for row in matches
    }
    if match_ids and not actual_match_ids.issubset(set(match_ids.values())):
        raise FixtureError("Fixture match IDs no longer match the manifest.")
    if require_terminal and (players or matches):
        expected_last_game_at: dict[int, datetime | None] = {
            player_id: None for player_id in player_ids
        }
        # Direct fixture seeding deliberately leaves player activity at its
        # authored baseline. An accepted exclusion operation runs a full replay,
        # after which activity must match the surviving scored fixture history.
        if rows["operations"]:
            for match in matches:
                if match.get("deleted_at") not in (None, ""):
                    continue
                try:
                    score_t1 = int(match.get("score_t1") or 0)
                    score_t2 = int(match.get("score_t2") or 0)
                except (TypeError, ValueError):
                    raise FixtureError(
                        "A fixture match has an invalid activity score."
                    ) from None
                if score_t1 + score_t2 <= 0:
                    continue
                match_time = _utc_datetime(
                    match.get("date"),
                    label="fixture match activity date",
                )
                if match_time is None:
                    raise FixtureError(
                        "A scored fixture match has no activity date."
                    )
                for column in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
                    player_id = _positive_int(
                        match.get(column),
                        label="match player ID",
                    )
                    if player_id not in expected_last_game_at:
                        continue
                    previous = expected_last_game_at[player_id]
                    expected_last_game_at[player_id] = (
                        match_time
                        if previous is None
                        else max(previous, match_time)
                    )
        actual_players_by_id = {
            _positive_int(player.get("id"), label="player ID"): player
            for player in players
        }
        if set(actual_players_by_id) != set(expected_last_game_at):
            raise FixtureError(
                "Terminal player activity scope does not match all fixture players."
            )
        for player_id, expected_activity in expected_last_game_at.items():
            actual_activity = _utc_datetime(
                actual_players_by_id[player_id].get("last_game_at"),
                label="fixture player last_game_at",
            )
            if actual_activity != expected_activity:
                raise FixtureError(
                    "Fixture player last_game_at does not match the latest "
                    "active scored fixture match."
                )
    if require_complete:
        if (
            len(clubs) != 1
            or len(clubs_by_slug) != 1
            or len(roles) != 1
            or len(players) != 4
            or len(matches) != 3
            or set(match_ids) != {
                "duplicate_keep",
                "duplicate_target",
                "distinct",
            }
        ):
            raise FixtureError("The disposable match-exclusion fixture is incomplete.")
        by_id = {
            _positive_int(row.get("id"), label="match ID"): row for row in matches
        }
        for name, match_id in match_ids.items():
            row = by_id[match_id]
            if (
                _positive_int(row.get("row_version"), label="match row version")
                != row_versions[name]
                or row.get("deleted_at") not in (None, "")
            ):
                raise FixtureError("A prepared fixture match is not at its exact active version.")
        duplicate_a = next(
            row for row in matches if row.get("context_id") == context_ids["duplicate_a"]
        )
        duplicate_b = next(
            row for row in matches if row.get("context_id") == context_ids["duplicate_b"]
        )
        duplicate_fields = (
            "league",
            "week_tag",
            "match_type",
            "t1_p1",
            "t1_p2",
            "t2_p1",
            "t2_p2",
            "score_t1",
            "score_t2",
        )
        if any(duplicate_a.get(key) != duplicate_b.get(key) for key in duplicate_fields):
            raise FixtureError("The prepared duplicate rows are not canonical duplicates.")
        distinct = next(
            row for row in matches if row.get("context_id") == context_ids["distinct"]
        )
        if all(distinct.get(key) == duplicate_a.get(key) for key in duplicate_fields):
            raise FixtureError("The prepared distinct row is still a canonical duplicate.")
    allowed_operation_keys = {
        idempotency_keys["duplicate_cleanup"],
        idempotency_keys["direct_exclusion"],
    }
    if rows["operations"] and (
        len(player_ids) != 4
        or set(match_ids) != {"duplicate_keep", "duplicate_target", "distinct"}
    ):
        raise FixtureError(
            "Refusing cleanup because operation evidence has no complete fixture manifest."
        )
    expected_operation_targets: dict[str, tuple[str, set[int]]] = {}
    if match_ids:
        expected_operation_targets = {
            idempotency_keys["duplicate_cleanup"]: (
                "duplicate_cleanup",
                {match_ids["duplicate_target"]},
            ),
            idempotency_keys["direct_exclusion"]: (
                "exclude",
                {match_ids["distinct"]},
            ),
        }
    operation_ids: set[str] = set()
    operation_keys: set[str] = set()
    replay_ids: set[str] = set()
    expected_progress_pairs: set[tuple[str, int]] = set()
    for operation in rows["operations"]:
        operation_id = _uuid_text(operation.get("id"), label="operation ID")
        operation_key = _uuid_text(
            operation.get("idempotency_key"),
            label="operation idempotency key",
        )
        replay_id = _uuid_text(operation.get("replay_job_id"), label="replay job ID")
        raw_excluded_ids = operation.get("excluded_match_ids")
        raw_affected_player_ids = operation.get("affected_player_ids")
        if not isinstance(raw_excluded_ids, list) or not isinstance(
            raw_affected_player_ids, list
        ):
            raise FixtureError(
                "A fixture exclusion operation has invalid exact-ID evidence."
            )
        excluded_ids = {
            _positive_int(value, label="excluded match ID")
            for value in raw_excluded_ids
        }
        affected_player_ids = {
            _positive_int(value, label="affected player ID")
            for value in raw_affected_player_ids
        }
        if (
            len(raw_excluded_ids) != len(excluded_ids)
            or len(raw_affected_player_ids) != len(affected_player_ids)
        ):
            raise FixtureError(
                "A fixture exclusion operation contains duplicate exact-ID evidence."
            )
        expected_mode, expected_excluded_ids = expected_operation_targets.get(
            operation_key,
            ("", set()),
        )
        if (
            str(operation.get("club_id") or "") != club_id
            or operation_key not in allowed_operation_keys
            or operation_key in operation_keys
            or str(operation.get("source") or "") != FIXTURE_SOURCE
            or str(operation.get("delete_note") or "")
            != f"Disposable staging fixture {club_id}"
            or str(operation.get("mode") or "") != expected_mode
            or excluded_ids != expected_excluded_ids
            or affected_player_ids != set(player_ids)
        ):
            raise FixtureError("The isolated club contains an unowned exclusion operation.")
        operation_ids.add(operation_id)
        operation_keys.add(operation_key)
        replay_ids.add(replay_id)
        expected_progress_pairs.update(
            (operation_id, player_id) for player_id in affected_player_ids
        )
        if require_terminal and (
            str(operation.get("status") or "") != TERMINAL_OPERATION_STATUS
            or operation.get("recovery_stage") not in (None, "")
            or not operation.get("finished_at")
        ):
            raise FixtureError(
                "Refusing cleanup because a match exclusion operation is not terminal."
            )
    actual_replay_ids: set[str] = set()
    for replay in rows["replay_jobs"]:
        replay_id = _uuid_text(replay.get("id"), label="replay job ID")
        actual_replay_ids.add(replay_id)
        if (
            str(replay.get("club_id") or "") != club_id
            or replay_id not in replay_ids
            or str(replay.get("target_reset") or "") != "ALL (Full System Reset)"
            or str(replay.get("source") or "") != FIXTURE_SOURCE
        ):
            raise FixtureError("The isolated club contains an unowned replay job.")
        if require_terminal and (
            str(replay.get("status") or "") != TERMINAL_REPLAY_STATUS
            or not replay.get("finished_at")
            or int(replay.get("attempt_count") or 0) != 1
            or replay.get("lease_token") not in (None, "")
            or replay.get("leased_by") not in (None, "")
            or replay.get("lease_expires_at") not in (None, "")
            or replay.get("heartbeat_at") not in (None, "")
        ):
            raise FixtureError(
                "Refusing cleanup because a fixture replay job lacks exact terminal lease evidence."
            )
    if actual_replay_ids != replay_ids:
        raise FixtureError("Fixture exclusion operations and replay jobs do not match.")
    actual_progress_pairs: set[tuple[str, int]] = set()
    for progress in rows["badge_progress"]:
        progress_operation_id = _uuid_text(
            progress.get("operation_id"),
            label="badge progress operation ID",
        )
        player_id = _positive_int(progress.get("player_id"), label="badge player ID")
        progress_pair = (progress_operation_id, player_id)
        if (
            str(progress.get("club_id") or "") != club_id
            or progress_operation_id not in operation_ids
            or progress_pair in actual_progress_pairs
            or (player_ids and player_id not in set(player_ids))
        ):
            raise FixtureError("The isolated club contains unowned badge progress.")
        actual_progress_pairs.add(progress_pair)
        if require_terminal and (
            str(progress.get("status") or "") != TERMINAL_BADGE_PROGRESS_STATUS
            or not progress.get("finished_at")
        ):
            raise FixtureError(
                "Refusing cleanup because badge reconciliation is not terminal."
            )
    if require_terminal and actual_progress_pairs != expected_progress_pairs:
        raise FixtureError(
            "Refusing cleanup because badge reconciliation progress is incomplete."
        )
    for rating in rows["league_ratings"]:
        if (
            not player_ids
            or _positive_int(rating.get("player_id"), label="league-rating player ID")
            not in set(player_ids)
        ):
            raise FixtureError("The isolated club contains an unowned league rating.")
    for badge in rows["player_badges"]:
        if (
            not player_ids
            or _positive_int(badge.get("player_id"), label="badge player ID")
            not in set(player_ids)
        ):
            raise FixtureError("The isolated club contains an unowned player badge.")


def _append_github_env(path: Path, manifest: Mapping[str, Any]) -> None:
    _fixture_id, club_id, _slug, _names, _contexts, idempotency_keys = (
        _manifest_contract(manifest)
    )
    _players, match_ids, versions, _role_id = _optional_manifest_ids(manifest)
    values = {
        "JUPR_MATCH_EXCLUSION_FIXTURE_CLUB_ID": club_id,
        "JUPR_MATCH_EXCLUSION_DUPLICATE_KEEP_ID": str(
            match_ids["duplicate_keep"]
        ),
        "JUPR_MATCH_EXCLUSION_DUPLICATE_TARGET_ID": str(
            match_ids["duplicate_target"]
        ),
        "JUPR_MATCH_EXCLUSION_DUPLICATE_TARGET_ROW_VERSION": str(
            versions["duplicate_target"]
        ),
        "JUPR_MATCH_EXCLUSION_DISTINCT_MATCH_ID": str(match_ids["distinct"]),
        "JUPR_MATCH_EXCLUSION_DISTINCT_ROW_VERSION": str(versions["distinct"]),
        "JUPR_MATCH_EXCLUSION_STALE_IDEMPOTENCY_KEY": idempotency_keys["stale"],
        "JUPR_MATCH_EXCLUSION_DUPLICATE_IDEMPOTENCY_KEY": idempotency_keys[
            "duplicate_cleanup"
        ],
        "JUPR_MATCH_EXCLUSION_DIRECT_IDEMPOTENCY_KEY": idempotency_keys[
            "direct_exclusion"
        ],
        "JUPR_MATCH_EXCLUSION_ALLOW_MUTATION_E2E": "1",
    }
    try:
        with path.open("a", encoding="utf-8") as handle:
            for name, value in values.items():
                if "\r" in value or "\n" in value:
                    raise FixtureError("A prepared fixture value is invalid.")
                handle.write(f"{name}={value}\n")
    except FixtureError:
        raise
    except OSError:
        raise FixtureError("Could not append fixture values to GITHUB_ENV.") from None


def _preflight_empty_scope(
    client: _RestClient,
    manifest: Mapping[str, Any],
) -> None:
    rows = _fixture_rows(client, manifest)
    if any(rows.values()):
        raise FixtureError(
            "A generated fixture scope already exists; no write was attempted."
        )


def prepare_fixture(
    *,
    report_dir: Path,
    github_env: Path,
    env: Mapping[str, str] | None = None,
    transport: Transport = _default_transport,
    uuid_factory: UuidFactory = uuid4,
) -> dict[str, Any]:
    environment = os.environ if env is None else env
    origin, service_role_key, candidate_sha = _validate_staging_environment(
        environment
    )
    operator_email, operator_user_id = _decode_jwt_identity(environment)
    manifest_path = report_dir / MANIFEST_NAME
    manifest = _new_manifest(
        environment,
        candidate_sha=candidate_sha,
        operator_email=operator_email,
        uuid_factory=uuid_factory,
    )
    _manifest_contract(manifest)
    _optional_manifest_ids(manifest)
    _write_json(manifest_path, manifest)
    client = _RestClient(
        origin=origin,
        service_role_key=service_role_key,
        transport=transport,
    )
    _assert_operator_assignment(
        client,
        email=operator_email,
        user_id=operator_user_id,
    )
    _preflight_empty_scope(client, manifest)
    (
        _fixture_id,
        club_id,
        club_slug,
        player_names,
        context_ids,
        _keys,
    ) = _manifest_contract(manifest)
    marker = str(manifest["marker"])

    club_rows = client.rows(
        "clubs",
        "POST",
        payload={
            "id": club_id,
            "slug": club_slug,
            "name": f"{FIXTURE_NAME_PREFIX} club",
            "tagline": marker,
            "support_email": "staging-fixture@example.invalid",
            "public_base_url": None,
            "is_active": False,
        },
        operation="Disposable isolated club creation",
    )
    if (
        len(club_rows) != 1
        or str(club_rows[0].get("id") or "") != club_id
        or str(club_rows[0].get("slug") or "") != club_slug
    ):
        raise FixtureError("Disposable isolated club creation was not exact.")

    role_rows = client.rows(
        "admin_role_assignments",
        "POST",
        payload={
            "club_id": club_id,
            "email": operator_email,
            "user_id": operator_user_id,
            "role": "super_admin",
        },
        operation="Disposable super-admin assignment creation",
    )
    _assert_exact_generated_rows(
        role_rows,
        count=1,
        club_id=club_id,
        operation="Disposable super-admin assignment creation",
    )
    role_id = _positive_int(role_rows[0].get("id"), label="role assignment ID")

    player_rows = client.rows(
        "players",
        "POST",
        payload=_player_payloads(club_id=club_id, player_names=player_names),
        operation="Disposable player creation",
    )
    _assert_exact_generated_rows(
        player_rows,
        count=4,
        club_id=club_id,
        operation="Disposable player creation",
    )
    players_by_name = {str(row.get("name") or ""): row for row in player_rows}
    if set(players_by_name) != set(player_names):
        raise FixtureError("Disposable player creation returned unexpected names.")
    player_ids = [
        _positive_int(players_by_name[name].get("id"), label="player ID")
        for name in player_names
    ]

    match_rows = client.rows(
        "matches",
        "POST",
        payload=_match_payloads(
            club_id=club_id,
            marker=marker,
            context_ids=context_ids,
            player_ids=player_ids,
        ),
        operation="Disposable rated match creation",
    )
    _assert_exact_generated_rows(
        match_rows,
        count=3,
        club_id=club_id,
        operation="Disposable rated match creation",
    )
    by_context = {str(row.get("context_id") or ""): row for row in match_rows}
    if set(by_context) != set(context_ids.values()):
        raise FixtureError("Disposable rated match creation returned unexpected contexts.")
    duplicate_rows = [
        by_context[context_ids["duplicate_a"]],
        by_context[context_ids["duplicate_b"]],
    ]
    duplicate_ids = sorted(
        _positive_int(row.get("id"), label="duplicate match ID")
        for row in duplicate_rows
    )
    distinct_row = by_context[context_ids["distinct"]]
    match_ids = {
        "duplicate_keep": duplicate_ids[0],
        "duplicate_target": duplicate_ids[1],
        "distinct": _positive_int(distinct_row.get("id"), label="distinct match ID"),
    }
    rows_by_id = {
        _positive_int(row.get("id"), label="match ID"): row for row in match_rows
    }
    row_versions = {
        name: _positive_int(
            rows_by_id[match_id].get("row_version"),
            label="match row version",
        )
        for name, match_id in match_ids.items()
    }

    prepared = {
        **manifest,
        "status": "prepared",
        "prepared_at": _utc_now(),
        "temporary_role_assignment_id": role_id,
        "player_ids": player_ids,
        "match_ids": match_ids,
        "match_row_versions": row_versions,
    }
    _write_json(manifest_path, prepared)
    readback = _fixture_rows(client, prepared)
    _validate_owned_rows(
        readback,
        prepared,
        operator_email=operator_email,
        operator_user_id=operator_user_id,
        require_complete=True,
        require_terminal=False,
    )
    prepared = {
        **prepared,
        "verified_counts": {
            "clubs": 1,
            "temporary_roles": 1,
            "players": 4,
            "rated_doubles_matches": 3,
            "canonical_duplicate_targets": 1,
            "distinct_targets": 1,
            "operations": 0,
            "replay_jobs": 0,
            "badge_progress": 0,
        },
    }
    _write_json(manifest_path, prepared)
    _append_github_env(github_env, prepared)
    return prepared


def _delete_exact_set(
    client: _RestClient,
    table: str,
    *,
    filters: Sequence[tuple[str, str]],
    expected_ids: set[str],
    operation: str,
) -> int:
    if not expected_ids:
        return 0
    ordered_ids = sorted(expected_ids)
    if any(DELETE_ID_RE.fullmatch(value) is None for value in ordered_ids):
        raise FixtureError(f"{operation} has an invalid exact row ID.")
    exact_filters = tuple(filters)
    existing_id_filters = [
        value for name, value in exact_filters if name == "id"
    ]
    if existing_id_filters:
        if (
            len(existing_id_filters) != 1
            or len(ordered_ids) != 1
            or existing_id_filters[0] != f"eq.{ordered_ids[0]}"
        ):
            raise FixtureError(f"{operation} has an ambiguous exact-ID filter.")
    else:
        exact_filters = (
            *exact_filters,
            ("id", f"in.({','.join(ordered_ids)})"),
        )
    rows = client.rows(
        table,
        "DELETE",
        query=exact_filters,
        operation=operation,
    )
    returned_ids = {str(row.get("id") or "") for row in rows}
    if len(rows) != len(expected_ids) or returned_ids != expected_ids:
        raise FixtureError(f"{operation} did not delete the exact owned rows.")
    return len(rows)


def _cleanup_fixture(
    *,
    report_dir: Path,
    env: Mapping[str, str] | None = None,
    transport: Transport = _default_transport,
) -> dict[str, Any]:
    environment = os.environ if env is None else env
    manifest_path = report_dir / MANIFEST_NAME
    cleanup_path = report_dir / CLEANUP_REPORT_NAME
    if not manifest_path.exists():
        report = {
            "contract": FIXTURE_CONTRACT,
            "status": "not_prepared",
            "cleaned_at": _utc_now(),
            "deleted_counts": {},
        }
        _write_json(cleanup_path, report)
        return report

    manifest = _load_manifest(manifest_path)
    origin, service_role_key, candidate_sha = _validate_staging_environment(
        environment
    )
    if str(manifest.get("candidate_sha") or "") != candidate_sha:
        raise FixtureError("The fixture manifest belongs to another candidate.")
    operator_email, operator_user_id = _decode_jwt_identity(environment)
    expected_email_hash = hashlib.sha256(operator_email.encode("utf-8")).hexdigest()
    if manifest.get("operator_email_sha256") != expected_email_hash:
        raise FixtureError("The cleanup identity does not own this fixture.")
    client = _RestClient(
        origin=origin,
        service_role_key=service_role_key,
        transport=transport,
    )
    _assert_operator_assignment(
        client,
        email=operator_email,
        user_id=operator_user_id,
    )
    before = _fixture_rows(client, manifest)
    _validate_owned_rows(
        before,
        manifest,
        operator_email=operator_email,
        operator_user_id=operator_user_id,
        require_complete=False,
        require_terminal=True,
    )
    (
        _fixture_id,
        club_id,
        _club_slug,
        _names,
        _contexts,
        _keys,
    ) = _manifest_contract(manifest)
    player_ids, _match_ids, _row_versions, role_id = _optional_manifest_ids(manifest)

    if not before["clubs"]:
        if any(
            before[name]
            for name in (
                "roles",
                "players",
                "matches",
                "league_ratings",
                "player_badges",
            )
        ):
            raise FixtureError("Fixture children exist without the owned staging club.")
        report = {
            "contract": FIXTURE_CONTRACT,
            "status": "already_absent",
            "cleaned_at": _utc_now(),
            "candidate_sha": candidate_sha,
            "club_id": club_id,
            "deleted_counts": {},
            "retained_operation_ids": sorted(
                str(row.get("id") or "") for row in before["operations"]
            ),
            "retained_replay_job_ids": sorted(
                str(row.get("id") or "") for row in before["replay_jobs"]
            ),
        }
        _write_json(cleanup_path, report)
        return report

    deleted_counts: dict[str, int] = {}
    deleted_counts["player_badges"] = _delete_exact_set(
        client,
        "player_badges",
        filters=(("club_id", f"eq.{club_id}"),),
        expected_ids={str(row.get("id") or "") for row in before["player_badges"]},
        operation="Disposable player-badge cleanup",
    )
    deleted_counts["league_ratings"] = _delete_exact_set(
        client,
        "league_ratings",
        filters=(("club_id", f"eq.{club_id}"),),
        expected_ids={str(row.get("id") or "") for row in before["league_ratings"]},
        operation="Disposable league-rating cleanup",
    )
    deleted_counts["matches"] = _delete_exact_set(
        client,
        "matches",
        filters=(("club_id", f"eq.{club_id}"),),
        expected_ids={str(row.get("id") or "") for row in before["matches"]},
        operation="Disposable match cleanup",
    )
    deleted_counts["players"] = _delete_exact_set(
        client,
        "players",
        filters=(("club_id", f"eq.{club_id}"),),
        expected_ids={str(row.get("id") or "") for row in before["players"]},
        operation="Disposable player cleanup",
    )
    expected_role_ids = (
        {str(role_id)}
        if role_id is not None and before["roles"]
        else {str(row.get("id") or "") for row in before["roles"]}
    )
    deleted_counts["temporary_roles"] = _delete_exact_set(
        client,
        "admin_role_assignments",
        filters=(("club_id", f"eq.{club_id}"),),
        expected_ids=expected_role_ids,
        operation="Disposable role cleanup",
    )
    deleted_counts["clubs"] = _delete_exact_set(
        client,
        "clubs",
        filters=(("id", f"eq.{club_id}"),),
        expected_ids={club_id},
        operation="Disposable club cleanup",
    )

    after = _fixture_rows(client, manifest)
    if any(
        after[name]
        for name in (
            "clubs",
            "clubs_by_slug",
            "roles",
            "players",
            "matches",
            "league_ratings",
            "player_badges",
        )
    ):
        raise FixtureError("Disposable fixture cleanup left a core or derived row behind.")
    _validate_owned_rows(
        after,
        manifest,
        operator_email=operator_email,
        operator_user_id=operator_user_id,
        require_complete=False,
        require_terminal=True,
    )
    report = {
        "contract": FIXTURE_CONTRACT,
        "status": "cleaned",
        "cleaned_at": _utc_now(),
        "candidate_sha": candidate_sha,
        "club_id": club_id,
        "deleted_counts": deleted_counts,
        "verified_remaining_core_rows": 0,
        "retained_operation_ids": sorted(
            str(row.get("id") or "") for row in after["operations"]
        ),
        "retained_replay_job_ids": sorted(
            str(row.get("id") or "") for row in after["replay_jobs"]
        ),
        "retained_badge_progress_count": len(after["badge_progress"]),
    }
    _write_json(cleanup_path, report)
    return report


def cleanup_fixture(
    *,
    report_dir: Path,
    env: Mapping[str, str] | None = None,
    transport: Transport = _default_transport,
) -> dict[str, Any]:
    try:
        return _cleanup_fixture(
            report_dir=report_dir,
            env=env,
            transport=transport,
        )
    except FixtureError as exc:
        report: dict[str, Any] = {
            "contract": FIXTURE_CONTRACT,
            "status": "cleanup_refused",
            "failed_at": _utc_now(),
            "failure": str(exc),
        }
        try:
            manifest = _load_manifest(report_dir / MANIFEST_NAME)
            _fixture_id, club_id, _slug, _names, _contexts, _keys = (
                _manifest_contract(manifest)
            )
            report.update(
                {
                    "candidate_sha": manifest["candidate_sha"],
                    "club_id": club_id,
                }
            )
        except FixtureError:
            pass
        try:
            _write_json(report_dir / CLEANUP_REPORT_NAME, report)
        except FixtureError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "cleanup"))
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="Directory for sanitized fixture and cleanup evidence.",
    )
    parser.add_argument(
        "--github-env",
        type=Path,
        help="GITHUB_ENV path; required only for prepare.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "prepare":
            if args.github_env is None:
                raise FixtureError("--github-env is required for prepare.")
            prepared = prepare_fixture(
                report_dir=args.report_dir,
                github_env=args.github_env,
            )
            print(
                "Prepared one isolated disposable match-exclusion fixture "
                f"for {prepared['supabase_project_ref']}."
            )
        else:
            cleaned = cleanup_fixture(report_dir=args.report_dir)
            print(f"Match-exclusion fixture cleanup status: {cleaned['status']}.")
    except FixtureError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
