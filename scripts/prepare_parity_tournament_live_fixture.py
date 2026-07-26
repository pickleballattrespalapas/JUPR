#!/usr/bin/env python3
"""Create and remove the one disposable Tournament Live parity score fixture.

This helper is intentionally staging-specific. It uses the Supabase service
role only in server-side GitHub Actions steps; the browser receives fixture IDs
and score values through ``GITHUB_ENV``, never the service-role credential.

The manifest is written before the first database mutation so an ``always()``
cleanup step can target an exact partially-created fixture. Cleanup refuses
unknown rows, official matches, podium/playoff state, unresolved operations, or
an unrestored score. Completed operation and audit rows are retained as
evidence because they have no foreign key to the disposable core rows.
"""

from __future__ import annotations

import argparse
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
EXPECTED_CLUB_ID = "tres_palapas"
FIXTURE_CONTRACT = "jupr:parity-tournament-live-score-fixture:v1"
FIXTURE_NAME_PREFIX = "JUPR parity live fixture"
FIXTURE_SOURCE = "staging-parity-tournament-live"
MANIFEST_NAME = "tournament-live-fixture.json"
CLEANUP_REPORT_NAME = "tournament-live-fixture-cleanup.json"
ORIGINAL_SCORE_A = 11
ORIGINAL_SCORE_B = 7
EXERCISE_SCORE_A = 11
EXERCISE_SCORE_B = 8
ACTIVE_OPERATION_STATUSES = frozenset({"intent", "mutated", "recovery_required"})
UNOWNED_TOURNAMENT_CHILD_TABLES = (
    "tournament_registration_settings",
    "tournament_registration_days",
    "tournament_event_options",
    "tournament_registrations",
    "tournament_registration_selections",
    "tournament_registration_partner_requests",
    "tournament_registration_team_links",
    "tournament_registration_team_members",
)
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
CANDIDATE_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

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


def _required_env(env: Mapping[str, str], name: str) -> str:
    value = str(env.get(name) or "").strip()
    if not value:
        raise FixtureError(f"Required environment value {name} is missing.")
    if "\r" in value or "\n" in value:
        raise FixtureError(f"Required environment value {name} is invalid.")
    return value


def _validate_staging_environment(env: Mapping[str, str]) -> tuple[str, str]:
    origin = _required_env(env, "STAGING_SUPABASE_URL").rstrip("/")
    project_ref = _required_env(env, "STAGING_SUPABASE_PROJECT_REF")
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
    return origin, _required_env(env, "STAGING_SUPABASE_SERVICE_ROLE_KEY")


def _default_transport(request: Request) -> tuple[int, bytes]:
    with _STAGING_OPENER.open(request, timeout=20) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
        status = int(getattr(response, "status", response.getcode()))
    return status, body


def _service_headers(service_role_key: str, *, return_representation: bool) -> dict[str, str]:
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
    except Exception:  # noqa: BLE001 - never expose connector response details
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


def _select(
    client: "_RestClient",
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
        raise FixtureError("Could not write the sanitized fixture evidence.") from None


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise FixtureError("The fixture manifest is unavailable or invalid.") from None
    if not isinstance(payload, dict) or payload.get("contract") != FIXTURE_CONTRACT:
        raise FixtureError("The fixture manifest has an unexpected contract.")
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


def _new_manifest(env: Mapping[str, str], uuid_factory: UuidFactory) -> dict[str, Any]:
    tournament_id = str(uuid_factory())
    draw_id = str(uuid_factory())
    team_ids = [str(uuid_factory()) for _ in range(4)]
    game_id = str(uuid_factory())
    run_id = str(env.get("GITHUB_RUN_ID") or "local").strip()[:40] or "local"
    run_attempt = str(env.get("GITHUB_RUN_ATTEMPT") or "1").strip()[:12] or "1"
    candidate_sha = _required_env(env, "CANDIDATE_SHA")
    if CANDIDATE_SHA_RE.fullmatch(candidate_sha) is None:
        raise FixtureError("CANDIDATE_SHA must be an exact lowercase commit SHA.")
    marker = f"{FIXTURE_SOURCE}:{run_id}:{run_attempt}:{tournament_id}"
    return {
        "contract": FIXTURE_CONTRACT,
        "status": "planned",
        "created_at": _utc_now(),
        "candidate_sha": candidate_sha,
        "github_run_id": run_id,
        "github_run_attempt": run_attempt,
        "supabase_project_ref": EXPECTED_SUPABASE_PROJECT_REF,
        "club_id": EXPECTED_CLUB_ID,
        "marker": marker,
        "tournament_id": tournament_id,
        "draw_id": draw_id,
        "team_ids": team_ids,
        "game_id": game_id,
        "original_score": [ORIGINAL_SCORE_A, ORIGINAL_SCORE_B],
        "exercise_score": [EXERCISE_SCORE_A, EXERCISE_SCORE_B],
    }


def _manifest_ids(manifest: Mapping[str, Any]) -> tuple[str, str, list[str], str]:
    if manifest.get("supabase_project_ref") != EXPECTED_SUPABASE_PROJECT_REF:
        raise FixtureError("The fixture manifest belongs to another Supabase project.")
    if manifest.get("club_id") != EXPECTED_CLUB_ID:
        raise FixtureError("The fixture manifest belongs to another club.")
    if CANDIDATE_SHA_RE.fullmatch(str(manifest.get("candidate_sha") or "")) is None:
        raise FixtureError("The fixture manifest is not bound to an exact candidate.")
    marker = str(manifest.get("marker") or "")
    if not marker.startswith(f"{FIXTURE_SOURCE}:") or "\n" in marker or "\r" in marker:
        raise FixtureError("The fixture manifest has an invalid ownership marker.")
    tournament_id = _uuid_text(manifest.get("tournament_id"), label="tournament ID")
    draw_id = _uuid_text(manifest.get("draw_id"), label="draw ID")
    raw_team_ids = manifest.get("team_ids")
    if not isinstance(raw_team_ids, list) or len(raw_team_ids) != 4:
        raise FixtureError("The fixture manifest must contain four team IDs.")
    team_ids = [_uuid_text(value, label="team ID") for value in raw_team_ids]
    if len(set(team_ids)) != 4:
        raise FixtureError("The fixture manifest contains duplicate team IDs.")
    game_id = _uuid_text(manifest.get("game_id"), label="game ID")
    all_ids = [tournament_id, draw_id, *team_ids, game_id]
    if len(set(all_ids)) != len(all_ids):
        raise FixtureError("The fixture manifest contains colliding IDs.")
    return tournament_id, draw_id, team_ids, game_id


def _assert_exact_insert(
    rows: list[dict[str, Any]],
    *,
    expected_ids: set[str],
    operation: str,
) -> None:
    returned_ids = {str(row.get("id") or "") for row in rows}
    if len(rows) != len(expected_ids) or returned_ids != expected_ids:
        raise FixtureError(f"{operation} did not return the exact planned rows.")


def _event_tags(marker: str) -> dict[str, list[str]]:
    return {"skill_levels": [], "date_tags": ["staging-only", marker]}


def _marker_present(row: Mapping[str, Any], marker: str) -> bool:
    tags = row.get("event_tags")
    return (
        isinstance(tags, Mapping)
        and isinstance(tags.get("date_tags"), list)
        and marker in tags.get("date_tags", [])
    )


def _fixture_rows(
    client: _RestClient,
    manifest: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    tournament_id, draw_id, team_ids, game_id = _manifest_ids(manifest)
    rows: dict[str, list[dict[str, Any]]] = {}
    rows["tournaments"] = _select(
        client,
        "tournaments",
        filters=(("id", f"eq.{tournament_id}"),),
        operation="Tournament fixture readback",
    )
    rows["draws"] = _select(
        client,
        "tournament_event_draws",
        filters=(("tournament_id", f"eq.{tournament_id}"),),
        operation="Tournament draw fixture readback",
    )
    rows["teams"] = _select(
        client,
        "tournament_teams",
        filters=(("tournament_id", f"eq.{tournament_id}"),),
        operation="Tournament team fixture readback",
    )
    rows["games"] = _select(
        client,
        "tournament_games",
        filters=(("tournament_id", f"eq.{tournament_id}"),),
        operation="Tournament game fixture readback",
    )
    rows["podium"] = _select(
        client,
        "tournament_podium",
        filters=(("tournament_id", f"eq.{tournament_id}"),),
        operation="Tournament podium safety readback",
    )
    rows["matches_by_tournament"] = _select(
        client,
        "matches",
        filters=(("tournament_id", f"eq.{tournament_id}"),),
        select="id,tournament_id,tournament_game_id",
        operation="Official tournament-match safety readback",
    )
    rows["matches_by_game"] = _select(
        client,
        "matches",
        filters=(("tournament_game_id", f"eq.{game_id}"),),
        select="id,tournament_id,tournament_game_id",
        operation="Official game-match safety readback",
    )
    rows["operations"] = _select(
        client,
        "tournament_admin_operations",
        filters=(
            ("club_id", f"eq.{EXPECTED_CLUB_ID}"),
            ("entity_id", f"eq.{draw_id}"),
        ),
        select=(
            "operation_key,surface,action,status,entity_id,"
            "client_idempotency_key"
        ),
        operation="Tournament Live operation safety readback",
    )
    for table in UNOWNED_TOURNAMENT_CHILD_TABLES:
        rows[f"unowned_child:{table}"] = _select(
            client,
            table,
            filters=(("tournament_id", f"eq.{tournament_id}"),),
            select="id,tournament_id",
            operation="Unexpected tournament-child safety readback",
        )
    rows["_expected_team_ids"] = [{"id": value} for value in team_ids]
    rows["_expected_game_id"] = [{"id": game_id}]
    return rows


def _validate_owned_rows(
    rows: Mapping[str, list[dict[str, Any]]],
    manifest: Mapping[str, Any],
    *,
    require_complete: bool,
    require_restored: bool,
) -> None:
    tournament_id, draw_id, team_ids, game_id = _manifest_ids(manifest)
    marker = str(manifest["marker"])
    tournaments = rows["tournaments"]
    draws = rows["draws"]
    teams = rows["teams"]
    games = rows["games"]
    if len(tournaments) > 1 or len(draws) > 1 or len(games) > 1 or len(teams) > 4:
        raise FixtureError("The disposable fixture scope contains unexpected duplicate rows.")
    if tournaments:
        tournament = tournaments[0]
        if (
            str(tournament.get("id") or "") != tournament_id
            or str(tournament.get("club_id") or "") != EXPECTED_CLUB_ID
            or str(tournament.get("name") or "") != f"{FIXTURE_NAME_PREFIX} {tournament_id[:8]}"
            or str(tournament.get("status") or "").upper() != "DRAFT"
            or int(tournament.get("team_count") or 0) != 4
            or not _marker_present(tournament, marker)
        ):
            raise FixtureError("The tournament row does not match the fixture ownership marker.")
    if draws:
        draw = draws[0]
        if (
            str(draw.get("id") or "") != draw_id
            or str(draw.get("tournament_id") or "") != tournament_id
            or str(draw.get("name") or "") != f"Parity score {draw_id[:8]}"
            or str(draw.get("status") or "").lower() != "draft"
        ):
            raise FixtureError("The draw row does not match the fixture manifest.")
    expected_team_ids = set(team_ids)
    if {str(row.get("id") or "") for row in teams} - expected_team_ids:
        raise FixtureError("The tournament contains an unowned team row.")
    expected_team_numbers = {
        team_id: index for index, team_id in enumerate(team_ids, start=1)
    }
    for team in teams:
        team_id = str(team.get("id") or "")
        if (
            str(team.get("tournament_id") or "") != tournament_id
            or str(team.get("draw_id") or "") != draw_id
            or str(team.get("club_id") or "") != EXPECTED_CLUB_ID
            or int(team.get("team_number") or 0) != expected_team_numbers.get(team_id)
            or str(team.get("source") or "") != FIXTURE_SOURCE
            or str(team.get("notes") or "") != marker
            or team.get("player1_id") is not None
            or team.get("player2_id") is not None
        ):
            raise FixtureError("A team row does not match the fixture manifest.")
    if games:
        game = games[0]
        if (
            str(game.get("id") or "") != game_id
            or str(game.get("tournament_id") or "") != tournament_id
            or str(game.get("draw_id") or "") != draw_id
            or str(game.get("club_id") or "") != EXPECTED_CLUB_ID
            or str(game.get("stage") or "").upper() != "ROUND_ROBIN"
            or str(game.get("team_a_id") or "") != team_ids[0]
            or str(game.get("team_b_id") or "") != team_ids[1]
        ):
            raise FixtureError("The game row does not match the fixture manifest.")
        if require_restored and (
            int(game.get("score_a") or -1) != ORIGINAL_SCORE_A
            or int(game.get("score_b") or -1) != ORIGINAL_SCORE_B
            or str(game.get("winner_team_id") or "") != team_ids[0]
            or str(game.get("loser_team_id") or "") != team_ids[1]
            or not game.get("finalized_at")
        ):
            raise FixtureError("The disposable game was not restored to its baseline.")
    if require_complete and (
        len(tournaments) != 1
        or len(draws) != 1
        or len(teams) != 4
        or {str(row.get("id") or "") for row in teams} != expected_team_ids
        or len(games) != 1
    ):
        raise FixtureError("The disposable fixture readback is incomplete.")
    if rows["podium"]:
        raise FixtureError("Refusing cleanup because the disposable fixture has podium rows.")
    if any(str(row.get("stage") or "").upper() == "PLAYOFF" for row in games):
        raise FixtureError("Refusing cleanup because the disposable fixture has playoff rows.")
    if rows["matches_by_tournament"] or rows["matches_by_game"]:
        raise FixtureError("Refusing cleanup because an official match references the fixture.")
    if any(
        rows[f"unowned_child:{table}"]
        for table in UNOWNED_TOURNAMENT_CHILD_TABLES
    ):
        raise FixtureError(
            "Refusing cleanup because an unowned tournament child references the fixture."
        )
    active = [
        row
        for row in rows["operations"]
        if str(row.get("status") or "").strip().lower() in ACTIVE_OPERATION_STATUSES
    ]
    if active:
        raise FixtureError("Refusing cleanup because a tournament operation is unresolved.")


def _append_github_env(path: Path, manifest: Mapping[str, Any]) -> None:
    tournament_id, draw_id, _team_ids, game_id = _manifest_ids(manifest)
    values = {
        "JUPR_TOURNAMENT_LIVE_TOURNAMENT_ID": tournament_id,
        "JUPR_TOURNAMENT_LIVE_DRAW_ID": draw_id,
        "JUPR_TOURNAMENT_LIVE_GAME_ID": game_id,
        "JUPR_TOURNAMENT_LIVE_ORIGINAL_SCORE_A": str(ORIGINAL_SCORE_A),
        "JUPR_TOURNAMENT_LIVE_ORIGINAL_SCORE_B": str(ORIGINAL_SCORE_B),
        "JUPR_TOURNAMENT_LIVE_EXERCISE_SCORE_A": str(EXERCISE_SCORE_A),
        "JUPR_TOURNAMENT_LIVE_EXERCISE_SCORE_B": str(EXERCISE_SCORE_B),
        "JUPR_TOURNAMENT_LIVE_ALLOW_MUTATION_E2E": "1",
    }
    try:
        with path.open("a", encoding="utf-8") as handle:
            for name, value in values.items():
                if "\n" in value or "\r" in value:
                    raise FixtureError("A prepared fixture value is invalid.")
                handle.write(f"{name}={value}\n")
    except FixtureError:
        raise
    except OSError:
        raise FixtureError("Could not append fixture values to GITHUB_ENV.") from None


def prepare_fixture(
    *,
    report_dir: Path,
    github_env: Path,
    env: Mapping[str, str] | None = None,
    transport: Transport = _default_transport,
    uuid_factory: UuidFactory = uuid4,
) -> dict[str, Any]:
    environment = os.environ if env is None else env
    origin, service_role_key = _validate_staging_environment(environment)
    manifest_path = report_dir / MANIFEST_NAME
    manifest = _new_manifest(environment, uuid_factory)
    _manifest_ids(manifest)
    _write_json(manifest_path, manifest)
    client = _RestClient(
        origin=origin,
        service_role_key=service_role_key,
        transport=transport,
    )
    tournament_id, draw_id, team_ids, game_id = _manifest_ids(manifest)
    for table, row_id in (
        ("tournaments", tournament_id),
        ("tournament_event_draws", draw_id),
        *((("tournament_teams", team_id) for team_id in team_ids)),
        ("tournament_games", game_id),
    ):
        collision = _select(
            client,
            table,
            filters=(("id", f"eq.{row_id}"),),
            select="id",
            operation="Fixture collision preflight",
        )
        if collision:
            raise FixtureError("A generated fixture ID already exists; no write was attempted.")

    now = _utc_now()
    marker = str(manifest["marker"])
    tournament_rows = client.rows(
        "tournaments",
        "POST",
        payload={
            "id": tournament_id,
            "club_id": EXPECTED_CLUB_ID,
            "name": f"{FIXTURE_NAME_PREFIX} {tournament_id[:8]}",
            "status": "DRAFT",
            "team_count": 4,
            "playoff_advance_count": None,
            "created_by_admin_id": None,
            "event_tags": _event_tags(marker),
            "created_at": now,
            "updated_at": now,
        },
        operation="Disposable tournament creation",
    )
    _assert_exact_insert(
        tournament_rows,
        expected_ids={tournament_id},
        operation="Disposable tournament creation",
    )
    draw_rows = client.rows(
        "tournament_event_draws",
        "POST",
        payload={
            "id": draw_id,
            "tournament_id": tournament_id,
            "registration_day_id": None,
            "event_option_id": None,
            "name": f"Parity score {draw_id[:8]}",
            "status": "draft",
            "created_at": now,
            "updated_at": now,
        },
        operation="Disposable draw creation",
    )
    _assert_exact_insert(
        draw_rows,
        expected_ids={draw_id},
        operation="Disposable draw creation",
    )
    team_payload = [
        {
            "id": team_id,
            "tournament_id": tournament_id,
            "draw_id": draw_id,
            "club_id": EXPECTED_CLUB_ID,
            "registration_day_id": None,
            "event_option_id": None,
            "team_number": index,
            "player1_id": None,
            "player2_id": None,
            "seed": index,
            "source": FIXTURE_SOURCE,
            "notes": marker,
            "created_at": now,
            "updated_at": now,
        }
        for index, team_id in enumerate(team_ids, start=1)
    ]
    team_rows = client.rows(
        "tournament_teams",
        "POST",
        payload=team_payload,
        operation="Disposable team creation",
    )
    _assert_exact_insert(
        team_rows,
        expected_ids=set(team_ids),
        operation="Disposable team creation",
    )
    game_rows = client.rows(
        "tournament_games",
        "POST",
        payload={
            "id": game_id,
            "tournament_id": tournament_id,
            "draw_id": draw_id,
            "club_id": EXPECTED_CLUB_ID,
            "registration_day_id": None,
            "event_option_id": None,
            "stage": "ROUND_ROBIN",
            "rr_round_number": 1,
            "rr_slot_number": 1,
            "playoff_game_code": None,
            "playoff_round": None,
            "team_a_id": team_ids[0],
            "team_b_id": team_ids[1],
            "team_a_source": None,
            "team_b_source": None,
            "score_a": ORIGINAL_SCORE_A,
            "score_b": ORIGINAL_SCORE_B,
            "winner_team_id": team_ids[0],
            "loser_team_id": team_ids[1],
            "finalized_at": now,
            "created_at": now,
            "updated_at": now,
        },
        operation="Disposable game creation",
    )
    _assert_exact_insert(
        game_rows,
        expected_ids={game_id},
        operation="Disposable game creation",
    )
    readback = _fixture_rows(client, manifest)
    _validate_owned_rows(
        readback,
        manifest,
        require_complete=True,
        require_restored=True,
    )
    prepared = {
        **manifest,
        "status": "prepared",
        "prepared_at": _utc_now(),
        "verified_counts": {
            "tournaments": 1,
            "draws": 1,
            "teams": 4,
            "games": 1,
            "official_matches": 0,
            "podium": 0,
            "playoffs": 0,
        },
    }
    _write_json(manifest_path, prepared)
    _append_github_env(github_env, prepared)
    return prepared


def _delete_exact(
    client: _RestClient,
    table: str,
    *,
    row_id: str,
    extra_filters: Sequence[tuple[str, str]] = (),
    operation: str,
) -> None:
    rows = client.rows(
        table,
        "DELETE",
        query=(("id", f"eq.{row_id}"), *extra_filters),
        operation=operation,
    )
    if len(rows) != 1 or str(rows[0].get("id") or "") != row_id:
        raise FixtureError(f"{operation} did not delete the exact owned row.")


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
            "deleted_counts": {
                "tournaments": 0,
                "draws": 0,
                "teams": 0,
                "games": 0,
            },
        }
        _write_json(cleanup_path, report)
        return report

    manifest = _load_manifest(manifest_path)
    origin, service_role_key = _validate_staging_environment(environment)
    client = _RestClient(
        origin=origin,
        service_role_key=service_role_key,
        transport=transport,
    )
    tournament_id, draw_id, team_ids, game_id = _manifest_ids(manifest)
    before = _fixture_rows(client, manifest)
    _validate_owned_rows(
        before,
        manifest,
        require_complete=False,
        require_restored=bool(before["games"]),
    )
    if not before["tournaments"]:
        if before["draws"] or before["teams"] or before["games"]:
            raise FixtureError("Fixture children exist without the owned tournament.")
        report = {
            "contract": FIXTURE_CONTRACT,
            "status": "already_absent",
            "cleaned_at": _utc_now(),
            "tournament_id": tournament_id,
            "draw_id": draw_id,
            "game_id": game_id,
            "deleted_counts": {
                "tournaments": 0,
                "draws": 0,
                "teams": 0,
                "games": 0,
            },
            "retained_completed_operation_keys": sorted(
                str(row.get("operation_key") or "")
                for row in before["operations"]
                if str(row.get("status") or "").lower() == "completed"
                and row.get("operation_key")
            ),
        }
        _write_json(cleanup_path, report)
        return report

    deleted_counts = {"tournaments": 0, "draws": 0, "teams": 0, "games": 0}
    if before["games"]:
        _delete_exact(
            client,
            "tournament_games",
            row_id=game_id,
            extra_filters=(
                ("tournament_id", f"eq.{tournament_id}"),
                ("club_id", f"eq.{EXPECTED_CLUB_ID}"),
            ),
            operation="Disposable game cleanup",
        )
        deleted_counts["games"] = 1
    existing_team_ids = {str(row.get("id") or "") for row in before["teams"]}
    for team_id in team_ids:
        if team_id not in existing_team_ids:
            continue
        _delete_exact(
            client,
            "tournament_teams",
            row_id=team_id,
            extra_filters=(
                ("tournament_id", f"eq.{tournament_id}"),
                ("draw_id", f"eq.{draw_id}"),
                ("club_id", f"eq.{EXPECTED_CLUB_ID}"),
            ),
            operation="Disposable team cleanup",
        )
        deleted_counts["teams"] += 1
    if before["draws"]:
        _delete_exact(
            client,
            "tournament_event_draws",
            row_id=draw_id,
            extra_filters=(("tournament_id", f"eq.{tournament_id}"),),
            operation="Disposable draw cleanup",
        )
        deleted_counts["draws"] = 1
    _delete_exact(
        client,
        "tournaments",
        row_id=tournament_id,
        extra_filters=(("club_id", f"eq.{EXPECTED_CLUB_ID}"),),
        operation="Disposable tournament cleanup",
    )
    deleted_counts["tournaments"] = 1

    after = _fixture_rows(client, manifest)
    if any(after[name] for name in ("tournaments", "draws", "teams", "games", "podium")):
        raise FixtureError("Disposable fixture cleanup left a core row behind.")
    if after["matches_by_tournament"] or after["matches_by_game"]:
        raise FixtureError("Disposable fixture cleanup found an official match reference.")
    if any(
        after[f"unowned_child:{table}"]
        for table in UNOWNED_TOURNAMENT_CHILD_TABLES
    ):
        raise FixtureError("Disposable fixture cleanup left a tournament child behind.")
    report = {
        "contract": FIXTURE_CONTRACT,
        "status": "cleaned",
        "cleaned_at": _utc_now(),
        "tournament_id": tournament_id,
        "draw_id": draw_id,
        "game_id": game_id,
        "deleted_counts": deleted_counts,
        "verified_remaining_core_rows": 0,
        "verified_official_matches": 0,
        "retained_completed_operation_keys": sorted(
            str(row.get("operation_key") or "")
            for row in after["operations"]
            if str(row.get("status") or "").lower() == "completed"
            and row.get("operation_key")
        ),
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
            tournament_id, draw_id, _team_ids, game_id = _manifest_ids(manifest)
            report.update(
                {
                    "candidate_sha": manifest["candidate_sha"],
                    "tournament_id": tournament_id,
                    "draw_id": draw_id,
                    "game_id": game_id,
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
                "Prepared one disposable Tournament Live score fixture "
                f"for {prepared['supabase_project_ref']}."
            )
        else:
            cleaned = cleanup_fixture(report_dir=args.report_dir)
            print(f"Tournament Live fixture cleanup status: {cleaned['status']}.")
    except FixtureError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
