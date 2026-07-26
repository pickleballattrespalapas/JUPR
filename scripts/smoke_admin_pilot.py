#!/usr/bin/env python3
"""Non-mutating smoke checks for the closed-club Match Log + Replay History pilot.

This script proves the runtime flags, admin JWT authorization, and role permissions are
ready before an operator performs the first real Match Log apply or Replay History run.
It intentionally uses invalid/no-op write bodies that should fail *after* auth and role
checks, so it does not mutate match, rating, or replay state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Iterable

DEFAULT_API_BASE_URL = "https://api.juprleagues.com"
DEFAULT_CLUB_ID = "tres_palapas"
API_BASE_ENV_NAMES = ("JUPR_API_BASE_URL", "STAGING_JUPR_API_BASE_URL")
ADMIN_TOKEN_ENV_NAMES = ("JUPR_ADMIN_BEARER_TOKEN", "STAGING_ADMIN_BEARER_TOKEN", "SUPABASE_ACCESS_TOKEN")


@dataclass(frozen=True)
class PilotCheck:
    name: str
    method: str
    path: str
    expected_statuses: tuple[int, ...]
    body: dict[str, Any] | None = None
    require_admin_token: bool = False
    required_json_paths: tuple[tuple[str, Any], ...] = ()
    required_text: str | None = None


@dataclass
class PilotResult:
    name: str
    method: str
    url: str
    expected_statuses: tuple[int, ...]
    ok: bool
    status: int | None = None
    elapsed_ms: int | None = None
    error: str | None = None


def _first_env(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _clean_base_url(value: str | None) -> str:
    return (value or DEFAULT_API_BASE_URL).strip().rstrip("/")


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _read_error_body(exc: urllib.error.HTTPError) -> tuple[int, bytes, str]:
    body = exc.read(1024 * 1024)
    content_type = exc.headers.get("content-type", "") if exc.headers else ""
    return int(exc.code), body, content_type


def _json_value(payload: Any, dotted_path: str) -> Any:
    current = payload
    for part in dotted_path.split("."):
      if isinstance(current, dict):
          current = current.get(part)
      elif isinstance(current, list):
          if part == "length":
              current = len(current)
          else:
              current = current[int(part)]
      else:
          return None
    return current


def _request(check: PilotCheck, *, api_base_url: str, admin_token: str | None, timeout_seconds: float) -> PilotResult:
    url = _join_url(api_base_url, check.path)
    started = time.perf_counter()
    headers = {
        "User-Agent": "admin-pilot-smoke/1.0",
        "Accept": "application/json",
    }
    body_bytes: bytes | None = None
    if check.body is not None:
        headers["Content-Type"] = "application/json"
        body_bytes = json.dumps(check.body).encode("utf-8")
    if check.require_admin_token:
        if not admin_token:
            return PilotResult(check.name, check.method, url, check.expected_statuses, False, error="Missing admin bearer token. Set JUPR_ADMIN_BEARER_TOKEN, STAGING_ADMIN_BEARER_TOKEN, or SUPABASE_ACCESS_TOKEN.")
        headers["Authorization"] = f"Bearer {admin_token}"

    request = urllib.request.Request(url, data=body_bytes, headers=headers, method=check.method)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            raw_body = response.read(1024 * 1024)
            content_type = response.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        status, raw_body, content_type = _read_error_body(exc)
    except Exception as exc:  # noqa: BLE001 - compact smoke output
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return PilotResult(check.name, check.method, url, check.expected_statuses, False, elapsed_ms=elapsed_ms, error=f"{exc.__class__.__name__}: {exc}")

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    ok = status in check.expected_statuses
    error = None
    payload: Any = None
    text = raw_body.decode("utf-8", "replace")

    if ok:
        try:
            payload = json.loads(text or "{}")
        except Exception as exc:
            ok = False
            error = f"Expected JSON response, but parsing failed: {exc}"
        else:
            if "json" not in content_type.lower():
                error = f"JSON parsed, but content-type was {content_type!r}"

    if ok and check.required_text:
        if check.required_text not in text:
            ok = False
            error = f"Expected response text {check.required_text!r} was missing."

    if ok and check.required_json_paths:
        for dotted_path, expected in check.required_json_paths:
            actual = _json_value(payload, dotted_path)
            if actual != expected:
                ok = False
                error = f"Expected JSON path {dotted_path!r} to equal {expected!r}, got {actual!r}."
                break

    if not ok and error is None:
        preview = text.replace("\n", " ")[:300]
        error = f"Expected status {check.expected_statuses}, got {status}. {preview}".strip()

    return PilotResult(
        name=check.name,
        method=check.method,
        url=url,
        expected_statuses=check.expected_statuses,
        ok=ok,
        status=status,
        elapsed_ms=elapsed_ms,
        error=error,
    )


def _build_checks(club_id: str) -> list[PilotCheck]:
    return [
        PilotCheck("api: health", "GET", "/health", (200,), required_json_paths=(("ok", True),)),
        PilotCheck(
            "pilot: operations mode",
            "GET",
            f"/admin/operations/status?club_id={urllib.parse.quote(club_id)}",
            (200,),
            require_admin_token=True,
            required_json_paths=(("write_pilot_enabled", True),),
        ),
        PilotCheck("pilot: match log enabled", "GET", f"/admin/clubs/{club_id}/match-log?limit=25", (200,), required_json_paths=(("enabled", True), ("apply_enabled", True))),
        PilotCheck("pilot: replay enabled", "GET", f"/admin/clubs/{club_id}/replay-history", (200,), required_json_paths=(("enabled", True),)),
        PilotCheck(
            "auth: match log apply permission preflight",
            "PATCH",
            f"/admin/clubs/{club_id}/match-log/edits",
            (400,),
            body={"patches": [], "confirmation_text": "APPLY", "correction_note": "admin pilot preflight", "source": "next_admin_pilot_preflight_noop"},
            require_admin_token=True,
            required_text="No patches provided",
        ),
        PilotCheck(
            "auth: replay permission preflight",
            "POST",
            f"/admin/clubs/{club_id}/replay-history",
            (400,),
            body={"target_reset": "ALL (Full System Reset)", "confirmation_text": "NOT_REPLAY", "source": "next_admin_pilot_preflight_noop"},
            require_admin_token=True,
            required_text="Type REPLAY",
        ),
    ]


def _print_text(results: list[PilotResult]) -> None:
    for result in results:
        mark = "PASS" if result.ok else "FAIL"
        status = result.status if result.status is not None else "n/a"
        elapsed = f"{result.elapsed_ms}ms" if result.elapsed_ms is not None else "n/a"
        print(f"[{mark}] {result.name} {result.method} {result.url} -> {status} ({elapsed})")
        if result.error:
            print(f"       {result.error}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default=_first_env(API_BASE_ENV_NAMES) or DEFAULT_API_BASE_URL)
    parser.add_argument("--club-id", default=os.getenv("JUPR_SMOKE_CLUB_ID", DEFAULT_CLUB_ID))
    parser.add_argument("--admin-token", default=_first_env(ADMIN_TOKEN_ENV_NAMES), help="Supabase access token for a staff admin. Prefer env vars over shell history.")
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    api_base_url = _clean_base_url(args.api_base_url)
    club_id = str(args.club_id or DEFAULT_CLUB_ID).strip() or DEFAULT_CLUB_ID
    admin_token = str(args.admin_token or "").strip() or None
    results = [_request(check, api_base_url=api_base_url, admin_token=admin_token, timeout_seconds=float(args.timeout_seconds)) for check in _build_checks(club_id)]
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
    else:
        _print_text(results)
    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
