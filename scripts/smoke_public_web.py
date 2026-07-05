#!/usr/bin/env python3
"""Smoke-test the read-only Pickleball Club Sandwich public SaaS surface.

The script is intentionally dependency-free so it can run from a laptop,
GitHub Actions, or a deployment shell before a Vercel/custom-domain cutover.
It checks public/read-only routes, status-only admin migration routes, and the guard
that Next admin score entry remains disabled by default.
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
from typing import BinaryIO, Iterable


DEFAULT_CLUB_SLUG = "tres-palapas"
DEFAULT_CLUB_ID = "tres_palapas"
PREVIEW_BODY_BYTES = 2048
MAX_JSON_BODY_BYTES = 10 * 1024 * 1024

KNOWN_PUBLIC_WEB_HOSTS = {
    "pickleballclubsandwich.com",
    "www.pickleballclubsandwich.com",
    "juprleagues.com",
    "www.juprleagues.com",
}
KNOWN_FASTAPI_BASE_URL = "https://api.juprleagues.com"

API_BASE_ENV_NAMES = (
    "JUPR_API_BASE_URL",
    "STAGING_JUPR_API_BASE_URL",
    "NEXT_PUBLIC_JUPR_API_BASE_URL",
)
WEB_BASE_ENV_NAMES = (
    "JUPR_WEB_BASE_URL",
    "STAGING_WEB_BASE_URL",
    "NEXT_PUBLIC_JUPR_WEB_BASE_URL",
)


@dataclass(frozen=True)
class SmokeCheck:
    name: str
    url: str
    expected_statuses: tuple[int, ...]
    method: str = "GET"
    body: str | None = None
    require_json: bool = False


@dataclass
class SmokeResult:
    name: str
    url: str
    method: str
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


def _clean_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().rstrip("/")
    return cleaned or None


def _join_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def _host(value: str | None) -> str:
    if not value:
        return ""
    try:
        return urllib.parse.urlparse(value).hostname or ""
    except Exception:
        return ""


def _origin(value: str | None) -> tuple[str, str, int | None]:
    if not value:
        return ("", "", None)
    try:
        parsed = urllib.parse.urlparse(value)
    except Exception:
        return ("", "", None)
    return (parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port)


def _base_url_validation_errors(api_base_url: str | None, web_base_url: str | None) -> list[str]:
    errors: list[str] = []
    api_host = _host(api_base_url).lower()

    if api_base_url and api_host in KNOWN_PUBLIC_WEB_HOSTS:
        errors.append(
            "STAGING_JUPR_API_BASE_URL/api_base_url points at a public Next/Vercel web domain "
            f"({api_base_url}). Use the FastAPI origin {KNOWN_FASTAPI_BASE_URL}."
        )

    if api_base_url and web_base_url and _origin(api_base_url) == _origin(web_base_url):
        errors.append(
            "STAGING_JUPR_API_BASE_URL/api_base_url and STAGING_WEB_BASE_URL/web_base_url have the same origin. "
            f"Use {KNOWN_FASTAPI_BASE_URL} for the API and the public website domain for the web app."
        )

    return errors


def _looks_like_html(body: bytes) -> bool:
    preview = body[:512].lower().lstrip()
    return preview.startswith(b"<!doctype html") or preview.startswith(b"<html") or b"<html" in preview[:160]


def _api_base_hint(url: str) -> str:
    return (
        f"{url} returned HTML instead of FastAPI JSON. "
        "The API base URL likely points at the Next/Vercel web domain. "
        "Set STAGING_JUPR_API_BASE_URL/api_base_url to the FastAPI origin, "
        f"for example {KNOWN_FASTAPI_BASE_URL}, and use STAGING_WEB_BASE_URL/web_base_url for the public website."
    )


def _read_response_body(response: BinaryIO, *, require_json: bool) -> tuple[bytes, bool]:
    """Read enough response body for validation without unbounded memory growth."""

    if require_json:
        body = response.read(MAX_JSON_BODY_BYTES + 1)
        return body[:MAX_JSON_BODY_BYTES], len(body) > MAX_JSON_BODY_BYTES

    return response.read(PREVIEW_BODY_BYTES), False


def _request(check: SmokeCheck, timeout_seconds: float) -> SmokeResult:
    started = time.perf_counter()
    data = check.body.encode("utf-8") if check.body is not None else None
    headers = {
        "User-Agent": "public-web-smoke/1.0",
        "Accept": "application/json,text/html,application/xml,text/xml;q=0.9,*/*;q=0.8",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(
        check.url,
        data=data,
        headers=headers,
        method=check.method,
    )

    body_truncated = False

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status = int(response.status)
            body, body_truncated = _read_response_body(response, require_json=check.require_json)
            content_type = response.headers.get("content-type", "")
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        body, body_truncated = _read_response_body(exc, require_json=check.require_json)
        content_type = exc.headers.get("content-type", "") if exc.headers else ""
    except Exception as exc:  # noqa: BLE001 - report smoke failures compactly
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return SmokeResult(
            name=check.name,
            url=check.url,
            method=check.method,
            expected_statuses=check.expected_statuses,
            ok=False,
            elapsed_ms=elapsed_ms,
            error=f"{exc.__class__.__name__}: {exc}",
        )

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    ok = status in check.expected_statuses
    error = None

    if ok and check.require_json:
        looks_json = "json" in content_type.lower()
        if body_truncated:
            ok = False
            error = f"Expected JSON response, but body exceeded {MAX_JSON_BODY_BYTES} bytes"
        else:
            try:
                json.loads(body.decode("utf-8") or "{}")
            except Exception as exc:
                ok = False
                if _looks_like_html(body):
                    error = _api_base_hint(check.url)
                else:
                    error = f"Expected JSON response, but parsing failed: {exc}"
            else:
                if not looks_json:
                    error = f"JSON parsed, but content-type was {content_type!r}"

    if not ok and error is None:
        preview = body.decode("utf-8", "replace").replace("\n", " ")[:240]
        if check.require_json and _looks_like_html(body):
            error = _api_base_hint(check.url)
        else:
            error = f"Expected status {check.expected_statuses}, got {status}. {preview}".strip()

    return SmokeResult(
        name=check.name,
        url=check.url,
        method=check.method,
        expected_statuses=check.expected_statuses,
        ok=ok,
        status=status,
        elapsed_ms=elapsed_ms,
        error=error,
    )


def _build_checks(
    *,
    api_base_url: str | None,
    web_base_url: str | None,
    club_slug: str,
    club_id: str,
    allow_live_unconfigured: bool,
) -> list[SmokeCheck]:
    checks: list[SmokeCheck] = []

    if api_base_url:
        live_statuses = (200, 503) if allow_live_unconfigured else (200,)
        checks.extend(
            [
                SmokeCheck("api: health", _join_url(api_base_url, "/health"), (200,), require_json=True),
                SmokeCheck("api: admin operations status", _join_url(api_base_url, "/admin/operations/status"), (200,), require_json=True),
                SmokeCheck("api: admin match log", _join_url(api_base_url, f"/admin/clubs/{club_id}/match-log"), (200,), require_json=True),
                SmokeCheck("api: admin replay", _join_url(api_base_url, f"/admin/clubs/{club_id}/replay-history"), (200,), require_json=True),
                SmokeCheck("api: admin match uploader", _join_url(api_base_url, f"/admin/clubs/{club_id}/match-uploader/status"), (200,), require_json=True),
                SmokeCheck("api: admin player editor", _join_url(api_base_url, f"/admin/clubs/{club_id}/players/editor/status"), (200,), require_json=True),
                SmokeCheck("api: admin league manager", _join_url(api_base_url, f"/admin/clubs/{club_id}/league-manager/status"), (200,), require_json=True),
                SmokeCheck("api: club", _join_url(api_base_url, f"/clubs/{club_slug}"), (200,), require_json=True),
                SmokeCheck("api: leaderboards", _join_url(api_base_url, f"/clubs/{club_slug}/leaderboards"), (200,), require_json=True),
                SmokeCheck("api: league results", _join_url(api_base_url, f"/clubs/{club_slug}/league-results"), (200,), require_json=True),
                SmokeCheck("api: badge codex", _join_url(api_base_url, f"/clubs/{club_slug}/badges"), (200,), require_json=True),
                SmokeCheck("api: challenge ladder", _join_url(api_base_url, f"/clubs/{club_slug}/challenge-ladder"), (200,), require_json=True),
                SmokeCheck("api: weekly recaps", _join_url(api_base_url, f"/clubs/{club_slug}/weekly-recaps"), (200,), require_json=True),
                SmokeCheck("api: tournament registration", _join_url(api_base_url, f"/clubs/{club_slug}/tournament-registration"), (200,), require_json=True),
                SmokeCheck("api: tournament roster", _join_url(api_base_url, f"/clubs/{club_slug}/tournament-roster"), (200,), require_json=True),
                SmokeCheck("api: players", _join_url(api_base_url, f"/clubs/{club_slug}/players"), (200,), require_json=True),
                SmokeCheck("api: matches", _join_url(api_base_url, f"/clubs/{club_slug}/matches"), (200,), require_json=True),
                SmokeCheck("api: match explorer", _join_url(api_base_url, f"/clubs/{club_slug}/match-explorer"), (200,), require_json=True),
                SmokeCheck("api: live sessions", _join_url(api_base_url, f"/clubs/{club_slug}/live-sessions"), live_statuses, require_json=True),
                SmokeCheck(
                    "api: admin score entry disabled",
                    _join_url(api_base_url, f"/admin/clubs/{club_id}/matches/batch"),
                    (403,),
                    method="POST",
                    body=json.dumps({"matches": [], "source": "public_smoke_guard"}),
                    require_json=True,
                ),
            ]
        )

    if web_base_url:
        for label, path in [
            ("web: home", "/"),
            ("web: sitemap xml", "/sitemap.xml"),
            ("web: site map", "/site-map"),
            ("web: admin operations", "/admin"),
            ("web: admin login", "/admin/login"),
            ("web: admin match log", "/admin/match-log"),
            ("web: admin replay", "/admin/replay-history"),
            ("web: admin match uploader", "/admin/match-uploader"),
            ("web: admin players", "/admin/players"),
            ("web: admin league manager", "/admin/league-manager"),
            ("web: club home", f"/clubs/{club_slug}"),
            ("web: leaderboards", f"/clubs/{club_slug}/leaderboards"),
            ("web: league results", f"/clubs/{club_slug}/league-results"),
            ("web: badge codex", f"/clubs/{club_slug}/badge-codex"),
            ("web: challenge ladder", f"/clubs/{club_slug}/challenge-ladder"),
            ("web: weekly recap", f"/clubs/{club_slug}/weekly-recap"),
            ("web: tournament registration", f"/clubs/{club_slug}/tournament-registration"),
            ("web: tournament roster", f"/clubs/{club_slug}/tournament-roster"),
            ("web: tournament partner board", f"/clubs/{club_slug}/tournament-partner-board"),
            ("web: match explorer", f"/clubs/{club_slug}/match-explorer"),
            ("web: players", f"/clubs/{club_slug}/players"),
            ("web: matches", f"/clubs/{club_slug}/matches"),
            ("web: live", f"/clubs/{club_slug}/live"),
            ("web: ratings explainer", "/how-ratings-work"),
            ("web: faq", "/faq"),
            ("web: privacy", "/privacy"),
            ("web: terms", "/terms"),
            ("web: support", "/support"),
            ("web: contact", "/contact"),
            ("web: data corrections", "/data-corrections"),
        ]:
            checks.append(SmokeCheck(label, _join_url(web_base_url, path), (200,)))
        checks.append(
            SmokeCheck(
                "web: match explorer preview proxy validation",
                _join_url(web_base_url, f"/api/clubs/{club_slug}/match-explorer/preview"),
                (422,),
                require_json=True,
            )
        )

    return checks


def _print_text(results: list[SmokeResult]) -> None:
    for result in results:
        mark = "PASS" if result.ok else "FAIL"
        status = result.status if result.status is not None else "n/a"
        elapsed = f"{result.elapsed_ms}ms" if result.elapsed_ms is not None else "n/a"
        print(f"[{mark}] {result.name} {result.method} {result.url} -> {status} ({elapsed})")
        if result.error:
            print(f"       {result.error}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base-url", default=_first_env(API_BASE_ENV_NAMES))
    parser.add_argument("--web-base-url", default=_first_env(WEB_BASE_ENV_NAMES))
    parser.add_argument("--club-slug", default=os.getenv("JUPR_SMOKE_CLUB_SLUG", DEFAULT_CLUB_SLUG))
    parser.add_argument("--club-id", default=os.getenv("JUPR_SMOKE_CLUB_ID", DEFAULT_CLUB_ID))
    parser.add_argument("--timeout-seconds", type=float, default=10.0)
    parser.add_argument(
        "--allow-live-unconfigured",
        action="store_true",
        help="Treat 503 from /live-sessions as acceptable while staging live_sessions migrations are being applied.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    api_base_url = _clean_base_url(args.api_base_url)
    web_base_url = _clean_base_url(args.web_base_url)

    if not api_base_url and not web_base_url:
        print(
            "Set JUPR_API_BASE_URL/STAGING_JUPR_API_BASE_URL and/or "
            "JUPR_WEB_BASE_URL/STAGING_WEB_BASE_URL, or pass --api-base-url/--web-base-url.",
            file=sys.stderr,
        )
        return 2

    config_errors = _base_url_validation_errors(api_base_url, web_base_url)
    if config_errors:
        print("Invalid public smoke URL configuration:", file=sys.stderr)
        for error in config_errors:
            print(f"- {error}", file=sys.stderr)
        return 2

    checks = _build_checks(
        api_base_url=api_base_url,
        web_base_url=web_base_url,
        club_slug=str(args.club_slug).strip() or DEFAULT_CLUB_SLUG,
        club_id=str(args.club_id).strip() or DEFAULT_CLUB_ID,
        allow_live_unconfigured=bool(args.allow_live_unconfigured),
    )
    results = [_request(check, timeout_seconds=float(args.timeout_seconds)) for check in checks]

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2, sort_keys=True))
    else:
        _print_text(results)

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
