from __future__ import annotations

import argparse
import json
import os
from html.parser import HTMLParser
from typing import Any
from urllib import error, parse, request


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data.strip())

    def text(self) -> str:
        return " ".join(self.parts)


class SmokeFailure(Exception):
    pass


def _http_get(url: str) -> tuple[int, str, str | None]:
    try:
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.status, body, None
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return exc.code, body, f"HTTP {exc.code}"
    except Exception as exc:
        return 0, "", str(exc)


def _get_json(url: str) -> tuple[int, Any, str | None]:
    status, body, err = _http_get(url)
    if err:
        return status, None, err
    try:
        return status, json.loads(body), None
    except json.JSONDecodeError:
        return status, None, "Invalid JSON response"


def _check_api(base_url: str, club_slug: str) -> tuple[list[dict[str, Any]], list[str]]:
    base = base_url.rstrip("/")
    endpoints = [
        ("/health", f"{base}/health"),
        (f"/clubs/{club_slug}", f"{base}/clubs/{club_slug}"),
        (f"/clubs/{club_slug}/leaderboards", f"{base}/clubs/{club_slug}/leaderboards"),
    ]

    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    for path, url in endpoints:
        status, payload, err = _get_json(url)
        check = {"kind": "api", "path": path, "status_code": status, "ok": True}
        if err:
            check["ok"] = False
            check["error"] = err
            failures.append(f"API {path} failed: {err}")
            checks.append(check)
            continue

        if status != 200:
            check["ok"] = False
            failures.append(f"API {path} returned unexpected status {status}")

        if path == "/health":
            if not isinstance(payload, dict) or "ok" not in payload:
                check["ok"] = False
                failures.append("API /health missing expected 'ok' field")
        elif path.endswith("/leaderboards"):
            if not isinstance(payload, dict) or "club" not in payload or "leaderboard" not in payload:
                check["ok"] = False
                failures.append(f"API {path} missing expected keys: club, leaderboard")
        else:
            if not isinstance(payload, dict) or not {"id", "slug", "name"}.issubset(payload.keys()):
                check["ok"] = False
                failures.append(f"API {path} missing expected keys: id, slug, name")

        checks.append(check)

    return checks, failures


def _check_web(base_url: str, club_slug: str) -> tuple[list[dict[str, Any]], list[str]]:
    base = base_url.rstrip("/")
    checks_to_run = [
        ("/", f"{base}/", None),
        (f"/clubs/{club_slug}", f"{base}/clubs/{club_slug}", club_slug.replace("-", " ")),
        (
            f"/clubs/{club_slug}/leaderboards",
            f"{base}/clubs/{club_slug}/leaderboards",
            "leaderboard",
        ),
    ]
    checks: list[dict[str, Any]] = []
    failures: list[str] = []

    for path, url, expected_text in checks_to_run:
        status, body, err = _http_get(url)
        check = {"kind": "web", "path": path, "status_code": status, "ok": True}
        if err:
            check["ok"] = False
            check["error"] = err
            failures.append(f"Web {path} failed: {err}")
            checks.append(check)
            continue

        if status != 200:
            check["ok"] = False
            failures.append(f"Web {path} returned unexpected status {status}")

        if expected_text:
            parser = _TextExtractor()
            parser.feed(body)
            page_text = parser.text().lower()
            if expected_text.lower() not in page_text:
                check["ok"] = False
                failures.append(f"Web {path} missing expected text: {expected_text}")

        checks.append(check)

    return checks, failures


def run_smoke(api_base_url: str, web_base_url: str | None, club_slug: str) -> tuple[int, dict[str, Any]]:
    summary: dict[str, Any] = {
        "ok": True,
        "club_slug": club_slug,
        "api_base_url": parse.urlsplit(api_base_url).netloc,
        "web_base_url": parse.urlsplit(web_base_url).netloc if web_base_url else None,
        "checks": [],
        "failures": [],
    }

    api_checks, api_failures = _check_api(api_base_url, club_slug)
    summary["checks"].extend(api_checks)
    summary["failures"].extend(api_failures)

    if web_base_url:
        web_checks, web_failures = _check_web(web_base_url, club_slug)
        summary["checks"].extend(web_checks)
        summary["failures"].extend(web_failures)

    summary["ok"] = not summary["failures"]
    return (0 if summary["ok"] else 1), summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only staging API/web smoke tests.")
    parser.add_argument("--api-base-url", default=os.getenv("STAGING_JUPR_API_BASE_URL", ""))
    parser.add_argument("--web-base-url", default=os.getenv("STAGING_WEB_BASE_URL", ""))
    parser.add_argument("--club-slug", default="tres-palapas")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    api_base_url = args.api_base_url.strip()
    web_base_url = args.web_base_url.strip() or None

    if not api_base_url:
        summary = {
            "ok": False,
            "club_slug": args.club_slug,
            "checks": [],
            "failures": ["API base URL is required via --api-base-url or STAGING_JUPR_API_BASE_URL."],
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 1

    rc, summary = run_smoke(api_base_url=api_base_url, web_base_url=web_base_url, club_slug=args.club_slug)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
