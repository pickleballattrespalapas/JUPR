from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

from jupr_app.data.client import make_supabase

PROD_MARKERS = ("prod", "production", "live")
SUPABASE_OBJECTS = (
    "clubs",
    "public_leaderboards",
    "admin_role_assignments",
    "admin_activity_log",
    "replay_jobs",
)


@dataclass
class CheckResult:
    status: str
    detail: str



def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}



def _mask_url(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        parts = parse.urlsplit(raw)
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        path = parts.path if parts.path else ""
        return parse.urlunsplit((parts.scheme, host, path, "", ""))
    except Exception:
        return "<unparseable-url>"



def _looks_production(value: str | None) -> bool:
    if not value:
        return False
    lower = value.lower()
    return any(marker in lower for marker in PROD_MARKERS)



def _check_supabase_objects(summary: dict[str, Any], require_supabase: bool) -> None:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not (url and key):
        if require_supabase:
            summary["errors"].append("Supabase checks requested but SUPABASE_URL/key is missing.")
        return

    try:
        supabase = make_supabase(url, key)
    except Exception as exc:
        summary["errors"].append(f"Failed to initialize Supabase client: {exc}")
        return

    checked: dict[str, dict[str, str]] = {}
    for obj in SUPABASE_OBJECTS:
        try:
            supabase.table(obj).select("*").limit(1).execute()
            checked[obj] = {"status": "ok", "detail": "readable"}
        except Exception as exc:
            checked[obj] = {"status": "error", "detail": str(exc)}
            summary["errors"].append(f"Supabase object check failed: {obj}")
    summary["checked_tables"] = checked



def _http_get_json(url: str) -> tuple[int, Any, str | None]:
    try:
        req = request.Request(url, method="GET")
        with request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body), None
    except error.HTTPError as exc:
        return exc.code, None, f"HTTP {exc.code}"
    except Exception as exc:
        return 0, None, str(exc)



def _check_api(summary: dict[str, Any], base_url: str, club_slug: str) -> None:
    base = base_url.rstrip("/")
    endpoints = {
        "/health": f"{base}/health",
        f"/clubs/{club_slug}": f"{base}/clubs/{club_slug}",
        f"/clubs/{club_slug}/leaderboards": f"{base}/clubs/{club_slug}/leaderboards",
    }
    results: dict[str, dict[str, Any]] = {}

    for path, url in endpoints.items():
        status, payload, err = _http_get_json(url)
        if err:
            results[path] = {"status": "error", "http_status": status, "detail": err}
            summary["errors"].append(f"API check failed: {path}")
            continue
        results[path] = {"status": "ok", "http_status": status}
        if path == "/health" and payload.get("ok") is not True:
            results[path]["status"] = "warning"
            summary["warnings"].append("/health reachable but ok=true not present.")
        if path.startswith("/clubs/") and path.endswith("/leaderboards"):
            if not isinstance(payload, dict) or "club" not in payload or "leaderboard" not in payload:
                results[path]["status"] = "warning"
                summary["warnings"].append("Leaderboard response missing expected keys.")
        if path.startswith("/clubs/") and not path.endswith("/leaderboards"):
            if not isinstance(payload, dict) or not {"id", "slug", "name"}.issubset(payload.keys()):
                results[path]["status"] = "warning"
                summary["warnings"].append("Club response missing one or more of id/slug/name.")

    summary["checked_endpoints"] = results



def run_checks(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    env = os.getenv("JUPR_ENV")
    summary: dict[str, Any] = {
        "ok": True,
        "warnings": [],
        "errors": [],
        "environment": {
            "JUPR_ENV": env,
            "SUPABASE_URL": _mask_url(os.getenv("SUPABASE_URL")),
            "DATABASE_URL": _mask_url(os.getenv("DATABASE_URL")),
            "SUPABASE_TEST_DATABASE_URL": _mask_url(os.getenv("SUPABASE_TEST_DATABASE_URL")),
            "JUPR_API_BASE_URL": _mask_url(args.api_base_url or os.getenv("JUPR_API_BASE_URL")),
        },
        "checked_tables": {},
        "checked_endpoints": {},
    }

    if env == "production":
        summary["errors"].append("Refusing to run: JUPR_ENV=production")
        summary["ok"] = False
        return 2, summary

    if env != "staging":
        summary["errors"].append("JUPR_ENV must be set to staging.")

    if args.require_supabase:
        if not os.getenv("SUPABASE_URL"):
            summary["errors"].append("SUPABASE_URL is required when --require-supabase is passed.")
        if not (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")):
            summary["errors"].append(
                "SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY is required when --require-supabase is passed."
            )

    if not args.allow_next_admin_score_entry:
        if _truthy(os.getenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY")):
            summary["errors"].append("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY must be disabled for staging verification.")
        if _truthy(os.getenv("NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY")):
            summary["errors"].append(
                "NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY must be disabled for staging verification."
            )

    for var_name in ("SUPABASE_URL", "DATABASE_URL", "SUPABASE_TEST_DATABASE_URL"):
        value = os.getenv(var_name)
        if _looks_production(value):
            summary["warnings"].append(f"{var_name} appears to include production markers.")
    api_base = args.api_base_url or os.getenv("JUPR_API_BASE_URL")
    if _looks_production(api_base):
        summary["warnings"].append("JUPR_API_BASE_URL appears to include production markers.")

    _check_supabase_objects(summary, require_supabase=args.require_supabase)

    if api_base:
        _check_api(summary, api_base, args.club_slug)
    elif args.require_api:
        summary["errors"].append("API checks required but no API base URL provided.")

    summary["ok"] = not summary["errors"]
    return (0 if summary["ok"] else 1), summary



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read-only staging environment verification.")
    parser.add_argument("--allow-next-admin-score-entry", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--api-base-url")
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument("--require-supabase", action="store_true")
    parser.add_argument("--club-slug", default="tres-palapas")
    parser.add_argument("--club-id", default="tres_palapas")
    return parser



def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc, summary = run_checks(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return rc

    print("[staging-check] Verification summary")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
