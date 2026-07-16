from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

from jupr_app.data.client import make_supabase

PROD_MARKERS = ("prod", "production", "live")
SUPABASE_OBJECTS = (
    "clubs",
    "players",
    "matches",
    "public_leaderboards",
    "leagues_metadata",
    "league_ratings",
    "admin_role_assignments",
    "admin_activity_log",
    "admin_match_log_duplicate_resolutions",
    "replay_jobs",
    "live_sessions",
    "league_live_sessions",
    "league_live_rounds",
    "league_live_courts",
    "public_support_requests",
    "player_profile_update_subscriptions",
    "player_profile_update_outbox",
    "player_weekly_profile_digests",
    "badges",
    "player_badges",
    "badge_eval_queue",
    "badge_recompute_runs",
    "tournaments",
    "tournament_registration_settings",
    "tournament_registration_days",
    "tournament_event_options",
    "tournament_registrations",
    "tournament_registration_selections",
    "tournament_registration_partner_requests",
    "tournament_registration_team_links",
    "tournament_registration_team_members",
    "tournament_event_draws",
    "tournament_teams",
    "tournament_games",
    "tournament_podium",
    "weekly_recaps",
    "ladder_settings",
    "ladder_roster",
    "ladder_challenges",
)

FULL_NEXT_ADMIN_FLAGS = (
    "JUPR_ENABLE_NEXT_ADMIN_SHELL",
    "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG",
    "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY",
    "JUPR_ENABLE_NEXT_ADMIN_REPLAY",
    "JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY",
    "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER",
    "JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR",
    "JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES",
    "JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS",
    "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
    "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS",
    "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP",
    "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS",
    "JUPR_ENABLE_NEXT_ADMIN_MONEYBALL",
    "JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE",
    "JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER",
    "JUPR_ENABLE_NEXT_ADMIN_MATCH_CANONICAL_AUDIT",
    "JUPR_ENABLE_NEXT_ADMIN_TOOLS",
)

FULL_NEXT_STAGING_OPTIONAL_FLAGS = (
    "JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS",
)

ADMIN_STATUS_PATHS = (
    "/admin/clubs/{club_id}/match-uploader/status",
    "/admin/clubs/{club_id}/players/editor/status",
    "/admin/clubs/{club_id}/player-updates/status",
    "/admin/clubs/{club_id}/verified-updates/status",
    "/admin/clubs/{club_id}/support-requests/status",
    "/admin/clubs/{club_id}/league-manager/status",
    "/admin/clubs/{club_id}/league-manager/live/status",
    "/admin/clubs/{club_id}/tournaments/admin/status",
    "/admin/clubs/{club_id}/tournaments/setup/status",
    "/admin/clubs/{club_id}/weekly-recap/status",
    "/admin/clubs/{club_id}/badges/status",
    "/admin/clubs/{club_id}/moneyball/status",
    "/admin/clubs/{club_id}/jupr-live/status",
    "/admin/clubs/{club_id}/challenge-ladder/status",
    "/admin/clubs/{club_id}/match-canonical-audit/status",
    "/admin/clubs/{club_id}/tools/status",
)


@dataclass
class CheckResult:
    status: str
    detail: str


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


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


def _supabase_project_ref(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        host = (parse.urlsplit(raw).hostname or "").strip().lower()
    except Exception:
        return None
    if not host.endswith(".supabase.co"):
        return None
    project_ref = host.split(".", 1)[0]
    return project_ref or None


def _check_supabase_isolation(
    summary: dict[str, Any],
    *,
    expected_project_ref: str | None,
    require_isolation: bool,
) -> None:
    expected = str(expected_project_ref or os.getenv("STAGING_SUPABASE_PROJECT_REF") or "").strip().lower()
    actual = _supabase_project_ref(os.getenv("SUPABASE_URL"))
    summary["supabase_isolation"] = {
        "expected_project_ref": expected or None,
        "actual_project_ref": actual,
        "verified": bool(expected and actual and expected == actual),
    }
    if require_isolation and not expected:
        summary["errors"].append(
            "Supabase isolation verification requires --expected-supabase-project-ref or STAGING_SUPABASE_PROJECT_REF."
        )
    if expected and not actual:
        summary["errors"].append("SUPABASE_URL is not a recognizable *.supabase.co project URL.")
    elif expected and actual != expected:
        summary["errors"].append(
            f"Supabase staging project mismatch: expected {expected!r}, got {actual!r}."
        )

    staging_url = str(os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    for prod_name in ("SUPABASE_PROD_URL", "SUPABASE_PRODUCTION_URL"):
        production_url = str(os.getenv(prod_name) or "").strip().rstrip("/")
        if staging_url and production_url and staging_url == production_url:
            summary["errors"].append(f"SUPABASE_URL matches {prod_name}; refusing staging verification.")


def _flag_status(names: tuple[str, ...]) -> dict[str, bool]:
    return {name: _truthy(os.getenv(name)) for name in names}


def _check_full_next_flags(summary: dict[str, Any], *, expect_full_next_admin: bool) -> None:
    required = _flag_status(FULL_NEXT_ADMIN_FLAGS)
    optional = _flag_status(FULL_NEXT_STAGING_OPTIONAL_FLAGS)
    summary["next_admin_flags"] = {
        "required": required,
        "optional": optional,
        "required_enabled_count": sum(1 for value in required.values() if value),
        "required_total_count": len(required),
    }
    missing = [name for name, enabled in required.items() if not enabled]
    if missing and expect_full_next_admin:
        summary["errors"].append("Full Next admin staging requested, but these flags are disabled: " + ", ".join(missing))
    elif missing:
        summary["warnings"].append("Some Next admin workflow flags are disabled: " + ", ".join(missing))


def _check_email_mode(summary: dict[str, Any]) -> None:
    mode = os.getenv("JUPR_EMAIL_MODE", "").strip().lower()
    redirect = os.getenv("JUPR_STAGING_EMAIL_REDIRECT_TO", "").strip()
    summary["email"] = {
        "JUPR_EMAIL_MODE": mode or None,
        "JUPR_STAGING_EMAIL_REDIRECT_TO_present": bool(redirect),
        "SMTP_HOST_present": bool(os.getenv("SMTP_HOST", "").strip()),
        "SMTP_FROM_EMAIL_present": bool(os.getenv("SMTP_FROM_EMAIL", "").strip()),
    }
    if mode == "live":
        summary["warnings"].append("JUPR_EMAIL_MODE=live. For staging validation, dry_run or staging_redirect is safer until final email approval.")
    if mode == "staging_redirect" and not redirect:
        summary["errors"].append("JUPR_EMAIL_MODE=staging_redirect requires JUPR_STAGING_EMAIL_REDIRECT_TO.")


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
        except Exception as exc:  # noqa: BLE001 - environment report should continue
            checked[obj] = {"status": "error", "detail": str(exc)}
            summary["errors"].append(f"Supabase object check failed: {obj}")
    summary["checked_tables"] = checked


def _http_get_json(url: str) -> tuple[int, Any, str | None]:
    try:
        req = request.Request(url, method="GET", headers={"Accept": "application/json"})
        with request.urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body), None
    except error.HTTPError as exc:
        return exc.code, None, f"HTTP {exc.code}"
    except Exception as exc:
        return 0, None, str(exc)


def _check_api(
    summary: dict[str, Any],
    base_url: str,
    club_slug: str,
    club_id: str,
    *,
    expect_full_next_admin: bool,
) -> None:
    base = base_url.rstrip("/")
    endpoints = {
        "/health": f"{base}/health",
        "/admin/operations/status": f"{base}/admin/operations/status",
        f"/clubs/{club_slug}": f"{base}/clubs/{club_slug}",
        f"/clubs/{club_slug}/leaderboards": f"{base}/clubs/{club_slug}/leaderboards",
    }
    for template in ADMIN_STATUS_PATHS:
        path = template.format(club_id=club_id)
        endpoints[path] = f"{base}{path}"
    results: dict[str, dict[str, Any]] = {}
    for path, url in endpoints.items():
        status, payload, err = _http_get_json(url)
        if err:
            results[path] = {"status": "error", "http_status": status, "detail": err}
            summary["errors"].append(f"API check failed: {path}")
            continue
        results[path] = {"status": "ok", "http_status": status}
        if path == "/health" and isinstance(payload, dict) and payload.get("ok") is not True:
            results[path]["status"] = "warning"
            summary["warnings"].append("/health reachable but ok=true not present.")
        if path == "/admin/operations/status" and isinstance(payload, dict):
            enabled = payload.get("enabled_workflows") or []
            results[path]["enabled_workflows"] = enabled
            results[path]["environment"] = payload.get("environment")
            results[path]["write_pilot_enabled"] = payload.get("write_pilot_enabled")
            if expect_full_next_admin and payload.get("environment") != "staging":
                results[path]["status"] = "error"
                summary["errors"].append("Admin operations status is not reporting environment=staging.")
            if expect_full_next_admin and payload.get("write_pilot_enabled") is not True:
                results[path]["status"] = "error"
                summary["errors"].append("Admin operations status is not reporting write_pilot_enabled=true.")
        if path in {template.format(club_id=club_id) for template in ADMIN_STATUS_PATHS}:
            enabled = payload.get("enabled") if isinstance(payload, dict) else None
            results[path]["enabled"] = enabled
            if expect_full_next_admin and enabled is not True:
                results[path]["status"] = "error"
                summary["errors"].append(f"Full Next admin staging requested, but API status is not enabled: {path}")
        if path.startswith("/clubs/") and path.endswith("/leaderboards"):
            if not isinstance(payload, dict) or "club" not in payload or "leaderboard" not in payload:
                results[path]["status"] = "warning"
                summary["warnings"].append("Leaderboard response missing expected keys.")
        elif path.startswith("/clubs/"):
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
        "supabase_isolation": {},
    }
    if env == "production":
        summary["errors"].append("Refusing to run full staging verification with JUPR_ENV=production")
        summary["ok"] = False
        return 2, summary
    if env != "staging":
        summary["errors"].append("JUPR_ENV must be set to staging.")

    if args.require_supabase:
        if not os.getenv("SUPABASE_URL"):
            summary["errors"].append("SUPABASE_URL is required when --require-supabase is passed.")
        if not (os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")):
            summary["errors"].append("SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY is required when --require-supabase is passed.")

    for var_name in ("SUPABASE_URL", "DATABASE_URL", "SUPABASE_TEST_DATABASE_URL"):
        value = os.getenv(var_name)
        if _looks_production(value):
            summary["warnings"].append(f"{var_name} appears to include production markers.")
    api_base = args.api_base_url or os.getenv("JUPR_API_BASE_URL")
    if _looks_production(api_base):
        summary["warnings"].append("JUPR_API_BASE_URL appears to include production markers.")

    _check_supabase_isolation(
        summary,
        expected_project_ref=args.expected_supabase_project_ref,
        require_isolation=bool(args.require_supabase_isolation),
    )
    _check_full_next_flags(summary, expect_full_next_admin=bool(args.expect_full_next_admin))
    _check_email_mode(summary)
    _check_supabase_objects(summary, require_supabase=args.require_supabase)
    if api_base:
        _check_api(
            summary,
            api_base,
            args.club_slug,
            args.club_id,
            expect_full_next_admin=bool(args.expect_full_next_admin),
        )
    elif args.require_api:
        summary["errors"].append("API checks required but no API base URL provided.")
    summary["ok"] = not summary["errors"]
    return (0 if summary["ok"] else 1), summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full Next/FastAPI staging environment verification.")
    parser.add_argument("--expect-full-next-admin", action="store_true", help="Require every Next admin workflow flag needed for full staging validation to be enabled.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--api-base-url")
    parser.add_argument("--require-api", action="store_true")
    parser.add_argument("--require-supabase", action="store_true")
    parser.add_argument("--require-supabase-isolation", action="store_true")
    parser.add_argument("--expected-supabase-project-ref")
    parser.add_argument("--club-slug", default="tres-palapas")
    parser.add_argument("--club-id", default="tres_palapas")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    rc, summary = run_checks(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
