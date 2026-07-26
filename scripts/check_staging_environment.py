from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any
from urllib import error, parse, request

from jupr_app.data.client import make_supabase
from scripts.staging_write_waves import (
    ALL_STAGING_WRITE_FLAGS,
    ALWAYS_DISABLED_FLAGS,
    NO_WRITE_WAVE,
    REGISTRATION_SECRET_WAVES,
    STAGING_WRITE_WAVES,
    expected_write_flags,
)

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
    "admin_guarded_operations",
    "admin_player_merge_operations",
    "communications_admin_operations",
    "match_edit_operations",
    "replay_jobs",
    "live_sessions",
    "league_live_sessions",
    "league_live_rounds",
    "league_live_courts",
    "league_live_guest_players",
    "league_live_publish_operations",
    "public_support_requests",
    "player_profile_update_subscriptions",
    "player_profile_update_outbox",
    "player_weekly_profile_digests",
    "badges",
    "player_badges",
    "badge_eval_queue",
    "badge_eval_runs",
    "worker_run_log",
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
    "tournament_admin_operations",
    "weekly_recaps",
    "ladder_settings",
    "ladder_roster",
    "ladder_challenges",
    "live_ladder_admin_operations",
    "public_live_operations",
)

FULL_NEXT_ADMIN_FLAGS = (
    "JUPR_ENABLE_NEXT_ADMIN_SHELL",
    "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG",
    "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
    "JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES",
    "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS",
    "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP",
    "JUPR_ENABLE_NEXT_ADMIN_MONEYBALL",
    "JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE",
    "JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER",
    "JUPR_ENABLE_NEXT_ADMIN_MATCH_CANONICAL_AUDIT",
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
    "/admin/clubs/{club_id}/tournament-live/status",
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
        parsed = parse.urlsplit(raw)
        host = (parsed.hostname or "").strip().lower()
    except Exception:
        return None
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or not host.endswith(".supabase.co")
    ):
        return None
    project_ref = host.removesuffix(".supabase.co")
    if not project_ref or "." in project_ref:
        return None
    return project_ref


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
    summary["next_admin_flags"] = {
        "required": required,
        "required_enabled_count": sum(1 for value in required.values() if value),
        "required_total_count": len(required),
    }
    missing = [name for name, enabled in required.items() if not enabled]
    if missing and expect_full_next_admin:
        summary["errors"].append("Full Next admin staging requested, but these flags are disabled: " + ", ".join(missing))
    elif missing:
        summary["warnings"].append("Some Next admin workflow flags are disabled: " + ", ".join(missing))


def _check_staging_write_wave(
    summary: dict[str, Any],
    *,
    expected_wave: str | None,
    expect_full_next_admin: bool,
) -> str | None:
    actual_wave = os.getenv("JUPR_STAGING_WRITE_WAVE", "").strip() or None
    requested_wave = str(expected_wave or "").strip() or None
    if requested_wave is not None and requested_wave not in STAGING_WRITE_WAVES:
        summary["errors"].append(f"Unknown expected staging write wave: {requested_wave}")
        requested_wave = None
    if expect_full_next_admin and requested_wave is None:
        summary["errors"].append(
            "Full Next admin staging verification requires an explicit --write-wave selector."
        )
    if requested_wave is not None and actual_wave != requested_wave:
        summary["errors"].append(
            f"Staging write wave mismatch: expected {requested_wave!r}, got {actual_wave!r}."
        )

    expected = (
        expected_write_flags(requested_wave)
        if requested_wave is not None
        else {name: False for name in ALL_STAGING_WRITE_FLAGS}
    )
    actual = _flag_status(ALL_STAGING_WRITE_FLAGS)
    mismatched = [name for name in ALL_STAGING_WRITE_FLAGS if actual[name] is not expected[name]]
    if requested_wave is not None and mismatched:
        summary["errors"].append(
            "Staging write gates do not exactly match the selected wave: " + ", ".join(mismatched)
        )

    always_disabled = _flag_status(ALWAYS_DISABLED_FLAGS)
    unsafe_enabled = [name for name, enabled in always_disabled.items() if enabled]
    if unsafe_enabled:
        summary["errors"].append(
            "Staging safety flags must remain disabled: " + ", ".join(unsafe_enabled)
        )

    edit_secret = os.getenv("JUPR_REGISTRATION_EDIT_SECRET", "").strip()
    confirmation_secret = os.getenv("JUPR_REGISTRATION_CONFIRMATION_SECRET", "").strip()
    public_live_token_secret = os.getenv("JUPR_PUBLIC_LIVE_TOKEN_SECRET", "").strip()
    public_live_rate_secret = os.getenv("JUPR_PUBLIC_LIVE_RATE_LIMIT_SECRET", "").strip()
    secrets_required = requested_wave in REGISTRATION_SECRET_WAVES
    if secrets_required:
        if len(edit_secret) < 32:
            summary["errors"].append(
                "The selected write wave requires JUPR_REGISTRATION_EDIT_SECRET with at least 32 characters."
            )
        if len(confirmation_secret) < 32:
            summary["errors"].append(
                "The selected write wave requires JUPR_REGISTRATION_CONFIRMATION_SECRET with at least 32 characters."
            )
        if edit_secret and edit_secret == confirmation_secret:
            summary["errors"].append("Registration edit and confirmation secrets must be distinct.")
    if requested_wave == "public-live":
        if len(public_live_token_secret) < 32:
            summary["errors"].append(
                "The public-live wave requires JUPR_PUBLIC_LIVE_TOKEN_SECRET with at least 32 characters."
            )
        if len(public_live_rate_secret) < 32:
            summary["errors"].append(
                "The public-live wave requires JUPR_PUBLIC_LIVE_RATE_LIMIT_SECRET with at least 32 characters."
            )
        if public_live_token_secret and public_live_token_secret == public_live_rate_secret:
            summary["errors"].append("Public Live token and rate-limit secrets must be distinct.")

    summary["staging_write_wave"] = {
        "expected": requested_wave,
        "actual": actual_wave,
        "known": actual_wave in STAGING_WRITE_WAVES,
        "flags": actual,
        "expected_flags": expected,
        "always_disabled": always_disabled,
        "registration_secrets_required": secrets_required,
        "registration_edit_secret_configured": len(edit_secret) >= 32,
        "registration_confirmation_secret_configured": len(confirmation_secret) >= 32,
        "public_live_token_secret_configured": len(public_live_token_secret) >= 32,
        "public_live_rate_limit_secret_configured": len(public_live_rate_secret) >= 32,
    }
    return requested_wave


def _check_email_mode(summary: dict[str, Any], *, require_dry_run: bool) -> None:
    mode = os.getenv("JUPR_EMAIL_MODE", "").strip().lower()
    redirect = os.getenv("JUPR_STAGING_EMAIL_REDIRECT_TO", "").strip()
    summary["email"] = {
        "JUPR_EMAIL_MODE": mode or None,
        "JUPR_STAGING_EMAIL_REDIRECT_TO_present": bool(redirect),
        "SMTP_HOST_present": bool(os.getenv("SMTP_HOST", "").strip()),
        "SMTP_FROM_EMAIL_present": bool(os.getenv("SMTP_FROM_EMAIL", "").strip()),
    }
    if mode == "live":
        summary["errors"].append("JUPR_EMAIL_MODE=live is forbidden during parity staging waves.")
    if require_dry_run and mode != "dry_run":
        summary["errors"].append("Full parity staging verification requires JUPR_EMAIL_MODE=dry_run.")
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
    expected_write_wave: str | None,
    expected_git_sha: str | None,
    expected_fly_app_name: str | None,
    expected_supabase_project_ref: str | None,
    expected_web_origin: str | None,
) -> None:
    base = base_url.rstrip("/")
    endpoints = {
        "/health": f"{base}/health",
        "/health/live-sessions": f"{base}/health/live-sessions",
        "/admin/operations/status": f"{base}/admin/operations/status",
        f"/clubs/{club_slug}": f"{base}/clubs/{club_slug}",
        f"/clubs/{club_slug}/leaderboards": f"{base}/clubs/{club_slug}/leaderboards",
        f"/clubs/{club_slug}/live-sessions": f"{base}/clubs/{club_slug}/live-sessions",
    }
    for template in ADMIN_STATUS_PATHS:
        path = template.format(club_id=club_id)
        endpoints[path] = f"{base}{path}"
    results: dict[str, dict[str, Any]] = {}
    expected_gates = (
        expected_write_flags(expected_write_wave)
        if expected_write_wave in STAGING_WRITE_WAVES
        else {name: False for name in ALL_STAGING_WRITE_FLAGS}
    )
    status_paths = {template.format(club_id=club_id) for template in ADMIN_STATUS_PATHS}

    def expected_surface_flag(flag_name: str) -> bool:
        """Project read-surface flags separately from controlled write gates."""

        if flag_name in expected_gates:
            return expected_gates[flag_name]
        return _truthy(os.getenv(flag_name))

    def require_bool(path: str, payload: Any, key: str, expected: bool) -> None:
        actual = payload.get(key) if isinstance(payload, dict) else None
        results[path][key] = actual
        if actual is not expected:
            results[path]["status"] = "error"
            summary["errors"].append(
                f"Staging status attestation mismatch for {path} {key}: expected {expected}."
            )

    for path, url in endpoints.items():
        status, payload, err = _http_get_json(url)
        if path == "/admin/operations/status":
            if status == 401:
                results[path] = {
                    "status": "ok",
                    "http_status": status,
                    "protected": True,
                }
            else:
                results[path] = {
                    "status": "error",
                    "http_status": status,
                    "protected": False,
                    "detail": err,
                }
                summary["errors"].append(
                    "Admin operations status must reject unauthenticated requests."
                )
            continue
        if err:
            results[path] = {"status": "error", "http_status": status, "detail": err}
            summary["errors"].append(f"API check failed: {path}")
            continue
        results[path] = {"status": "ok", "http_status": status}
        if path == "/health":
            if not isinstance(payload, dict) or payload.get("ok") is not True:
                results[path]["status"] = "error" if expect_full_next_admin else "warning"
                message = "/health reachable but ok=true not present."
                (summary["errors"] if expect_full_next_admin else summary["warnings"]).append(message)
            if expect_full_next_admin and isinstance(payload, dict):
                expected_identity = {
                    "environment": "staging",
                    "git_commit_sha": str(expected_git_sha or "").strip().lower() or None,
                    "image_build_git_sha": str(expected_git_sha or "").strip().lower()
                    or None,
                    "fly_app_name": str(expected_fly_app_name or "").strip() or None,
                    "supabase_project_ref": str(expected_supabase_project_ref or "").strip().lower() or None,
                    "web_origin": str(expected_web_origin or "").strip().rstrip("/") or None,
                    "staging_write_wave": expected_write_wave,
                    "business_data_write_wave_active": expected_write_wave != NO_WRITE_WAVE,
                    "security_denial_audit_logging_required": True,
                    "jwt_verification_configured": True,
                    "jwt_verification_mode": "jwks",
                    "jwt_verification_project_ref": str(
                        expected_supabase_project_ref or ""
                    ).strip().lower()
                    or None,
                    "public_live_writes_enabled": expected_gates["JUPR_ENABLE_PUBLIC_LIVE_WRITES"],
                    "public_live_production_override_enabled": False,
                }
                for key, expected in expected_identity.items():
                    actual = payload.get(key)
                    if key in {
                        "git_commit_sha",
                        "image_build_git_sha",
                        "supabase_project_ref",
                    } and isinstance(actual, str):
                        actual = actual.lower()
                    results[path][key] = actual
                    if expected is None or actual != expected:
                        results[path]["status"] = "error"
                        summary["errors"].append(
                            f"Staging health identity mismatch for {key}: expected {expected!r}."
                        )
                projected_flags = payload.get("controlled_write_flags")
                results[path]["controlled_write_flags"] = projected_flags
                if projected_flags != expected_gates:
                    results[path]["status"] = "error"
                    summary["errors"].append(
                        "Staging health controlled-write gate projection does not exactly match the selected wave."
                    )
                expected_fingerprint = hashlib.sha256(
                    "\n".join(
                        f"{name}={1 if enabled else 0}"
                        for name, enabled in sorted(expected_gates.items())
                    ).encode("utf-8")
                ).hexdigest()
                results[path]["controlled_write_flag_fingerprint"] = payload.get(
                    "controlled_write_flag_fingerprint"
                )
                if payload.get("controlled_write_flag_fingerprint") != expected_fingerprint:
                    results[path]["status"] = "error"
                    summary["errors"].append(
                        "Staging health controlled-write gate fingerprint is invalid."
                    )
                prerequisites = payload.get("write_prerequisites") or {}
                for key in (
                    "service_role_configured",
                    "api_audit_required",
                    "worker_run_log_required",
                ):
                    require_bool(path, prerequisites, key, True)
                require_bool(path, prerequisites, "live_player_update_email_enabled", False)
                if prerequisites.get("email_mode") != "dry_run":
                    results[path]["status"] = "error"
                    summary["errors"].append(
                        "Staging health write prerequisites do not attest email_mode=dry_run."
                    )
                fly_image_ref = payload.get("fly_image_ref")
                results[path]["fly_image_ref"] = fly_image_ref
                if not isinstance(fly_image_ref, str) or not fly_image_ref.strip():
                    results[path]["status"] = "error"
                    summary["errors"].append("Staging health identity is missing fly_image_ref.")
                if expected_write_wave in REGISTRATION_SECRET_WAVES:
                    require_bool(path, payload, "registration_edit_secret_configured", True)
                    require_bool(path, payload, "registration_confirmation_secret_configured", True)
        if (
            path == "/health/live-sessions"
            and expect_full_next_admin
            and expected_write_wave == "public-live"
        ):
            for key in (
                "ok",
                "service_role_configured",
                "live_sessions_query_ok",
                "operation_ledger_query_ok",
                "durability_schema_ready",
                "token_secret_configured",
                "rate_limit_secret_configured",
            ):
                require_bool(path, payload, key, True)
        if path in status_paths:
            enabled = payload.get("enabled") if isinstance(payload, dict) else None
            results[path]["enabled"] = enabled
            status_gate_names = {
                f"/admin/clubs/{club_id}/match-uploader/status": "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER",
                f"/admin/clubs/{club_id}/players/editor/status": "JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR",
                f"/admin/clubs/{club_id}/player-updates/status": "JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES",
                f"/admin/clubs/{club_id}/verified-updates/status": "JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES",
                f"/admin/clubs/{club_id}/support-requests/status": "JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS",
                f"/admin/clubs/{club_id}/league-manager/live/status": "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN",
                f"/admin/clubs/{club_id}/weekly-recap/status": "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP",
                f"/admin/clubs/{club_id}/badges/status": "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS",
                f"/admin/clubs/{club_id}/tools/status": "JUPR_ENABLE_NEXT_ADMIN_TOOLS",
            }
            gate_name = status_gate_names.get(path)
            expected_enabled = expected_surface_flag(gate_name) if gate_name else True
            if expect_full_next_admin and enabled is not expected_enabled:
                results[path]["status"] = "error"
                summary["errors"].append(
                    f"Full Next admin staging status mismatch for {path}: expected enabled={expected_enabled}."
                )

        if expect_full_next_admin and expected_write_wave is not None and isinstance(payload, dict):
            if path == f"/admin/clubs/{club_id}/player-updates/status":
                require_bool(
                    path,
                    payload,
                    "mutations_enabled",
                    expected_gates["JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS"],
                )
                require_bool(
                    path,
                    payload,
                    "auto_send_enabled",
                    expected_gates["JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS"],
                )
            elif path == f"/admin/clubs/{club_id}/verified-updates/status":
                require_bool(
                    path,
                    payload,
                    "mutations_enabled",
                    expected_gates["JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS"],
                )
            elif path == f"/admin/clubs/{club_id}/weekly-recap/status":
                require_bool(
                    path,
                    payload,
                    "mutations_enabled",
                    expected_gates["JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS"],
                )
            elif path == f"/admin/clubs/{club_id}/league-manager/status":
                require_bool(
                    path,
                    payload,
                    "awards_write_enabled",
                    expected_gates["JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE"],
                )
                require_bool(
                    path,
                    payload,
                    "league_manager_writes_enabled",
                    expected_gates["JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES"],
                )
            elif path == f"/admin/clubs/{club_id}/league-manager/live/status":
                require_bool(
                    path,
                    payload,
                    "submit_enabled",
                    expected_gates["JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT"],
                )
                if expected_gates["JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN"]:
                    require_bool(path, payload, "service_role_configured", True)
            elif path == f"/admin/clubs/{club_id}/tournaments/admin/status":
                mutation_runtime = payload.get("mutation_runtime") or {}
                surface_flags = mutation_runtime.get("surface_flags") or {}
                surface_gate_names = {
                    "tournament": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS",
                    "setup": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS",
                    "registration": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS",
                    "import_handoff": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF",
                    "operations": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
                    "tournament_live": "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES",
                }
                for surface, flag_name in surface_gate_names.items():
                    projected = surface_flags.get(surface) or {}
                    actual_name = projected.get("name")
                    actual_enabled = projected.get("enabled")
                    results[path][f"mutation_runtime.{surface}"] = projected
                    if actual_name != flag_name or actual_enabled is not expected_gates[flag_name]:
                        results[path]["status"] = "error"
                        summary["errors"].append(
                            f"Tournament mutation status mismatch for {surface}: expected {flag_name}={expected_gates[flag_name]}."
                        )
                require_bool(path, mutation_runtime, "service_role_ready", True)

                operations_runtime = payload.get("operations_runtime") or {}
                for key, flag_name in {
                    "operations_mutations_enabled": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
                    "official_publish_enabled": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH",
                    "email_handoff_enabled": "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_EMAIL_HANDOFF",
                    "auto_player_updates_enabled": "JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS",
                }.items():
                    require_bool(path, operations_runtime, key, expected_gates[flag_name])
                if operations_runtime.get("email_mode") != "dry_run":
                    results[path]["status"] = "error"
                    summary["errors"].append("Tournament Operations status is not attesting email_mode=dry_run.")
            elif path == f"/admin/clubs/{club_id}/tournament-live/status":
                expected = expected_gates["JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES"]
                write_flag = payload.get("write_flag") or {}
                results[path]["write_flag"] = write_flag
                if (
                    write_flag.get("name") != "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES"
                    or write_flag.get("enabled") is not expected
                ):
                    results[path]["status"] = "error"
                    summary["errors"].append("Tournament Live write-flag attestation mismatch.")
                require_bool(path, payload, "writes_enabled", expected)
            elif path == f"/admin/clubs/{club_id}/challenge-ladder/status":
                require_bool(
                    path,
                    payload,
                    "writes_enabled",
                    expected_gates["JUPR_ENABLE_STAGING_NEXT_ADMIN_CHALLENGE_LADDER_WRITES"],
                )
            elif path == f"/admin/clubs/{club_id}/moneyball/status":
                require_bool(
                    path,
                    payload,
                    "writes_enabled",
                    expected_gates["JUPR_ENABLE_STAGING_NEXT_ADMIN_MONEYBALL_WRITES"],
                )
            elif path == f"/admin/clubs/{club_id}/jupr-live/status":
                require_bool(
                    path,
                    payload,
                    "writes_enabled",
                    expected_gates["JUPR_ENABLE_STAGING_NEXT_ADMIN_JUPR_LIVE_WRITES"],
                )
            elif path == f"/admin/clubs/{club_id}/match-canonical-audit/status":
                require_bool(
                    path,
                    payload,
                    "normalize_writes_enabled",
                    expected_gates[
                        "JUPR_ENABLE_STAGING_NEXT_ADMIN_MATCH_CANONICAL_NORMALIZE_WRITES"
                    ],
                )
        if path.startswith("/clubs/") and path.endswith("/leaderboards"):
            if not isinstance(payload, dict) or "club" not in payload or "leaderboard" not in payload:
                results[path]["status"] = "warning"
                summary["warnings"].append("Leaderboard response missing expected keys.")
        elif path.startswith("/clubs/") and path.endswith("/live-sessions"):
            if expect_full_next_admin and expected_write_wave is not None:
                expected = expected_gates["JUPR_ENABLE_PUBLIC_LIVE_WRITES"]
                require_bool(path, payload, "write_enabled", expected)
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
    expected_wave = _check_staging_write_wave(
        summary,
        expected_wave=args.write_wave,
        expect_full_next_admin=bool(args.expect_full_next_admin),
    )
    _check_email_mode(summary, require_dry_run=bool(args.expect_full_next_admin))
    _check_supabase_objects(summary, require_supabase=args.require_supabase)
    if api_base:
        _check_api(
            summary,
            api_base,
            args.club_slug,
            args.club_id,
            expect_full_next_admin=bool(args.expect_full_next_admin),
            expected_write_wave=expected_wave,
            expected_git_sha=args.expected_git_sha,
            expected_fly_app_name=args.expected_fly_app_name,
            expected_supabase_project_ref=args.expected_supabase_project_ref,
            expected_web_origin=args.expected_web_origin,
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
    parser.add_argument("--expected-git-sha")
    parser.add_argument("--expected-fly-app-name")
    parser.add_argument("--expected-web-origin")
    parser.add_argument("--write-wave", choices=tuple(STAGING_WRITE_WAVES))
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
