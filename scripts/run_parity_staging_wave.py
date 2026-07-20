#!/usr/bin/env python3
"""Run one candidate-bound parity staging wave and reject weak evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.staging_write_waves import expected_write_flags  # noqa: E402

WEB_ROOT = ROOT / "apps" / "web"
EXPECTED_STAGING_PROJECT_REF = "sijpxjxvdtrehmqvirfi"
EXPECTED_STAGING_WEB_ORIGIN = (
    "https://jupr-git-staging-pickleballattrespalapas1.vercel.app"
)
EXPECTED_STAGING_API_ORIGIN = "https://juprleagues-api-staging.fly.dev"
EXPECTED_STAGING_AUTH_ORIGIN = f"https://{EXPECTED_STAGING_PROJECT_REF}.supabase.co"
EXPECTED_FLY_APP_NAME = "juprleagues-api-staging"
MUTATION_CONFIRMATION = "RUN DISPOSABLE STAGING WRITES"
EXPECTED_WRITE_WAVE_BY_EVIDENCE_MODE = {
    "public-read": "none",
    "public-intake-auth": "public-intake-auth",
    "admin-read-export": "none",
    "match-rating-writes": "tournament-live",
}
IMMUTABLE_VERCEL_HOST_RE = re.compile(
    r"[a-z0-9](?:[a-z0-9-]{0,180}[a-z0-9])?-[a-z0-9]{8,64}"
    r"-pickleballattrespalapas1\.vercel\.app"
)

# Order 29 automates only committed route-specific suites. Generic JSON mutation
# plans are intentionally absent: creates, publishes, finalizers, and flows whose
# inverse depends on live response state remain manual-book procedures.
REQUIRED_REAL_SPECS: dict[str, tuple[str, ...]] = {
    "public-intake-auth": ("e2e/admin-auth.real.staging.spec.ts",),
    "match-rating-writes": ("e2e/tournament-live.staging.spec.ts",),
}

WAVES: dict[str, tuple[dict[str, object], ...]] = {
    "public-read": (
        {
            "name": "public-read-routes",
            "specs": (
                "e2e/staging.smoke.spec.ts",
                "e2e/leaderboards.staging.spec.ts",
                "e2e/league-results.parity.spec.ts",
                "e2e/players.staging.spec.ts",
                "e2e/public-explorer-recap.spec.ts",
            ),
        },
    ),
    "public-intake-auth": (
        {
            "name": "public-intake-auth-readiness",
            "specs": REQUIRED_REAL_SPECS["public-intake-auth"],
        },
        {
            "name": "static-support-read-only",
            "specs": ("e2e/static-support.parity.spec.ts",),
            "grep": "rating rules, FAQ, and policy request links are complete",
        },
        {
            "name": "tournament-registration-readiness",
            "specs": ("e2e/tournament-registration.parity.spec.ts",),
        },
        {
            "name": "tournament-partner-board-read-only",
            "specs": ("e2e/tournament-partner-board.parity.spec.ts",),
            "grep": "partner board renders the explicit privacy boundary",
        },
    ),
    "admin-read-export": (
        {
            "name": "admin-read-export-routes",
            "specs": (
                "e2e/league-core.staging.spec.ts",
                "e2e/match-durability.staging.spec.ts",
                "e2e/match-player-flows.parity.spec.ts",
                "e2e/communications.staging.spec.ts",
                "e2e/league-live-domain.staging.spec.ts",
                "e2e/league-live-submit.staging.spec.ts",
            ),
        },
        {
            "name": "live-ladder-read-only",
            "specs": ("e2e/live-ladder.staging.spec.ts",),
            "grep": "live-ladder admin surfaces render without issuing a write",
        },
        {
            "name": "tournament-admin-read-only",
            "specs": ("e2e/tournament-admin.staging.spec.ts",),
            "grep": "dedicated setup, management, handoff, and recovery surfaces are explicit",
        },
        {
            "name": "tournament-operations-read-only",
            "specs": ("e2e/tournament-operations.staging.spec.ts",),
            "grep": "route-specific operations surfaces|DUPR preview",
        },
    ),
    "match-rating-writes": (
        {
            "name": "tournament-live-reversible-score-command",
            "specs": REQUIRED_REAL_SPECS["match-rating-writes"],
        },
    ),
}

COMMON_REQUIRED_ENV = {
    "STAGING_WEB_BASE_URL",
    "STAGING_API_BASE_URL",
    "NEXT_PUBLIC_JUPR_API_BASE_URL",
    "VERCEL_AUTOMATION_BYPASS_SECRET",
    "JUPR_EXPECTED_STAGING_API_ORIGIN",
    "JUPR_EXPECTED_STAGING_AUTH_ORIGIN",
    "STAGING_SUPABASE_URL",
    "STAGING_SUPABASE_PROJECT_REF",
}
WAVE_REQUIRED_ENV: dict[str, set[str]] = {
    "public-read": set(),
    "public-intake-auth": {
        "STAGING_ADMIN_EMAIL",
        "STAGING_ADMIN_PASSWORD",
        "JUPR_TOURNAMENT_REGISTRATION_FIXTURE_SLUG",
    },
    "admin-read-export": {
        "STAGING_ADMIN_BEARER_TOKEN",
        "JUPR_STAGING_ADMIN_ACCESS_TOKEN",
        "JUPR_COMMUNICATIONS_DRAFT_WEEK_START",
        "JUPR_TOURNAMENT_OPS_TOURNAMENT_ID",
        "JUPR_TOURNAMENT_OPS_DRAW_ID",
    },
    "match-rating-writes": {
        "STAGING_ADMIN_BEARER_TOKEN",
        "JUPR_TOURNAMENT_LIVE_TOURNAMENT_ID",
        "JUPR_TOURNAMENT_LIVE_DRAW_ID",
        "JUPR_TOURNAMENT_LIVE_GAME_ID",
        "JUPR_TOURNAMENT_LIVE_ORIGINAL_SCORE_A",
        "JUPR_TOURNAMENT_LIVE_ORIGINAL_SCORE_B",
        "JUPR_TOURNAMENT_LIVE_EXERCISE_SCORE_A",
        "JUPR_TOURNAMENT_LIVE_EXERCISE_SCORE_B",
        "JUPR_TOURNAMENT_LIVE_ALLOW_MUTATION_E2E",
    },
}
MUTATING_WAVES = {"match-rating-writes"}


def _canonical_origin(value: object) -> str | None:
    raw = str(value).strip()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed.port not in {None, 443}
    ):
        return None
    return f"https://{parsed.hostname.lower()}"


def _immutable_vercel_origin(value: object) -> str | None:
    """Accept only a unique Vercel deployment origin owned by the staging team."""
    if not isinstance(value, str) or value != value.strip():
        return None
    canonical = _canonical_origin(value)
    if canonical != value or canonical == EXPECTED_STAGING_WEB_ORIGIN:
        return None
    hostname = urlsplit(canonical).hostname or ""
    if not IMMUTABLE_VERCEL_HOST_RE.fullmatch(hostname):
        return None
    return canonical


def environment_errors(wave: str, env: Mapping[str, str]) -> list[str]:
    errors: list[str] = []
    required = COMMON_REQUIRED_ENV | WAVE_REQUIRED_ENV.get(wave, set())
    missing = sorted(name for name in required if not str(env.get(name, "")).strip())
    if missing:
        errors.append("Missing required staging environment values: " + ", ".join(missing))

    exact_origins = {
        "STAGING_WEB_BASE_URL": EXPECTED_STAGING_WEB_ORIGIN,
        "STAGING_API_BASE_URL": EXPECTED_STAGING_API_ORIGIN,
        "NEXT_PUBLIC_JUPR_API_BASE_URL": EXPECTED_STAGING_API_ORIGIN,
        "JUPR_EXPECTED_STAGING_API_ORIGIN": EXPECTED_STAGING_API_ORIGIN,
        "JUPR_EXPECTED_STAGING_AUTH_ORIGIN": EXPECTED_STAGING_AUTH_ORIGIN,
        "STAGING_SUPABASE_URL": EXPECTED_STAGING_AUTH_ORIGIN,
    }
    for name, expected in exact_origins.items():
        value = str(env.get(name, "")).strip()
        if value and _canonical_origin(value) != expected:
            errors.append(f"Refusing non-allowlisted staging origin in {name}.")

    project_ref = str(env.get("STAGING_SUPABASE_PROJECT_REF", "")).strip()
    if project_ref and project_ref != EXPECTED_STAGING_PROJECT_REF:
        errors.append(
            f"Refusing project {project_ref}; expected JUPR Staging {EXPECTED_STAGING_PROJECT_REF}."
        )

    confirmation = str(env.get("JUPR_PARITY_MUTATION_CONFIRMATION", ""))
    if wave in MUTATING_WAVES and confirmation != MUTATION_CONFIRMATION:
        errors.append(f"Wave {wave} requires exact confirmation: {MUTATION_CONFIRMATION}")
    elif wave not in MUTATING_WAVES and confirmation:
        errors.append(f"Wave {wave} must not receive a mutation confirmation.")
    return errors


def candidate_errors(candidate_sha: str, actual_sha: str) -> list[str]:
    candidate_sha = candidate_sha.strip().lower()
    actual_sha = actual_sha.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{40}", candidate_sha):
        return ["Candidate SHA must be a full 40-character hexadecimal Git SHA."]
    if candidate_sha != actual_sha:
        return [f"Checked-out SHA {actual_sha} does not match candidate SHA {candidate_sha}."]
    return []


def manifest_errors(wave: str, web_root: Path = WEB_ROOT) -> list[str]:
    errors: list[str] = []
    names: list[str] = []
    required: list[str] = []
    for invocation in WAVES[wave]:
        name = str(invocation.get("name", "")).strip()
        names.append(name)
        specs_value = invocation.get("specs")
        specs = (
            tuple(str(spec).strip() for spec in specs_value)
            if isinstance(specs_value, tuple)
            else ()
        )
        if not name or not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,100}", name):
            errors.append(f"Wave {wave} contains an invalid invocation name.")
        if not specs or any(
            not re.fullmatch(r"e2e/[A-Za-z0-9_.-]+\.spec\.ts", spec) for spec in specs
        ):
            errors.append(
                f"Wave {wave} invocation {name or '(missing)'} has an invalid spec manifest."
            )
        if len(specs) != len(set(specs)):
            errors.append(f"Wave {wave} invocation {name or '(missing)'} repeats a spec.")
        required.extend(specs)
        if "grep" in invocation:
            grep = invocation.get("grep")
            if not isinstance(grep, str) or not grep.strip():
                errors.append(
                    f"Wave {wave} invocation {name or '(missing)'} has an empty grep contract."
                )
            elif all((web_root / spec).is_file() for spec in specs):
                try:
                    pattern = re.compile(grep)
                except re.error:
                    errors.append(f"Wave {wave} invocation {name} has an invalid grep contract.")
                else:
                    source = "\n".join(
                        (web_root / spec).read_text(encoding="utf-8") for spec in specs
                    )
                    if not pattern.search(source):
                        errors.append(
                            f"Wave {wave} invocation {name} grep contract selects no committed test."
                        )
    if len(names) != len(set(names)):
        errors.append(f"Wave {wave} invocation names must be unique.")
    missing = sorted(spec for spec in required if not (web_root / spec).is_file())
    if missing:
        errors.append(
            f"Wave {wave} is blocked until real staging spec(s) exist: " + ", ".join(missing)
        )
    return errors


def integrated_manifest_errors(web_root: Path = WEB_ROOT) -> list[str]:
    """Validate every invocation in WAVES against a final stacked web root."""
    errors = [error for wave in WAVES for error in manifest_errors(wave, web_root)]
    names = [
        str(invocation.get("name", ""))
        for wave in WAVES.values()
        for invocation in wave
    ]
    if len(names) != len(set(names)):
        errors.append("Integrated WAVES invocation names must be globally unique.")
    return errors


def deployment_identity_errors(
    web: object,
    api: object,
    *,
    candidate_sha: str,
    vercel_deployment_id: str,
    fly_image_ref: str,
    expected_write_wave: str = "none",
    expected_web_origin: str | None = None,
) -> list[str]:
    if not isinstance(web, dict):
        return ["Vercel /api/environment did not return a JSON object."]
    if not isinstance(api, dict):
        return ["Fly /health did not return a JSON object."]

    expected_web = {
        "environment": "staging",
        "vercel_environment": "preview",
        "api_origin": EXPECTED_STAGING_API_ORIGIN,
        "auth_origin": EXPECTED_STAGING_AUTH_ORIGIN,
        "preview_isolation_active": True,
        "preview_auth_isolation_active": True,
        "git_commit_sha": candidate_sha.lower(),
        "vercel_deployment_id": vercel_deployment_id,
    }
    expected_api = {
        "ok": True,
        "environment": "staging",
        "git_commit_sha": candidate_sha.lower(),
        "fly_app_name": EXPECTED_FLY_APP_NAME,
        "fly_image_ref": fly_image_ref,
        "web_origin": EXPECTED_STAGING_WEB_ORIGIN,
        "supabase_project_ref": EXPECTED_STAGING_PROJECT_REF,
        "jwt_verification_configured": True,
        "jwt_verification_mode": "jwks",
        "jwt_verification_project_ref": EXPECTED_STAGING_PROJECT_REF,
        "staging_write_wave": expected_write_wave,
        "business_data_write_wave_active": expected_write_wave != "none",
        "security_denial_audit_logging_required": True,
        "public_live_writes_enabled": expected_write_flags(expected_write_wave)[
            "JUPR_ENABLE_PUBLIC_LIVE_WRITES"
        ],
        "public_live_production_override_enabled": False,
    }
    errors: list[str] = []
    attested_web_origin = _immutable_vercel_origin(web.get("vercel_deployment_origin"))
    if attested_web_origin is None:
        errors.append("Vercel deployment identity has no canonical immutable deployment origin.")
    elif expected_web_origin is not None and attested_web_origin != expected_web_origin:
        errors.append("Vercel deployment identity mismatch for vercel_deployment_origin.")
    for key, expected in expected_web.items():
        actual = web.get(key)
        if key == "git_commit_sha" and isinstance(actual, str):
            actual = actual.lower()
        if actual != expected:
            errors.append(f"Vercel deployment identity mismatch for {key}.")
    for key, expected in expected_api.items():
        actual = api.get(key)
        if key == "git_commit_sha" and isinstance(actual, str):
            actual = actual.lower()
        if actual != expected:
            errors.append(f"Fly deployment identity mismatch for {key}.")
    expected_flags = expected_write_flags(expected_write_wave)
    if api.get("controlled_write_flags") != expected_flags:
        errors.append(
            "Fly deployment identity controlled_write_flags do not exactly match the required evidence wave."
        )
    expected_fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}={1 if enabled else 0}"
            for name, enabled in sorted(expected_flags.items())
        ).encode("utf-8")
    ).hexdigest()
    if api.get("controlled_write_flag_fingerprint") != expected_fingerprint:
        errors.append(
            "Fly deployment identity controlled_write_flag_fingerprint does not match the required evidence wave."
        )
    prerequisites = api.get("write_prerequisites")
    if not isinstance(prerequisites, dict):
        errors.append("Fly deployment identity has no write_prerequisites object.")
    else:
        for key in (
            "service_role_configured",
            "api_audit_required",
            "worker_run_log_required",
        ):
            if prerequisites.get(key) is not True:
                errors.append(f"Fly deployment identity write prerequisite {key} is not true.")
        if prerequisites.get("email_mode") != "dry_run":
            errors.append(
                "Fly deployment identity write prerequisite email_mode is not dry_run."
            )
        if prerequisites.get("live_player_update_email_enabled") is not False:
            errors.append(
                "Fly deployment identity live player-update email safety gate is not false."
            )
    if expected_write_wave == "public-intake-auth":
        for key in (
            "registration_edit_secret_configured",
            "registration_confirmation_secret_configured",
        ):
            if api.get(key) is not True:
                errors.append(f"Fly deployment identity {key} is not true for public intake.")
    return errors


class _SameOriginRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        old = urlsplit(req.full_url)
        new = urlsplit(newurl)
        if (old.scheme, old.netloc) != (new.scheme, new.netloc):
            raise HTTPError(newurl, code, "Cross-origin redirect refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get_json(url: str, headers: Mapping[str, str] | None = None) -> object:
    request = Request(url, headers=dict(headers or {}), method="GET")
    with build_opener(_SameOriginRedirectHandler()).open(request, timeout=20) as response:
        if response.status != 200:
            raise RuntimeError(f"identity endpoint returned HTTP {response.status}")
        return json.loads(response.read().decode("utf-8"))


def _deployment_identity(
    env: Mapping[str, str],
    *,
    candidate_sha: str,
    vercel_deployment_id: str,
    fly_image_ref: str,
    expected_write_wave: str = "none",
    web_origin: str = EXPECTED_STAGING_WEB_ORIGIN,
    expected_web_origin: str | None = None,
    phase: str = "preflight",
) -> tuple[dict[str, object], list[str]]:
    try:
        web = _get_json(
            f"{web_origin}/api/environment",
            {
                "x-vercel-protection-bypass": str(env["VERCEL_AUTOMATION_BYPASS_SECRET"]),
                "x-vercel-set-bypass-cookie": "true",
            },
        )
        api = _get_json(f"{EXPECTED_STAGING_API_ORIGIN}/health")
    except (HTTPError, URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"Deployment identity {phase} failed: {exc.__class__.__name__}."]
    except RuntimeError as exc:
        return {}, [f"Deployment identity {phase} failed: {exc}."]

    errors = deployment_identity_errors(
        web,
        api,
        candidate_sha=candidate_sha,
        vercel_deployment_id=vercel_deployment_id,
        fly_image_ref=fly_image_ref,
        expected_write_wave=expected_write_wave,
        expected_web_origin=expected_web_origin,
    )
    evidence = {
        "candidate_sha": candidate_sha.lower(),
        "vercel_deployment_id": vercel_deployment_id,
        "fly_image_ref": fly_image_ref,
        "requested_web_origin": web_origin,
        "web": web,
        "api": api,
    }
    return evidence, errors


def report_errors(report: object) -> list[str]:
    if not isinstance(report, dict):
        return ["Playwright JSON report is not an object."]
    stats = report.get("stats")
    if not isinstance(stats, dict):
        return ["Playwright JSON report has no stats object."]

    errors: list[str] = []
    expected = int(stats.get("expected", 0) or 0)
    skipped = int(stats.get("skipped", 0) or 0)
    unexpected = int(stats.get("unexpected", 0) or 0)
    flaky = int(stats.get("flaky", 0) or 0)
    if expected <= 0:
        errors.append("Playwright invocation executed zero passing/expected tests.")
    if skipped:
        errors.append(f"Playwright invocation skipped {skipped} test(s).")
    if unexpected:
        errors.append(f"Playwright invocation had {unexpected} unexpected result(s).")
    if flaky:
        errors.append(f"Playwright invocation had {flaky} flaky result(s).")
    return errors


def _head_sha() -> str:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _write_summary(report_dir: Path, summary: object) -> None:
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run_wave(
    wave: str,
    candidate_sha: str,
    vercel_deployment_id: str,
    vercel_deployment_origin: str,
    fly_image_ref: str,
    report_dir: Path,
    identity_only: bool = False,
    expected_write_wave: str | None = None,
) -> list[str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    errors = environment_errors(wave, os.environ)
    errors.extend(candidate_errors(candidate_sha, _head_sha()))
    errors.extend(manifest_errors(wave))
    required_write_wave = EXPECTED_WRITE_WAVE_BY_EVIDENCE_MODE[wave]
    if expected_write_wave is not None and expected_write_wave != required_write_wave:
        errors.append(
            f"Evidence mode {wave} requires staging write wave {required_write_wave}, "
            f"not {expected_write_wave}."
        )
    candidate_web_origin = _immutable_vercel_origin(vercel_deployment_origin)
    if candidate_web_origin is None:
        errors.append(
            "Vercel deployment origin must be a canonical immutable deployment origin "
            "owned by pickleballattrespalapas1, not the mutable staging alias."
        )
    summary: dict[str, object] = {
        "wave": wave,
        "required_staging_write_wave": required_write_wave,
        "candidate_sha": candidate_sha,
        "vercel_deployment_id": vercel_deployment_id,
        "vercel_deployment_origin": vercel_deployment_origin,
        "fly_image_ref": fly_image_ref,
        "staging_supabase_project_ref": EXPECTED_STAGING_PROJECT_REF,
        "invocations": [],
    }
    if errors:
        summary["preflight_errors"] = errors
        _write_summary(report_dir, summary)
        return errors
    assert candidate_web_origin is not None

    identity, identity_errors = _deployment_identity(
        os.environ,
        candidate_sha=candidate_sha,
        vercel_deployment_id=vercel_deployment_id,
        fly_image_ref=fly_image_ref,
        expected_write_wave=required_write_wave,
        web_origin=candidate_web_origin,
        expected_web_origin=candidate_web_origin,
    )
    (report_dir / "deployment-identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if identity_errors:
        summary["preflight_errors"] = identity_errors
        _write_summary(report_dir, summary)
        return identity_errors

    summary["deployment_identity_artifact"] = "deployment-identity.json"
    attested_web_origin = candidate_web_origin

    if not identity_only:
        playwright_env = os.environ.copy()
        playwright_env["STAGING_WEB_BASE_URL"] = attested_web_origin
        playwright_env["JUPR_ATTESTED_VERCEL_DEPLOYMENT_ORIGIN"] = attested_web_origin
        summary["browser_base_origin"] = attested_web_origin
        for invocation in WAVES[wave]:
            name = str(invocation["name"])
            specs = tuple(str(spec) for spec in invocation["specs"])
            command = ["npx", "playwright", "test", *specs, "--reporter=json"]
            if grep := invocation.get("grep"):
                command.extend(("--grep", str(grep)))
            result = subprocess.run(
                command,
                cwd=WEB_ROOT,
                env=playwright_env,
                capture_output=True,
                text=True,
                check=False,
            )
            report_path = report_dir / f"{name}.json"
            stderr_path = report_dir / f"{name}.stderr.log"
            report_path.write_text(result.stdout, encoding="utf-8")
            stderr_path.write_text(result.stderr, encoding="utf-8")

            invocation_errors: list[str] = []
            try:
                report = json.loads(result.stdout)
            except json.JSONDecodeError as exc:
                report = {}
                invocation_errors.append(f"Playwright emitted invalid JSON: {exc}")
            else:
                invocation_errors.extend(report_errors(report))
            if result.returncode != 0:
                invocation_errors.append(f"Playwright exited {result.returncode}.")

            summary["invocations"].append(
                {
                    "name": name,
                    "command": command,
                    "report": str(report_path),
                    "errors": invocation_errors,
                }
            )
            errors.extend(f"{name}: {error}" for error in invocation_errors)
    else:
        summary["identity_only"] = True

    post_identity, post_identity_errors = _deployment_identity(
        os.environ,
        candidate_sha=candidate_sha,
        vercel_deployment_id=vercel_deployment_id,
        fly_image_ref=fly_image_ref,
        expected_write_wave=required_write_wave,
        web_origin=attested_web_origin,
        expected_web_origin=attested_web_origin,
        phase="post-wave re-attestation",
    )
    (report_dir / "deployment-identity-post-wave.json").write_text(
        json.dumps(post_identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary["post_wave_deployment_identity_artifact"] = (
        "deployment-identity-post-wave.json"
    )
    if post_identity_errors:
        summary["post_wave_identity_errors"] = post_identity_errors
        errors.extend(f"post-wave identity: {error}" for error in post_identity_errors)

    _write_summary(report_dir, summary)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wave", nargs="?", choices=sorted(WAVES))
    parser.add_argument("--candidate-sha")
    parser.add_argument("--vercel-deployment-id")
    parser.add_argument(
        "--vercel-deployment-origin",
        help="Exact immutable Vercel origin for the candidate, never the staging alias.",
    )
    parser.add_argument("--fly-image-ref")
    parser.add_argument(
        "--expected-write-wave",
        help="Explicit dispatch-bound staging write wave; must match the selected evidence mode.",
    )
    parser.add_argument("--report-dir", type=Path, default=ROOT / "parity-staging-artifacts")
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="Verify candidate/deployment identity and write its artifact without running Playwright.",
    )
    parser.add_argument("--list", action="store_true", help="Print the selected manifest without running it.")
    parser.add_argument(
        "--check-integrated-manifest",
        action="store_true",
        help="Fail unless every WAVES spec and grep contract exists in the current stacked tree.",
    )
    args = parser.parse_args()

    if args.check_integrated_manifest:
        errors = integrated_manifest_errors()
        if errors:
            for error in errors:
                print(f"ERROR: {error}", file=sys.stderr)
            return 1
        print("Integrated parity staging manifest contains every WAVES spec and grep contract.")
        return 0

    if not args.wave:
        parser.error("wave is required unless --check-integrated-manifest is used")
    if args.list:
        print(json.dumps({args.wave: WAVES[args.wave]}, indent=2))
        return 0
    for argument, value in (
        ("--candidate-sha", args.candidate_sha),
        ("--vercel-deployment-id", args.vercel_deployment_id),
        ("--vercel-deployment-origin", args.vercel_deployment_origin),
        ("--fly-image-ref", args.fly_image_ref),
    ):
        if not value:
            parser.error(f"{argument} is required when running a wave")

    errors = run_wave(
        args.wave,
        args.candidate_sha,
        args.vercel_deployment_id,
        args.vercel_deployment_origin,
        args.fly_image_ref,
        args.report_dir,
        identity_only=args.identity_only,
        expected_write_wave=args.expected_write_wave,
    )
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"Parity staging wave {args.wave} passed with zero skips, flakes, or failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
