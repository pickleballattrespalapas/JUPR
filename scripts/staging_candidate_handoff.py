#!/usr/bin/env python3
"""Create an exact-candidate, staging-only deployment handoff.

The collector makes GET requests only. Targets are constants rather than user
inputs so this workflow cannot be repurposed for production or another project.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_parity_staging_wave import (  # noqa: E402
    _immutable_vercel_origin,
    deployment_identity_errors,
)
from scripts.staging_write_waves import expected_write_flags  # noqa: E402

REPOSITORY = "pickleballattrespalapas/JUPR"
STAGING_BRANCH = "staging"
EXPECTED_WRITE_WAVE = "open"
STAGING_SUPABASE_PROJECT_REF = "sijpxjxvdtrehmqvirfi"
PRODUCTION_SUPABASE_PROJECT_REF = "dnoockbwfenunhcibwfn"
STAGING_AUTH_ORIGIN = f"https://{STAGING_SUPABASE_PROJECT_REF}.supabase.co"
STAGING_FLY_APP = "juprleagues-api-staging"
STAGING_FLY_ORIGIN = f"https://{STAGING_FLY_APP}.fly.dev"
STAGING_VERCEL_ALIAS = (
    "https://jupr-git-staging-pickleballattrespalapas1.vercel.app"
)
PRODUCTION_FLY_APP = "juprleagues-api"
PRODUCTION_ORIGINS = frozenset(
    {
        "https://api.juprleagues.com",
        "https://juprleagues-api.fly.dev",
        "https://juprleagues.com",
        "https://www.juprleagues.com",
        "https://pickleballclubsandwich.com",
        "https://www.pickleballclubsandwich.com",
        f"https://{PRODUCTION_SUPABASE_PROJECT_REF}.supabase.co",
    }
)
FLY_HEALTH_URL = f"{STAGING_FLY_ORIGIN}/health"
VERCEL_ENVIRONMENT_URL = f"{STAGING_VERCEL_ALIAS}/api/environment"
SHA_RE = re.compile(r"[0-9a-f]{40}")
VERCEL_DEPLOYMENT_ID_RE = re.compile(r"dpl_[A-Za-z0-9]{8,128}")
STAGING_FLY_IMAGE_RE = re.compile(
    r"registry\.fly\.io/juprleagues-api-staging:"
    r"deployment-[A-Za-z0-9]{10,128}"
)
WORKFLOW_RUN_URL_RE = re.compile(
    rf"https://github\.com/{re.escape(REPOSITORY)}/actions/runs/[1-9][0-9]*"
)
MAX_RESPONSE_BYTES = 10 * 1024 * 1024

READ_ONLY_CHECKS = (
    ("api_health", FLY_HEALTH_URL, True, False),
    ("api_club", f"{STAGING_FLY_ORIGIN}/clubs/tres-palapas", True, False),
    (
        "api_tournament_registration",
        f"{STAGING_FLY_ORIGIN}/clubs/tres-palapas/tournament-registration",
        True,
        False,
    ),
    (
        "api_league_results",
        f"{STAGING_FLY_ORIGIN}/clubs/tres-palapas/league-results",
        True,
        False,
    ),
    ("web_home", f"{STAGING_VERCEL_ALIAS}/", False, True),
    (
        "web_tournament_home",
        f"{STAGING_VERCEL_ALIAS}/clubs/tres-palapas/tournaments",
        False,
        True,
    ),
    (
        "web_tournament_registration",
        f"{STAGING_VERCEL_ALIAS}/clubs/tres-palapas/tournament-registration",
        False,
        True,
    ),
    (
        "web_admin_league_manager",
        f"{STAGING_VERCEL_ALIAS}/admin/league-manager",
        False,
        True,
    ),
)


class HandoffError(RuntimeError):
    """A staging identity or read-only smoke contract failed closed."""


class _SameOriginRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001, ANN201
        resolved = urljoin(req.full_url, newurl)
        old = urlsplit(req.full_url)
        new = urlsplit(resolved)
        if (old.scheme, old.netloc) != (new.scheme, new.netloc):
            raise HTTPError(resolved, code, "Cross-origin redirect refused", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _full_sha(value: object) -> str:
    candidate = str(value or "").strip()
    if SHA_RE.fullmatch(candidate) is None:
        raise HandoffError("Candidate SHA must be 40 lowercase hexadecimal characters.")
    return candidate


def _workflow_run_url(value: object) -> str:
    url = str(value or "").strip()
    if WORKFLOW_RUN_URL_RE.fullmatch(url) is None:
        raise HandoffError("Workflow run URL is not canonical for the JUPR repository.")
    return url


def _canonical_origin(value: object) -> str | None:
    try:
        parsed = urlsplit(str(value or "").strip())
        port = parsed.port
    except (TypeError, ValueError):
        return None
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    return f"https://{parsed.hostname.lower()}"


def _production_target_errors(fly: Mapping[str, object], web: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    if fly.get("supabase_project_ref") == PRODUCTION_SUPABASE_PROJECT_REF:
        errors.append("Fly attests the forbidden production Supabase project.")
    if fly.get("jwt_verification_project_ref") == PRODUCTION_SUPABASE_PROJECT_REF:
        errors.append("Fly JWT verification attests the forbidden production project.")
    if fly.get("fly_app_name") == PRODUCTION_FLY_APP:
        errors.append("Fly attests the forbidden production app.")

    origins = {
        _canonical_origin(fly.get("web_origin")),
        _canonical_origin(web.get("api_origin")),
        _canonical_origin(web.get("auth_origin")),
    }
    forbidden = sorted(origin for origin in origins if origin in PRODUCTION_ORIGINS)
    if forbidden:
        errors.append("Deployment identity contains a forbidden production origin.")
    return errors


def validate_identity(
    *,
    candidate_sha: str,
    fly: object,
    web: object,
) -> dict[str, object]:
    candidate = _full_sha(candidate_sha)
    if not isinstance(fly, dict) or not isinstance(web, dict):
        raise HandoffError("Fly and Vercel identities must be JSON objects.")

    deployment_id = str(web.get("vercel_deployment_id") or "")
    deployment_origin = _immutable_vercel_origin(web.get("vercel_deployment_origin"))
    fly_image = str(fly.get("fly_image_ref") or "")
    errors = deployment_identity_errors(
        web,
        fly,
        candidate_sha=candidate,
        vercel_deployment_id=deployment_id,
        fly_image_ref=fly_image,
        expected_write_wave=EXPECTED_WRITE_WAVE,
        expected_web_origin=deployment_origin,
    )
    errors.extend(_production_target_errors(fly, web))
    if VERCEL_DEPLOYMENT_ID_RE.fullmatch(deployment_id) is None:
        errors.append("Vercel deployment ID is not canonical.")
    if STAGING_FLY_IMAGE_RE.fullmatch(fly_image) is None:
        errors.append("Fly image is not canonical for the staging app.")

    exact = {
        "Fly image build SHA": fly.get("image_build_git_sha") == candidate,
        "Fly write wave": fly.get("write_wave") == EXPECTED_WRITE_WAVE,
        "Fly persistent-open posture": fly.get("business_data_write_wave_active") is True,
        "Fly staging app": fly.get("fly_app_name") == STAGING_FLY_APP,
        "Fly staging project": fly.get("supabase_project_ref")
        == STAGING_SUPABASE_PROJECT_REF,
        "Fly staging web alias": _canonical_origin(fly.get("web_origin"))
        == STAGING_VERCEL_ALIAS,
        "Vercel staging API": _canonical_origin(web.get("api_origin"))
        == STAGING_FLY_ORIGIN,
        "Vercel staging Auth": _canonical_origin(web.get("auth_origin"))
        == STAGING_AUTH_ORIGIN,
    }
    errors.extend(label for label, accepted in exact.items() if not accepted)

    prerequisites = fly.get("write_prerequisites")
    if not isinstance(prerequisites, dict):
        errors.append("Fly write prerequisites are missing.")
    else:
        if prerequisites.get("email_mode") != "dry_run":
            errors.append("Fly email mode is not dry_run.")
        if prerequisites.get("live_player_update_email_enabled") is not False:
            errors.append("Live player-update email is not disabled.")

    expected_flags = expected_write_flags(EXPECTED_WRITE_WAVE)
    expected_fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}={1 if enabled else 0}"
            for name, enabled in sorted(expected_flags.items())
        ).encode("utf-8")
    ).hexdigest()
    if fly.get("controlled_write_flags") != expected_flags:
        errors.append("Fly controlled-write flags do not match persistent open.")
    if fly.get("controlled_write_flag_fingerprint") != expected_fingerprint:
        errors.append("Fly controlled-write fingerprint does not match persistent open.")
    if fly.get("public_live_production_override_enabled") is not False:
        errors.append("The production public-live override is not disabled.")

    if errors:
        raise HandoffError("Staging deployment identity rejected: " + " ".join(errors))

    return {
        "candidate_sha": candidate,
        "fly_image_ref": fly_image,
        "vercel_deployment_id": deployment_id,
        "vercel_deployment_origin": deployment_origin,
        "supabase_project_ref": STAGING_SUPABASE_PROJECT_REF,
        "write_wave": EXPECTED_WRITE_WAVE,
        "email_mode": "dry_run",
    }


def _headers(*, vercel: bool) -> dict[str, str]:
    headers = {
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        "User-Agent": "jupr-staging-handoff/1.0",
    }
    if vercel:
        secret = os.getenv("VERCEL_AUTOMATION_BYPASS_SECRET", "").strip()
        if not secret:
            raise HandoffError("Vercel automation bypass secret is required.")
        headers["x-vercel-protection-bypass"] = secret
    return headers


def _get(url: str, *, expect_json: bool, vercel: bool, timeout_seconds: float) -> tuple[int, object | None]:
    allowed_origins = {STAGING_FLY_ORIGIN, STAGING_VERCEL_ALIAS}
    origin = _canonical_origin(urlsplit(url)._replace(path="", query="", fragment="").geturl())
    if origin not in allowed_origins:
        raise HandoffError("Read-only check refused a non-staging origin.")
    request = Request(url, headers=_headers(vercel=vercel), method="GET")
    try:
        with build_opener(_SameOriginRedirectHandler()).open(
            request, timeout=timeout_seconds
        ) as response:
            body = response.read(MAX_RESPONSE_BYTES + 1)
            status = int(response.status)
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise HandoffError(f"GET check failed for {url}: {exc.__class__.__name__}.") from exc
    if len(body) > MAX_RESPONSE_BYTES:
        raise HandoffError(f"GET check response was too large for {url}.")
    if status != 200:
        raise HandoffError(f"GET check returned HTTP {status} for {url}.")
    if not expect_json:
        return status, None
    try:
        return status, json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"GET check returned invalid JSON for {url}.") from exc


def wait_for_identity(
    candidate_sha: str,
    *,
    timeout_seconds: float,
    poll_seconds: float,
    get: Callable[..., tuple[int, object | None]] = _get,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, object], object, object]:
    candidate = _full_sha(candidate_sha)
    deadline = monotonic() + max(0.0, timeout_seconds)
    last_error = "deployment identities were not ready"
    while True:
        try:
            _fly_status, fly = get(
                FLY_HEALTH_URL,
                expect_json=True,
                vercel=False,
                timeout_seconds=20,
            )
            _web_status, web = get(
                VERCEL_ENVIRONMENT_URL,
                expect_json=True,
                vercel=True,
                timeout_seconds=20,
            )
            identity = validate_identity(candidate_sha=candidate, fly=fly, web=web)
            return identity, fly, web
        except HandoffError as exc:
            last_error = str(exc)
        now = monotonic()
        if now >= deadline:
            raise HandoffError(
                "Exact staging candidate did not become ready: " + last_error
            )
        sleep(min(max(0.1, poll_seconds), max(0.0, deadline - now)))


def run_read_only_checks(
    *,
    get: Callable[..., tuple[int, object | None]] = _get,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for name, url, expect_json, vercel in READ_ONLY_CHECKS:
        status, _payload = get(
            url,
            expect_json=expect_json,
            vercel=vercel,
            timeout_seconds=20,
        )
        results.append(
            {
                "name": name,
                "method": "GET",
                "url": url,
                "status": status,
                "ok": status == 200,
            }
        )
    return results


def build_handoff(
    *,
    candidate_sha: str,
    identity: Mapping[str, object],
    checks: Sequence[Mapping[str, object]],
    workflow_run_url: str,
    generated_at: str | None = None,
) -> dict[str, object]:
    candidate = _full_sha(candidate_sha)
    run_url = _workflow_run_url(workflow_run_url)
    if identity.get("candidate_sha") != candidate:
        raise HandoffError("Identity and handoff candidate SHAs differ.")
    if not checks or any(
        check.get("method") != "GET" or check.get("ok") is not True
        for check in checks
    ):
        raise HandoffError("Every handoff smoke check must be a successful GET.")
    timestamp = generated_at or datetime.now(timezone.utc).isoformat().replace(
        "+00:00", "Z"
    )
    return {
        "schema_version": 1,
        "status": "ready_for_manual_testing",
        "generated_at": timestamp,
        "repository": REPOSITORY,
        "branch": STAGING_BRANCH,
        "candidate_sha": candidate,
        "workflow_run_url": run_url,
        "deployment": dict(identity),
        "read_only_checks": [dict(check) for check in checks],
        "safety": {
            "requests": "GET only",
            "staging_targets_only": True,
            "production_targets_contacted": False,
            "production_supabase_project_forbidden": PRODUCTION_SUPABASE_PROJECT_REF,
            "email_mode": "dry_run",
        },
    }


def render_markdown(handoff: Mapping[str, object]) -> str:
    deployment = handoff.get("deployment")
    checks = handoff.get("read_only_checks")
    if not isinstance(deployment, dict) or not isinstance(checks, list):
        raise HandoffError("Handoff is missing deployment or smoke details.")
    lines = [
        "# Staging candidate handoff",
        "",
        f"- Status: `{handoff.get('status')}`",
        f"- Candidate SHA: `{handoff.get('candidate_sha')}`",
        f"- Branch: `{handoff.get('branch')}`",
        f"- Supabase project: `{deployment.get('supabase_project_ref')}`",
        f"- Fly image: `{deployment.get('fly_image_ref')}`",
        f"- Vercel deployment: `{deployment.get('vercel_deployment_id')}`",
        f"- Vercel origin: {deployment.get('vercel_deployment_origin')}",
        f"- Write posture: `{deployment.get('write_wave')}` (persistent staging testing)",
        f"- Email mode: `{deployment.get('email_mode')}`",
        f"- Workflow: {handoff.get('workflow_run_url')}",
        "",
        "## Read-only checks",
        "",
    ]
    for check in checks:
        if isinstance(check, dict):
            lines.append(
                f"- PASS `{check.get('method')}` {check.get('name')} — HTTP {check.get('status')}"
            )
    lines.extend(
        [
            "",
            "Production targets were not contacted. This artifact authorizes manual testing on staging only.",
            "",
        ]
    )
    return "\n".join(lines)


def write_handoff(output_dir: Path, handoff: Mapping[str, object]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "staging-handoff.json"
    markdown_path = output_dir / "staging-handoff.md"
    json_path.write_text(
        json.dumps(handoff, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(handoff), encoding="utf-8")
    return json_path, markdown_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-sha", required=True)
    parser.add_argument("--workflow-run-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    parser.add_argument("--poll-seconds", type=float, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        identity, _fly, _web = wait_for_identity(
            args.candidate_sha,
            timeout_seconds=args.timeout_seconds,
            poll_seconds=args.poll_seconds,
        )
        checks = run_read_only_checks()
        handoff = build_handoff(
            candidate_sha=args.candidate_sha,
            identity=identity,
            checks=checks,
            workflow_run_url=args.workflow_run_url,
        )
        json_path, markdown_path = write_handoff(args.output_dir, handoff)
    except (HandoffError, OSError) as exc:
        print(f"Staging handoff rejected: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "candidate_sha": handoff["candidate_sha"],
                "json": str(json_path),
                "markdown": str(markdown_path),
                "status": handoff["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
