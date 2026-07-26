#!/usr/bin/env python3
"""Protected GitHub controller helpers for candidate-bound staging evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_parity_staging_wave import (  # noqa: E402
    EXPECTED_FLY_APP_NAME,
    EXPECTED_STAGING_API_ORIGIN,
    EXPECTED_STAGING_AUTH_ORIGIN,
    EXPECTED_STAGING_PROJECT_REF,
    EXPECTED_STAGING_WEB_ORIGIN,
    MUTATION_CONFIRMATION,
    _SameOriginRedirectHandler,
    _get_json,
    _immutable_vercel_origin,
    deployment_identity_errors,
)
from scripts.staging_write_waves import expected_write_flags  # noqa: E402
from scripts.staging_write_waves import STAGING_WRITE_WAVES  # noqa: E402

REPOSITORY = "pickleballattrespalapas/JUPR"
REPOSITORY_ID = 1120897513
OWNER_ID = 250933369
OWNER_LOGIN = "pickleballattrespalapas"
DEFAULT_BRANCH = "rollback-feb8"
STAGING_BRANCH = "staging"
API_VERSION = "2026-03-10"
SHA_RE = re.compile(r"[0-9a-f]{40}")
DEPLOYMENT_ID_RE = re.compile(r"dpl_[A-Za-z0-9]{8,128}")

WRITE_WAVE_BY_MODE = {
    "public-read": "none",
    "public-intake-auth": "public-intake-auth",
    "admin-read-export": "none",
    "match-rating-writes": "tournament-live",
    "match-exclusion-recovery": "match-exclusion-recovery",
    "complete-book": "none",
}
MUTATING_MODES = {"match-rating-writes", "match-exclusion-recovery"}
WORKFLOW_PATHS = {
    "fly_api_staging_deploy.yml": ".github/workflows/fly_api_staging_deploy.yml",
    "parity-final-evidence.yml": ".github/workflows/parity-final-evidence.yml",
}
WORKFLOW_INPUTS = {
    "fly_api_staging_deploy.yml": {
        "app_name",
        "primary_region",
        "fly_org",
        "club_slug",
        "club_id",
        "write_wave",
        "expected_candidate_sha",
        "orchestration_run_id",
    },
    "parity-final-evidence.yml": {
        "mode",
        "candidate_sha",
        "vercel_deployment_id",
        "vercel_deployment_origin",
        "fly_image_ref",
        "mutation_confirmation",
        "orchestration_run_id",
    },
}

JsonRequest = Callable[[str, str, object | None], object]
JsonGet = Callable[[str, Mapping[str, str] | None], object]


class ContractError(RuntimeError):
    """A fail-closed automation contract violation."""


class GitHubClient:
    def __init__(self, token: str, api_origin: str = "https://api.github.com") -> None:
        token = token.strip()
        if not token:
            raise ContractError("A GitHub token is required.")
        self._token = token
        self._api_origin = api_origin.rstrip("/")

    def request(self, method: str, path: str, payload: object | None = None) -> object:
        if not path.startswith("/"):
            raise ContractError("GitHub API paths must be absolute.")
        body = None
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": API_VERSION,
        }
        if payload is not None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            f"{self._api_origin}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with build_opener(_SameOriginRedirectHandler()).open(
                request, timeout=30
            ) as response:
                if not 200 <= response.status < 300:
                    raise ContractError(
                        f"GitHub API returned HTTP {response.status}."
                    )
                raw = response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            raise ContractError(
                f"GitHub API request failed: {exc.__class__.__name__}."
            ) from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("GitHub API returned invalid JSON.") from exc


def _dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object.")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{label} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be a positive integer.") from exc
    if parsed <= 0:
        raise ContractError(f"{label} must be a positive integer.")
    return parsed


def _full_sha(value: object, label: str = "candidate SHA") -> str:
    text = str(value or "").strip()
    if SHA_RE.fullmatch(text) is None:
        raise ContractError(f"{label} must be a lowercase full Git SHA.")
    return text


def _ref_sha(request_json: JsonRequest, branch: str = STAGING_BRANCH) -> str:
    ref = _dict(
        request_json(
            "GET",
            f"/repos/{REPOSITORY}/git/ref/heads/{branch}",
            None,
        ),
        "GitHub ref",
    )
    return _full_sha(_dict(ref.get("object"), "GitHub ref object").get("sha"))


def authorize_event(
    event: object,
    *,
    issue_modes: Mapping[int, str],
    run_attempt: int,
    request_json: JsonRequest,
) -> dict[str, object]:
    payload = _dict(event, "GitHub event")
    issue = _dict(payload.get("issue"), "GitHub issue")
    repository = _dict(payload.get("repository"), "GitHub repository")
    sender = _dict(payload.get("sender"), "GitHub sender")
    owner = _dict(repository.get("owner"), "GitHub repository owner")
    issue_user = _dict(issue.get("user"), "GitHub issue author")
    number = _positive_int(issue.get("number"), "Issue number")

    checks = {
        "event action": payload.get("action") == "reopened",
        "repository name": repository.get("full_name") == REPOSITORY,
        "repository ID": repository.get("id") == REPOSITORY_ID,
        "repository owner": owner.get("id") == OWNER_ID,
        "sender ID": sender.get("id") == OWNER_ID,
        "sender login": sender.get("login") == OWNER_LOGIN,
        "issue author": issue_user.get("id") == OWNER_ID,
        "owner association": issue.get("author_association") == "OWNER",
        "issue state": issue.get("state") == "open",
        "issue lock": issue.get("locked") is True,
        "issue kind": "pull_request" not in issue,
        "run attempt": run_attempt == 1,
        "control issue": number in issue_modes,
    }
    failed = [name for name, accepted in checks.items() if not accepted]
    if failed:
        raise ContractError("Controller authorization rejected: " + ", ".join(failed))

    mode = str(issue_modes[number])
    if mode not in WRITE_WAVE_BY_MODE:
        raise ContractError("Control issue maps to an unsupported evidence mode.")

    live = _dict(
        request_json("GET", f"/repos/{REPOSITORY}/issues/{number}", None),
        "Live GitHub issue",
    )
    live_user = _dict(live.get("user"), "Live GitHub issue author")
    live_checks = {
        "live issue number": live.get("number") == number,
        "live issue state": live.get("state") == "open",
        "live issue lock": live.get("locked") is True,
        "live issue author": live_user.get("id") == OWNER_ID,
        "live owner association": live.get("author_association") == "OWNER",
        "live issue kind": "pull_request" not in live,
    }
    failed = [name for name, accepted in live_checks.items() if not accepted]
    if failed:
        raise ContractError(
            "Live controller authorization rejected: " + ", ".join(failed)
        )

    candidate_sha = _ref_sha(request_json)
    return {
        "authorized": True,
        "issue_number": number,
        "mode": mode,
        "write_wave": WRITE_WAVE_BY_MODE[mode],
        "mutation_confirmation": (
            MUTATION_CONFIRMATION if mode in MUTATING_MODES else ""
        ),
        "candidate_sha": candidate_sha,
    }


def _web_identity_errors(web: object, candidate_sha: str) -> list[str]:
    if not isinstance(web, dict):
        return ["Vercel identity is not a JSON object."]
    expected = {
        "environment": "staging",
        "vercel_environment": "preview",
        "git_commit_sha": candidate_sha,
        "api_origin": EXPECTED_STAGING_API_ORIGIN,
        "auth_origin": EXPECTED_STAGING_AUTH_ORIGIN,
        "preview_isolation_active": True,
        "preview_auth_isolation_active": True,
    }
    errors = [
        f"Vercel identity mismatch for {key}."
        for key, value in expected.items()
        if web.get(key) != value
    ]
    if DEPLOYMENT_ID_RE.fullmatch(str(web.get("vercel_deployment_id") or "")) is None:
        errors.append("Vercel identity has no canonical deployment ID.")
    if _immutable_vercel_origin(web.get("vercel_deployment_origin")) is None:
        errors.append("Vercel identity has no canonical immutable origin.")
    return errors


def resolve_vercel_identity(
    *,
    candidate_sha: str,
    bypass_secret: str,
    web_origin: str = EXPECTED_STAGING_WEB_ORIGIN,
    timeout_seconds: float = 600,
    poll_seconds: float = 10,
    get_json: JsonGet = _get_json,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    candidate_sha = _full_sha(candidate_sha)
    bypass_secret = bypass_secret.strip()
    if not bypass_secret:
        raise ContractError("The Vercel automation bypass secret is required.")
    headers = {"x-vercel-protection-bypass": bypass_secret}
    deadline = monotonic() + max(timeout_seconds, 0)
    last_error = "identity was not ready"

    while True:
        try:
            alias_web = get_json(f"{web_origin}/api/environment", headers)
            errors = _web_identity_errors(alias_web, candidate_sha)
            if errors:
                raise ContractError("; ".join(errors))
            alias = _dict(alias_web, "Vercel alias identity")
            deployment_id = str(alias["vercel_deployment_id"])
            immutable_origin = _immutable_vercel_origin(
                alias["vercel_deployment_origin"]
            )
            if immutable_origin is None:
                raise ContractError("Vercel immutable origin is invalid.")
            immutable_web = get_json(
                f"{immutable_origin}/api/environment", headers
            )
            errors = _web_identity_errors(immutable_web, candidate_sha)
            if errors:
                raise ContractError("; ".join(errors))
            immutable = _dict(immutable_web, "Vercel immutable identity")
            if immutable.get("vercel_deployment_id") != deployment_id:
                raise ContractError("Vercel deployment ID changed across origins.")
            if (
                _immutable_vercel_origin(immutable.get("vercel_deployment_origin"))
                != immutable_origin
            ):
                raise ContractError("Vercel immutable origin did not re-attest itself.")
            return {
                "candidate_sha": candidate_sha,
                "vercel_deployment_id": deployment_id,
                "vercel_deployment_origin": immutable_origin,
                "web": immutable,
            }
        except (
            ContractError,
            HTTPError,
            URLError,
            TimeoutError,
            ValueError,
            KeyError,
            json.JSONDecodeError,
        ) as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
        if monotonic() >= deadline:
            raise ContractError(
                f"Vercel candidate identity did not become ready: {last_error}"
            )
        sleep(min(poll_seconds, max(0, deadline - monotonic())))


def dispatch_workflow(
    request_json: JsonRequest,
    *,
    workflow: str,
    inputs: Mapping[str, str],
) -> dict[str, object]:
    path = WORKFLOW_PATHS.get(workflow)
    if path is None:
        raise ContractError("Workflow dispatch target is not allowlisted.")
    if not all(isinstance(key, str) and isinstance(value, str) for key, value in inputs.items()):
        raise ContractError("Workflow dispatch inputs must be strings.")
    unknown = sorted(set(inputs) - WORKFLOW_INPUTS[workflow])
    if unknown:
        raise ContractError(
            "Workflow dispatch has unknown input(s): " + ", ".join(unknown)
        )
    orchestration_run_id = inputs.get("orchestration_run_id", "")
    if orchestration_run_id and (
        re.fullmatch(r"[1-9][0-9]{0,19}", orchestration_run_id) is None
    ):
        raise ContractError("orchestration_run_id must be a positive run ID.")
    if workflow == "fly_api_staging_deploy.yml":
        if inputs.get("write_wave") not in STAGING_WRITE_WAVES:
            raise ContractError("Fly dispatch write_wave is not allowlisted.")
        _full_sha(inputs.get("expected_candidate_sha"))
    else:
        required = WORKFLOW_INPUTS[workflow]
        missing = sorted(required - set(inputs))
        if missing:
            raise ContractError(
                "Parity dispatch is missing input(s): " + ", ".join(missing)
            )
        mode = inputs["mode"]
        if mode not in WRITE_WAVE_BY_MODE:
            raise ContractError("Parity dispatch mode is not allowlisted.")
        _full_sha(inputs["candidate_sha"])
        if DEPLOYMENT_ID_RE.fullmatch(inputs["vercel_deployment_id"]) is None:
            raise ContractError("Parity dispatch Vercel deployment ID is invalid.")
        if (
            _immutable_vercel_origin(inputs["vercel_deployment_origin"])
            != inputs["vercel_deployment_origin"]
        ):
            raise ContractError("Parity dispatch Vercel origin is not immutable.")
        if re.fullmatch(
            r"registry\.fly\.io/juprleagues-api-staging:"
            r"deployment-[A-Za-z0-9]{10,128}",
            inputs["fly_image_ref"],
        ) is None:
            raise ContractError("Parity dispatch Fly image is invalid.")
        expected_confirmation = (
            MUTATION_CONFIRMATION if mode in MUTATING_MODES else ""
        )
        if inputs["mutation_confirmation"] != expected_confirmation:
            raise ContractError("Parity dispatch mutation confirmation is invalid.")
        if not orchestration_run_id:
            raise ContractError("Parity controller dispatch requires its run ID.")
    metadata = _dict(
        request_json(
            "GET",
            f"/repos/{REPOSITORY}/actions/workflows/{workflow}",
            None,
        ),
        "Workflow metadata",
    )
    workflow_id = _positive_int(metadata.get("id"), "Workflow ID")
    if metadata.get("path") != path or metadata.get("state") != "active":
        raise ContractError("Workflow registry metadata is not active and exact.")
    response = _dict(
        request_json(
            "POST",
            f"/repos/{REPOSITORY}/actions/workflows/{workflow}/dispatches",
            {
                "ref": STAGING_BRANCH,
                "inputs": dict(inputs),
            },
        ),
        "Workflow dispatch response",
    )
    run_id = _positive_int(response.get("workflow_run_id"), "Workflow run ID")
    return {
        "workflow": workflow,
        "workflow_id": workflow_id,
        "workflow_run_id": run_id,
        "html_url": response.get("html_url"),
    }


def wait_for_workflow_run(
    request_json: JsonRequest,
    *,
    workflow: str,
    run_id: int,
    candidate_sha: str,
    timeout_seconds: float = 5400,
    poll_seconds: float = 15,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, object]:
    path = WORKFLOW_PATHS.get(workflow)
    if path is None:
        raise ContractError("Workflow wait target is not allowlisted.")
    candidate_sha = _full_sha(candidate_sha)
    metadata = _dict(
        request_json(
            "GET",
            f"/repos/{REPOSITORY}/actions/workflows/{workflow}",
            None,
        ),
        "Workflow metadata",
    )
    workflow_id = _positive_int(metadata.get("id"), "Workflow ID")
    if metadata.get("path") != path:
        raise ContractError("Workflow path does not match the allowlist.")
    deadline = monotonic() + max(timeout_seconds, 0)

    while True:
        run = _dict(
            request_json(
                "GET",
                f"/repos/{REPOSITORY}/actions/runs/{run_id}",
                None,
            ),
            "Workflow run",
        )
        exact = {
            "run ID": run.get("id") == run_id,
            "workflow ID": run.get("workflow_id") == workflow_id,
            "event": run.get("event") == "workflow_dispatch",
            "head branch": run.get("head_branch") == STAGING_BRANCH,
            "head SHA": run.get("head_sha") == candidate_sha,
            "repository ID": _dict(
                run.get("repository"), "Workflow run repository"
            ).get("id")
            == REPOSITORY_ID,
        }
        failed = [name for name, accepted in exact.items() if not accepted]
        if failed:
            raise ContractError(
                "Workflow run identity rejected: " + ", ".join(failed)
            )
        status = str(run.get("status") or "")
        if status == "completed":
            if run.get("conclusion") != "success":
                raise ContractError(
                    f"Workflow run concluded {run.get('conclusion') or 'without a conclusion'}."
                )
            return {
                "workflow": workflow,
                "workflow_run_id": run_id,
                "candidate_sha": candidate_sha,
                "status": status,
                "conclusion": "success",
                "html_url": run.get("html_url"),
            }
        if status not in {
            "queued",
            "in_progress",
            "pending",
            "requested",
            "waiting",
        }:
            raise ContractError(f"Workflow run has unexpected status {status!r}.")
        if monotonic() >= deadline:
            raise ContractError("Workflow run timed out.")
        sleep(min(poll_seconds, max(0, deadline - monotonic())))


def fetch_fly_health(get_json: JsonGet = _get_json) -> dict[str, object]:
    health = get_json(f"{EXPECTED_STAGING_API_ORIGIN}/health", None)
    return _dict(health, "Fly health")


def verify_deployment_identity(
    *,
    candidate_sha: str,
    vercel: object,
    fly: object,
    expected_write_wave: str,
) -> dict[str, object]:
    candidate_sha = _full_sha(candidate_sha)
    vercel_obj = _dict(vercel, "Vercel evidence")
    web = _dict(vercel_obj.get("web"), "Vercel identity")
    fly_obj = _dict(fly, "Fly identity")
    deployment_id = str(vercel_obj.get("vercel_deployment_id") or "")
    deployment_origin = str(vercel_obj.get("vercel_deployment_origin") or "")
    fly_image = str(fly_obj.get("fly_image_ref") or "")
    errors = deployment_identity_errors(
        web,
        fly_obj,
        candidate_sha=candidate_sha,
        vercel_deployment_id=deployment_id,
        fly_image_ref=fly_image,
        expected_write_wave=expected_write_wave,
        expected_web_origin=deployment_origin,
    )
    if errors:
        raise ContractError(" ".join(errors))
    return {
        "candidate_sha": candidate_sha,
        "vercel_deployment_id": deployment_id,
        "vercel_deployment_origin": deployment_origin,
        "fly_image_ref": fly_image,
        "write_wave": expected_write_wave,
    }


def verify_final_none(fly: object) -> dict[str, object]:
    health = _dict(fly, "Fly health")
    expected_flags = expected_write_flags("none")
    expected_fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}={1 if enabled else 0}"
            for name, enabled in sorted(expected_flags.items())
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "ok": True,
        "environment": "staging",
        "fly_app_name": EXPECTED_FLY_APP_NAME,
        "supabase_project_ref": EXPECTED_STAGING_PROJECT_REF,
        "staging_write_wave": "none",
        "business_data_write_wave_active": False,
        "controlled_write_flags": expected_flags,
        "controlled_write_flag_fingerprint": expected_fingerprint,
        "public_live_writes_enabled": False,
        "public_live_production_override_enabled": False,
        "web_origin": EXPECTED_STAGING_WEB_ORIGIN,
        "jwt_verification_configured": True,
        "jwt_verification_mode": "jwks",
        "jwt_verification_project_ref": EXPECTED_STAGING_PROJECT_REF,
        "security_denial_audit_logging_required": True,
    }
    failed = [
        key for key, expected_value in expected.items() if health.get(key) != expected_value
    ]
    if failed:
        raise ContractError(
            "Final Fly no-write verification rejected: " + ", ".join(failed)
        )
    prerequisites = _dict(health.get("write_prerequisites"), "Fly write prerequisites")
    required_prerequisites = {
        "service_role_configured": True,
        "api_audit_required": True,
        "worker_run_log_required": True,
        "email_mode": "dry_run",
        "live_player_update_email_enabled": False,
    }
    failed_prerequisites = [
        key
        for key, expected_value in required_prerequisites.items()
        if prerequisites.get(key) != expected_value
    ]
    if failed_prerequisites:
        raise ContractError(
            "Final Fly prerequisites rejected: " + ", ".join(failed_prerequisites)
        )
    fly_image_ref = str(health.get("fly_image_ref") or "")
    if re.fullmatch(
        r"registry\.fly\.io/juprleagues-api-staging:"
        r"deployment-[A-Za-z0-9]{10,128}",
        fly_image_ref,
    ) is None:
        raise ContractError("Final Fly image reference is invalid.")
    return {
        "safe": True,
        "candidate_sha": _full_sha(health.get("git_commit_sha"), "Fly candidate SHA"),
        "fly_image_ref": fly_image_ref,
        "write_wave": "none",
    }


def _load_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_issue_modes(values: Sequence[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for value in values:
        number_text, separator, mode = value.partition("=")
        if not separator:
            raise ContractError("--issue-mode must use NUMBER=MODE.")
        number = _positive_int(number_text, "Control issue number")
        if number in result:
            raise ContractError("Control issue numbers must be unique.")
        result[number] = mode
    return result


def _parse_inputs(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or key in result:
            raise ContractError("--input must use one unique KEY=VALUE pair.")
        result[key] = item
    return result


def _client() -> GitHubClient:
    return GitHubClient(
        os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN") or ""
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    authorize = commands.add_parser("authorize")
    authorize.add_argument("--event-path", required=True)
    authorize.add_argument("--run-attempt", type=int, required=True)
    authorize.add_argument("--issue-mode", action="append", default=[], required=True)

    commands.add_parser("candidate-sha")

    vercel = commands.add_parser("resolve-vercel")
    vercel.add_argument("--candidate-sha", required=True)
    vercel.add_argument("--web-origin", default=EXPECTED_STAGING_WEB_ORIGIN)
    vercel.add_argument("--timeout-seconds", type=float, default=600)
    vercel.add_argument("--poll-seconds", type=float, default=10)

    dispatch = commands.add_parser("dispatch")
    dispatch.add_argument("--workflow", required=True, choices=tuple(WORKFLOW_PATHS))
    dispatch.add_argument("--input", action="append", default=[])

    wait = commands.add_parser("wait-run")
    wait.add_argument("--workflow", required=True, choices=tuple(WORKFLOW_PATHS))
    wait.add_argument("--run-id", type=int, required=True)
    wait.add_argument("--candidate-sha", required=True)
    wait.add_argument("--timeout-seconds", type=float, default=5400)
    wait.add_argument("--poll-seconds", type=float, default=15)

    commands.add_parser("fly-health")

    identity = commands.add_parser("verify-identity")
    identity.add_argument("--candidate-sha", required=True)
    identity.add_argument("--vercel-json", required=True)
    identity.add_argument("--fly-json", required=True)
    identity.add_argument("--expected-write-wave", required=True)

    final_none = commands.add_parser("verify-final-none")
    final_none.add_argument("--fly-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "authorize":
            result = authorize_event(
                _load_json(args.event_path),
                issue_modes=_parse_issue_modes(args.issue_mode),
                run_attempt=args.run_attempt,
                request_json=_client().request,
            )
        elif args.command == "candidate-sha":
            result = {"candidate_sha": _ref_sha(_client().request)}
        elif args.command == "resolve-vercel":
            result = resolve_vercel_identity(
                candidate_sha=args.candidate_sha,
                bypass_secret=os.environ.get(
                    "VERCEL_AUTOMATION_BYPASS_SECRET", ""
                ),
                web_origin=args.web_origin,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
        elif args.command == "dispatch":
            result = dispatch_workflow(
                _client().request,
                workflow=args.workflow,
                inputs=_parse_inputs(args.input),
            )
        elif args.command == "wait-run":
            result = wait_for_workflow_run(
                _client().request,
                workflow=args.workflow,
                run_id=args.run_id,
                candidate_sha=args.candidate_sha,
                timeout_seconds=args.timeout_seconds,
                poll_seconds=args.poll_seconds,
            )
        elif args.command == "fly-health":
            result = fetch_fly_health()
        elif args.command == "verify-identity":
            result = verify_deployment_identity(
                candidate_sha=args.candidate_sha,
                vercel=_load_json(args.vercel_json),
                fly=_load_json(args.fly_json),
                expected_write_wave=args.expected_write_wave,
            )
        elif args.command == "verify-final-none":
            result = verify_final_none(_load_json(args.fly_json))
        else:  # pragma: no cover
            raise ContractError("Unsupported command.")
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"staging evidence automation rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
