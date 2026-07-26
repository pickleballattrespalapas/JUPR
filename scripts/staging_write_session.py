#!/usr/bin/env python3
"""Fail-closed control contracts for short-lived staging write sessions."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Callable, Mapping
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_parity_staging_wave import (  # noqa: E402
    EXPECTED_FLY_APP_NAME,
    EXPECTED_STAGING_PROJECT_REF,
    EXPECTED_STAGING_WEB_ORIGIN,
)
from scripts.staging_evidence_automation import (  # noqa: E402
    ContractError,
    GitHubClient,
    OWNER_ID,
    OWNER_LOGIN,
    REPOSITORY,
    REPOSITORY_ID,
    STAGING_BRANCH,
)
from scripts.staging_write_waves import (  # noqa: E402
    NO_WRITE_WAVE,
    STAGING_WRITE_WAVES,
    expected_write_flags,
)

CONTROL_ISSUE_NUMBER = 1062
CONTROL_ISSUE_TITLE = "Protected staging write session control"
CONTROL_BLOCK_LANGUAGE = "yaml"
CONTROL_COMMANDS = ("open", "advance", "close")
MIN_LEASE_SECONDS = 5 * 60
MAX_LEASE_SECONDS = 60 * 60
MAX_CLOCK_SKEW_SECONDS = 2 * 60
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
ACTIVE_WRITE_WAVES = tuple(
    wave for wave in STAGING_WRITE_WAVES if wave != NO_WRITE_WAVE
)

JsonRequest = Callable[[str, str, object | None], object]


@dataclass(frozen=True)
class SessionCommand:
    command: str
    candidate_sha: str
    expected_write_wave: str
    write_wave: str
    session_nonce: str
    lease_started_at: str = ""
    lease_expires_at: str = ""


def _dict(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object.")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{label} must be a positive integer.")
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be a positive integer.") from exc
    if parsed < 1:
        raise ContractError(f"{label} must be a positive integer.")
    return parsed


def _full_sha(value: object, label: str = "Candidate SHA") -> str:
    raw = str(value or "")
    if re.fullmatch(r"[0-9a-f]{40}", raw) is None:
        raise ContractError(f"{label} must be a lowercase full Git SHA.")
    return raw


def _canonical_uuid(value: object) -> str:
    raw = str(value or "")
    try:
        parsed = UUID(raw)
    except (ValueError, AttributeError) as exc:
        raise ContractError("session_nonce must be a canonical UUID.") from exc
    if parsed.version != 4 or str(parsed) != raw:
        raise ContractError("session_nonce must be a canonical UUIDv4.")
    return raw


def _parse_timestamp(value: object, label: str) -> datetime:
    raw = str(value or "")
    try:
        parsed = datetime.strptime(raw, TIMESTAMP_FORMAT).replace(
            tzinfo=timezone.utc
        )
    except ValueError as exc:
        raise ContractError(
            f"{label} must use exact UTC seconds: YYYY-MM-DDTHH:MM:SSZ."
        ) from exc
    if parsed.strftime(TIMESTAMP_FORMAT) != raw:
        raise ContractError(
            f"{label} must use exact UTC seconds: YYYY-MM-DDTHH:MM:SSZ."
        )
    return parsed


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _yaml_block(body: object) -> dict[str, str]:
    if not isinstance(body, str) or len(body) > 20_000:
        raise ContractError("Control issue body is missing or too large.")
    fence = f"```{CONTROL_BLOCK_LANGUAGE}"
    if body.count(fence) != 1 or body.count("```") != 2:
        raise ContractError(
            "Control issue body must contain exactly one fenced yaml command."
        )
    match = re.search(r"```yaml\n(?P<command>.*?)\n```", body, re.DOTALL)
    if match is None:
        raise ContractError("Control issue yaml command fence is malformed.")
    values: dict[str, str] = {}
    for line in match.group("command").splitlines():
        if not line or line != line.strip() or ":" not in line:
            raise ContractError("Control issue yaml command lines are malformed.")
        key, raw_value = line.split(":", 1)
        value = raw_value[1:] if raw_value.startswith(" ") else raw_value
        if (
            re.fullmatch(r"[a-z][a-z0-9_]{1,40}", key) is None
            or not value
            or value != value.strip()
            or key in values
        ):
            raise ContractError("Control issue yaml command fields are malformed.")
        values[key] = value
    return values


def parse_session_command(body: object) -> SessionCommand:
    values = _yaml_block(body)
    command = values.get("command", "")
    if command not in CONTROL_COMMANDS:
        raise ContractError("Control command must be open, advance, or close.")

    common = {
        "command",
        "candidate_sha",
        "expected_write_wave",
        "write_wave",
        "session_nonce",
    }
    lease = {"lease_started_at", "lease_expires_at"}
    required = common | (lease if command in {"open", "advance"} else set())
    if set(values) != required:
        missing = sorted(required - set(values))
        unknown = sorted(set(values) - required)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unknown:
            detail.append("unknown " + ", ".join(unknown))
        raise ContractError(
            "Control command fields must be exact"
            + (": " + "; ".join(detail) if detail else ".")
        )

    candidate_sha = _full_sha(values["candidate_sha"])
    session_nonce = _canonical_uuid(values["session_nonce"])
    expected_wave = values["expected_write_wave"]
    write_wave = values["write_wave"]
    if expected_wave not in STAGING_WRITE_WAVES:
        raise ContractError("expected_write_wave is not allowlisted.")
    if write_wave not in STAGING_WRITE_WAVES:
        raise ContractError("write_wave is not allowlisted.")

    if command == "open":
        if expected_wave != NO_WRITE_WAVE or write_wave == NO_WRITE_WAVE:
            raise ContractError(
                "open must transition from none to one active allowlisted wave."
            )
    elif command == "advance":
        if (
            expected_wave == NO_WRITE_WAVE
            or write_wave == NO_WRITE_WAVE
            or expected_wave == write_wave
        ):
            raise ContractError(
                "advance must name two distinct active allowlisted waves."
            )
    elif write_wave != NO_WRITE_WAVE:
        raise ContractError("close must target write_wave none.")

    return SessionCommand(
        command=command,
        candidate_sha=candidate_sha,
        expected_write_wave=expected_wave,
        write_wave=write_wave,
        session_nonce=session_nonce,
        lease_started_at=values.get("lease_started_at", ""),
        lease_expires_at=values.get("lease_expires_at", ""),
    )


def validate_lease(
    command: SessionCommand,
    *,
    now: datetime,
    require_fresh_start: bool,
) -> tuple[datetime, datetime]:
    if command.command not in {"open", "advance"}:
        raise ContractError("Only open and advance commands carry a lease.")
    started = _parse_timestamp(command.lease_started_at, "lease_started_at")
    expires = _parse_timestamp(command.lease_expires_at, "lease_expires_at")
    duration = (expires - started).total_seconds()
    if not MIN_LEASE_SECONDS <= duration <= MAX_LEASE_SECONDS:
        raise ContractError("Lease duration must be between 5 and 60 minutes.")
    if expires <= now:
        raise ContractError("Lease is already expired.")
    if started > now and (started - now).total_seconds() > MAX_CLOCK_SKEW_SECONDS:
        raise ContractError("Lease start is too far in the future.")
    if require_fresh_start and abs((now - started).total_seconds()) > (
        MAX_CLOCK_SKEW_SECONDS
    ):
        raise ContractError("Lease start is not fresh for this owner command.")
    return started, expires


def _label_names(issue: Mapping[str, object]) -> set[str]:
    labels = issue.get("labels")
    if not isinstance(labels, list):
        return set()
    names: set[str] = set()
    for item in labels:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, dict) and isinstance(item.get("name"), str):
            names.add(str(item["name"]))
    return names


def _issue_errors(issue: Mapping[str, object], *, require_open: bool) -> list[str]:
    user = issue.get("user")
    user_id = user.get("id") if isinstance(user, dict) else None
    errors: list[str] = []
    expected = {
        "issue number": issue.get("number") == CONTROL_ISSUE_NUMBER,
        "issue title": issue.get("title") == CONTROL_ISSUE_TITLE,
        "issue state": issue.get("state") == ("open" if require_open else "closed"),
        "issue lock": issue.get("locked") is True,
        "issue author": user_id == OWNER_ID,
        "owner association": issue.get("author_association") == "OWNER",
        "issue kind": "pull_request" not in issue,
    }
    errors.extend(name for name, accepted in expected.items() if not accepted)
    return errors


def _ref_sha(request_json: JsonRequest) -> str:
    response = _dict(
        request_json(
            "GET",
            f"/repos/{REPOSITORY}/git/ref/heads/{STAGING_BRANCH}",
            None,
        ),
        "GitHub staging ref",
    )
    obj = _dict(response.get("object"), "GitHub staging ref object")
    return _full_sha(obj.get("sha"), "Canonical staging SHA")


def authorize_event(
    event: object,
    *,
    run_attempt: int,
    request_json: JsonRequest,
    now: datetime | None = None,
) -> dict[str, object]:
    payload = _dict(event, "GitHub event")
    issue = _dict(payload.get("issue"), "GitHub issue")
    repository = _dict(payload.get("repository"), "GitHub repository")
    repository_owner = _dict(repository.get("owner"), "GitHub repository owner")
    sender = _dict(payload.get("sender"), "GitHub sender")
    action = str(payload.get("action") or "")

    checks = {
        "event action": action in {"reopened", "edited"},
        "repository name": repository.get("full_name") == REPOSITORY,
        "repository ID": repository.get("id") == REPOSITORY_ID,
        "repository owner": repository_owner.get("id") == OWNER_ID,
        "sender ID": sender.get("id") == OWNER_ID,
        "sender login": sender.get("login") == OWNER_LOGIN,
        "run attempt": run_attempt == 1,
    }
    failed = [name for name, accepted in checks.items() if not accepted]
    failed.extend(_issue_errors(issue, require_open=True))
    if failed:
        raise ContractError(
            "Write-session authorization rejected: " + ", ".join(failed)
        )

    event_command = parse_session_command(issue.get("body"))
    if event_command.command == "open":
        if action == "edited":
            changes = payload.get("changes")
            if (
                not isinstance(changes, dict)
                or not isinstance(changes.get("body"), dict)
                or "from" not in changes["body"]
            ):
                raise ContractError("open requires an owner reopen event.")
            previous_command = parse_session_command(
                changes["body"].get("from")
            )
            if previous_command.command != "close":
                raise ContractError(
                    "An edited open companion must follow a close command."
                )
            return {
                "authorized": False,
                "superseded": True,
                "issue_number": CONTROL_ISSUE_NUMBER,
            }
        if action != "reopened":
            raise ContractError("open requires an owner reopen event.")
    if event_command.command in {"advance", "close"}:
        changes = payload.get("changes")
        if (
            action != "edited"
            or not isinstance(changes, dict)
            or not isinstance(changes.get("body"), dict)
            or "from" not in changes["body"]
        ):
            raise ContractError(
                "advance and close require an owner issue-body edit event."
            )
        previous_body = changes["body"].get("from")
        previous_command = parse_session_command(previous_body)
        continuity_errors = []
        if previous_command.command not in {"open", "advance"}:
            continuity_errors.append("previous command is not an active lease")
        if previous_command.candidate_sha != event_command.candidate_sha:
            continuity_errors.append("candidate changed")
        if previous_command.session_nonce != event_command.session_nonce:
            continuity_errors.append("session nonce changed")
        if previous_command.write_wave != event_command.expected_write_wave:
            continuity_errors.append("expected wave does not continue the session")
        if continuity_errors:
            raise ContractError(
                "Control command does not continue the active issue ledger: "
                + ", ".join(continuity_errors)
            )

    live = _dict(
        request_json(
            "GET",
            f"/repos/{REPOSITORY}/issues/{CONTROL_ISSUE_NUMBER}",
            None,
        ),
        "Live control issue",
    )
    live_errors = _issue_errors(live, require_open=True)
    if live_errors:
        raise ContractError(
            "Live write-session authorization rejected: "
            + ", ".join(live_errors)
        )
    if live.get("body") != issue.get("body"):
        return {
            "authorized": False,
            "superseded": True,
            "issue_number": CONTROL_ISSUE_NUMBER,
        }
    live_command = parse_session_command(live.get("body"))
    if live_command != event_command:
        raise ContractError("Live control command does not match the event command.")

    current_sha = _ref_sha(request_json)
    if event_command.candidate_sha != current_sha:
        raise ContractError(
            "Control candidate_sha does not equal canonical staging."
        )
    if event_command.command in {"open", "advance"}:
        validate_lease(
            event_command,
            now=now or _now(),
            require_fresh_start=True,
        )
    return {
        "authorized": True,
        "superseded": False,
        "issue_number": CONTROL_ISSUE_NUMBER,
        **asdict(event_command),
    }


def _leased_health_errors(
    fly: object,
    *,
    command: SessionCommand,
    current_wave: str,
) -> list[str]:
    if not isinstance(fly, dict):
        return ["Fly health is not a JSON object."]
    flags = expected_write_flags(current_wave)
    fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}={1 if enabled else 0}"
            for name, enabled in sorted(flags.items())
        ).encode("utf-8")
    ).hexdigest()
    expected = {
        "ok": True,
        "environment": "staging",
        "git_commit_sha": command.candidate_sha,
        "fly_app_name": EXPECTED_FLY_APP_NAME,
        "web_origin": EXPECTED_STAGING_WEB_ORIGIN,
        "supabase_project_ref": EXPECTED_STAGING_PROJECT_REF,
        "staging_write_wave": current_wave,
        "business_data_write_wave_active": current_wave != NO_WRITE_WAVE,
        "controlled_write_flags": flags,
        "controlled_write_flag_fingerprint": fingerprint,
        "public_live_writes_enabled": flags["JUPR_ENABLE_PUBLIC_LIVE_WRITES"],
        "public_live_production_override_enabled": False,
        "jwt_verification_configured": True,
        "jwt_verification_mode": "jwks",
        "jwt_verification_project_ref": EXPECTED_STAGING_PROJECT_REF,
        "security_denial_audit_logging_required": True,
    }
    errors = [
        f"Fly health mismatch for {key}."
        for key, expected_value in expected.items()
        if fly.get(key) != expected_value
    ]
    prerequisites = fly.get("write_prerequisites")
    required_prerequisites = {
        "service_role_configured": True,
        "api_audit_required": True,
        "worker_run_log_required": True,
        "email_mode": "dry_run",
        "live_player_update_email_enabled": False,
    }
    if not isinstance(prerequisites, dict):
        errors.append("Fly health has no write_prerequisites object.")
    else:
        errors.extend(
            f"Fly write prerequisite mismatch for {key}."
            for key, expected_value in required_prerequisites.items()
            if prerequisites.get(key) != expected_value
        )
    if current_wave == "public-intake-auth":
        for key in (
            "registration_edit_secret_configured",
            "registration_confirmation_secret_configured",
        ):
            if fly.get(key) is not True:
                errors.append(f"Fly health mismatch for {key}.")
    return errors


def inspect_active_lease(
    *,
    issue: object,
    current_candidate_sha: str,
    fly: object,
    now: datetime | None = None,
) -> dict[str, object]:
    try:
        issue_obj = _dict(issue, "Live control issue")
        issue_errors = _issue_errors(issue_obj, require_open=True)
        if issue_errors:
            raise ContractError(", ".join(issue_errors))
        command = parse_session_command(issue_obj.get("body"))
        if command.command not in {"open", "advance"}:
            raise ContractError("Control issue is not leasing an active wave.")
        if command.candidate_sha != _full_sha(
            current_candidate_sha, "Canonical staging SHA"
        ):
            raise ContractError("Lease candidate does not equal canonical staging.")
        _, expires = validate_lease(
            command,
            now=now or _now(),
            require_fresh_start=False,
        )
        allowed_waves = (
            {NO_WRITE_WAVE, command.write_wave}
            if command.command == "open"
            else {
                command.expected_write_wave,
                NO_WRITE_WAVE,
                command.write_wave,
            }
        )
        current_wave = (
            str(fly.get("staging_write_wave") or "")
            if isinstance(fly, dict)
            else ""
        )
        if current_wave not in allowed_waves:
            raise ContractError(
                "Fly health wave is outside this command's bounded transition."
            )
        health_errors = _leased_health_errors(
            fly,
            command=command,
            current_wave=current_wave,
        )
        if health_errors:
            raise ContractError(" ".join(health_errors))
    except ContractError as exc:
        return {
            "keep_active": False,
            "issue_number": CONTROL_ISSUE_NUMBER,
            "reason": str(exc),
        }
    return {
        "keep_active": True,
        "issue_number": CONTROL_ISSUE_NUMBER,
        "candidate_sha": command.candidate_sha,
        "write_wave": command.write_wave,
        "current_write_wave": current_wave,
        "transition_phase": (
            "ready"
            if current_wave == command.write_wave
            else "restored_none"
            if current_wave == NO_WRITE_WAVE
            else "expected_prestate"
        ),
        "session_nonce": command.session_nonce,
        "lease_expires_at": command.lease_expires_at,
        "seconds_remaining": max(
            0, int((expires - (now or _now())).total_seconds())
        ),
        "email_mode": "dry_run",
    }


def inspect_live_lease(
    *,
    fly: object,
    request_json: JsonRequest,
    now: datetime | None = None,
) -> dict[str, object]:
    issue = request_json(
        "GET",
        f"/repos/{REPOSITORY}/issues/{CONTROL_ISSUE_NUMBER}",
        None,
    )
    candidate_sha = _ref_sha(request_json)
    return inspect_active_lease(
        issue=issue,
        current_candidate_sha=candidate_sha,
        fly=fly,
        now=now,
    )


def should_expire_lease(
    *,
    issue: object,
    current_candidate_sha: str,
    command: SessionCommand,
    now: datetime | None = None,
) -> dict[str, object]:
    current = now or _now()
    try:
        issue_obj = _dict(issue, "Live control issue")
        live_command = parse_session_command(issue_obj.get("body"))
        if live_command != command:
            raise ContractError("Lease was superseded by a newer owner command.")
    except ContractError as exc:
        return {
            "expire": False,
            "issue_number": CONTROL_ISSUE_NUMBER,
            "reason": str(exc),
        }

    # Once the exact body and nonce still match this watcher, every broken
    # safety invariant must restore no-write mode. In particular, an owner
    # closing the issue or canonical staging moving must not leave an active
    # wave waiting for the next scheduled recovery.
    safety_errors = _issue_errors(issue_obj, require_open=True)
    try:
        current_sha = _full_sha(
            current_candidate_sha, "Canonical staging SHA"
        )
    except ContractError as exc:
        safety_errors.append(str(exc))
        current_sha = ""
    if current_sha and command.candidate_sha != current_sha:
        safety_errors.append(
            "Lease candidate does not equal canonical staging."
        )
    try:
        _, expires = validate_lease(
            command,
            now=current,
            require_fresh_start=False,
        )
    except ContractError as exc:
        if str(exc) != "Lease is already expired.":
            safety_errors.append(str(exc))
        expires = _parse_timestamp(
            command.lease_expires_at, "lease_expires_at"
        )

    if safety_errors or current >= expires:
        result: dict[str, object] = {
            "expire": True,
            "issue_number": CONTROL_ISSUE_NUMBER,
            "session_nonce": command.session_nonce,
            "write_wave": command.write_wave,
        }
        if safety_errors:
            result["reason"] = " ".join(safety_errors)
        return result
    return {
        "expire": False,
        "issue_number": CONTROL_ISSUE_NUMBER,
        "session_nonce": command.session_nonce,
        "write_wave": command.write_wave,
    }


def _load_json(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _command_from_args(args: argparse.Namespace) -> SessionCommand:
    return SessionCommand(
        command=str(args.command),
        candidate_sha=_full_sha(args.candidate_sha),
        expected_write_wave=str(args.expected_write_wave),
        write_wave=str(args.write_wave),
        session_nonce=_canonical_uuid(args.session_nonce),
        lease_started_at=str(args.lease_started_at),
        lease_expires_at=str(args.lease_expires_at),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Authorize and inspect protected staging write sessions."
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    authorize = subparsers.add_parser("authorize")
    authorize.add_argument("--event-path", required=True)
    authorize.add_argument("--run-attempt", type=int, required=True)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--fly-json", required=True)

    expiry = subparsers.add_parser("should-expire")
    expiry.add_argument("--command", choices=("open", "advance"), required=True)
    expiry.add_argument("--candidate-sha", required=True)
    expiry.add_argument("--expected-write-wave", required=True)
    expiry.add_argument("--write-wave", required=True)
    expiry.add_argument("--session-nonce", required=True)
    expiry.add_argument("--lease-started-at", required=True)
    expiry.add_argument("--lease-expires-at", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = GitHubClient(os.environ.get("GH_TOKEN", ""))
    if args.action == "authorize":
        result = authorize_event(
            _load_json(args.event_path),
            run_attempt=args.run_attempt,
            request_json=client.request,
        )
    elif args.action == "inspect":
        result = inspect_live_lease(
            fly=_load_json(args.fly_json),
            request_json=client.request,
        )
    else:
        command = _command_from_args(args)
        issue = client.request(
            "GET",
            f"/repos/{REPOSITORY}/issues/{CONTROL_ISSUE_NUMBER}",
            None,
        )
        candidate_sha = _ref_sha(client.request)
        result = should_expire_lease(
            issue=issue,
            current_candidate_sha=candidate_sha,
            command=command,
        )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, json.JSONDecodeError, OSError) as exc:
        print(f"Protected staging write session rejected: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
