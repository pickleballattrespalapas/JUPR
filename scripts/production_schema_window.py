from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from scripts.deployment_verifier import PRODUCTION_FEATURE_FLAGS
from scripts.staging_write_waves import ALL_STAGING_WRITE_FLAGS


PRODUCTION_SCHEMA_WINDOW_TRIGGER_PATH = (
    ".github/production-schema-window.trigger"
)
PRODUCTION_SCHEMA_WINDOW_ACTIONS = frozenset(
    {"quiesce", "restore_baseline"}
)
PRODUCTION_SCHEMA_WINDOW_CONFIRMATIONS = {
    "quiesce": "QUIESCE PRODUCTION FOR SCHEMA MIGRATION",
    "restore_baseline": "RESTORE PRODUCTION TO LIVE TOURNAMENT BASELINE",
}
PRODUCTION_SCHEMA_WINDOW_TRIGGER_KEYS = frozenset(
    {"action", "confirmation", "release_parent_sha", "schema_version"}
)
PRODUCTION_FLY_APP = "juprleagues-api"
PRODUCTION_ENVIRONMENT = "production"
PRODUCTION_WEB_ORIGIN = "https://pickleballclubsandwich.com"
PRODUCTION_ALLOWED_ORIGINS = (
    "https://juprleagues.com",
    "https://www.juprleagues.com",
    "https://pickleballclubsandwich.com",
    "https://www.pickleballclubsandwich.com",
)
DISALLOWED_PRODUCTION_SUPABASE_PROJECT_REFS = frozenset(
    {"sijpxjxvdtrehmqvirfi"}
)

# This is the exact tournament feature projection that was live before the
# staging-to-production release candidate. Candidate-only League Manager and
# League Live flags are intentionally absent. Keeping the baseline local to the
# transition controller lets it restore the old image without depending on the
# candidate image being deployed.
PRODUCTION_LIVE_TOURNAMENT_BASELINE_ENABLED_FLAGS = frozenset(
    {
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS",
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF",
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS",
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH",
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OPERATIONS_MUTATIONS",
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS",
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS",
        "JUPR_ENABLE_PUBLIC_LIVE_WRITES",
        "JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION",
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES",
        "JUPR_ENABLE_STAGING_PUBLIC_INTAKE_WRITES",
        "JUPR_ENABLE_STAGING_TOURNAMENT_COMMERCE_WRITES",
        "JUPR_ENABLE_TOURNAMENT_COMMERCE",
        "JUPR_ENABLE_TOURNAMENT_TEAM_COMPETITION",
        "JUPR_ENABLE_TOURNAMENT_WRITES_PRODUCTION",
    }
)
PRODUCTION_CANDIDATE_ONLY_LEAGUE_FLAGS = frozenset(
    {
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
    }
)
TRANSITION_BASE_SETTINGS = {
    "JUPR_EMAIL_MODE": "dry_run",
    "JUPR_STAGING_WRITE_WAVE": "none",
}
PRESERVED_HEALTH_FIELDS = (
    "service",
    "environment",
    "git_commit_sha",
    "image_build_git_sha",
    "fly_app_name",
    "fly_image_ref",
    "web_origin",
    "jwt_verification_mode",
    "jwt_verification_project_ref",
    "expected_migration_head",
    "expected_migration_contract",
    "expected_migration_profile",
    "cors_allowed_origins",
    "cors_allowed_origin_regex",
    "supabase_project_ref",
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PROJECT_REF_RE = re.compile(r"^[a-z0-9]{20}$")


def _clean_parent_shas(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value or "").strip().lower() for value in values)


def schema_window_trigger_errors(
    payload: Any,
    *,
    head_sha: str,
    parent_shas: Iterable[str],
    changed_status_lines: Iterable[str],
) -> tuple[list[str], dict[str, str]]:
    """Validate the sole push shape allowed to transition production runtime."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["Production schema-window trigger must be a JSON object."], {}

    unknown_keys = sorted(set(payload) - PRODUCTION_SCHEMA_WINDOW_TRIGGER_KEYS)
    missing_keys = sorted(PRODUCTION_SCHEMA_WINDOW_TRIGGER_KEYS - set(payload))
    if unknown_keys:
        errors.append(
            "Production schema-window trigger has unknown keys: "
            + ", ".join(unknown_keys)
        )
    if missing_keys:
        errors.append(
            "Production schema-window trigger is missing keys: "
            + ", ".join(missing_keys)
        )

    if type(payload.get("schema_version")) is not int or payload.get(
        "schema_version"
    ) != 1:
        errors.append(
            "Production schema-window trigger must use schema_version=1."
        )
    for name in ("action", "confirmation", "release_parent_sha"):
        if name in payload and not isinstance(payload.get(name), str):
            errors.append(f"Production schema-window trigger {name} must be a string.")

    action = str(payload.get("action") or "").strip()
    if action not in PRODUCTION_SCHEMA_WINDOW_ACTIONS:
        errors.append(
            "Production schema-window action must be quiesce or restore_baseline."
        )
    confirmation = str(payload.get("confirmation") or "").strip()
    if action in PRODUCTION_SCHEMA_WINDOW_CONFIRMATIONS and confirmation != (
        PRODUCTION_SCHEMA_WINDOW_CONFIRMATIONS[action]
    ):
        errors.append(
            "Production schema-window approval phrase does not match its action."
        )

    clean_head = str(head_sha or "").strip().lower()
    if not _SHA_RE.fullmatch(clean_head):
        errors.append(
            "Production schema-window HEAD must be an exact lowercase Git SHA."
        )
    clean_parents = _clean_parent_shas(parent_shas)
    if len(clean_parents) != 1 or not _SHA_RE.fullmatch(
        clean_parents[0] if clean_parents else ""
    ):
        errors.append(
            "Production schema-window trigger commit must have exactly one exact Git parent."
        )
    reviewed_parent = str(payload.get("release_parent_sha") or "").strip().lower()
    if not _SHA_RE.fullmatch(reviewed_parent):
        errors.append(
            "Production schema-window release_parent_sha is invalid."
        )
    elif len(clean_parents) == 1 and reviewed_parent != clean_parents[0]:
        errors.append(
            "Production schema-window release_parent_sha does not match its commit parent."
        )

    status_lines = tuple(str(line).rstrip("\r\n") for line in changed_status_lines)
    allowed_statuses = {
        f"A\t{PRODUCTION_SCHEMA_WINDOW_TRIGGER_PATH}",
        f"M\t{PRODUCTION_SCHEMA_WINDOW_TRIGGER_PATH}",
    }
    if len(status_lines) != 1 or status_lines[0] not in allowed_statuses:
        errors.append(
            "Production schema-window trigger commit must only add or modify "
            "the exact trigger file."
        )

    return errors, {
        "action": action,
        "confirmation": confirmation,
        "head_sha": clean_head,
        "release_parent_sha": reviewed_parent,
        "schema_version": "1",
    }


def transition_plan(action: str) -> dict[str, Any]:
    if action not in PRODUCTION_SCHEMA_WINDOW_ACTIONS:
        raise ValueError(f"Unknown production schema-window action: {action}")
    unknown_baseline = (
        PRODUCTION_LIVE_TOURNAMENT_BASELINE_ENABLED_FLAGS
        - set(PRODUCTION_FEATURE_FLAGS)
    )
    if unknown_baseline:
        raise ValueError(
            "Production tournament baseline contains unknown feature flags: "
            + ", ".join(sorted(unknown_baseline))
        )

    if action == "quiesce":
        feature_flags = {name: False for name in PRODUCTION_FEATURE_FLAGS}
        write_policy = "read_only"
    else:
        feature_flags = {
            name: name in PRODUCTION_LIVE_TOURNAMENT_BASELINE_ENABLED_FLAGS
            for name in PRODUCTION_FEATURE_FLAGS
        }
        write_policy = "enabled"

    settings = {
        **TRANSITION_BASE_SETTINGS,
        "JUPR_PRODUCTION_WRITE_POLICY": write_policy,
        **{
            name: "1" if enabled else "0"
            for name, enabled in sorted(feature_flags.items())
        },
    }
    controlled_write_flags = {
        name: feature_flags[name]
        for name in ALL_STAGING_WRITE_FLAGS
    }
    return {
        "schema_version": 1,
        "action": action,
        "fly_app": PRODUCTION_FLY_APP,
        "production_write_policy": write_policy,
        "settings": dict(sorted(settings.items())),
        "feature_flags": dict(sorted(feature_flags.items())),
        "controlled_write_flags": dict(sorted(controlled_write_flags.items())),
        "candidate_only_league_flags": sorted(
            PRODUCTION_CANDIDATE_ONLY_LEAGUE_FLAGS
        ),
    }


def _secret_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("secrets", "Secrets"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def secret_deployment_state(
    payload: Any,
    *,
    allowed_pending_names: Iterable[str] = (),
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Split pending Fly secrets into known transition and unknown names."""

    items = _secret_items(payload)
    if not items:
        return (
            ["Fly secret deployment-status inventory is missing or unrecognized."],
            [],
            {"recognized_secret_count": 0, "pending": []},
        )
    allowed = {str(name).strip() for name in allowed_pending_names if str(name).strip()}
    known_pending: list[str] = []
    unknown_pending: list[str] = []
    for item in items:
        name = next(
            (
                str(item.get(key) or "").strip()
                for key in ("name", "Name", "NAME")
                if str(item.get(key) or "").strip()
            ),
            "<unnamed>",
        )
        status = next(
            (
                str(item.get(key) or "").strip().lower()
                for key in (
                    "deployment_status",
                    "DeploymentStatus",
                    "status",
                    "Status",
                )
                if str(item.get(key) or "").strip()
            ),
            "unknown",
        )
        if status == "deployed":
            continue
        rendered = f"{name}={status}"
        if name in allowed:
            known_pending.append(rendered)
        else:
            unknown_pending.append(rendered)

    errors = []
    if unknown_pending:
        errors.append(
            "Fly has pending secrets outside the exact schema-window bundle: "
            + ", ".join(sorted(unknown_pending))
        )
    return errors, sorted(known_pending), {
        "recognized_secret_count": len(items),
        "known_pending": sorted(known_pending),
        "unknown_pending": sorted(unknown_pending),
    }


def _feature_flag_fingerprint(flags: Mapping[str, bool]) -> str:
    return hashlib.sha256(
        "\n".join(
            f"{name}={1 if bool(enabled) else 0}"
            for name, enabled in sorted(flags.items())
        ).encode("utf-8")
    ).hexdigest()


def _active_machine_images(payload: Any) -> tuple[set[str], list[str]]:
    if isinstance(payload, dict):
        for key in ("machines", "Machines"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return set(), ["Fly machine inventory is missing or unrecognized."]
    images: set[str] = set()
    errors: list[str] = []
    for index, machine in enumerate(payload):
        if not isinstance(machine, dict):
            errors.append(f"Fly machine inventory entry {index} is invalid.")
            continue
        state = str(machine.get("state") or machine.get("State") or "").strip().lower()
        if state in {"destroyed", "replaced", "migrated"}:
            continue
        config = machine.get("config") or machine.get("Config") or {}
        image = config.get("image") if isinstance(config, dict) else None
        if not isinstance(image, str) or not image.strip():
            errors.append(f"Active Fly machine entry {index} has no image identity.")
            continue
        images.add(image.strip())
    if not images:
        errors.append("Fly has no active machine image identity.")
    return images, errors


def production_health_invariant_errors(
    payload: Any,
    *,
    expected_project_ref: str,
) -> list[str]:
    if not isinstance(payload, dict):
        return ["Production /health payload is not a JSON object."]
    errors: list[str] = []
    clean_ref = str(expected_project_ref or "").strip().lower()
    if not _PROJECT_REF_RE.fullmatch(clean_ref):
        errors.append("Expected production Supabase project ref is invalid.")
    elif clean_ref in DISALLOWED_PRODUCTION_SUPABASE_PROJECT_REFS:
        errors.append("The staging Supabase project is forbidden in production.")

    expected_values = {
        "ok": True,
        "service": "jupr-api",
        "environment": PRODUCTION_ENVIRONMENT,
        "fly_app_name": PRODUCTION_FLY_APP,
        "web_origin": PRODUCTION_WEB_ORIGIN,
        "jwt_verification_mode": "jwks",
        "jwt_verification_project_ref": clean_ref,
        "supabase_project_ref": clean_ref,
        "cors_allowed_origins": list(PRODUCTION_ALLOWED_ORIGINS),
        "cors_allowed_origin_regex": None,
    }
    for name, expected in expected_values.items():
        if payload.get(name) != expected:
            errors.append(
                f"Production /health invariant {name} does not match its protected value."
            )
    prerequisites = payload.get("write_prerequisites")
    if not isinstance(prerequisites, dict):
        errors.append("Production /health write prerequisites are missing.")
    else:
        if prerequisites.get("email_mode") != "dry_run":
            errors.append("Production email mode must remain dry_run.")
        if prerequisites.get("live_player_update_email_enabled") is not False:
            errors.append("Live player-update email must remain disabled.")
        for name in (
            "service_role_configured",
            "api_audit_required",
            "worker_run_log_required",
        ):
            if prerequisites.get(name) is not True:
                errors.append(f"Production write prerequisite {name} is not protected.")
    if not str(payload.get("fly_image_ref") or "").strip():
        errors.append("Production /health has no Fly image identity.")
    return errors


def transition_verification_errors(
    *,
    before_health: Any,
    after_health: Any,
    before_machines: Any,
    after_machines: Any,
    plan: Mapping[str, Any],
    expected_project_ref: str,
) -> tuple[list[str], dict[str, Any]]:
    errors = production_health_invariant_errors(
        before_health,
        expected_project_ref=expected_project_ref,
    )
    errors.extend(
        production_health_invariant_errors(
            after_health,
            expected_project_ref=expected_project_ref,
        )
    )
    if not isinstance(before_health, dict) or not isinstance(after_health, dict):
        return errors, {}

    changed_invariants = [
        name
        for name in PRESERVED_HEALTH_FIELDS
        if before_health.get(name) != after_health.get(name)
    ]
    if changed_invariants:
        errors.append(
            "Production identity/CORS/Supabase invariants changed: "
            + ", ".join(changed_invariants)
        )

    before_images, before_image_errors = _active_machine_images(before_machines)
    after_images, after_image_errors = _active_machine_images(after_machines)
    errors.extend(before_image_errors)
    errors.extend(after_image_errors)
    if before_images and after_images and before_images != after_images:
        errors.append(
            "Production Fly machine image identity changed during the schema window transition."
        )

    expected_flags = plan.get("feature_flags")
    expected_controlled = plan.get("controlled_write_flags")
    expected_policy = plan.get("production_write_policy")
    if not isinstance(expected_flags, dict) or not isinstance(
        expected_controlled, dict
    ):
        errors.append("Production schema-window plan has no exact feature projection.")
    else:
        actual_flags = after_health.get("feature_flags")
        actual_controlled = after_health.get("controlled_write_flags")
        if not isinstance(actual_flags, dict):
            errors.append("Production /health feature-flag inventory is missing.")
            actual_flags = {}
        if not isinstance(actual_controlled, dict):
            errors.append("Production /health controlled-write inventory is missing.")
            actual_controlled = {}
        flag_projection_mismatches = sorted(
            name
            for name, enabled in actual_flags.items()
            if name not in expected_flags or expected_flags[name] is not enabled
        )
        controlled_projection_mismatches = sorted(
            name
            for name, enabled in actual_controlled.items()
            if name not in expected_controlled
            or expected_controlled[name] is not enabled
        )
        if flag_projection_mismatches:
            errors.append(
                "Production /health feature flags do not match the exact transition plan: "
                + ", ".join(flag_projection_mismatches)
            )
        if controlled_projection_mismatches:
            errors.append(
                "Production /health controlled write flags do not match the exact transition plan: "
                + ", ".join(controlled_projection_mismatches)
            )
        if after_health.get("feature_flag_fingerprint") != (
            _feature_flag_fingerprint(actual_flags)
        ):
            errors.append("Production feature-flag fingerprint is inconsistent.")
        if after_health.get("controlled_write_flag_fingerprint") != (
            _feature_flag_fingerprint(actual_controlled)
        ):
            errors.append(
                "Production controlled-write fingerprint is inconsistent."
            )
    if after_health.get("production_business_write_policy") != expected_policy:
        errors.append("Production write policy does not match the transition plan.")
    if after_health.get("write_wave") != "none" or after_health.get(
        "business_data_write_wave_active"
    ) is not False:
        errors.append("Production staging write wave is not fail-closed.")

    return errors, {
        "action": plan.get("action"),
        "after_feature_flag_fingerprint": after_health.get(
            "feature_flag_fingerprint"
        ),
        "after_production_write_policy": after_health.get(
            "production_business_write_policy"
        ),
        "fly_image_identity_preserved": bool(
            before_images and before_images == after_images
        ),
        "preserved_health_fields": list(PRESERVED_HEALTH_FIELDS),
        "supabase_project_ref": after_health.get("supabase_project_ref"),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path | None, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(rendered, end="")
    else:
        path.write_text(rendered, encoding="utf-8")


def _validate_trigger_command(args: argparse.Namespace) -> int:
    status_lines = args.changed_status_file.read_text(encoding="utf-8").splitlines()
    errors, resolved = schema_window_trigger_errors(
        _read_json(args.trigger_json),
        head_sha=args.head_sha,
        parent_shas=args.parent_sha,
        changed_status_lines=status_lines,
    )
    _write_json(args.output_json, {**resolved, "ok": not errors, "errors": errors})
    return 0 if not errors else 1


def _build_plan_command(args: argparse.Namespace) -> int:
    plan = transition_plan(args.action)
    _write_json(args.output_json, plan)
    args.output_settings.write_text(
        "\n".join(
            f"{name}={value}" for name, value in plan["settings"].items()
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _secrets_command(args: argparse.Namespace) -> int:
    allowed_names: set[str] = set()
    if args.allowed_settings is not None:
        for line in args.allowed_settings.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                allowed_names.add(line.split("=", 1)[0].strip())
    errors, known_pending, summary = secret_deployment_state(
        _read_json(args.fly_secrets_json),
        allowed_pending_names=allowed_names,
    )
    _write_json(args.output_json, {**summary, "ok": not errors and not known_pending})
    if errors:
        for error in errors:
            print(error)
        return 2
    return 3 if known_pending else 0


def _verify_before_command(args: argparse.Namespace) -> int:
    health = _read_json(args.health_json)
    machines = _read_json(args.fly_machines_json)
    errors = production_health_invariant_errors(
        health,
        expected_project_ref=args.expected_project_ref,
    )
    _, machine_errors = _active_machine_images(machines)
    errors.extend(machine_errors)
    _write_json(
        args.output_json,
        {
            "ok": not errors,
            "errors": errors,
            "fly_app": health.get("fly_app_name") if isinstance(health, dict) else None,
            "fly_image_ref": health.get("fly_image_ref") if isinstance(health, dict) else None,
            "supabase_project_ref": (
                health.get("supabase_project_ref")
                if isinstance(health, dict)
                else None
            ),
        },
    )
    return 0 if not errors else 1


def _verify_transition_command(args: argparse.Namespace) -> int:
    errors, summary = transition_verification_errors(
        before_health=_read_json(args.before_health_json),
        after_health=_read_json(args.after_health_json),
        before_machines=_read_json(args.before_machines_json),
        after_machines=_read_json(args.after_machines_json),
        plan=_read_json(args.plan_json),
        expected_project_ref=args.expected_project_ref,
    )
    _write_json(args.output_json, {**summary, "ok": not errors, "errors": errors})
    return 0 if not errors else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and attest the protected production schema window."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    trigger = commands.add_parser("validate-trigger")
    trigger.add_argument("--trigger-json", type=Path, required=True)
    trigger.add_argument("--head-sha", required=True)
    trigger.add_argument("--parent-sha", action="append", default=[])
    trigger.add_argument("--changed-status-file", type=Path, required=True)
    trigger.add_argument("--output-json", type=Path)
    trigger.set_defaults(handler=_validate_trigger_command)

    plan = commands.add_parser("build-plan")
    plan.add_argument(
        "--action",
        choices=tuple(sorted(PRODUCTION_SCHEMA_WINDOW_ACTIONS)),
        required=True,
    )
    plan.add_argument("--output-json", type=Path, required=True)
    plan.add_argument("--output-settings", type=Path, required=True)
    plan.set_defaults(handler=_build_plan_command)

    secrets = commands.add_parser("secrets")
    secrets.add_argument("--fly-secrets-json", type=Path, required=True)
    secrets.add_argument("--allowed-settings", type=Path)
    secrets.add_argument("--output-json", type=Path)
    secrets.set_defaults(handler=_secrets_command)

    before = commands.add_parser("verify-before")
    before.add_argument("--health-json", type=Path, required=True)
    before.add_argument("--fly-machines-json", type=Path, required=True)
    before.add_argument("--expected-project-ref", required=True)
    before.add_argument("--output-json", type=Path)
    before.set_defaults(handler=_verify_before_command)

    verify = commands.add_parser("verify-transition")
    verify.add_argument("--before-health-json", type=Path, required=True)
    verify.add_argument("--after-health-json", type=Path, required=True)
    verify.add_argument("--before-machines-json", type=Path, required=True)
    verify.add_argument("--after-machines-json", type=Path, required=True)
    verify.add_argument("--plan-json", type=Path, required=True)
    verify.add_argument("--expected-project-ref", required=True)
    verify.add_argument("--output-json", type=Path)
    verify.set_defaults(handler=_verify_transition_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
