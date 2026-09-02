from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tomllib
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import unquote, urlsplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.staging_write_waves import (
    ALL_STAGING_WRITE_FLAGS,
    ALWAYS_DISABLED_FLAGS,
)

PRODUCTION_FLY_APP = "juprleagues-api"
PRODUCTION_FLY_REGION = "dfw"
PRODUCTION_ENVIRONMENT = "production"
PRODUCTION_WRITE_POLICY = "enabled"
NO_WRITE_WAVE = "none"
PRODUCTION_WEB_ORIGIN = "https://pickleballclubsandwich.com"
PRODUCTION_API_ORIGIN = "https://api.juprleagues.com"
PRODUCTION_FLY_ORIGIN = f"https://{PRODUCTION_FLY_APP}.fly.dev"
PRODUCTION_PUBLIC_CLUB_SLUG = "tres-palapas"
PRODUCTION_RELEASE_TRIGGER_PATH = ".github/production-api-release.trigger"
PRODUCTION_RELEASE_CONFIRMATION = "DEPLOY PRODUCTION API"
LEGACY_BASELINE_CONFIRMATION = "BOOTSTRAP REVIEWED LEGACY ROLLBACK"
PRODUCTION_RELEASE_TRIGGER_KEYS = frozenset(
    {
        "confirmation",
        "legacy_baseline_config_sha256",
        "legacy_baseline_confirmation",
        "legacy_baseline_image_digest",
        "release_parent_sha",
        "schema_version",
    }
)
DEFAULT_MIGRATION_CONTRACT_PATH = Path(
    "config/production_migration_contract.json"
)
PRODUCTION_ALLOWED_ORIGINS = (
    "https://juprleagues.com",
    "https://www.juprleagues.com",
    "https://pickleballclubsandwich.com",
    "https://www.pickleballclubsandwich.com",
)
DISALLOWED_PRODUCTION_SUPABASE_PROJECT_REFS = frozenset(
    {
        # Dedicated JUPR staging project. A production deploy must never target it.
        "sijpxjxvdtrehmqvirfi",
    }
)
NON_DEPLOYABLE_MIGRATION_FILES = frozenset(
    {
        # Historical backport source retained for reference; Supabase cannot use
        # the XX placeholder as a migration-ledger version.
        "2026XX_backport_tournament_engine.sql",
    }
)
SCHEMA_CONTRACT_ONLY_MIGRATION_NAMES = frozenset(
    {
        # This idempotent hotfix can be applied by a connector-assigned migration
        # name/version. Its exact database shape is therefore probed directly.
        "tournament_registrations_player_id_postgrest_reload",
    }
)
MIGRATION_SCHEMA_CONTRACT_KEYS = (
    "tournament_registrations_player_id_column",
    "idx_tournament_registrations_player_id",
    "uq_tournament_registrations_tournament_player",
    "tournament_player_duplicate_groups",
)

# Read surfaces are also projected so a production deploy cannot silently enable
# an admin replacement before its guarded staging acceptance is complete.
PRODUCTION_READ_FEATURE_FLAGS = (
    "JUPR_ENABLE_NEXT_ADMIN_BADGE_DIAGNOSTICS",
    "JUPR_ENABLE_NEXT_ADMIN_CHALLENGE_LADDER",
    "JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE",
    "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
    "JUPR_ENABLE_NEXT_ADMIN_MATCH_CANONICAL_AUDIT",
    "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG",
    "JUPR_ENABLE_NEXT_ADMIN_MONEYBALL",
    "JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES",
    "JUPR_ENABLE_NEXT_ADMIN_SHELL",
    "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS",
    "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP",
    "JUPR_ENABLE_TEAM_LEAGUES",
    "JUPR_ENABLE_TOURNAMENT_COMMERCE",
    "JUPR_ENABLE_TOURNAMENT_TEAM_COMPETITION",
)
PRODUCTION_TOURNAMENT_FEATURE_FLAGS = ("JUPR_ENABLE_TOURNAMENT_WRITES_PRODUCTION",)
PRODUCTION_FEATURE_FLAGS = tuple(
    sorted(
        set(PRODUCTION_READ_FEATURE_FLAGS)
        | set(PRODUCTION_TOURNAMENT_FEATURE_FLAGS)
        | set(ALL_STAGING_WRITE_FLAGS)
        | set(ALWAYS_DISABLED_FLAGS)
    )
)

# This is the reviewed production activation projection already serving live
# tournament traffic.  Every feature not named here remains explicitly off.
# Keeping one closed-world mapping prevents a routine API deploy from silently
# disabling established production workflows or opening an unreviewed surface.
PRODUCTION_LIVE_BASELINE_ENABLED_FEATURE_FLAGS = frozenset(
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
# These three reviewed League gates open only after the candidate image is
# deployed. A rejected candidate always returns to the exact live baseline.
PRODUCTION_ENABLED_FEATURE_FLAGS = frozenset(
    {
        *PRODUCTION_LIVE_BASELINE_ENABLED_FEATURE_FLAGS,
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
    }
)
PRODUCTION_FEATURE_PROFILES = {
    "baseline": PRODUCTION_LIVE_BASELINE_ENABLED_FEATURE_FLAGS,
    "release": PRODUCTION_ENABLED_FEATURE_FLAGS,
}
if not PRODUCTION_ENABLED_FEATURE_FLAGS.issubset(PRODUCTION_FEATURE_FLAGS):
    raise RuntimeError(
        "Reviewed production feature flags must belong to the closed-world inventory."
    )
PRODUCTION_RUNTIME_SECRET_NAMES = tuple(
    sorted(
        {
            "JUPR_ALLOWED_ORIGINS",
            "JUPR_EMAIL_MODE",
            "JUPR_ENV",
            "JUPR_EXPECTED_MIGRATION_CONTRACT",
            "JUPR_EXPECTED_MIGRATION_HEAD",
            "JUPR_EXPECTED_MIGRATION_PROFILE",
            "JUPR_PRODUCTION_WRITE_POLICY",
            "JUPR_REQUIRE_API_AUDIT_LOG",
            "JUPR_REQUIRE_WORKER_RUN_LOG",
            "JUPR_STAGING_WRITE_WAVE",
            "JUPR_SUPABASE_JWT_MODE",
            "JUPR_WEB_BASE_URL",
            "SUPABASE_ANON_KEY",
            "SUPABASE_JWKS_URL",
            "SUPABASE_SERVICE_ROLE_KEY",
            "SUPABASE_URL",
        }
        | set(PRODUCTION_FEATURE_FLAGS)
    )
)
FORBIDDEN_PRODUCTION_RUNTIME_SECRET_NAMES = (
    "JUPR_ALLOWED_ORIGIN_REGEX",
)
REQUIRED_GITHUB_ENV_NAMES = (
    "EXPECTED_MIGRATION_HEAD",
    "EXPECTED_SUPABASE_PROJECT_REF",
    "FLY_API_TOKEN",
    "FLY_SSH_TOKEN",
    "GITHUB_SHA",
    "SUPABASE_ANON_KEY",
    "SUPABASE_DATABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "SUPABASE_URL",
)
SECONDARY_HEALTH_IDENTITY_KEYS = (
    "ok",
    "service",
    "environment",
    "git_commit_sha",
    "image_build_git_sha",
    "fly_app_name",
    "fly_image_ref",
    "web_origin",
    "supabase_project_ref",
    "jwt_verification_mode",
    "jwt_verification_project_ref",
    "write_wave",
    "business_data_write_wave_active",
    "production_business_write_policy",
    "expected_migration_contract",
    "expected_migration_head",
    "expected_migration_profile",
    "cors_allowed_origins",
    "cors_allowed_origin_regex",
    "feature_flags",
    "feature_flag_fingerprint",
    "controlled_write_flags",
    "controlled_write_flag_fingerprint",
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PROJECT_REF_RE = re.compile(r"^[a-z0-9]{20}$")
_MIGRATION_FILE_RE = re.compile(
    r"^(?P<version>\d{8}(?:\d{6})?)_"
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9_]*)\.sql$"
)
_MIGRATION_VERSION_RE = re.compile(r"^\d{8}(?:\d{6})?$")
_FLY_IMAGE_RE = re.compile(
    rf"^registry\.fly\.io/{re.escape(PRODUCTION_FLY_APP)}"
    r"(?::[A-Za-z0-9._-]+(?:@sha256:[0-9a-f]{64})?|@sha256:[0-9a-f]{64})$"
)


def canonical_https_origin(raw: str | None) -> str | None:
    try:
        parsed = urlsplit(str(raw or "").strip())
        port = parsed.port
    except (TypeError, ValueError):
        return None
    host = (parsed.hostname or "").strip().lower()
    if (
        parsed.scheme.lower() != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    return f"https://{host}"


def supabase_project_ref(raw: str | None) -> str | None:
    origin = canonical_https_origin(raw)
    if origin is None:
        return None
    host = urlsplit(origin).hostname or ""
    suffix = ".supabase.co"
    if not host.endswith(suffix):
        return None
    project_ref = host.removesuffix(suffix)
    if not _PROJECT_REF_RE.fullmatch(project_ref):
        return None
    return project_ref


def database_url_project_ref(raw: str | None) -> str | None:
    """Extract a Supabase project ref from a direct or transaction-pooler URL."""

    try:
        parsed = urlsplit(str(raw or "").strip())
        port = parsed.port
    except (TypeError, ValueError):
        return None
    host = (parsed.hostname or "").strip().lower()
    username = unquote(parsed.username or "").strip().lower()
    database = (parsed.path or "").removeprefix("/")
    if (
        parsed.scheme not in {"postgres", "postgresql"}
        or not parsed.password
        or not host
        or port is None
        or database != "postgres"
        or parsed.query
        or parsed.fragment
    ):
        return None

    direct = re.fullmatch(r"db\.([a-z0-9]{20})\.supabase\.co", host)
    if direct and username == "postgres":
        return direct.group(1)

    if host.endswith(".pooler.supabase.com"):
        pooled = re.fullmatch(r"postgres\.([a-z0-9]{20})", username)
        if pooled:
            return pooled.group(1)
    return None


def production_release_trigger_errors(
    payload: Any,
    *,
    head_sha: str,
    parent_shas: Iterable[str],
    changed_status_lines: Iterable[str],
) -> tuple[list[str], dict[str, str]]:
    """Validate the only push shape authorized to start a production deploy."""

    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["Production release trigger must be a JSON object."], {}

    unknown_keys = sorted(set(payload) - PRODUCTION_RELEASE_TRIGGER_KEYS)
    missing_keys = sorted(
        {"schema_version", "confirmation", "release_parent_sha"} - set(payload)
    )
    if unknown_keys:
        errors.append(
            "Production release trigger has unknown keys: "
            + ", ".join(unknown_keys)
        )
    if missing_keys:
        errors.append(
            "Production release trigger is missing keys: "
            + ", ".join(missing_keys)
        )
    if type(payload.get("schema_version")) is not int or payload.get(
        "schema_version"
    ) != 1:
        errors.append("Production release trigger must use schema_version=1.")

    string_keys = PRODUCTION_RELEASE_TRIGGER_KEYS - {"schema_version"}
    for name in sorted(string_keys & set(payload)):
        if not isinstance(payload.get(name), str):
            errors.append(f"Production release trigger {name} must be a string.")

    raw_head = str(head_sha or "").strip()
    clean_head = raw_head.lower()
    if raw_head != clean_head or not _SHA_RE.fullmatch(clean_head):
        errors.append("Production trigger HEAD must be an exact lowercase Git SHA.")
    raw_parents = tuple(str(sha or "").strip() for sha in parent_shas)
    clean_parents = tuple(sha.lower() for sha in raw_parents)
    if (
        len(clean_parents) != 1
        or not _SHA_RE.fullmatch(clean_parents[0] if clean_parents else "")
        or raw_parents != clean_parents
    ):
        errors.append(
            "Production trigger commit must have exactly one exact Git parent."
        )
    raw_reviewed_parent = str(payload.get("release_parent_sha") or "").strip()
    reviewed_parent = raw_reviewed_parent.lower()
    if (
        raw_reviewed_parent != reviewed_parent
        or not _SHA_RE.fullmatch(reviewed_parent)
    ):
        errors.append("Production trigger release_parent_sha is invalid.")
    elif len(clean_parents) == 1 and reviewed_parent != clean_parents[0]:
        errors.append(
            "Production trigger release_parent_sha does not match its commit parent."
        )

    status_lines = tuple(str(line).rstrip("\r\n") for line in changed_status_lines)
    allowed_statuses = {
        f"A\t{PRODUCTION_RELEASE_TRIGGER_PATH}",
        f"M\t{PRODUCTION_RELEASE_TRIGGER_PATH}",
    }
    if len(status_lines) != 1 or status_lines[0] not in allowed_statuses:
        errors.append(
            "Production trigger commit must only add or modify the exact trigger file."
        )

    confirmation = str(payload.get("confirmation") or "").strip()
    if confirmation != PRODUCTION_RELEASE_CONFIRMATION:
        errors.append("Production trigger approval phrase is incorrect.")

    raw_legacy_digest = str(
        payload.get("legacy_baseline_image_digest") or ""
    ).strip()
    legacy_digest = raw_legacy_digest.lower()
    raw_legacy_config = str(
        payload.get("legacy_baseline_config_sha256") or ""
    ).strip()
    legacy_config = raw_legacy_config.lower()
    legacy_confirmation = str(
        payload.get("legacy_baseline_confirmation") or ""
    ).strip()
    if legacy_digest or legacy_config or legacy_confirmation:
        if (
            raw_legacy_digest != legacy_digest
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", legacy_digest)
        ):
            errors.append(
                "Production trigger legacy image digest is missing or invalid."
            )
        if (
            raw_legacy_config != legacy_config
            or not re.fullmatch(r"[0-9a-f]{64}", legacy_config)
        ):
            errors.append(
                "Production trigger legacy config fingerprint is missing or invalid."
            )
        if legacy_confirmation != LEGACY_BASELINE_CONFIRMATION:
            errors.append(
                "Production trigger legacy baseline approval phrase is incorrect."
            )

    resolved = {
        "candidate_sha": clean_head,
        "confirmation": confirmation,
        "legacy_baseline_config_sha256": legacy_config,
        "legacy_baseline_confirmation": legacy_confirmation,
        "legacy_baseline_image_digest": legacy_digest,
        "release_parent_sha": reviewed_parent,
    }
    return errors, resolved


def expected_migration_inventory(
    migrations_dir: Path,
) -> tuple[tuple[str, str], ...]:
    inventory: list[tuple[str, str]] = []
    invalid_files: list[str] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        match = _MIGRATION_FILE_RE.fullmatch(path.name)
        if match:
            inventory.append((match.group("version"), match.group("name")))
        elif path.name not in NON_DEPLOYABLE_MIGRATION_FILES:
            invalid_files.append(path.name)
    if invalid_files:
        raise ValueError(
            "Unrecognized non-deployable Supabase migration filenames: "
            + ", ".join(invalid_files)
        )
    if not inventory:
        raise ValueError(f"No deployable Supabase migrations found in {migrations_dir}")
    versions = [version for version, _ in inventory]
    names = [name for _, name in inventory]
    if len(set(versions)) != len(versions):
        raise ValueError("Deployable Supabase migration versions must be unique")
    if len(set(names)) != len(names):
        raise ValueError("Deployable Supabase logical migration names must be unique")
    return tuple(sorted(inventory, key=lambda item: _migration_sort_key(item[0])))


def expected_migration_versions(
    migrations_dir: Path,
) -> tuple[str, ...]:
    return tuple(
        version for version, _ in expected_migration_inventory(migrations_dir)
    )


def expected_migration_names(
    migrations_dir: Path,
) -> tuple[str, ...]:
    return tuple(name for _, name in expected_migration_inventory(migrations_dir))


def repository_migration_content_fingerprint(migrations_dir: Path) -> str:
    records = [
        f"{path.name}\t{hashlib.sha256(path.read_bytes()).hexdigest()}"
        for path in sorted(migrations_dir.glob("*.sql"))
    ]
    if not records:
        raise ValueError(f"No Supabase migration SQL found in {migrations_dir}")
    return hashlib.sha256("\n".join(records).encode("utf-8")).hexdigest()


def load_migration_contract(
    contract_path: Path,
    migrations_dir: Path,
) -> dict[str, Any]:
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Production migration contract must use schema_version=1")
    profile = str(payload.get("profile") or "").strip()
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]+", profile):
        raise ValueError("Production migration contract has an invalid profile")

    required_names = payload.get("required_ledger_names")
    if not isinstance(required_names, list) or not required_names:
        raise ValueError("Production migration contract has no required ledger names")
    clean_required = [str(name).strip() for name in required_names]
    if any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]*", name)
        for name in clean_required
    ):
        raise ValueError("Production migration contract has an invalid ledger name")
    if clean_required != sorted(set(clean_required)):
        raise ValueError(
            "Production migration contract ledger names must be unique and sorted"
        )

    deployment_order = payload.get("deployment_order")
    if not isinstance(deployment_order, list) or not deployment_order:
        raise ValueError(
            "Production migration contract has no canonical deployment order"
        )
    clean_deployment_order = [
        str(name).strip() for name in deployment_order
    ]
    if any(
        not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]*", name)
        for name in clean_deployment_order
    ):
        raise ValueError(
            "Production migration contract deployment order has an invalid "
            "ledger name"
        )
    if len(clean_deployment_order) != len(set(clean_deployment_order)):
        raise ValueError(
            "Production migration contract deployment order must be unique"
        )
    if set(clean_deployment_order) != set(clean_required):
        raise ValueError(
            "Production migration contract deployment order must contain "
            "exactly the required ledger names"
        )

    contract_only = payload.get("schema_contract_only_repository_migrations")
    if not isinstance(contract_only, list) or set(contract_only) != set(
        SCHEMA_CONTRACT_ONLY_MIGRATION_NAMES
    ):
        raise ValueError(
            "Production migration contract has the wrong schema-only migration set"
        )
    repository_names = expected_migration_names(migrations_dir)
    if not set(contract_only).issubset(repository_names):
        raise ValueError(
            "Production migration contract references an absent repository migration"
        )
    repository_fingerprint = repository_migration_content_fingerprint(
        migrations_dir
    )
    if (
        payload.get("repository_migration_content_sha256")
        != repository_fingerprint
    ):
        raise ValueError(
            "Production migration contract is stale for repository migration SQL"
        )
    if not isinstance(payload.get("allow_additional_ledger_names"), bool):
        raise ValueError(
            "Production migration contract must declare allow_additional_ledger_names"
        )
    allowed_duplicates = payload.get("allowed_duplicate_ledger_names")
    if not isinstance(allowed_duplicates, list):
        raise ValueError(
            "Production migration contract must declare allowed_duplicate_ledger_names"
        )
    clean_allowed_duplicates = [
        str(name).strip() for name in allowed_duplicates
    ]
    if (
        clean_allowed_duplicates != sorted(set(clean_allowed_duplicates))
        or not set(clean_allowed_duplicates).issubset(clean_required)
    ):
        raise ValueError(
            "Production migration contract duplicate ledger names must be "
            "unique, sorted, and required"
        )
    return {
        "profile": profile,
        "required_ledger_names": tuple(clean_required),
        "deployment_order": tuple(clean_deployment_order),
        "allow_additional_ledger_names": payload[
            "allow_additional_ledger_names"
        ],
        "allowed_duplicate_ledger_names": tuple(
            clean_allowed_duplicates
        ),
        "repository_logical_names": repository_names,
        "repository_migration_content_sha256": repository_fingerprint,
        "schema_contract_only_repository_migrations": tuple(contract_only),
    }


def _migration_sort_key(version: str) -> str:
    clean = str(version).strip()
    if not _MIGRATION_VERSION_RE.fullmatch(clean):
        raise ValueError(f"Invalid Supabase migration version: {version!r}")
    return clean if len(clean) == 14 else f"{clean}000000"


def parse_remote_migration_ledger(
    lines: Iterable[str],
) -> tuple[list[tuple[str, str]], list[str]]:
    entries: list[tuple[str, str]] = []
    invalid: list[str] = []
    for raw in lines:
        clean = str(raw).strip()
        if not clean:
            continue
        pieces = clean.split("\t")
        if (
            len(pieces) != 2
            or not _MIGRATION_VERSION_RE.fullmatch(pieces[0].strip())
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_]*", pieces[1].strip())
        ):
            invalid.append(clean)
            continue
        entries.append((pieces[0].strip(), pieces[1].strip()))
    return entries, invalid


def pending_required_migration_names(
    deployment_order: Iterable[str],
    remote_entries: Iterable[tuple[str, str]],
) -> tuple[str, ...]:
    """Return missing required migrations in the reviewed dependency order."""
    remote_names = {
        str(name).strip()
        for _, name in remote_entries
        if str(name).strip()
    }
    return tuple(
        name
        for raw_name in deployment_order
        if (name := str(raw_name).strip()) and name not in remote_names
    )


def migration_ledger_errors(
    required_names: Iterable[str],
    remote_entries: Iterable[tuple[str, str]],
    *,
    invalid_remote_rows: Iterable[str] = (),
    allow_additional_names: bool = False,
    allowed_duplicate_names: Iterable[str] = (),
) -> list[str]:
    expected = {
        str(value).strip()
        for value in required_names
        if str(value).strip()
    }
    remote = [
        (str(version).strip(), str(name).strip())
        for version, name in remote_entries
    ]
    errors: list[str] = []
    if not expected:
        return ["Repository logical migration inventory is empty."]
    if not remote:
        return ["Remote Supabase migration ledger has no valid named entries."]
    invalid_remote = sorted(set(invalid_remote_rows))
    if invalid_remote:
        errors.append(
            "Remote Supabase migration ledger contains invalid rows: "
            + ", ".join(invalid_remote)
        )

    remote_names = [name for _, name in remote]
    reviewed_duplicates = {
        str(name).strip()
        for name in allowed_duplicate_names
        if str(name).strip()
    }
    duplicate_names = sorted(
        {name for name in remote_names if remote_names.count(name) > 1}
        - reviewed_duplicates
    )
    if duplicate_names:
        errors.append(
            "Remote Supabase migration ledger repeats logical names: "
            + ", ".join(duplicate_names)
        )
    missing = sorted(expected - set(remote_names))
    if missing:
        errors.append(
            "Remote Supabase migration ledger is missing repository logical names: "
            + ", ".join(missing)
        )
    unexpected = sorted(set(remote_names) - expected)
    if unexpected and not allow_additional_names:
        errors.append(
            "Remote Supabase migration ledger contains names outside the reviewed "
            "production profile: "
            + ", ".join(unexpected)
        )
    return errors


def migration_schema_contract_errors(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["Production migration schema-contract payload is not a JSON object."]
    errors: list[str] = []
    for key in MIGRATION_SCHEMA_CONTRACT_KEYS[:3]:
        if payload.get(key) is not True:
            errors.append(f"Production schema contract is missing {key}.")
    duplicate_groups = payload.get("tournament_player_duplicate_groups")
    if duplicate_groups != 0:
        errors.append(
            "Production schema contract has duplicate tournament/player registrations."
        )
    return errors


def migration_contract_fingerprint(
    *,
    profile: str,
    required_ledger_names: Iterable[str],
    deployment_order: Iterable[str],
    repository_migration_content_sha256: str,
    allowed_duplicate_ledger_names: Iterable[str] = (),
) -> str:
    material = [
        f"profile:{profile}",
        f"repository-content:{repository_migration_content_sha256}",
        *(
            f"ledger-name:{name}"
            for name in sorted(set(required_ledger_names))
        ),
        *(
            f"deployment-order:{position}:{name}"
            for position, name in enumerate(deployment_order, start=1)
        ),
        *(
            f"allowed-duplicate-ledger-name:{name}"
            for name in sorted(set(allowed_duplicate_ledger_names))
        ),
        *(f"schema-probe:{key}" for key in MIGRATION_SCHEMA_CONTRACT_KEYS),
    ]
    return hashlib.sha256("\n".join(material).encode("utf-8")).hexdigest()


def expected_production_feature_flags(
    *, profile: str = "release"
) -> dict[str, bool]:
    enabled_flags = PRODUCTION_FEATURE_PROFILES.get(profile)
    if enabled_flags is None:
        raise ValueError(f"Unknown production feature profile: {profile}")
    return {name: name in enabled_flags for name in PRODUCTION_FEATURE_FLAGS}


def expected_production_controlled_write_flags(
    *, profile: str = "release"
) -> dict[str, bool]:
    enabled_flags = PRODUCTION_FEATURE_PROFILES.get(profile)
    if enabled_flags is None:
        raise ValueError(f"Unknown production feature profile: {profile}")
    return {name: name in enabled_flags for name in ALL_STAGING_WRITE_FLAGS}


def production_feature_profile_from_health(health: Any) -> str | None:
    """Identify an exact reviewed live profile without accepting flag drift."""

    if not isinstance(health, dict):
        return None
    prerequisites = health.get("write_prerequisites")
    if (
        health.get("production_business_write_policy") != PRODUCTION_WRITE_POLICY
        or health.get("write_wave") != NO_WRITE_WAVE
        or health.get("staging_write_wave") != NO_WRITE_WAVE
        or health.get("business_data_write_wave_active") is not False
        or health.get("public_live_writes_enabled") is not True
        or health.get("public_live_production_override_enabled") is not True
        or not isinstance(prerequisites, dict)
        or prerequisites.get("email_mode") != "dry_run"
        or prerequisites.get("live_player_update_email_enabled") is not False
    ):
        return None
    for profile in PRODUCTION_FEATURE_PROFILES:
        expected_flags = expected_production_feature_flags(profile=profile)
        expected_controlled = expected_production_controlled_write_flags(
            profile=profile
        )
        if (
            health.get("feature_flags") == expected_flags
            and health.get("feature_flag_fingerprint")
            == feature_flag_fingerprint(expected_flags)
            and health.get("controlled_write_flags") == expected_controlled
            and health.get("controlled_write_flag_fingerprint")
            == feature_flag_fingerprint(expected_controlled)
        ):
            return profile
    return None


def feature_flag_fingerprint(flags: Mapping[str, bool]) -> str:
    return hashlib.sha256(
        "\n".join(
            f"{name}={1 if bool(enabled) else 0}"
            for name, enabled in sorted(flags.items())
        ).encode("utf-8")
    ).hexdigest()


def production_fly_config_errors(config_path: Path) -> list[str]:
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - verifier returns deterministic errors
        return [f"Could not parse production Fly config: {exc}"]

    errors: list[str] = []
    if config.get("app") != PRODUCTION_FLY_APP:
        errors.append(f"Fly app must be exactly {PRODUCTION_FLY_APP}.")
    if config.get("primary_region") != PRODUCTION_FLY_REGION:
        errors.append(f"Fly primary region must be exactly {PRODUCTION_FLY_REGION}.")
    build = config.get("build") or {}
    if build.get("dockerfile") != "Dockerfile.api":
        errors.append("Production Fly config must build Dockerfile.api.")

    env = config.get("env") or {}
    required_values = {
        "PORT": "8080",
        "JUPR_ENV": PRODUCTION_ENVIRONMENT,
        "JUPR_EMAIL_MODE": "dry_run",
        "JUPR_PRODUCTION_WRITE_POLICY": PRODUCTION_WRITE_POLICY,
        "JUPR_STAGING_WRITE_WAVE": NO_WRITE_WAVE,
        "JUPR_WEB_BASE_URL": PRODUCTION_WEB_ORIGIN,
        "JUPR_REQUIRE_API_AUDIT_LOG": "1",
        "JUPR_REQUIRE_WORKER_RUN_LOG": "1",
        "JUPR_SUPABASE_JWT_MODE": "jwks",
        "JUPR_ALLOWED_ORIGINS": ",".join(PRODUCTION_ALLOWED_ORIGINS),
    }
    for name, expected in required_values.items():
        if env.get(name) != expected:
            errors.append(f"Production Fly config must set {name}={expected!r}.")
    if str(env.get("JUPR_ALLOWED_ORIGIN_REGEX") or "").strip():
        errors.append("Production Fly config must not allow a CORS origin regex.")
    for name, enabled in expected_production_feature_flags().items():
        expected = "1" if enabled else "0"
        if env.get(name) != expected:
            errors.append(
                "Production Fly config must set the reviewed feature projection "
                f"{name}={expected!r}."
            )

    http_service = config.get("http_service") or {}
    if http_service.get("force_https") is not True:
        errors.append("Production Fly config must force HTTPS.")
    if http_service.get("auto_stop_machines") != "off":
        errors.append("Production Fly config must keep the API machine running.")
    if int(http_service.get("min_machines_running") or 0) < 1:
        errors.append("Production Fly config must keep at least one machine running.")
    return errors


def preflight_errors(
    env: Mapping[str, str],
    *,
    config_path: Path,
    migrations_dir: Path,
    migration_contract_path: Path = DEFAULT_MIGRATION_CONTRACT_PATH,
) -> tuple[list[str], tuple[str, ...]]:
    errors: list[str] = []
    missing = [name for name in REQUIRED_GITHUB_ENV_NAMES if not str(env.get(name) or "").strip()]
    if missing:
        errors.append("Missing required protected production configuration: " + ", ".join(missing))

    candidate_sha = str(env.get("GITHUB_SHA") or "").strip().lower()
    if not _SHA_RE.fullmatch(candidate_sha):
        errors.append("GITHUB_SHA must be an exact lowercase 40-character commit SHA.")
    if str(env.get("JUPR_ENV") or "").strip() != PRODUCTION_ENVIRONMENT:
        errors.append("JUPR_ENV must be exactly production.")
    if str(env.get("FLY_APP_NAME") or "").strip() != PRODUCTION_FLY_APP:
        errors.append(f"FLY_APP_NAME must be exactly {PRODUCTION_FLY_APP}.")
    if not _MIGRATION_VERSION_RE.fullmatch(
        str(env.get("EXPECTED_MIGRATION_HEAD") or "").strip()
    ):
        errors.append(
            "EXPECTED_MIGRATION_HEAD must be the reviewed connector ledger head."
        )

    expected_ref = str(env.get("EXPECTED_SUPABASE_PROJECT_REF") or "").strip().lower()
    if not _PROJECT_REF_RE.fullmatch(expected_ref):
        errors.append(
            "EXPECTED_SUPABASE_PROJECT_REF must be the exact 20-character production project ref."
        )
    elif expected_ref in DISALLOWED_PRODUCTION_SUPABASE_PROJECT_REFS:
        errors.append("The dedicated staging Supabase project is forbidden for production.")

    actual_ref = supabase_project_ref(env.get("SUPABASE_URL"))
    if actual_ref is None:
        errors.append("SUPABASE_URL must be a canonical Supabase HTTPS project origin.")
    elif expected_ref and actual_ref != expected_ref:
        errors.append("SUPABASE_URL does not target the protected production project ref.")

    database_ref = database_url_project_ref(env.get("SUPABASE_DATABASE_URL"))
    if database_ref is None:
        errors.append(
            "SUPABASE_DATABASE_URL must be a canonical production Supabase direct or pooler URL."
        )
    elif expected_ref and database_ref != expected_ref:
        errors.append("SUPABASE_DATABASE_URL does not target the protected production project ref.")

    errors.extend(production_fly_config_errors(config_path))
    try:
        migrations = expected_migration_versions(migrations_dir)
    except ValueError as exc:
        errors.append(str(exc))
        migrations = ()
    try:
        load_migration_contract(migration_contract_path, migrations_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    return errors, migrations


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _secret_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        for key in ("secrets", "Secrets"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def _secret_names(payload: Any) -> set[str]:
    names: set[str] = set()
    for item in _secret_items(payload):
        for key in ("name", "Name", "NAME"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                names.add(value.strip())
                break
    return names


def pending_secret_errors(payload: Any) -> list[str]:
    items = _secret_items(payload)
    if not items:
        return ["Fly secret deployment-status inventory is missing or unrecognized."]
    pending: list[str] = []
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
        if status != "deployed":
            pending.append(f"{name}={status}")
    if pending:
        return [
            "Fly has staged, partial, or unknown secret deployments: "
            + ", ".join(sorted(pending))
        ]
    return []


def safe_secret_convergence_errors(payload: Any) -> list[str]:
    """Allow retrying only the workflow's complete fail-closed secret bundle."""

    items = _secret_items(payload)
    if not items:
        return ["Fly secret deployment-status inventory is missing or unrecognized."]
    unsafe_pending: list[str] = []
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
        if status != "deployed" and name not in PRODUCTION_RUNTIME_SECRET_NAMES:
            unsafe_pending.append(f"{name}={status}")
    if unsafe_pending:
        return [
            "Fly has pending secrets outside the exact safe convergence bundle: "
            + ", ".join(sorted(unsafe_pending))
        ]
    return []


def secret_inventory_errors(payload: Any) -> list[str]:
    actual = _secret_names(payload)
    if not actual:
        return ["Fly runtime secret inventory is missing or unrecognized."]
    errors: list[str] = []
    missing = [name for name in PRODUCTION_RUNTIME_SECRET_NAMES if name not in actual]
    if missing:
        errors.append(
            "Fly runtime secret inventory is missing required names: "
            + ", ".join(missing)
        )
    forbidden = [
        name for name in FORBIDDEN_PRODUCTION_RUNTIME_SECRET_NAMES if name in actual
    ]
    if forbidden:
        errors.append(
            "Fly runtime secret inventory contains forbidden names: "
            + ", ".join(forbidden)
        )
    errors.extend(pending_secret_errors(payload))
    return errors


def _nonterminal_machine_images(
    payload: Any,
) -> tuple[list[dict[str, str]], list[str]]:
    if isinstance(payload, dict):
        for key in ("machines", "Machines"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list):
        return [], ["Fly machine inventory is missing or unrecognized."]
    identities: list[dict[str, str]] = []
    errors: list[str] = []
    for index, machine in enumerate(payload):
        if not isinstance(machine, dict):
            errors.append(f"Fly machine inventory entry {index} is invalid.")
            continue
        state = str(machine.get("state") or machine.get("State") or "").strip().lower()
        machine_id = str(machine.get("id") or machine.get("ID") or index)
        if not state:
            errors.append(f"Fly machine {machine_id} has no state.")
            continue
        if state in {"destroyed", "replaced", "migrated"}:
            continue
        config = machine.get("config") or machine.get("Config") or {}
        candidates = [config.get("image") if isinstance(config, dict) else None]
        candidates.extend([machine.get("image"), machine.get("image_ref")])
        image_ref = machine.get("image_ref")
        if isinstance(image_ref, dict):
            candidates.extend(
                [
                    image_ref.get("ref"),
                    image_ref.get("Ref"),
                ]
            )
        resolved = next(
            (
                candidate.strip()
                for candidate in candidates
                if isinstance(candidate, str) and candidate.strip()
            ),
            None,
        )
        if resolved is None:
            errors.append(
                f"Nonterminal Fly machine {machine_id} has no deployed image reference."
            )
            continue
        if not _FLY_IMAGE_RE.fullmatch(resolved):
            errors.append(
                f"Nonterminal Fly machine {machine_id} has an invalid or foreign image reference."
            )
            continue

        if not isinstance(image_ref, dict):
            errors.append(
                f"Nonterminal Fly machine {machine_id} has no structured "
                "immutable image identity."
            )
            continue
        registry = str(
            image_ref.get("registry") or image_ref.get("Registry") or ""
        ).strip().lower()
        repository = str(
            image_ref.get("repository")
            or image_ref.get("Repository")
            or ""
        ).strip()
        if registry != "registry.fly.io" or repository != PRODUCTION_FLY_APP:
            errors.append(
                f"Nonterminal Fly machine {machine_id} reports a foreign "
                "image registry or repository."
            )
            continue
        configured_digest = ""
        if "@sha256:" in resolved:
            configured_digest = "sha256:" + resolved.rsplit("@sha256:", 1)[1]
        raw_digest = str(
            image_ref.get("digest") or image_ref.get("Digest") or ""
        ).strip().lower()
        if raw_digest and not raw_digest.startswith("sha256:"):
            raw_digest = f"sha256:{raw_digest}"
        if (
            configured_digest
            and raw_digest
            and configured_digest != raw_digest
        ):
            errors.append(
                f"Nonterminal Fly machine {machine_id} reports conflicting "
                "configured and immutable image digests."
            )
            continue
        digest = raw_digest or configured_digest
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            errors.append(
                f"Nonterminal Fly machine {machine_id} has no immutable image digest."
            )
            continue
        digest_ref = f"registry.fly.io/{PRODUCTION_FLY_APP}@{digest}"
        identities.append(
            {
                "id": machine_id,
                "state": state,
                "configured_ref": resolved,
                "digest": digest,
                "immutable_ref": digest_ref,
            }
        )
    if not identities and not errors:
        errors.append("Fly machine inventory has no nonterminal production machines.")
    return identities, errors


def predeploy_fly_config_fingerprint(path: Path) -> str:
    """Validate and fingerprint the exact remote config used for rollback."""

    raw = path.read_bytes()
    payload = tomllib.loads(raw.decode("utf-8"))
    if payload.get("app") != PRODUCTION_FLY_APP:
        raise ValueError(
            "Saved pre-deploy Fly config does not target the exact production app."
        )
    return hashlib.sha256(raw).hexdigest()


def predeploy_rollback_snapshot(
    health: Any,
    fly_status: Any,
    *,
    fly_config_sha256: str,
    reviewed_legacy_image_digest: str | None = None,
    reviewed_legacy_config_sha256: str | None = None,
    capture_unreviewed_legacy_evidence: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Build the exact rollback pair before any production runtime mutation."""

    errors: list[str] = []
    if not isinstance(health, dict):
        return ["Pre-deploy production health payload is not a JSON object."], {}
    if not re.fullmatch(r"[0-9a-f]{64}", str(fly_config_sha256).strip().lower()):
        errors.append("Pre-deploy Fly config fingerprint is invalid.")
    identities, machine_errors = _nonterminal_machine_images(fly_status)
    errors.extend(machine_errors)
    configured_refs = {item["configured_ref"] for item in identities}
    digests = {item["digest"] for item in identities}
    immutable_refs = {item["immutable_ref"] for item in identities}
    if len(configured_refs) != 1 or len(digests) != 1:
        errors.append(
            "Pre-deploy production machines must all use one exact image reference."
        )
    image_ref = next(iter(configured_refs), None)
    image_digest = next(iter(digests), None)
    immutable_ref = next(iter(immutable_refs), None)
    git_sha = str(health.get("git_commit_sha") or "").strip().lower()
    image_sha = str(health.get("image_build_git_sha") or "").strip().lower()
    git_identity_is_exact = bool(
        _SHA_RE.fullmatch(git_sha)
        and _SHA_RE.fullmatch(image_sha)
        and git_sha == image_sha
    )
    git_identity_is_legacy = (
        git_sha in {"", "unknown"} and image_sha in {"", "unknown"}
    )
    reviewed_digest = str(reviewed_legacy_image_digest or "").strip().lower()
    reviewed_config = str(reviewed_legacy_config_sha256 or "").strip().lower()
    legacy_review_requested = bool(reviewed_digest or reviewed_config)
    live_feature_profile = production_feature_profile_from_health(health)
    if live_feature_profile is None:
        errors.append(
            "Pre-deploy production feature flags do not match a reviewed live profile."
        )
    expected_identity = {
        "ok": True,
        "service": "jupr-api",
        "environment": PRODUCTION_ENVIRONMENT,
        "fly_app_name": PRODUCTION_FLY_APP,
        "fly_image_ref": image_ref,
    }
    for key, expected in expected_identity.items():
        if health.get(key) != expected:
            errors.append(
                f"Pre-deploy production health identity mismatch for {key}."
            )
    identity_mode = "exact-git"
    if git_identity_is_exact:
        if legacy_review_requested:
            errors.append(
                "Legacy baseline review values are forbidden once production "
                "reports an exact Git identity."
            )
    elif git_identity_is_legacy:
        if capture_unreviewed_legacy_evidence and not legacy_review_requested:
            identity_mode = "legacy-unreviewed-evidence"
        else:
            identity_mode = "legacy-immutable-bootstrap"
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", reviewed_digest):
                errors.append(
                    "Legacy baseline bootstrap requires the reviewed immutable "
                    "Fly image digest."
                )
            elif reviewed_digest != image_digest:
                errors.append(
                    "Live legacy Fly image digest does not match the reviewed baseline."
                )
            if not re.fullmatch(r"[0-9a-f]{64}", reviewed_config):
                errors.append(
                    "Legacy baseline bootstrap requires the reviewed Fly config "
                    "fingerprint."
                )
            elif reviewed_config != str(fly_config_sha256).strip().lower():
                errors.append(
                    "Live legacy Fly config does not match the reviewed baseline."
                )
    else:
        errors.append(
            "Pre-deploy production health has an inconsistent or invalid Git identity."
        )
    snapshot = {
        "fly_app": PRODUCTION_FLY_APP,
        "identity_mode": identity_mode,
        "feature_profile": live_feature_profile,
        "git_commit_sha": git_sha if git_identity_is_exact else None,
        "image_build_git_sha": image_sha if git_identity_is_exact else None,
        "fly_image_ref": image_ref,
        "fly_image_digest": image_digest,
        "fly_immutable_image_ref": immutable_ref,
        "fly_config_sha256": str(fly_config_sha256).strip().lower() or None,
        "reviewed_legacy_image_digest": (
            reviewed_digest if identity_mode == "legacy-immutable-bootstrap" else None
        ),
        "reviewed_legacy_config_sha256": (
            reviewed_config if identity_mode == "legacy-immutable-bootstrap" else None
        ),
    }
    return errors, snapshot


def runtime_identity_errors(
    health: Any,
    *,
    candidate_sha: str,
    expected_project_ref: str,
    expected_migration_head: str,
    expected_migration_contract: str,
    expected_migration_profile: str,
    fly_machines: Any,
    fly_secrets: Any,
    allow_legacy_git_identity: bool = False,
    feature_profile: str = "release",
) -> list[str]:
    if not isinstance(health, dict):
        return ["Production health payload is not a JSON object."]

    errors: list[str] = []
    if not _SHA_RE.fullmatch(str(candidate_sha).strip().lower()):
        errors.append("Expected production candidate SHA is invalid.")
    if not _PROJECT_REF_RE.fullmatch(str(expected_project_ref).strip().lower()):
        errors.append("Expected production Supabase project ref is invalid.")
    if expected_project_ref in DISALLOWED_PRODUCTION_SUPABASE_PROJECT_REFS:
        errors.append("Expected production Supabase project ref is the staging project.")
    if not _MIGRATION_VERSION_RE.fullmatch(str(expected_migration_head).strip()):
        errors.append("Expected production migration head is invalid.")
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(expected_migration_contract).strip().lower()
    ):
        errors.append("Expected production migration contract is invalid.")
    if not re.fullmatch(
        r"[a-z0-9][a-z0-9_-]+", str(expected_migration_profile).strip()
    ):
        errors.append("Expected production migration profile is invalid.")

    expected_flags = expected_production_feature_flags(profile=feature_profile)
    expected_controlled_write_flags = (
        expected_production_controlled_write_flags(profile=feature_profile)
    )
    expected_identity: dict[str, Any] = {
        "ok": True,
        "service": "jupr-api",
        "environment": PRODUCTION_ENVIRONMENT,
        "fly_app_name": PRODUCTION_FLY_APP,
        "web_origin": PRODUCTION_WEB_ORIGIN,
        "supabase_project_ref": expected_project_ref.lower(),
        "jwt_verification_configured": True,
        "jwt_verification_mode": "jwks",
        "jwt_verification_project_ref": expected_project_ref.lower(),
        "write_wave": NO_WRITE_WAVE,
        "staging_write_wave": NO_WRITE_WAVE,
        "business_data_write_wave_active": False,
        "production_business_write_policy": PRODUCTION_WRITE_POLICY,
        "security_denial_audit_logging_required": True,
        "public_live_writes_enabled": True,
        "public_live_production_override_enabled": True,
        "expected_migration_contract": expected_migration_contract.lower(),
        "expected_migration_head": expected_migration_head,
        "expected_migration_profile": expected_migration_profile,
        "cors_allowed_origins": list(PRODUCTION_ALLOWED_ORIGINS),
        "cors_allowed_origin_regex": None,
        "feature_flags": expected_flags,
        "feature_flag_fingerprint": feature_flag_fingerprint(expected_flags),
        "controlled_write_flags": expected_controlled_write_flags,
        "controlled_write_flag_fingerprint": feature_flag_fingerprint(
            expected_controlled_write_flags
        ),
    }
    if allow_legacy_git_identity:
        legacy_git_sha = str(health.get("git_commit_sha") or "").strip().lower()
        legacy_image_sha = str(
            health.get("image_build_git_sha") or ""
        ).strip().lower()
        if legacy_git_sha not in {"", "unknown"} or legacy_image_sha not in {
            "",
            "unknown",
        }:
            errors.append(
                "Legacy rollback health must retain its reviewed missing Git identity."
            )
    else:
        expected_identity.update(
            {
                "git_commit_sha": candidate_sha.lower(),
                "image_build_git_sha": candidate_sha.lower(),
            }
        )
    for key, expected in expected_identity.items():
        actual = health.get(key)
        if key in {
            "git_commit_sha",
            "image_build_git_sha",
            "supabase_project_ref",
            "jwt_verification_project_ref",
        } and isinstance(actual, str):
            actual = actual.lower()
        if actual != expected:
            errors.append(
                f"Production health identity mismatch for {key}: "
                f"expected {expected!r}, got {actual!r}."
            )

    image_ref = health.get("fly_image_ref")
    if not isinstance(image_ref, str) or not _FLY_IMAGE_RE.fullmatch(image_ref.strip()):
        errors.append("Production health identity has no valid production Fly image ref.")
    else:
        identities, machine_errors = _nonterminal_machine_images(fly_machines)
        errors.extend(machine_errors)
        configured_refs = {item["configured_ref"] for item in identities}
        immutable_refs = {item["immutable_ref"] for item in identities}
        digests = {item["digest"] for item in identities}
        if (
            len(configured_refs) != 1
            or len(immutable_refs) != 1
            or len(digests) != 1
        ):
            errors.append(
                "All nonterminal Fly machines must use one exact image and immutable digest."
            )
        elif image_ref.strip() not in configured_refs | immutable_refs:
            errors.append(
                "Fly health image ref does not exactly match the nonterminal "
                "production-machine image."
            )
    if not str(health.get("fly_machine_version") or "").strip():
        errors.append("Production health identity is missing fly_machine_version.")

    prerequisites = health.get("write_prerequisites")
    if not isinstance(prerequisites, dict):
        errors.append("Production health identity is missing write_prerequisites.")
    else:
        for key in (
            "service_role_configured",
            "api_audit_required",
            "worker_run_log_required",
        ):
            if prerequisites.get(key) is not True:
                errors.append(f"Production write prerequisite {key} must be true.")
        if prerequisites.get("email_mode") != "dry_run":
            errors.append("Production email mode must remain dry_run during promotion.")
        if prerequisites.get("live_player_update_email_enabled") is not False:
            errors.append("Production live player-update email delivery must remain disabled.")

    errors.extend(secret_inventory_errors(fly_secrets))
    return errors


def final_runtime_errors(
    health: Any,
    *,
    candidate_sha: str,
    promotion_accepted: bool,
    expected_project_ref: str,
    expected_migration_head: str,
    expected_migration_contract: str,
    expected_migration_profile: str,
    rollback_snapshot: Any,
    fly_machines: Any,
    fly_secrets: Any,
) -> list[str]:
    """Attest either the accepted candidate or an exact immutable rollback."""

    if not isinstance(health, dict):
        return ["Final production health payload is not a JSON object."]
    if not isinstance(rollback_snapshot, dict):
        return ["Production rollback snapshot is not a JSON object."]

    candidate_sha = str(candidate_sha).strip().lower()
    errors: list[str] = []
    if not _SHA_RE.fullmatch(candidate_sha):
        errors.append("Expected production candidate SHA is invalid.")

    rollback_sha = str(
        rollback_snapshot.get("image_build_git_sha")
        or rollback_snapshot.get("git_commit_sha")
        or ""
    ).strip().lower()
    rollback_digest = str(
        rollback_snapshot.get("fly_image_digest") or ""
    ).strip().lower()
    rollback_immutable_ref = str(
        rollback_snapshot.get("fly_immutable_image_ref") or ""
    ).strip()
    rollback_configured_ref = str(
        rollback_snapshot.get("fly_image_ref") or ""
    ).strip()
    rollback_identity_mode = str(
        rollback_snapshot.get("identity_mode") or ""
    ).strip()
    rollback_config_sha256 = str(
        rollback_snapshot.get("fly_config_sha256") or ""
    ).strip().lower()
    rollback_feature_profile = str(
        rollback_snapshot.get("feature_profile") or ""
    ).strip()
    legacy_rollback = rollback_identity_mode == "legacy-immutable-bootstrap"
    if rollback_identity_mode not in {"exact-git", "legacy-immutable-bootstrap"}:
        errors.append("Rollback snapshot has no recognized identity mode.")
    if not legacy_rollback and not _SHA_RE.fullmatch(rollback_sha):
        errors.append("Rollback snapshot has no exact image-build Git SHA.")
    if legacy_rollback and rollback_sha:
        errors.append("Legacy rollback snapshot must not claim an exact Git SHA.")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", rollback_digest):
        errors.append("Rollback snapshot has no immutable Fly image digest.")
    if (
        not _FLY_IMAGE_RE.fullmatch(rollback_immutable_ref)
        or f"@{rollback_digest}" not in rollback_immutable_ref
    ):
        errors.append("Rollback snapshot has no matching immutable Fly image reference.")
    if not re.fullmatch(r"[0-9a-f]{64}", rollback_config_sha256):
        errors.append("Rollback snapshot has no exact Fly config fingerprint.")
    if rollback_feature_profile not in PRODUCTION_FEATURE_PROFILES:
        errors.append("Rollback snapshot has no reviewed production feature profile.")
    if legacy_rollback:
        if (
            str(
                rollback_snapshot.get("reviewed_legacy_image_digest") or ""
            ).strip().lower()
            != rollback_digest
        ):
            errors.append(
                "Legacy rollback snapshot image digest is not the reviewed baseline."
            )
        if (
            str(
                rollback_snapshot.get("reviewed_legacy_config_sha256") or ""
            ).strip().lower()
            != rollback_config_sha256
        ):
            errors.append(
                "Legacy rollback snapshot config is not the reviewed baseline."
            )

    expected_sha = (
        candidate_sha if promotion_accepted or legacy_rollback else rollback_sha
    )
    final_feature_profile = "release" if promotion_accepted else (
        rollback_feature_profile
        if rollback_feature_profile in PRODUCTION_FEATURE_PROFILES
        else "baseline"
    )
    errors.extend(
        runtime_identity_errors(
            health,
            candidate_sha=expected_sha,
            expected_project_ref=expected_project_ref,
            expected_migration_head=expected_migration_head,
            expected_migration_contract=expected_migration_contract,
            expected_migration_profile=expected_migration_profile,
            fly_machines=fly_machines,
            fly_secrets=fly_secrets,
            allow_legacy_git_identity=(not promotion_accepted and legacy_rollback),
            feature_profile=final_feature_profile,
        )
    )

    if not promotion_accepted:
        health_image_ref = str(health.get("fly_image_ref") or "").strip()
        identities, machine_errors = _nonterminal_machine_images(fly_machines)
        errors.extend(machine_errors)
        configured_refs = {item["configured_ref"] for item in identities}
        immutable_refs = {item["immutable_ref"] for item in identities}
        digests = {item["digest"] for item in identities}
        if digests != {rollback_digest}:
            errors.append(
                "Failed promotion did not restore the snapshot's immutable image digest."
            )
        if immutable_refs != {rollback_immutable_ref}:
            errors.append(
                "Failed promotion did not restore the snapshot's immutable image reference."
            )
        allowed_rollback_health_refs = {
            rollback_configured_ref,
            rollback_immutable_ref,
        } | configured_refs
        if health_image_ref not in allowed_rollback_health_refs:
            errors.append(
                "Failed promotion health image does not resolve to the restored "
                "rollback digest."
            )
    return errors


def openapi_errors(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return ["OpenAPI payload is not a JSON object."]
    errors: list[str] = []
    if not str(payload.get("openapi") or "").startswith("3."):
        errors.append("OpenAPI document does not declare an OpenAPI 3.x version.")
    info = payload.get("info") or {}
    if not isinstance(info, dict) or info.get("title") != "JUPR API":
        errors.append("OpenAPI document does not identify the JUPR API.")
    paths = payload.get("paths") or {}
    if not isinstance(paths, dict) or "/health" not in paths:
        errors.append("OpenAPI document does not expose /health.")
    return errors


def public_database_read_errors(payload: Any) -> list[str]:
    """Validate the public response produced by required Supabase table reads."""

    if not isinstance(payload, dict):
        return ["Public database-backed API payload is not a JSON object."]
    errors: list[str] = []
    club = payload.get("club")
    if not isinstance(club, dict) or club.get("slug") != PRODUCTION_PUBLIC_CLUB_SLUG:
        errors.append("Public database-backed API payload has the wrong club identity.")
    if not isinstance(payload.get("leaderboard"), list):
        errors.append("Public database-backed API payload has no leaderboard list.")
    scopes = payload.get("scopes")
    if not isinstance(scopes, list) or not scopes:
        errors.append("Public database-backed API payload has no leaderboard scopes.")
    if not isinstance(payload.get("summary"), dict):
        errors.append("Public database-backed API payload has no summary object.")
    pagination = payload.get("pagination")
    if not isinstance(pagination, dict) or not isinstance(
        pagination.get("total"), int
    ):
        errors.append("Public database-backed API payload has invalid pagination.")
    return errors


def cors_header_errors(headers_by_origin: Mapping[str, str]) -> list[str]:
    errors: list[str] = []
    if set(headers_by_origin) != set(PRODUCTION_ALLOWED_ORIGINS):
        errors.append("CORS evidence does not cover the exact production origin allowlist.")
        return errors
    for origin, raw_headers in headers_by_origin.items():
        headers: dict[str, list[str]] = {}
        for line in raw_headers.splitlines():
            if ":" not in line:
                continue
            name, value = line.split(":", 1)
            headers.setdefault(name.strip().lower(), []).append(value.strip())
        if headers.get("access-control-allow-origin") != [origin]:
            errors.append(f"CORS preflight did not echo the exact allowed origin {origin}.")
        credentials = ",".join(headers.get("access-control-allow-credentials", [])).lower()
        if credentials != "true":
            errors.append(f"CORS preflight did not allow credentials for {origin}.")
        methods = ",".join(headers.get("access-control-allow-methods", [])).upper()
        if "GET" not in {value.strip() for value in methods.split(",")}:
            errors.append(f"CORS preflight did not allow GET for {origin}.")
    return errors


def disallowed_cors_header_errors(
    headers_by_api_origin: Mapping[str, str],
) -> list[str]:
    expected_api_origins = {PRODUCTION_FLY_ORIGIN, PRODUCTION_API_ORIGIN}
    if set(headers_by_api_origin) != expected_api_origins:
        return [
            "Disallowed-origin CORS evidence does not cover both production API origins."
        ]
    errors: list[str] = []
    for api_origin, raw_headers in headers_by_api_origin.items():
        allow_origin_values = [
            value.strip()
            for line in raw_headers.splitlines()
            if ":" in line
            for name, value in [line.split(":", 1)]
            if name.strip().lower() == "access-control-allow-origin"
            and value.strip()
        ]
        if allow_origin_values:
            errors.append(
                f"Production API origin {api_origin} allowed a disallowed CORS origin."
            )
    return errors


def _write_github_env(
    path: str | None,
    migrations: tuple[str, ...],
    migration_contract: Mapping[str, Any],
) -> None:
    if not path or not migrations or not migration_contract:
        return
    contract_fingerprint = migration_contract_fingerprint(
        profile=str(migration_contract["profile"]),
        required_ledger_names=migration_contract["required_ledger_names"],
        deployment_order=migration_contract["deployment_order"],
        repository_migration_content_sha256=str(
            migration_contract["repository_migration_content_sha256"]
        ),
        allowed_duplicate_ledger_names=migration_contract[
            "allowed_duplicate_ledger_names"
        ],
    )
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(f"EXPECTED_MIGRATION_COUNT={len(migrations)}\n")
        handle.write(f"EXPECTED_MIGRATION_CONTRACT={contract_fingerprint}\n")
        handle.write(
            f"EXPECTED_MIGRATION_PROFILE={migration_contract['profile']}\n"
        )


def _emit(errors: list[str], summary: Mapping[str, Any]) -> int:
    payload = {**summary, "ok": not errors, "errors": errors}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not errors else 1


def _preflight_command(args: argparse.Namespace) -> int:
    errors, migrations = preflight_errors(
        os.environ,
        config_path=args.config,
        migrations_dir=args.migrations_dir,
        migration_contract_path=args.migration_contract,
    )
    contract: dict[str, Any] = {}
    if not errors:
        contract = load_migration_contract(
            args.migration_contract,
            args.migrations_dir,
        )
    if not errors:
        _write_github_env(os.getenv("GITHUB_ENV"), migrations, contract)
    return _emit(
        errors,
        {
            "candidate_sha": str(os.getenv("GITHUB_SHA") or "").strip().lower() or None,
            "fly_app": PRODUCTION_FLY_APP,
            "migration_count": len(migrations),
            "migration_profile": contract.get("profile"),
            "production_write_policy": PRODUCTION_WRITE_POLICY,
            "required_configuration_present": not any(
                name
                for name in REQUIRED_GITHUB_ENV_NAMES
                if not str(os.getenv(name) or "").strip()
            ),
            "supabase_project_ref": str(
                os.getenv("EXPECTED_SUPABASE_PROJECT_REF") or ""
            ).strip()
            or None,
        },
    )


def _migrations_command(args: argparse.Namespace) -> int:
    contract = load_migration_contract(
        args.migration_contract,
        args.migrations_dir,
    )
    remote_lines = args.remote_ledger.read_text(encoding="utf-8").splitlines()
    remote_entries, invalid_rows = parse_remote_migration_ledger(remote_lines)
    errors = migration_ledger_errors(
        contract["required_ledger_names"],
        remote_entries,
        invalid_remote_rows=invalid_rows,
        allow_additional_names=bool(contract["allow_additional_ledger_names"]),
        allowed_duplicate_names=contract["allowed_duplicate_ledger_names"],
    )
    errors.extend(
        migration_schema_contract_errors(_read_json(args.schema_contract_json))
    )
    remote_versions = [version for version, _ in remote_entries]
    remote_head = (
        max(set(remote_versions), key=_migration_sort_key)
        if remote_versions
        else None
    )
    expected_remote_head = str(
        os.getenv("EXPECTED_MIGRATION_HEAD") or ""
    ).strip()
    if not _MIGRATION_VERSION_RE.fullmatch(expected_remote_head):
        errors.append(
            "Reviewed EXPECTED_MIGRATION_HEAD is missing or invalid."
        )
    elif remote_head != expected_remote_head:
        errors.append(
            "Remote Supabase migration head does not match the reviewed head: "
            f"expected {expected_remote_head}, got {remote_head}."
        )
    contract_fingerprint = migration_contract_fingerprint(
        profile=str(contract["profile"]),
        required_ledger_names=contract["required_ledger_names"],
        deployment_order=contract["deployment_order"],
        repository_migration_content_sha256=str(
            contract["repository_migration_content_sha256"]
        ),
        allowed_duplicate_ledger_names=contract[
            "allowed_duplicate_ledger_names"
        ],
    )
    expected_contract = str(
        os.getenv("EXPECTED_MIGRATION_CONTRACT") or ""
    ).strip()
    if expected_contract and expected_contract != contract_fingerprint:
        errors.append("Production migration contract fingerprint is inconsistent.")
    return _emit(
        errors,
        {
            "migration_contract": contract_fingerprint,
            "migration_profile": contract["profile"],
            "pending_required_ledger_names": list(
                pending_required_migration_names(
                    contract["deployment_order"], remote_entries
                )
            ),
            "required_ledger_name_count": len(
                contract["required_ledger_names"]
            ),
            "remote_valid_count": len(remote_entries),
            "remote_head": remote_head,
        },
    )


def _runtime_command(args: argparse.Namespace) -> int:
    health = _read_json(args.health_json)
    errors = runtime_identity_errors(
        health,
        candidate_sha=str(os.getenv("GITHUB_SHA") or "").strip().lower(),
        expected_project_ref=str(
            os.getenv("EXPECTED_SUPABASE_PROJECT_REF") or ""
        ).strip().lower(),
        expected_migration_head=str(
            os.getenv("EXPECTED_MIGRATION_HEAD") or ""
        ).strip(),
        expected_migration_contract=str(
            os.getenv("EXPECTED_MIGRATION_CONTRACT") or ""
        ).strip().lower(),
        expected_migration_profile=str(
            os.getenv("EXPECTED_MIGRATION_PROFILE") or ""
        ).strip(),
        fly_machines=_read_json(args.fly_machines_json),
        fly_secrets=_read_json(args.fly_secrets_json),
    )
    errors.extend(openapi_errors(_read_json(args.openapi_json)))
    if args.secondary_health_json is not None:
        secondary = _read_json(args.secondary_health_json)
        if not isinstance(secondary, dict):
            errors.append("Custom production API health payload is not a JSON object.")
        else:
            mismatched = [
                key
                for key in SECONDARY_HEALTH_IDENTITY_KEYS
                if secondary.get(key) != health.get(key)
            ]
            if mismatched:
                errors.append(
                    "Custom production API origin identity differs from Fly for: "
                    + ", ".join(mismatched)
                )
    return _emit(
        errors,
        {
            "candidate_sha": str(os.getenv("GITHUB_SHA") or "").strip().lower() or None,
            "fly_app": PRODUCTION_FLY_APP,
            "fly_image_ref": health.get("fly_image_ref") if isinstance(health, dict) else None,
            "migration_head": str(os.getenv("EXPECTED_MIGRATION_HEAD") or "").strip()
            or None,
            "migration_profile": str(
                os.getenv("EXPECTED_MIGRATION_PROFILE") or ""
            ).strip()
            or None,
            "supabase_project_ref": str(
                os.getenv("EXPECTED_SUPABASE_PROJECT_REF") or ""
            ).strip()
            or None,
        },
    )


def _secrets_command(args: argparse.Namespace) -> int:
    payload = _read_json(args.fly_secrets_json)
    if args.safe_convergence_only:
        errors = safe_secret_convergence_errors(payload)
        mode = "safe-convergence"
    elif args.no_pending_only:
        errors = pending_secret_errors(payload)
        mode = "no-pending"
    else:
        errors = secret_inventory_errors(payload)
        mode = "runtime"
    return _emit(
        errors,
        {
            "mode": mode,
            "required_secret_name_count": len(PRODUCTION_RUNTIME_SECRET_NAMES),
            "required_secret_names": list(PRODUCTION_RUNTIME_SECRET_NAMES),
        },
    )


def _snapshot_command(args: argparse.Namespace) -> int:
    fly_config_sha256 = predeploy_fly_config_fingerprint(args.fly_config)
    errors, snapshot = predeploy_rollback_snapshot(
        _read_json(args.health_json),
        _read_json(args.fly_status_json),
        fly_config_sha256=fly_config_sha256,
        reviewed_legacy_image_digest=args.reviewed_legacy_image_digest,
        reviewed_legacy_config_sha256=args.reviewed_legacy_config_sha256,
        capture_unreviewed_legacy_evidence=(
            args.capture_unreviewed_legacy_evidence
        ),
    )
    if not errors:
        args.output_json.write_text(
            json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return _emit(errors, {"rollback_snapshot": snapshot})


def _release_trigger_command(args: argparse.Namespace) -> int:
    errors, resolved = production_release_trigger_errors(
        _read_json(args.trigger_json),
        head_sha=args.head_sha,
        parent_shas=args.parent_sha,
        changed_status_lines=args.changed_status.read_text(
            encoding="utf-8"
        ).splitlines(),
    )
    if not errors:
        args.output_json.write_text(
            json.dumps(resolved, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return _emit(errors, {"release_trigger": resolved})


def _final_command(args: argparse.Namespace) -> int:
    promotion_accepted = args.promotion_accepted == "true"
    health = _read_json(args.health_json)
    errors = final_runtime_errors(
        health,
        candidate_sha=str(os.getenv("GITHUB_SHA") or "").strip().lower(),
        promotion_accepted=promotion_accepted,
        expected_project_ref=str(
            os.getenv("EXPECTED_SUPABASE_PROJECT_REF") or ""
        ).strip().lower(),
        expected_migration_head=str(
            os.getenv("EXPECTED_MIGRATION_HEAD") or ""
        ).strip(),
        expected_migration_contract=str(
            os.getenv("EXPECTED_MIGRATION_CONTRACT") or ""
        ).strip().lower(),
        expected_migration_profile=str(
            os.getenv("EXPECTED_MIGRATION_PROFILE") or ""
        ).strip(),
        rollback_snapshot=_read_json(args.rollback_snapshot_json),
        fly_machines=_read_json(args.fly_machines_json),
        fly_secrets=_read_json(args.fly_secrets_json),
    )
    return _emit(
        errors,
        {
            "candidate_sha": str(os.getenv("GITHUB_SHA") or "").strip().lower()
            or None,
            "final_git_sha": (
                health.get("git_commit_sha") if isinstance(health, dict) else None
            ),
            "fly_app": PRODUCTION_FLY_APP,
            "promotion_accepted": promotion_accepted,
        },
    )


def _cors_command(args: argparse.Namespace) -> int:
    api_origins = (PRODUCTION_FLY_ORIGIN, PRODUCTION_API_ORIGIN)
    errors: list[str] = []
    blocked_evidence: dict[str, str] = {}
    for api_index, api_origin in enumerate(api_origins):
        evidence = {
            origin: (
                args.headers_dir / f"{api_index}-{origin_index}.headers"
            ).read_text(encoding="utf-8")
            for origin_index, origin in enumerate(PRODUCTION_ALLOWED_ORIGINS)
        }
        errors.extend(
            f"{api_origin}: {error}" for error in cors_header_errors(evidence)
        )
        blocked_evidence[api_origin] = (
            args.headers_dir / f"{api_index}-blocked.headers"
        ).read_text(encoding="utf-8")
    errors.extend(disallowed_cors_header_errors(blocked_evidence))
    return _emit(
        errors,
        {
            "checked_api_origins": list(api_origins),
            "checked_allowed_origins": list(PRODUCTION_ALLOWED_ORIGINS),
            "checked_disallowed_origin": "https://not-allowed.invalid",
        },
    )


def _public_read_command(args: argparse.Namespace) -> int:
    payload = _read_json(args.payload_json)
    return _emit(
        public_database_read_errors(payload),
        {
            "club_slug": PRODUCTION_PUBLIC_CLUB_SLUG,
            "database_backed_endpoint": (
                f"/clubs/{PRODUCTION_PUBLIC_CLUB_SLUG}/leaderboards"
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed production Fly deployment policy verifier."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", type=Path, default=Path("fly.toml"))
    preflight.add_argument(
        "--migrations-dir", type=Path, default=Path("supabase/migrations")
    )
    preflight.add_argument(
        "--migration-contract",
        type=Path,
        default=DEFAULT_MIGRATION_CONTRACT_PATH,
    )
    preflight.set_defaults(handler=_preflight_command)

    migrations = subparsers.add_parser("migrations")
    migrations.add_argument("--remote-ledger", type=Path, required=True)
    migrations.add_argument("--schema-contract-json", type=Path, required=True)
    migrations.add_argument(
        "--migrations-dir", type=Path, default=Path("supabase/migrations")
    )
    migrations.add_argument(
        "--migration-contract",
        type=Path,
        default=DEFAULT_MIGRATION_CONTRACT_PATH,
    )
    migrations.set_defaults(handler=_migrations_command)

    runtime = subparsers.add_parser("runtime")
    runtime.add_argument("--health-json", type=Path, required=True)
    runtime.add_argument("--secondary-health-json", type=Path)
    runtime.add_argument("--openapi-json", type=Path, required=True)
    runtime.add_argument("--fly-machines-json", type=Path, required=True)
    runtime.add_argument("--fly-secrets-json", type=Path, required=True)
    runtime.set_defaults(handler=_runtime_command)

    secrets = subparsers.add_parser("secrets")
    secrets.add_argument("--fly-secrets-json", type=Path, required=True)
    secrets_modes = secrets.add_mutually_exclusive_group()
    secrets_modes.add_argument("--no-pending-only", action="store_true")
    secrets_modes.add_argument("--safe-convergence-only", action="store_true")
    secrets.set_defaults(handler=_secrets_command)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--health-json", type=Path, required=True)
    snapshot.add_argument("--fly-status-json", type=Path, required=True)
    snapshot.add_argument("--fly-config", type=Path, required=True)
    snapshot.add_argument("--output-json", type=Path, required=True)
    snapshot.add_argument("--reviewed-legacy-image-digest")
    snapshot.add_argument("--reviewed-legacy-config-sha256")
    snapshot.add_argument(
        "--capture-unreviewed-legacy-evidence",
        action="store_true",
    )
    snapshot.set_defaults(handler=_snapshot_command)

    release_trigger = subparsers.add_parser("release-trigger")
    release_trigger.add_argument("--trigger-json", type=Path, required=True)
    release_trigger.add_argument("--head-sha", required=True)
    release_trigger.add_argument(
        "--parent-sha",
        action="append",
        default=[],
    )
    release_trigger.add_argument("--changed-status", type=Path, required=True)
    release_trigger.add_argument("--output-json", type=Path, required=True)
    release_trigger.set_defaults(handler=_release_trigger_command)

    final = subparsers.add_parser("final")
    final.add_argument("--health-json", type=Path, required=True)
    final.add_argument("--fly-machines-json", type=Path, required=True)
    final.add_argument("--fly-secrets-json", type=Path, required=True)
    final.add_argument("--rollback-snapshot-json", type=Path, required=True)
    final.add_argument(
        "--promotion-accepted",
        choices=("true", "false"),
        required=True,
    )
    final.set_defaults(handler=_final_command)

    public_read = subparsers.add_parser("public-read")
    public_read.add_argument("--payload-json", type=Path, required=True)
    public_read.set_defaults(handler=_public_read_command)

    cors = subparsers.add_parser("cors")
    cors.add_argument("--headers-dir", type=Path, required=True)
    cors.set_defaults(handler=_cors_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return _emit([str(exc)], {"command": args.command})


if __name__ == "__main__":
    raise SystemExit(main())
