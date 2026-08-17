from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from scripts import deployment_verifier as verifier
from scripts.staging_write_waves import ALL_STAGING_WRITE_FLAGS


ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_REF = "abcdefghijklmnopqrst"
CANDIDATE_SHA = "a" * 40
IMAGE_REF = "registry.fly.io/juprleagues-api:deployment-01ABCDEF"
IMAGE_DIGEST = "sha256:" + ("1" * 64)
IMMUTABLE_IMAGE_REF = (
    f"registry.fly.io/{verifier.PRODUCTION_FLY_APP}@{IMAGE_DIGEST}"
)
FLY_CONFIG_SHA = "4" * 64
MIGRATION_PROFILE = "next-fastapi-tournament-acceptance-2026-08-03"
MIGRATION_CONTRACT = verifier.load_migration_contract(
    ROOT / "config/production_migration_contract.json",
    ROOT / "supabase/migrations",
)
MIGRATION_CONTRACT_FINGERPRINT = verifier.migration_contract_fingerprint(
    profile=MIGRATION_CONTRACT["profile"],
    required_ledger_names=MIGRATION_CONTRACT["required_ledger_names"],
    repository_migration_content_sha256=MIGRATION_CONTRACT[
        "repository_migration_content_sha256"
    ],
)
REMOTE_MIGRATION_HEAD = "20260720123402"


def _production_env(**overrides: str) -> dict[str, str]:
    env = {
        "EXPECTED_SUPABASE_PROJECT_REF": PRODUCTION_REF,
        "EXPECTED_MIGRATION_HEAD": REMOTE_MIGRATION_HEAD,
        "FLY_API_TOKEN": "fly-token-present",
        "FLY_SSH_TOKEN": "fly-ssh-token-present",
        "FLY_APP_NAME": verifier.PRODUCTION_FLY_APP,
        "GITHUB_SHA": CANDIDATE_SHA,
        "JUPR_ENV": "production",
        "SUPABASE_ANON_KEY": "anon-present",
        "SUPABASE_DATABASE_URL": (
            f"postgresql://postgres:secret@db.{PRODUCTION_REF}.supabase.co:5432/postgres"
        ),
        "SUPABASE_SERVICE_ROLE_KEY": "service-role-present",
        "SUPABASE_URL": f"https://{PRODUCTION_REF}.supabase.co",
    }
    env.update(overrides)
    return env


def _health_payload() -> dict:
    controlled = {name: False for name in ALL_STAGING_WRITE_FLAGS}
    features = verifier.expected_production_feature_flags()
    return {
        "ok": True,
        "service": "jupr-api",
        "environment": "production",
        "git_commit_sha": CANDIDATE_SHA,
        "image_build_git_sha": CANDIDATE_SHA,
        "fly_app_name": verifier.PRODUCTION_FLY_APP,
        "fly_image_ref": IMAGE_REF,
        "fly_machine_version": "42",
        "web_origin": verifier.PRODUCTION_WEB_ORIGIN,
        "supabase_project_ref": PRODUCTION_REF,
        "jwt_verification_configured": True,
        "jwt_verification_mode": "jwks",
        "jwt_verification_project_ref": PRODUCTION_REF,
        "write_wave": "none",
        "staging_write_wave": "none",
        "business_data_write_wave_active": False,
        "production_business_write_policy": "read_only",
        "security_denial_audit_logging_required": True,
        "public_live_writes_enabled": False,
        "public_live_production_override_enabled": False,
        "expected_migration_contract": MIGRATION_CONTRACT_FINGERPRINT,
        "expected_migration_head": REMOTE_MIGRATION_HEAD,
        "expected_migration_profile": MIGRATION_PROFILE,
        "cors_allowed_origins": list(verifier.PRODUCTION_ALLOWED_ORIGINS),
        "cors_allowed_origin_regex": None,
        "feature_flags": features,
        "feature_flag_fingerprint": verifier.feature_flag_fingerprint(features),
        "controlled_write_flags": controlled,
        "controlled_write_flag_fingerprint": verifier.feature_flag_fingerprint(
            controlled
        ),
        "write_prerequisites": {
            "service_role_configured": True,
            "api_audit_required": True,
            "worker_run_log_required": True,
            "email_mode": "dry_run",
            "live_player_update_email_enabled": False,
        },
    }


def _fly_secrets() -> list[dict[str, str]]:
    return [
        {
            "Name": name,
            "Digest": f"digest-{index}",
            "DeploymentStatus": "Deployed",
        }
        for index, name in enumerate(verifier.PRODUCTION_RUNTIME_SECRET_NAMES)
    ]


def _machine(
    *,
    machine_id: str = "one",
    state: str = "started",
    image: str = IMAGE_REF,
    digest: str = IMAGE_DIGEST,
) -> dict:
    return {
        "id": machine_id,
        "state": state,
        "config": {"image": image},
        "image_ref": {
            "registry": "registry.fly.io",
            "repository": verifier.PRODUCTION_FLY_APP,
            "digest": digest,
        },
    }


def test_production_fly_config_is_exact_read_only_policy() -> None:
    assert verifier.production_fly_config_errors(ROOT / "fly.toml") == []

    config = tomllib.loads((ROOT / "fly.toml").read_text(encoding="utf-8"))
    assert config["app"] == verifier.PRODUCTION_FLY_APP
    assert config["primary_region"] == verifier.PRODUCTION_FLY_REGION
    assert config["env"]["JUPR_PRODUCTION_WRITE_POLICY"] == "read_only"
    assert config["env"]["JUPR_STAGING_WRITE_WAVE"] == "none"
    assert all(
        config["env"][name] == "0" for name in verifier.PRODUCTION_FEATURE_FLAGS
    )


def test_predeploy_fly_config_fingerprint_binds_exact_production_config(
    tmp_path: Path,
) -> None:
    fingerprint = verifier.predeploy_fly_config_fingerprint(ROOT / "fly.toml")
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint)

    foreign = tmp_path / "fly.toml"
    foreign.write_text('app = "another-app"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="exact production app"):
        verifier.predeploy_fly_config_fingerprint(foreign)


def test_production_feature_projection_covers_every_runtime_flag() -> None:
    discovered: set[str] = set()
    for source_root in ("services", "jupr_app"):
        for path in (ROOT / source_root).rglob("*.py"):
            discovered.update(
                re.findall(
                    r"JUPR_ENABLE_[A-Z0-9_]+",
                    path.read_text(encoding="utf-8"),
                )
            )

    assert discovered == set(verifier.PRODUCTION_FEATURE_FLAGS)


def test_direct_singles_uploader_gate_is_projected_off_in_production() -> None:
    flag = "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES"

    assert flag in verifier.PRODUCTION_FEATURE_FLAGS
    assert verifier.expected_production_feature_flags()[flag] is False
    config = tomllib.loads((ROOT / "fly.toml").read_text(encoding="utf-8"))
    assert config["env"][flag] == "0"


def test_match_log_destructive_gate_is_projected_off_in_production() -> None:
    flag = "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE"

    assert flag in verifier.PRODUCTION_FEATURE_FLAGS
    assert verifier.expected_production_feature_flags()[flag] is False
    config = tomllib.loads((ROOT / "fly.toml").read_text(encoding="utf-8"))
    assert config["env"][flag] == "0"


def test_production_fly_config_rejects_enabled_or_missing_feature_flag(
    tmp_path: Path,
) -> None:
    text = (ROOT / "fly.toml").read_text(encoding="utf-8")
    changed = text.replace(
        'JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY = "0"',
        'JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY = "1"',
    )
    path = tmp_path / "fly.toml"
    path.write_text(changed, encoding="utf-8")

    assert any(
        "JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY" in error
        for error in verifier.production_fly_config_errors(path)
    )


def test_repository_migration_inventory_and_reviewed_profile_are_deterministic() -> None:
    versions = verifier.expected_migration_versions(ROOT / "supabase/migrations")
    names = verifier.expected_migration_names(ROOT / "supabase/migrations")
    contract = verifier.load_migration_contract(
        ROOT / "config/production_migration_contract.json",
        ROOT / "supabase/migrations",
    )

    assert len(versions) == 64
    assert versions[-10:] == (
        "20260807150000",
        "20261020000000",
        "20261021000000",
        "20261022000000",
        "20261023000000",
        "20261024000000",
        "20261025000000",
        "20261026000000",
        "20261027000000",
        "20261028000000",
    )
    assert len(names) == 64
    assert all("XX" not in version for version in versions)
    assert len(contract["required_ledger_names"]) == 64
    assert "tournament_complete_registration_editor" in contract[
        "required_ledger_names"
    ]
    assert contract["allow_additional_ledger_names"] is False
    assert contract["schema_contract_only_repository_migrations"] == (
        "tournament_registrations_player_id_postgrest_reload",
    )


def test_repository_migration_inventory_rejects_new_non_numeric_placeholders(
    tmp_path: Path,
) -> None:
    (tmp_path / "20260101000000_valid.sql").write_text("select 1;", encoding="utf-8")
    (tmp_path / "future_placeholder.sql").write_text("select 1;", encoding="utf-8")

    with pytest.raises(ValueError, match="future_placeholder.sql"):
        verifier.expected_migration_versions(tmp_path)


def test_reviewed_migration_contract_is_bound_to_sql_bytes(tmp_path: Path) -> None:
    migrations = tmp_path / "migrations"
    shutil.copytree(ROOT / "supabase/migrations", migrations)
    target = migrations / "20260719155515_server_only_data_api_lockdown.sql"
    target.write_text(
        target.read_text(encoding="utf-8") + "\n-- changed after review\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="stale for repository migration SQL"):
        verifier.load_migration_contract(
            ROOT / "config/production_migration_contract.json",
            migrations,
        )


def test_migration_ledger_uses_reviewed_logical_names_not_filename_versions() -> None:
    expected = ("first_contract", "second_contract")
    remote = [
        ("20260720033650", "first_contract"),
        ("20260720123402", "second_contract"),
    ]

    assert verifier.migration_ledger_errors(expected, remote) == []
    missing = verifier.migration_ledger_errors(expected, remote[:1])
    assert any("missing" in error.lower() for error in missing)

    additional = verifier.migration_ledger_errors(
        expected,
        [*remote, ("20250101000000", "unreviewed_contract")],
    )
    assert any("outside the reviewed" in error.lower() for error in additional)
    assert (
        verifier.migration_ledger_errors(
            expected,
            [*remote, ("20250101000000", "unreviewed_contract")],
            allow_additional_names=True,
        )
        == []
    )

    duplicate = verifier.migration_ledger_errors(
        expected,
        [*remote, ("20260720130000", "first_contract")],
    )
    assert any("repeats logical names" in error.lower() for error in duplicate)

    parsed, invalid_rows = verifier.parse_remote_migration_ledger(
        [
            "20260720033650\tfirst_contract",
            "2026XX\tbad",
        ]
    )
    assert parsed == [("20260720033650", "first_contract")]
    assert invalid_rows == ["2026XX\tbad"]


def test_migration_schema_contract_requires_hotfix_shape_and_no_duplicates() -> None:
    valid = {
        "tournament_registrations_player_id_column": True,
        "idx_tournament_registrations_player_id": True,
        "uq_tournament_registrations_tournament_player": True,
        "tournament_player_duplicate_groups": 0,
    }
    assert verifier.migration_schema_contract_errors(valid) == []
    assert verifier.migration_schema_contract_errors(
        {
            **valid,
            "uq_tournament_registrations_tournament_player": False,
            "tournament_player_duplicate_groups": 1,
        }
    )


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            f"postgresql://postgres:secret@db.{PRODUCTION_REF}.supabase.co:5432/postgres",
            PRODUCTION_REF,
        ),
        (
            "postgres://"
            f"postgres.{PRODUCTION_REF}:secret@aws-0-us-east-1.pooler.supabase.com:6543/postgres",
            PRODUCTION_REF,
        ),
        (f"postgresql://postgres:secret@db.{PRODUCTION_REF}.supabase.co/postgres", None),
        (
            f"postgresql://postgres:secret@db.{PRODUCTION_REF}.supabase.co:5432/other",
            None,
        ),
    ],
)
def test_database_url_project_ref_is_fail_closed(
    url: str, expected: str | None
) -> None:
    assert verifier.database_url_project_ref(url) == expected


def test_preflight_accepts_only_matching_protected_project_and_config() -> None:
    errors, migrations = verifier.preflight_errors(
        _production_env(),
        config_path=ROOT / "fly.toml",
        migrations_dir=ROOT / "supabase/migrations",
    )

    assert errors == []
    assert migrations[-1] == "20261028000000"

    wrong_project_errors, _ = verifier.preflight_errors(
        _production_env(
            SUPABASE_URL="https://zzzzzzzzzzzzzzzzzzzz.supabase.co"
        ),
        config_path=ROOT / "fly.toml",
        migrations_dir=ROOT / "supabase/migrations",
    )
    assert any("protected production project" in error for error in wrong_project_errors)

    staging_errors, _ = verifier.preflight_errors(
        _production_env(
            EXPECTED_SUPABASE_PROJECT_REF="sijpxjxvdtrehmqvirfi",
            SUPABASE_URL="https://sijpxjxvdtrehmqvirfi.supabase.co",
            SUPABASE_DATABASE_URL=(
                "postgresql://postgres:secret@"
                "db.sijpxjxvdtrehmqvirfi.supabase.co:5432/postgres"
            ),
        ),
        config_path=ROOT / "fly.toml",
        migrations_dir=ROOT / "supabase/migrations",
    )
    assert any("staging Supabase project" in error for error in staging_errors)


def test_runtime_identity_requires_exact_sha_image_project_and_read_only_policy() -> None:
    errors = verifier.runtime_identity_errors(
        _health_payload(),
        candidate_sha=CANDIDATE_SHA,
        expected_project_ref=PRODUCTION_REF,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_migration_contract=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_profile=MIGRATION_PROFILE,
        fly_machines=[
            _machine(),
            _machine(machine_id="stopped", state="stopped"),
            {
                "id": "destroyed",
                "state": "destroyed",
                "config": {"image": "registry.fly.io/juprleagues-api:old"},
            },
        ],
        fly_secrets=_fly_secrets(),
    )
    assert errors == []

    unsafe = _health_payload()
    unsafe["production_business_write_policy"] = "enabled"
    unsafe["business_data_write_wave_active"] = True
    unsafe["feature_flags"] = {
        **unsafe["feature_flags"],
        "JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY": True,
    }
    unsafe_errors = verifier.runtime_identity_errors(
        unsafe,
        candidate_sha=CANDIDATE_SHA,
        expected_project_ref=PRODUCTION_REF,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_migration_contract=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_profile=MIGRATION_PROFILE,
        fly_machines=[_machine()],
        fly_secrets=_fly_secrets(),
    )
    assert any("production_business_write_policy" in error for error in unsafe_errors)
    assert any("feature_flags" in error for error in unsafe_errors)


def test_predeploy_snapshot_preserves_exact_image_and_sha_pair() -> None:
    health = _health_payload()
    errors, snapshot = verifier.predeploy_rollback_snapshot(
        health,
        {"Machines": [_machine()]},
        fly_config_sha256=FLY_CONFIG_SHA,
    )

    assert errors == []
    assert snapshot == {
        "fly_app": verifier.PRODUCTION_FLY_APP,
        "git_commit_sha": CANDIDATE_SHA,
        "image_build_git_sha": CANDIDATE_SHA,
        "fly_image_ref": IMAGE_REF,
        "fly_image_digest": IMAGE_DIGEST,
        "fly_immutable_image_ref": IMMUTABLE_IMAGE_REF,
        "fly_config_sha256": FLY_CONFIG_SHA,
    }

    legacy = {**health, "git_commit_sha": None, "image_build_git_sha": None}
    legacy_errors, _ = verifier.predeploy_rollback_snapshot(
        legacy,
        {"Machines": [_machine()]},
        fly_config_sha256=FLY_CONFIG_SHA,
    )
    assert any("no exact git sha" in error.lower() for error in legacy_errors)
    assert any("image-build git sha" in error.lower() for error in legacy_errors)


def test_runtime_identity_rejects_image_or_secret_inventory_mismatch() -> None:
    errors = verifier.runtime_identity_errors(
        _health_payload(),
        candidate_sha=CANDIDATE_SHA,
        expected_project_ref=PRODUCTION_REF,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_migration_contract=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_profile=MIGRATION_PROFILE,
        fly_machines=[
            _machine(image="registry.fly.io/juprleagues-api:different")
        ],
        fly_secrets=[],
    )

    assert any("image ref" in error.lower() for error in errors)
    assert any("secret inventory" in error.lower() for error in errors)

    mixed_machine_errors = verifier.runtime_identity_errors(
        _health_payload(),
        candidate_sha=CANDIDATE_SHA,
        expected_project_ref=PRODUCTION_REF,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_migration_contract=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_profile=MIGRATION_PROFILE,
        fly_machines=[
            _machine(machine_id="valid"),
            _machine(
                machine_id="foreign",
                image="registry.fly.io/another-app:deployment-02",
            ),
        ],
        fly_secrets=_fly_secrets(),
    )
    assert any(
        "foreign image reference" in error.lower()
        for error in mixed_machine_errors
    )

    forbidden_secrets = [
        *_fly_secrets(),
        {"Name": "JUPR_ALLOWED_ORIGIN_REGEX", "Digest": "forbidden"},
    ]
    assert any(
        "forbidden names" in error.lower()
        for error in verifier.secret_inventory_errors(forbidden_secrets)
    )

    staged = _fly_secrets()
    staged[0] = {**staged[0], "DeploymentStatus": "Staged"}
    assert any(
        "staged, partial, or unknown" in error.lower()
        for error in verifier.pending_secret_errors(staged)
    )
    assert verifier.safe_secret_convergence_errors(staged) == []
    assert verifier.safe_secret_convergence_errors(
        [
            *staged,
            {
                "Name": "EXTERNAL_PENDING_SECRET",
                "DeploymentStatus": "Staged",
            },
        ]
    )


def test_runtime_identity_checks_starting_machines_and_immutable_digest() -> None:
    common = {
        "candidate_sha": CANDIDATE_SHA,
        "expected_project_ref": PRODUCTION_REF,
        "expected_migration_head": REMOTE_MIGRATION_HEAD,
        "expected_migration_contract": MIGRATION_CONTRACT_FINGERPRINT,
        "expected_migration_profile": MIGRATION_PROFILE,
        "fly_secrets": _fly_secrets(),
    }

    assert verifier.runtime_identity_errors(
        _health_payload(),
        fly_machines=[_machine(state="starting")],
        **common,
    ) == []

    mixed_errors = verifier.runtime_identity_errors(
        _health_payload(),
        fly_machines=[
            _machine(machine_id="started"),
            _machine(
                machine_id="starting",
                state="starting",
                image="registry.fly.io/juprleagues-api:different",
                digest="sha256:" + ("2" * 64),
            ),
        ],
        **common,
    )
    assert any("one exact image" in error.lower() for error in mixed_errors)

    stale_stopped_errors = verifier.runtime_identity_errors(
        _health_payload(),
        fly_machines=[
            _machine(machine_id="started"),
            _machine(
                machine_id="stopped",
                state="stopped",
                image="registry.fly.io/juprleagues-api:old",
                digest="sha256:" + ("2" * 64),
            ),
        ],
        **common,
    )
    assert any("one exact image" in error.lower() for error in stale_stopped_errors)

    missing_digest_errors = verifier.runtime_identity_errors(
        _health_payload(),
        fly_machines=[
            {
                "id": "starting",
                "state": "starting",
                "config": {"image": IMAGE_REF},
                "image_ref": {
                    "registry": "registry.fly.io",
                    "repository": verifier.PRODUCTION_FLY_APP,
                },
            }
        ],
        **common,
    )
    assert any("immutable image digest" in error.lower() for error in missing_digest_errors)

    foreign_registry = _machine()
    foreign_registry["image_ref"] = {
        **foreign_registry["image_ref"],
        "repository": "another-app",
    }
    foreign_registry_errors = verifier.runtime_identity_errors(
        _health_payload(),
        fly_machines=[foreign_registry],
        **common,
    )
    assert any(
        "foreign image registry or repository" in error.lower()
        for error in foreign_registry_errors
    )

    conflicting_digest = _machine(image=IMMUTABLE_IMAGE_REF)
    conflicting_digest["image_ref"] = {
        **conflicting_digest["image_ref"],
        "digest": "sha256:" + ("2" * 64),
    }
    conflicting_digest_errors = verifier.runtime_identity_errors(
        _health_payload(),
        fly_machines=[conflicting_digest],
        **common,
    )
    assert any(
        "conflicting configured and immutable image digests" in error.lower()
        for error in conflicting_digest_errors
    )


def test_final_runtime_attests_candidate_or_exact_immutable_rollback() -> None:
    snapshot = {
        "fly_app": verifier.PRODUCTION_FLY_APP,
        "git_commit_sha": "b" * 40,
        "image_build_git_sha": "b" * 40,
        "fly_image_ref": "registry.fly.io/juprleagues-api:previous",
        "fly_image_digest": "sha256:" + ("3" * 64),
        "fly_immutable_image_ref": (
            "registry.fly.io/juprleagues-api@sha256:" + ("3" * 64)
        ),
    }

    assert verifier.final_runtime_errors(
        _health_payload(),
        candidate_sha=CANDIDATE_SHA,
        promotion_accepted=True,
        expected_project_ref=PRODUCTION_REF,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_migration_contract=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_profile=MIGRATION_PROFILE,
        rollback_snapshot=snapshot,
        fly_machines=[_machine()],
        fly_secrets=_fly_secrets(),
    ) == []

    rolled_back_health = {
        **_health_payload(),
        "git_commit_sha": "b" * 40,
        "image_build_git_sha": "b" * 40,
        "fly_image_ref": "registry.fly.io/juprleagues-api:rollback-release",
    }
    rolled_back_machine = _machine(
        image=rolled_back_health["fly_image_ref"],
        digest=snapshot["fly_image_digest"],
    )
    assert verifier.final_runtime_errors(
        rolled_back_health,
        candidate_sha=CANDIDATE_SHA,
        promotion_accepted=False,
        expected_project_ref=PRODUCTION_REF,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_migration_contract=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_profile=MIGRATION_PROFILE,
        rollback_snapshot=snapshot,
        fly_machines=[rolled_back_machine],
        fly_secrets=_fly_secrets(),
    ) == []

    stale_sha_errors = verifier.final_runtime_errors(
        _health_payload(),
        candidate_sha=CANDIDATE_SHA,
        promotion_accepted=False,
        expected_project_ref=PRODUCTION_REF,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_migration_contract=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_profile=MIGRATION_PROFILE,
        rollback_snapshot=snapshot,
        fly_machines=[rolled_back_machine],
        fly_secrets=_fly_secrets(),
    )
    assert any("git_commit_sha" in error for error in stale_sha_errors)
    assert any("image_build_git_sha" in error for error in stale_sha_errors)

    unsafe_final = _health_payload()
    unsafe_final["feature_flags"] = {
        **unsafe_final["feature_flags"],
        "JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY": True,
    }
    unsafe_final["write_prerequisites"] = {
        **unsafe_final["write_prerequisites"],
        "email_mode": "live",
    }
    unsafe_final_errors = verifier.final_runtime_errors(
        unsafe_final,
        candidate_sha=CANDIDATE_SHA,
        promotion_accepted=True,
        expected_project_ref=PRODUCTION_REF,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_migration_contract=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_profile=MIGRATION_PROFILE,
        rollback_snapshot=snapshot,
        fly_machines=[_machine()],
        fly_secrets=_fly_secrets(),
    )
    assert any("feature_flags" in error for error in unsafe_final_errors)
    assert any("email mode" in error.lower() for error in unsafe_final_errors)


def test_openapi_and_cors_verifiers_require_exact_contract() -> None:
    assert (
        verifier.openapi_errors(
            {
                "openapi": "3.1.0",
                "info": {"title": "JUPR API"},
                "paths": {"/health": {"get": {}}},
            }
        )
        == []
    )
    assert verifier.openapi_errors({"openapi": "2.0", "paths": {}})

    headers = {
        origin: (
            "HTTP/2 200\n"
            f"access-control-allow-origin: {origin}\n"
            "access-control-allow-credentials: true\n"
            "access-control-allow-methods: GET, POST, OPTIONS\n"
        )
        for origin in verifier.PRODUCTION_ALLOWED_ORIGINS
    }
    assert verifier.cors_header_errors(headers) == []
    headers[verifier.PRODUCTION_ALLOWED_ORIGINS[0]] = "HTTP/2 200\n"
    assert verifier.cors_header_errors(headers)

    blocked = {
        verifier.PRODUCTION_FLY_ORIGIN: "HTTP/2 400\ncontent-type: text/plain\n",
        verifier.PRODUCTION_API_ORIGIN: "HTTP/2 400\ncontent-type: text/plain\n",
    }
    assert verifier.disallowed_cors_header_errors(blocked) == []
    blocked[verifier.PRODUCTION_API_ORIGIN] = (
        "HTTP/2 200\n"
        "access-control-allow-origin: https://not-allowed.invalid\n"
    )
    assert verifier.disallowed_cors_header_errors(blocked)


def test_public_database_read_verifier_requires_safe_leaderboard_contract() -> None:
    payload = {
        "club": {"id": "tres_palapas", "slug": "tres-palapas"},
        "scopes": [{"name": "OVERALL", "label": "Overall"}],
        "summary": {"players": 0},
        "leaderboard": [],
        "pagination": {"total": 0, "offset": 0, "limit": 1, "has_more": False},
    }

    assert verifier.public_database_read_errors(payload) == []
    assert verifier.public_database_read_errors(
        {**payload, "club": {"slug": "wrong"}, "leaderboard": None}
    )


def test_production_workflow_is_exact_candidate_and_never_creates_or_retargets_app() -> None:
    workflow = (
        ROOT / ".github/workflows/fly_api_deploy.yml"
    ).read_text(encoding="utf-8")

    assert workflow.startswith("name: Deploy FastAPI production to Fly\n")
    assert "environment: production" in workflow
    assert "FLY_APP_NAME: juprleagues-api" in workflow
    assert "PRODUCTION_SUPABASE_PROJECT_REF" in workflow
    assert "SUPABASE_PROD_DATABASE_URL" in workflow
    assert "FLY_SSH_TOKEN" in workflow
    assert "PRODUCTION_SOURCE_BRANCH: rollback-feb8" in workflow
    assert "ref: rollback-feb8" in workflow
    assert "github.event.repository.default_branch" not in workflow
    assert (
        "EXPECTED_MIGRATION_HEAD: "
        "${{ vars.PRODUCTION_MIGRATION_LEDGER_HEAD }}"
    ) in workflow
    assert 'if [ "$GITHUB_REF" != "refs/heads/$PRODUCTION_SOURCE_BRANCH" ]; then' in workflow
    assert '"$HEAD_SHA" != "$GITHUB_SHA"' in workflow
    assert '"$HEAD_SHA" != "$PRODUCTION_SHA"' in workflow
    assert '"$HEAD_SHA" != "$CANDIDATE_SHA_INPUT"' in workflow
    assert "apps create" not in workflow
    assert "setup-flyctl@master" not in workflow
    assert "app_name:" not in workflow
    assert "primary_region:" not in workflow
    assert "custom_domain:" not in workflow
    assert "|| true" not in workflow


def test_production_workflow_verifies_database_runtime_cors_and_final_write_policy() -> None:
    workflow = (
        ROOT / ".github/workflows/fly_api_deploy.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count(
        "select version || E'\\t' || coalesce(name, '') "
        "from supabase_migrations.schema_migrations order by version"
    ) == 2
    assert workflow.count(
        "'tournament_player_duplicate_groups'"
    ) == 2
    assert "config/production_migration_contract.json" in workflow
    assert "deployment_verifier.py preflight" in workflow
    assert "deployment_verifier.py migrations" in workflow
    assert "deployment_verifier.py snapshot" in workflow
    assert "deployment_verifier.py secrets" in workflow
    assert "deployment_verifier.py runtime" in workflow
    assert "deployment_verifier.py final" in workflow
    assert "deployment_verifier.py public-read" in workflow
    assert "deployment_verifier.py cors" in workflow
    assert '"JUPR_PRODUCTION_WRITE_POLICY=read_only"' in workflow
    assert '"JUPR_STAGING_WRITE_WAVE=none"' in workflow
    assert '--build-arg "JUPR_DEPLOYMENT_GIT_SHA=$GITHUB_SHA"' in workflow
    assert "flyctl machines list" in workflow
    assert "$PRODUCTION_FLY_ORIGIN/openapi.json" in workflow
    assert "$PRODUCTION_API_ORIGIN/health" in workflow
    assert 'api_origins=(' in workflow
    assert '"$PRODUCTION_FLY_ORIGIN"' in workflow
    assert '"$PRODUCTION_API_ORIGIN"' in workflow
    assert "https://not-allowed.invalid" in workflow
    assert "Establish and verify write_wave none before deploy" in workflow
    assert "Restore and attest final read-only production state" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "production-rollback-snapshot.json" in workflow
    assert "steps.quiesce.outcome != 'skipped'" in workflow
    assert "PROMOTION_ACCEPTED:" in workflow
    assert "steps.deploy.outcome == 'success'" in workflow
    assert "steps.cors_verify.outcome == 'success'" in workflow
    assert "--promotion-accepted \"$PROMOTION_ACCEPTED\"" in workflow
    assert workflow.index("deployment_verifier.py preflight") < workflow.index(
        "flyctl deploy"
    )
    assert workflow.index("deployment_verifier.py migrations") < workflow.index(
        "flyctl deploy"
    )
    assert workflow.index("deployment_verifier.py snapshot") < workflow.index(
        "Establish and verify write_wave none before deploy"
    )
    assert workflow.index("Establish and verify write_wave none before deploy") < (
        workflow.index("flyctl deploy")
    )
    assert workflow.count("flyctl secrets set") == 2
    safe_bundles = re.findall(
        r"runtime_secrets=\(\n(?P<body>.*?)\n\s+\)",
        workflow,
        flags=re.DOTALL,
    )
    assert len(safe_bundles) == 2
    assert [
        line.strip()
        for line in safe_bundles[0].splitlines()
        if line.strip()
    ] == [
        line.strip()
        for line in safe_bundles[1].splitlines()
        if line.strip()
    ]
    assert workflow.index("flyctl secrets set") < workflow.index("flyctl deploy")
    assert workflow.rindex("flyctl secrets set") > workflow.index("flyctl deploy")
    assert '--safe-convergence-only' in workflow
    assert 'if [ "$QUIESCE_OUTCOME" != "success" ]; then' in workflow
    assert "--stage" not in workflow
    assert workflow.count("JUPR_DEPLOYMENT_GIT_SHA=") == 1
    assert "fly_immutable_image_ref" in workflow
    assert '--image "$PREDEPLOY_IMMUTABLE_IMAGE"' in workflow
    assert "flyctl config save" in workflow
    assert "fly_config_sha256" in workflow
    assert '--config "$PREDEPLOY_CONFIG"' in workflow
    assert '&& [ "$DEPLOY_OUTCOME" != "skipped" ]; then' in workflow
    assert workflow.index("--no-pending-only") < workflow.index("flyctl secrets set")
    assert workflow.count("flyctl ssh console") == 2


def test_staging_deploy_bakes_the_exact_candidate_sha_into_the_image() -> None:
    workflow = (
        ROOT / ".github/workflows/fly_api_staging_deploy.yml"
    ).read_text(encoding="utf-8")

    assert '--build-arg "JUPR_DEPLOYMENT_GIT_SHA=$GITHUB_SHA"' in workflow


def test_preflight_cli_never_prints_secret_values(tmp_path: Path) -> None:
    github_env = tmp_path / "github-env"
    secret_marker = "do-not-print-739"
    env = {
        **os.environ,
        **_production_env(
            FLY_API_TOKEN=secret_marker,
            SUPABASE_SERVICE_ROLE_KEY=secret_marker,
        ),
        "GITHUB_ENV": str(github_env),
    }
    result = subprocess.run(
        [
            sys.executable,
            "scripts/deployment_verifier.py",
            "preflight",
            "--config",
            "fly.toml",
            "--migrations-dir",
            "supabase/migrations",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert secret_marker not in result.stdout + result.stderr
    github_env_text = github_env.read_text(encoding="utf-8")
    assert (
        f"EXPECTED_MIGRATION_CONTRACT={MIGRATION_CONTRACT_FINGERPRINT}"
        in github_env_text
    )
    assert f"EXPECTED_MIGRATION_PROFILE={MIGRATION_PROFILE}" in github_env_text
    assert "EXPECTED_MIGRATION_HEAD=" not in github_env_text


def test_api_image_bakes_candidate_revision_for_runtime_cross_check() -> None:
    dockerfile = (ROOT / "Dockerfile.api").read_text(encoding="utf-8")

    assert "ARG JUPR_DEPLOYMENT_GIT_SHA=unknown" in dockerfile
    assert 'org.opencontainers.image.revision="${JUPR_DEPLOYMENT_GIT_SHA}"' in dockerfile
    assert 'JUPR_IMAGE_BUILD_GIT_SHA="${JUPR_DEPLOYMENT_GIT_SHA}"' in dockerfile
