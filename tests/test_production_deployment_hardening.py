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
MIGRATION_PROFILE = "next-fastapi-team-competition-2026-07-28"
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
        "business_data_write_wave_source": "none",
        "staging_write_flags": controlled,
        "email_mode": "dry_run",
        "live_player_update_email_enabled": False,
        "public_live_production_override_enabled": False,
        "environment_guard": {
            "expected_project_ref": PRODUCTION_REF,
            "actual_project_ref": PRODUCTION_REF,
            "allowed": True,
        },
        "features": features,
    }


def _identity_payload() -> dict:
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
    }


def _write_release_candidate_contract(
    path: Path,
    *,
    candidate_sha: str = CANDIDATE_SHA,
    image_digest: str = IMAGE_DIGEST,
    fly_config_sha256: str = FLY_CONFIG_SHA,
    migration_profile: str = MIGRATION_PROFILE,
    migration_contract_fingerprint: str = MIGRATION_CONTRACT_FINGERPRINT,
    migration_head: str = REMOTE_MIGRATION_HEAD,
    supabase_project_ref: str = PRODUCTION_REF,
) -> None:
    path.write_text(
        "\n".join(
            [
                "schema_version: 1",
                f"candidate_sha: {candidate_sha}",
                f"fly_image_digest: {image_digest}",
                f"fly_config_sha256: {fly_config_sha256}",
                f"migration_profile: {migration_profile}",
                f"migration_contract_fingerprint: {migration_contract_fingerprint}",
                f"migration_head: {migration_head}",
                f"supabase_project_ref: {supabase_project_ref}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_expected_production_feature_flags_are_safe() -> None:
    flags = verifier.expected_production_feature_flags()

    assert flags
    assert all(value is False for value in flags.values())
    assert set(flags) == set(verifier.PRODUCTION_FEATURE_FLAG_DEFAULTS)


def test_production_health_rejects_any_staging_write_flag() -> None:
    payload = _health_payload()
    payload["staging_write_flags"]["JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER"] = True

    errors = verifier.production_health_errors(
        payload,
        expected_sha=CANDIDATE_SHA,
        expected_supabase_ref=PRODUCTION_REF,
        expected_image_ref=IMAGE_REF,
    )

    assert any("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER" in error for error in errors)


def test_production_health_rejects_permanent_write_posture() -> None:
    payload = _health_payload()
    payload["write_wave"] = "open"
    payload["staging_write_wave"] = "open"
    payload["business_data_write_wave_active"] = True

    errors = verifier.production_health_errors(
        payload,
        expected_sha=CANDIDATE_SHA,
        expected_supabase_ref=PRODUCTION_REF,
        expected_image_ref=IMAGE_REF,
    )

    assert any("write_wave" in error for error in errors)
    assert any("business_data_write_wave_active" in error for error in errors)


def test_production_health_rejects_live_email_and_public_override() -> None:
    payload = _health_payload()
    payload["email_mode"] = "live"
    payload["live_player_update_email_enabled"] = True
    payload["public_live_production_override_enabled"] = True

    errors = verifier.production_health_errors(
        payload,
        expected_sha=CANDIDATE_SHA,
        expected_supabase_ref=PRODUCTION_REF,
        expected_image_ref=IMAGE_REF,
    )

    assert any("email_mode" in error for error in errors)
    assert any("live_player_update_email_enabled" in error for error in errors)
    assert any("public_live_production_override_enabled" in error for error in errors)


def test_production_health_rejects_feature_flag_drift() -> None:
    payload = _health_payload()
    payload["features"]["admin_match_uploader_enabled"] = True

    errors = verifier.production_health_errors(
        payload,
        expected_sha=CANDIDATE_SHA,
        expected_supabase_ref=PRODUCTION_REF,
        expected_image_ref=IMAGE_REF,
    )

    assert any("admin_match_uploader_enabled" in error for error in errors)


def test_production_identity_rejects_staging_supabase() -> None:
    payload = _identity_payload()
    payload["supabase_project_ref"] = verifier.STAGING_SUPABASE_PROJECT_REF

    errors = verifier.production_identity_errors(
        payload,
        expected_sha=CANDIDATE_SHA,
        expected_supabase_ref=PRODUCTION_REF,
        expected_image_ref=IMAGE_REF,
    )

    assert any("Supabase" in error for error in errors)


def test_production_fly_config_rejects_staging_flags(tmp_path: Path) -> None:
    source = (ROOT / "fly.production.toml").read_text(encoding="utf-8")
    changed = source.replace(
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

    assert len(versions) == 54
    assert versions[-5:] == (
        "20260728040000",
        "20260728041000",
        "20260731033000",
        "20260731210000",
        "20261020000000",
    )
    assert len(names) == 54
    assert all("XX" not in version for version in versions)
    assert len(contract["required_ledger_names"]) == 54
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

    with pytest.raises(ValueError, match="stale"):
        verifier.load_migration_contract(
            ROOT / "config/production_migration_contract.json",
            migrations,
        )


def test_reviewed_migration_contract_rejects_missing_schema_only_entry(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "migration-contract.json"
    payload = dict(MIGRATION_CONTRACT)
    payload["required_ledger_names"] = list(payload["required_ledger_names"])
    payload["schema_contract_only_repository_migrations"] = []
    contract_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="wrong schema-only migration set"):
        verifier.load_migration_contract(
            contract_path,
            ROOT / "supabase/migrations",
        )


def test_migration_contract_rejects_unsorted_required_names(tmp_path: Path) -> None:
    contract_path = tmp_path / "migration-contract.json"
    payload = dict(MIGRATION_CONTRACT)
    payload["required_ledger_names"] = list(reversed(payload["required_ledger_names"]))
    contract_path.write_text(__import__("json").dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unique and sorted"):
        verifier.load_migration_contract(
            contract_path,
            ROOT / "supabase/migrations",
        )


def test_release_candidate_contract_rejects_changed_candidate(tmp_path: Path) -> None:
    contract_path = tmp_path / "release-candidate.yml"
    _write_release_candidate_contract(contract_path)
    payload = verifier.load_release_candidate_contract(contract_path)
    errors = verifier.release_candidate_contract_errors(
        payload,
        expected_candidate="b" * 40,
        expected_image_digest=IMAGE_DIGEST,
        expected_fly_config_sha256=FLY_CONFIG_SHA,
        expected_migration_profile=MIGRATION_PROFILE,
        expected_migration_contract_fingerprint=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_supabase_project_ref=PRODUCTION_REF,
    )
    assert any("candidate SHA" in error for error in errors)


def test_release_candidate_contract_rejects_changed_migration_fingerprint(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "release-candidate.yml"
    _write_release_candidate_contract(contract_path)
    payload = verifier.load_release_candidate_contract(contract_path)
    errors = verifier.release_candidate_contract_errors(
        payload,
        expected_candidate=CANDIDATE_SHA,
        expected_image_digest=IMAGE_DIGEST,
        expected_fly_config_sha256=FLY_CONFIG_SHA,
        expected_migration_profile=MIGRATION_PROFILE,
        expected_migration_contract_fingerprint="f" * 64,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_supabase_project_ref=PRODUCTION_REF,
    )
    assert any("migration contract fingerprint" in error for error in errors)


def test_release_candidate_contract_rejects_changed_migration_head(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "release-candidate.yml"
    _write_release_candidate_contract(contract_path)
    payload = verifier.load_release_candidate_contract(contract_path)
    errors = verifier.release_candidate_contract_errors(
        payload,
        expected_candidate=CANDIDATE_SHA,
        expected_image_digest=IMAGE_DIGEST,
        expected_fly_config_sha256=FLY_CONFIG_SHA,
        expected_migration_profile=MIGRATION_PROFILE,
        expected_migration_contract_fingerprint=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_head="20260720999999",
        expected_supabase_project_ref=PRODUCTION_REF,
    )
    assert any("migration head" in error for error in errors)


def test_release_candidate_contract_rejects_changed_supabase_project(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "release-candidate.yml"
    _write_release_candidate_contract(contract_path)
    payload = verifier.load_release_candidate_contract(contract_path)
    errors = verifier.release_candidate_contract_errors(
        payload,
        expected_candidate=CANDIDATE_SHA,
        expected_image_digest=IMAGE_DIGEST,
        expected_fly_config_sha256=FLY_CONFIG_SHA,
        expected_migration_profile=MIGRATION_PROFILE,
        expected_migration_contract_fingerprint=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_supabase_project_ref="different-project-ref",
    )
    assert any("Supabase project" in error for error in errors)


def test_release_candidate_contract_rejects_missing_field(tmp_path: Path) -> None:
    contract_path = tmp_path / "release-candidate.yml"
    _write_release_candidate_contract(contract_path)
    text = contract_path.read_text(encoding="utf-8").replace(
        f"migration_head: {REMOTE_MIGRATION_HEAD}\n",
        "",
    )
    contract_path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        verifier.load_release_candidate_contract(contract_path)


def test_release_candidate_contract_rejects_invalid_digest(tmp_path: Path) -> None:
    contract_path = tmp_path / "release-candidate.yml"
    _write_release_candidate_contract(contract_path, image_digest="sha256:123")

    with pytest.raises(ValueError, match="invalid fly_image_digest"):
        verifier.load_release_candidate_contract(contract_path)


def test_release_candidate_contract_rejects_migration_head_not_in_required_ledger(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "release-candidate.yml"
    _write_release_candidate_contract(contract_path, migration_head="20990101000000")
    payload = verifier.load_release_candidate_contract(contract_path)
    errors = verifier.release_candidate_contract_errors(
        payload,
        expected_candidate=CANDIDATE_SHA,
        expected_image_digest=IMAGE_DIGEST,
        expected_fly_config_sha256=FLY_CONFIG_SHA,
        expected_migration_profile=MIGRATION_PROFILE,
        expected_migration_contract_fingerprint=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_head="20990101000000",
        expected_supabase_project_ref=PRODUCTION_REF,
        required_migration_ledger=[
            ("20260720123401", "previous"),
            (REMOTE_MIGRATION_HEAD, "current"),
        ],
    )
    assert any("not present in the required migration ledger" in error for error in errors)


def test_release_candidate_contract_accepts_required_inputs(tmp_path: Path) -> None:
    contract_path = tmp_path / "release-candidate.yml"
    _write_release_candidate_contract(contract_path)
    payload = verifier.load_release_candidate_contract(contract_path)

    assert (
        verifier.release_candidate_contract_errors(
            payload,
            expected_candidate=CANDIDATE_SHA,
            expected_image_digest=IMAGE_DIGEST,
            expected_fly_config_sha256=FLY_CONFIG_SHA,
            expected_migration_profile=MIGRATION_PROFILE,
            expected_migration_contract_fingerprint=MIGRATION_CONTRACT_FINGERPRINT,
            expected_migration_head=REMOTE_MIGRATION_HEAD,
            expected_supabase_project_ref=PRODUCTION_REF,
            required_migration_ledger=[
                ("20260720123401", "previous"),
                (REMOTE_MIGRATION_HEAD, "current"),
            ],
        )
        == []
    )


def test_release_candidate_contract_allows_remote_head_after_schema_only_contract(
    tmp_path: Path,
) -> None:
    contract_path = tmp_path / "release-candidate.yml"
    _write_release_candidate_contract(contract_path)
    payload = verifier.load_release_candidate_contract(contract_path)
    repository_names = verifier.expected_migration_names(ROOT / "supabase/migrations")
    required_ledger_names = verifier.required_repository_ledger_names(
        repository_names,
        contract["schema_contract_only_repository_migrations"]
        if (contract := MIGRATION_CONTRACT)
        else (),
    )
    ledger = [
        (str(index + 1), name)
        for index, name in enumerate(required_ledger_names)
    ]
    ledger[-1] = (REMOTE_MIGRATION_HEAD, ledger[-1][1])

    assert (
        verifier.release_candidate_contract_errors(
            payload,
            expected_candidate=CANDIDATE_SHA,
            expected_image_digest=IMAGE_DIGEST,
            expected_fly_config_sha256=FLY_CONFIG_SHA,
            expected_migration_profile=MIGRATION_PROFILE,
            expected_migration_contract_fingerprint=MIGRATION_CONTRACT_FINGERPRINT,
            expected_migration_head=REMOTE_MIGRATION_HEAD,
            expected_supabase_project_ref=PRODUCTION_REF,
            required_migration_ledger=ledger,
        )
        == []
    )


def test_production_verifier_requires_remote_migration_contract() -> None:
    source = (ROOT / "scripts/deployment_verifier.py").read_text(encoding="utf-8")

    assert "supabase_migration_ledger" in source
    assert "verify_required_migration_ledger" in source
    assert "migration_contract_fingerprint" in source


def test_deployment_verifier_returns_nonzero_for_missing_env(tmp_path: Path) -> None:
    verifier_path = ROOT / "scripts/deployment_verifier.py"
    result = subprocess.run(
        [sys.executable, str(verifier_path), "production"],
        cwd=ROOT,
        env={"PATH": os.environ.get("PATH", "")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing required environment variable" in result.stderr


def test_production_deploy_workflow_blocks_wrong_targets() -> None:
    workflow = (ROOT / ".github/workflows/deploy-production.yml").read_text(
        encoding="utf-8"
    )

    assert "verify-production-target" in workflow
    assert "deployment_verifier.py production-target" in workflow
    assert "environment: production" in workflow
    assert "release_candidate_contract" in workflow
    assert "verify-migrations" in workflow
    assert "production-migration-contract" in workflow
    assert "Verify production migration ledger contract" in workflow
    assert "verify-release-candidate-contract" in workflow
    assert "Upload production release evidence" in workflow
    assert "environment: staging" not in workflow
    assert "juprleagues-api-staging" not in workflow
    assert verifier.PRODUCTION_FLY_APP in workflow


def test_production_deploy_has_dedicated_fly_config() -> None:
    workflow = (ROOT / ".github/workflows/deploy-production.yml").read_text(
        encoding="utf-8"
    )

    assert "fly.production.toml" in workflow
    assert (ROOT / "fly.production.toml").exists()
    assert verifier.production_fly_config_errors(ROOT / "fly.production.toml") == []


def test_staging_deploy_does_not_mutate_production() -> None:
    workflow = (ROOT / ".github/workflows/deploy-fly-staging.yml").read_text(
        encoding="utf-8"
    )

    assert "deployment_verifier.py staging-target" in workflow
    assert "juprleagues-api-staging" in workflow
    assert "environment: staging" in workflow
    assert "deployment_verifier.py production-target" not in workflow
    assert verifier.PRODUCTION_FLY_APP not in workflow


def test_production_deploy_uses_immutable_digest_for_final_verification() -> None:
    workflow = (ROOT / ".github/workflows/deploy-production.yml").read_text(
        encoding="utf-8"
    )

    assert "--image-digest" in workflow
    assert "--expected-image-digest" in workflow
    assert "fly_image_digest" in workflow
    assert "deployment_verifier.py production-identity" in workflow


def test_production_deploy_requires_manual_environment_approval() -> None:
    workflow = (ROOT / ".github/workflows/deploy-production.yml").read_text(
        encoding="utf-8"
    )

    assert "workflow_dispatch:" in workflow
    assert "environment: production" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
