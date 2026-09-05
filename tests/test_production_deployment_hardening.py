from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from scripts import deployment_verifier as verifier


ROOT = Path(__file__).resolve().parent.parent
PRODUCTION_REF = verifier.PRODUCTION_SUPABASE_PROJECT_REF
CANDIDATE_SHA = "a" * 40
IMAGE_REF = "registry.fly.io/juprleagues-api:deployment-01ABCDEF"
IMAGE_DIGEST = "sha256:" + ("1" * 64)
IMMUTABLE_IMAGE_REF = (
    f"registry.fly.io/{verifier.PRODUCTION_FLY_APP}@{IMAGE_DIGEST}"
)
FLY_CONFIG_SHA = "4" * 64
MIGRATION_PROFILE = "next-fastapi-tournament-acceptance-2026-08-25"
MIGRATION_CONTRACT = verifier.load_migration_contract(
    ROOT / "config/production_migration_contract.json",
    ROOT / "supabase/migrations",
)
MIGRATION_CONTRACT_FINGERPRINT = verifier.migration_contract_fingerprint(
    profile=MIGRATION_CONTRACT["profile"],
    required_ledger_names=MIGRATION_CONTRACT["required_ledger_names"],
    deployment_order=MIGRATION_CONTRACT["deployment_order"],
    repository_migration_content_sha256=MIGRATION_CONTRACT[
        "repository_migration_content_sha256"
    ],
    allowed_duplicate_ledger_names=MIGRATION_CONTRACT[
        "allowed_duplicate_ledger_names"
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
        "SUPABASE_PROD_DATABASE_READ_TOKEN": "sbp_fc_test-read-token-1234567890",
        "SUPABASE_ANON_KEY": "anon-present",
        "SUPABASE_SERVICE_ROLE_KEY": "service-role-present",
        "SUPABASE_URL": f"https://{PRODUCTION_REF}.supabase.co",
    }
    env.update(overrides)
    return env


def _health_payload(*, feature_profile: str = "release") -> dict:
    controlled = verifier.expected_production_controlled_write_flags(
        profile=feature_profile
    )
    features = verifier.expected_production_feature_flags(
        profile=feature_profile
    )
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
        "production_business_write_policy": "enabled",
        "security_denial_audit_logging_required": True,
        "public_live_writes_enabled": True,
        "public_live_production_override_enabled": True,
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


def test_production_fly_config_is_exact_reviewed_live_policy() -> None:
    assert verifier.production_fly_config_errors(ROOT / "fly.toml") == []

    config = tomllib.loads((ROOT / "fly.toml").read_text(encoding="utf-8"))
    assert config["app"] == verifier.PRODUCTION_FLY_APP
    assert config["primary_region"] == verifier.PRODUCTION_FLY_REGION
    assert config["env"]["JUPR_PRODUCTION_WRITE_POLICY"] == "enabled"
    assert config["env"]["JUPR_STAGING_WRITE_WAVE"] == "none"
    assert {
        name: config["env"][name] == "1"
        for name in verifier.PRODUCTION_FEATURE_FLAGS
    } == verifier.expected_production_feature_flags()
    assert {
        name
        for name, enabled in verifier.expected_production_feature_flags().items()
        if enabled
    } == set(verifier.PRODUCTION_ENABLED_FEATURE_FLAGS)


def _saved_fly_config(*, generated_at: str, region: str = "dfw") -> bytes:
    return (
        "# fly.toml app configuration file generated for "
        f"{verifier.PRODUCTION_FLY_APP} on {generated_at}\n"
        f'app = "{verifier.PRODUCTION_FLY_APP}"\n'
        f'primary_region = "{region}"\n'
    ).encode("utf-8")


def test_predeploy_fly_config_fingerprint_ignores_generated_at_timestamp(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    first_raw = _saved_fly_config(generated_at="2026-09-02T15:08:41Z")
    second_raw = _saved_fly_config(generated_at="2026-09-02T15:33:04Z")
    first.write_bytes(first_raw)
    second.write_bytes(second_raw)

    expected = hashlib.sha256(
        first_raw.replace(b"2026-09-02T15:08:41Z", b"<generated-at>", 1)
    ).hexdigest()
    assert verifier.predeploy_fly_config_fingerprint(first) == expected
    assert verifier.predeploy_fly_config_fingerprint(second) == expected


def test_predeploy_fly_config_fingerprint_binds_other_config_bytes(
    tmp_path: Path,
) -> None:
    original = tmp_path / "original.toml"
    changed = tmp_path / "changed.toml"
    changed_comment = tmp_path / "changed-comment.toml"
    original.write_bytes(
        _saved_fly_config(generated_at="2026-09-02T15:08:41Z")
    )
    changed.write_bytes(
        _saved_fly_config(
            generated_at="2026-09-02T15:33:04Z",
            region="iad",
        )
    )
    changed_comment.write_bytes(
        _saved_fly_config(generated_at="2026-09-02T15:33:04Z")
        + b"# reviewed operator note\n"
    )

    original_fingerprint = verifier.predeploy_fly_config_fingerprint(original)
    assert original_fingerprint != verifier.predeploy_fly_config_fingerprint(changed)
    assert original_fingerprint != verifier.predeploy_fly_config_fingerprint(
        changed_comment
    )


def test_predeploy_fly_config_fingerprint_normalizes_only_exact_leading_header(
    tmp_path: Path,
) -> None:
    exact = tmp_path / "exact.toml"
    nonleading = tmp_path / "nonleading.toml"
    malformed = tmp_path / "malformed.toml"
    exact_raw = _saved_fly_config(generated_at="2026-09-02T15:08:41Z")
    nonleading_raw = b"# operator note\n" + exact_raw
    malformed_raw = _saved_fly_config(generated_at="not-a-timestamp")
    exact.write_bytes(exact_raw)
    nonleading.write_bytes(nonleading_raw)
    malformed.write_bytes(malformed_raw)

    exact_fingerprint = verifier.predeploy_fly_config_fingerprint(exact)
    assert exact_fingerprint != verifier.predeploy_fly_config_fingerprint(nonleading)
    assert verifier.predeploy_fly_config_fingerprint(malformed) == hashlib.sha256(
        malformed_raw
    ).hexdigest()


def test_predeploy_fly_config_fingerprint_rejects_foreign_app(
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


def test_reviewed_projection_preserves_live_and_adds_only_league_core() -> None:
    assert verifier.PRODUCTION_ENABLED_FEATURE_FLAGS == {
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
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
    assert all(
        verifier.expected_production_feature_flags()[name] is False
        for name in (
            "JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS",
            "JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL",
            "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE",
            "JUPR_ENABLE_TEAM_LEAGUES",
        )
    )
    assert (
        verifier.PRODUCTION_ENABLED_FEATURE_FLAGS
        - verifier.PRODUCTION_LIVE_BASELINE_ENABLED_FEATURE_FLAGS
    ) == {
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
    }
    assert all(
        verifier.expected_production_feature_flags(profile="baseline")[name]
        is False
        for name in (
            "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN",
            "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT",
            "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
        )
    )
    assert verifier.feature_flag_fingerprint(
        verifier.expected_production_feature_flags(profile="baseline")
    ) == "8fb4e6ee26a71c5aeaa6cd4634e1541abee3d02b02588e2fa2331f6c4d1c2a85"
    assert verifier.feature_flag_fingerprint(
        verifier.expected_production_controlled_write_flags(profile="baseline")
    ) == "0545cd0d4e437114b8d93fba7de5aec8057a00c08d3417830d367cfba1756bbb"


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


def test_production_fly_config_rejects_any_feature_projection_drift(
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

    disabled_live = text.replace(
        'JUPR_ENABLE_TOURNAMENT_WRITES_PRODUCTION = "1"',
        'JUPR_ENABLE_TOURNAMENT_WRITES_PRODUCTION = "0"',
    )
    path.write_text(disabled_live, encoding="utf-8")
    assert any(
        "JUPR_ENABLE_TOURNAMENT_WRITES_PRODUCTION" in error
        for error in verifier.production_fly_config_errors(path)
    )


def test_repository_migration_inventory_and_reviewed_profile_are_deterministic() -> None:
    versions = verifier.expected_migration_versions(ROOT / "supabase/migrations")
    names = verifier.expected_migration_names(ROOT / "supabase/migrations")
    contract = verifier.load_migration_contract(
        ROOT / "config/production_migration_contract.json",
        ROOT / "supabase/migrations",
    )

    assert len(versions) == 112
    assert versions[-31:] == (
        "20261030010000",
        "20261101000000",
        "20261102000000",
        "20261103000000",
        "20261104000000",
        "20261105000000",
        "20261106000000",
        "20261107000000",
        "20261108000000",
        "20261108003000",
        "20261108010000",
        "20261108013000",
        "20261108014000",
        "20261108015000",
        "20261108016000",
        "20261108017000",
        "20261108018000",
        "20261108019000",
        "20261108020000",
        "20261108021000",
        "20261108022000",
        "20261108022500",
        "20261108023000",
        "20261108024000",
        "20261108025000",
        "20261108026000",
        "20261108027000",
        "20261108028000",
        "20261109000000",
        "20261109001000",
        "20261109002000",
    )
    assert len(names) == 112
    assert all("XX" not in version for version in versions)
    assert len(contract["required_ledger_names"]) == 99
    assert len(contract["deployment_order"]) == 99
    assert set(contract["deployment_order"]) == set(
        contract["required_ledger_names"]
    )
    assert "tournament_complete_registration_editor" in contract[
        "required_ledger_names"
    ]
    assert "tournament_day_roster_authority" in contract[
        "required_ledger_names"
    ]
    assert "manual_tournament_day_court_assignment" in contract[
        "required_ledger_names"
    ]
    assert "tournament_day_court_reservations" in contract[
        "required_ledger_names"
    ]
    assert "tournament_day_court_reservations_fk_index" in contract[
        "required_ledger_names"
    ]
    assert "tournament_team_retirement_results" in contract[
        "required_ledger_names"
    ]
    assert "tournament_podium_row_versions" in contract[
        "required_ledger_names"
    ]
    assert "tournament_podium_badge_catalog" in contract[
        "required_ledger_names"
    ]
    assert "tournament_playoff_round_scoring" in contract[
        "required_ledger_names"
    ]
    assert "tournament_best_of_three_game_scores" in contract[
        "required_ledger_names"
    ]
    assert "fix_tournament_day_score_id_types" in contract[
        "required_ledger_names"
    ]
    assert "independent_singles_registration_rating" in contract[
        "required_ledger_names"
    ]
    assert "transactional_bulk_tournament_check_in" in contract[
        "required_ledger_names"
    ]
    # Production retains cutover-era migration ledger entries that are not part
    # of the canonical repository set, so the release contract must tolerate
    # those reviewed additional names while still requiring every current one.
    assert contract["allow_additional_ledger_names"] is True
    assert contract["allowed_duplicate_ledger_names"] == (
        "tournament_admin_operations",
    )
    assert contract["schema_contract_only_repository_migrations"] == (
        "tournament_registrations_player_id_postgrest_reload",
    )


def test_reviewed_migration_order_binds_dependency_inversions() -> None:
    order = MIGRATION_CONTRACT["deployment_order"]
    position = {name: index for index, name in enumerate(order)}

    dependencies = (
        (
            "tournament_setup_courts_flexible_bundles",
            "tournament_venue_inventory",
        ),
        (
            "team_league_composition_settings",
            "team_league_normalized_rosters_substitute_pool",
        ),
        (
            "tournament_operator_special_form_repair",
            "tournament_terminal_completion_and_schedule_recovery",
        ),
        (
            "standard_tournament_substitute_policy",
            "repair_tournament_game_draw_scoped_uniqueness",
        ),
        (
            "tournament_podium_row_versions",
            "tournament_podium_badge_catalog",
        ),
        (
            "tournament_podium_badge_catalog",
            "tournament_team_retirement_results",
        ),
        (
            "tournament_team_retirement_results",
            "tournament_playoff_round_scoring",
        ),
        (
            "four_player_team_match_schema",
            "four_player_team_match_management_rpcs",
        ),
        (
            "four_player_team_match_management_rpcs",
            "four_player_team_match_security_hardening",
        ),
        (
            "server_only_view_security_hardening",
            "team_league_trigger_function_security_hardening",
        ),
    )
    assert all(
        position[dependency] < position[dependent]
        for dependency, dependent in dependencies
    )


def test_pending_migration_plan_preserves_reviewed_dependency_order() -> None:
    order = MIGRATION_CONTRACT["deployment_order"]
    expected_pending = (
        "tournament_setup_courts_flexible_bundles",
        "tournament_venue_inventory",
        "team_league_composition_settings",
        "team_league_normalized_rosters_substitute_pool",
        "four_player_team_match_schema",
        "four_player_team_match_security_hardening",
        "team_league_trigger_function_security_hardening",
    )
    pending = set(expected_pending)
    remote = [
        ("20260101000000", name)
        for name in order
        if name not in pending
    ]

    assert (
        verifier.pending_required_migration_names(order, remote)
        == expected_pending
    )


@pytest.mark.parametrize("malformation", ["duplicate", "set-mismatch"])
def test_reviewed_migration_contract_rejects_invalid_deployment_order(
    tmp_path: Path,
    malformation: str,
) -> None:
    payload = json.loads(
        (ROOT / "config/production_migration_contract.json").read_text(
            encoding="utf-8"
        )
    )
    if malformation == "duplicate":
        payload["deployment_order"][1] = payload["deployment_order"][0]
        match = "deployment order must be unique"
    else:
        payload["deployment_order"][-1] = "unreviewed_migration"
        match = "exactly the required ledger names"
    contract_path = tmp_path / "production_migration_contract.json"
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        verifier.load_migration_contract(
            contract_path,
            ROOT / "supabase/migrations",
        )


def test_migration_contract_fingerprint_binds_deployment_order() -> None:
    order = MIGRATION_CONTRACT["deployment_order"]
    common = {
        "profile": MIGRATION_CONTRACT["profile"],
        "required_ledger_names": MIGRATION_CONTRACT["required_ledger_names"],
        "repository_migration_content_sha256": MIGRATION_CONTRACT[
            "repository_migration_content_sha256"
        ],
        "allowed_duplicate_ledger_names": MIGRATION_CONTRACT[
            "allowed_duplicate_ledger_names"
        ],
    }

    assert verifier.migration_contract_fingerprint(
        **common,
        deployment_order=order,
    ) != verifier.migration_contract_fingerprint(
        **common,
        deployment_order=(*order[:-2], order[-1], order[-2]),
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
    assert (
        verifier.migration_ledger_errors(
            expected,
            [*remote, ("20260720130000", "first_contract")],
            allowed_duplicate_names=("first_contract",),
        )
        == []
    )

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


def _management_migration_attestation() -> list[dict]:
    return [
        {
            "release_attestation": {
                "ledger": [
                    {"version": "20250220", "name": "badges_v1"},
                    {
                        "version": REMOTE_MIGRATION_HEAD,
                        "name": "baseline_worker_run_log",
                    },
                ],
                "schema_contract": {
                    "tournament_registrations_player_id_column": True,
                    "idx_tournament_registrations_player_id": True,
                    "uq_tournament_registrations_tournament_player": True,
                    "tournament_player_duplicate_groups": 0,
                },
            }
        }
    ]


def test_supabase_read_only_migration_attestation_is_strictly_parsed() -> None:
    ledger, schema_contract = (
        verifier.parse_supabase_migration_attestation_response(
            _management_migration_attestation()
        )
    )

    assert ledger == [
        ("20250220", "badges_v1"),
        (REMOTE_MIGRATION_HEAD, "baseline_worker_run_log"),
    ]
    assert schema_contract["tournament_player_duplicate_groups"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        [{"release_attestation": {}}, {"release_attestation": {}}],
        [{"unexpected": {}}],
        [
            {
                "release_attestation": {
                    "ledger": [
                        {"version": "20250220", "name": "valid", "extra": 1}
                    ],
                    "schema_contract": {
                        "tournament_registrations_player_id_column": True,
                        "idx_tournament_registrations_player_id": True,
                        "uq_tournament_registrations_tournament_player": True,
                        "tournament_player_duplicate_groups": 0,
                    },
                }
            }
        ],
        [
            {
                "release_attestation": {
                    "ledger": [
                        {"version": "not-a-version", "name": "valid"}
                    ],
                    "schema_contract": {},
                }
            }
        ],
        [
            {
                "release_attestation": {
                    "ledger": [{"version": "20250220", "name": "bad\tname"}],
                    "schema_contract": {},
                }
            }
        ],
        [
            {
                "release_attestation": {
                    "ledger": [{"version": "20250220", "name": "valid"}],
                    "schema_contract": {
                        "tournament_registrations_player_id_column": 1,
                        "idx_tournament_registrations_player_id": True,
                        "uq_tournament_registrations_tournament_player": True,
                        "tournament_player_duplicate_groups": 0,
                    },
                }
            }
        ],
        [
            {
                "release_attestation": {
                    "ledger": [{"version": "20250220", "name": "valid"}],
                    "schema_contract": {
                        "tournament_registrations_player_id_column": True,
                        "idx_tournament_registrations_player_id": True,
                        "uq_tournament_registrations_tournament_player": True,
                        "tournament_player_duplicate_groups": -1,
                    },
                }
            }
        ],
    ],
)
def test_supabase_read_only_migration_attestation_rejects_other_shapes(
    payload: object,
) -> None:
    with pytest.raises(ValueError):
        verifier.parse_supabase_migration_attestation_response(payload)


def test_supabase_read_only_migration_query_is_select_only_and_schema_bound() -> None:
    query = verifier.PRODUCTION_MIGRATION_ATTESTATION_QUERY

    assert query.startswith("select pg_catalog.json_build_object(")
    assert "select pg_catalog.json_agg(" in query
    assert query.count("pg_catalog.json_build_object(") == 3
    assert query.count("pg_catalog.count(*)") == 2
    assert "from supabase_migrations.schema_migrations as sm" in query
    assert "from information_schema.columns as c" in query
    assert "from public.tournament_registrations as tr" in query
    assert not re.search(
        r"\b(insert|update|delete|alter|drop|truncate|grant|revoke)\b",
        query,
        flags=re.IGNORECASE,
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
    assert migrations[-1] == "20261109002000"

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
        ),
        config_path=ROOT / "fly.toml",
        migrations_dir=ROOT / "supabase/migrations",
    )
    assert any("staging Supabase project" in error for error in staging_errors)

    classic_token_errors, _ = verifier.preflight_errors(
        _production_env(
            SUPABASE_PROD_DATABASE_READ_TOKEN="sbp_classic-token"
        ),
        config_path=ROOT / "fly.toml",
        migrations_dir=ROOT / "supabase/migrations",
    )
    assert any("scoped Supabase" in error for error in classic_token_errors)

    short_scoped_token_errors, _ = verifier.preflight_errors(
        _production_env(SUPABASE_PROD_DATABASE_READ_TOKEN="sbp_fc_short"),
        config_path=ROOT / "fly.toml",
        migrations_dir=ROOT / "supabase/migrations",
    )
    assert any(
        "scoped Supabase" in error for error in short_scoped_token_errors
    )

    whitespace_token_errors, _ = verifier.preflight_errors(
        _production_env(
            SUPABASE_PROD_DATABASE_READ_TOKEN=(
                "sbp_fc_test read token must be rejected"
            )
        ),
        config_path=ROOT / "fly.toml",
        migrations_dir=ROOT / "supabase/migrations",
    )
    assert any(
        "scoped Supabase" in error for error in whitespace_token_errors
    )

    padded_token_errors, _ = verifier.preflight_errors(
        _production_env(
            SUPABASE_PROD_DATABASE_READ_TOKEN=(
                " sbp_fc_test-read-token-1234567890 "
            )
        ),
        config_path=ROOT / "fly.toml",
        migrations_dir=ROOT / "supabase/migrations",
    )
    assert any("scoped Supabase" in error for error in padded_token_errors)


def test_runtime_identity_requires_exact_sha_image_project_and_reviewed_policy() -> None:
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
    unsafe["production_business_write_policy"] = "read_only"
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
    health = _health_payload(feature_profile="baseline")
    errors, snapshot = verifier.predeploy_rollback_snapshot(
        health,
        {"Machines": [_machine()]},
        fly_config_sha256=FLY_CONFIG_SHA,
    )

    assert errors == []
    assert snapshot == {
        "fly_app": verifier.PRODUCTION_FLY_APP,
        "identity_mode": "exact-git",
        "feature_profile": "baseline",
        "git_commit_sha": CANDIDATE_SHA,
        "image_build_git_sha": CANDIDATE_SHA,
        "fly_image_ref": IMAGE_REF,
        "fly_image_digest": IMAGE_DIGEST,
        "fly_immutable_image_ref": IMMUTABLE_IMAGE_REF,
        "fly_config_sha256": FLY_CONFIG_SHA,
        "reviewed_legacy_image_digest": None,
        "reviewed_legacy_config_sha256": None,
    }

    release_profile_errors, release_profile_snapshot = (
        verifier.predeploy_rollback_snapshot(
            _health_payload(feature_profile="release"),
            {"Machines": [_machine()]},
            fly_config_sha256=FLY_CONFIG_SHA,
        )
    )
    assert release_profile_errors == []
    assert release_profile_snapshot["feature_profile"] == "release"

    legacy = {
        **health,
        "git_commit_sha": "unknown",
        "image_build_git_sha": "unknown",
    }
    unreviewed_errors, _ = verifier.predeploy_rollback_snapshot(
        legacy,
        {"Machines": [_machine()]},
        fly_config_sha256=FLY_CONFIG_SHA,
    )
    assert any("reviewed immutable" in error.lower() for error in unreviewed_errors)
    assert any("reviewed fly config" in error.lower() for error in unreviewed_errors)

    capture_errors, captured = verifier.predeploy_rollback_snapshot(
        legacy,
        {"Machines": [_machine()]},
        fly_config_sha256=FLY_CONFIG_SHA,
        capture_unreviewed_legacy_evidence=True,
    )
    assert capture_errors == []
    assert captured["identity_mode"] == "legacy-unreviewed-evidence"
    assert captured["feature_profile"] == "baseline"
    assert captured["fly_image_digest"] == IMAGE_DIGEST
    assert captured["fly_config_sha256"] == FLY_CONFIG_SHA
    assert captured["reviewed_legacy_image_digest"] is None
    assert captured["reviewed_legacy_config_sha256"] is None

    legacy_errors, legacy_snapshot = verifier.predeploy_rollback_snapshot(
        legacy,
        {"Machines": [_machine()]},
        fly_config_sha256=FLY_CONFIG_SHA,
        reviewed_legacy_image_digest=IMAGE_DIGEST,
        reviewed_legacy_config_sha256=FLY_CONFIG_SHA,
    )
    assert legacy_errors == []
    assert legacy_snapshot == {
        "fly_app": verifier.PRODUCTION_FLY_APP,
        "identity_mode": "legacy-immutable-bootstrap",
        "feature_profile": "baseline",
        "git_commit_sha": None,
        "image_build_git_sha": None,
        "fly_image_ref": IMAGE_REF,
        "fly_image_digest": IMAGE_DIGEST,
        "fly_immutable_image_ref": IMMUTABLE_IMAGE_REF,
        "fly_config_sha256": FLY_CONFIG_SHA,
        "reviewed_legacy_image_digest": IMAGE_DIGEST,
        "reviewed_legacy_config_sha256": FLY_CONFIG_SHA,
    }

    digest_drift, _ = verifier.predeploy_rollback_snapshot(
        legacy,
        {"Machines": [_machine()]},
        fly_config_sha256=FLY_CONFIG_SHA,
        reviewed_legacy_image_digest="sha256:" + ("9" * 64),
        reviewed_legacy_config_sha256=FLY_CONFIG_SHA,
    )
    assert any("digest does not match" in error.lower() for error in digest_drift)

    no_longer_legacy, _ = verifier.predeploy_rollback_snapshot(
        health,
        {"Machines": [_machine()]},
        fly_config_sha256=FLY_CONFIG_SHA,
        reviewed_legacy_image_digest=IMAGE_DIGEST,
        reviewed_legacy_config_sha256=FLY_CONFIG_SHA,
    )
    assert any("forbidden once" in error.lower() for error in no_longer_legacy)


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
        "identity_mode": "exact-git",
        "feature_profile": "baseline",
        "git_commit_sha": "b" * 40,
        "image_build_git_sha": "b" * 40,
        "fly_image_ref": "registry.fly.io/juprleagues-api:previous",
        "fly_image_digest": "sha256:" + ("3" * 64),
        "fly_immutable_image_ref": (
            "registry.fly.io/juprleagues-api@sha256:" + ("3" * 64)
        ),
        "fly_config_sha256": FLY_CONFIG_SHA,
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
        **_health_payload(feature_profile="baseline"),
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
        _health_payload(feature_profile="baseline"),
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


def test_final_runtime_allows_only_exact_reviewed_legacy_rollback() -> None:
    legacy_digest = "sha256:" + ("3" * 64)
    snapshot = {
        "fly_app": verifier.PRODUCTION_FLY_APP,
        "identity_mode": "legacy-immutable-bootstrap",
        "feature_profile": "baseline",
        "git_commit_sha": None,
        "image_build_git_sha": None,
        "fly_image_ref": "registry.fly.io/juprleagues-api:legacy",
        "fly_image_digest": legacy_digest,
        "fly_immutable_image_ref": (
            f"registry.fly.io/juprleagues-api@{legacy_digest}"
        ),
        "fly_config_sha256": FLY_CONFIG_SHA,
        "reviewed_legacy_image_digest": legacy_digest,
        "reviewed_legacy_config_sha256": FLY_CONFIG_SHA,
    }
    legacy_health = {
        **_health_payload(feature_profile="baseline"),
        "git_commit_sha": "unknown",
        "image_build_git_sha": "unknown",
        "fly_image_ref": snapshot["fly_image_ref"],
    }
    legacy_machine = _machine(
        image=snapshot["fly_image_ref"],
        digest=legacy_digest,
    )

    assert verifier.final_runtime_errors(
        legacy_health,
        candidate_sha=CANDIDATE_SHA,
        promotion_accepted=False,
        expected_project_ref=PRODUCTION_REF,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_migration_contract=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_profile=MIGRATION_PROFILE,
        rollback_snapshot=snapshot,
        fly_machines=[legacy_machine],
        fly_secrets=_fly_secrets(),
    ) == []

    unreviewed = {
        **snapshot,
        "identity_mode": "legacy-unreviewed-evidence",
        "reviewed_legacy_image_digest": None,
        "reviewed_legacy_config_sha256": None,
    }
    unreviewed_errors = verifier.final_runtime_errors(
        legacy_health,
        candidate_sha=CANDIDATE_SHA,
        promotion_accepted=False,
        expected_project_ref=PRODUCTION_REF,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_migration_contract=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_profile=MIGRATION_PROFILE,
        rollback_snapshot=unreviewed,
        fly_machines=[legacy_machine],
        fly_secrets=_fly_secrets(),
    )
    assert any("identity mode" in error.lower() for error in unreviewed_errors)

    claimed_sha = {**legacy_health, "git_commit_sha": CANDIDATE_SHA}
    claimed_sha_errors = verifier.final_runtime_errors(
        claimed_sha,
        candidate_sha=CANDIDATE_SHA,
        promotion_accepted=False,
        expected_project_ref=PRODUCTION_REF,
        expected_migration_head=REMOTE_MIGRATION_HEAD,
        expected_migration_contract=MIGRATION_CONTRACT_FINGERPRINT,
        expected_migration_profile=MIGRATION_PROFILE,
        rollback_snapshot=snapshot,
        fly_machines=[legacy_machine],
        fly_secrets=_fly_secrets(),
    )
    assert any("missing git identity" in error.lower() for error in claimed_sha_errors)


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


def test_release_trigger_requires_closed_content_parent_and_trigger_only_diff() -> None:
    parent_sha = "b" * 40
    payload = {
        "schema_version": 1,
        "confirmation": verifier.PRODUCTION_RELEASE_CONFIRMATION,
        "release_parent_sha": parent_sha,
    }
    status = [f"A\t{verifier.PRODUCTION_RELEASE_TRIGGER_PATH}"]
    errors, resolved = verifier.production_release_trigger_errors(
        payload,
        head_sha=CANDIDATE_SHA,
        parent_shas=[parent_sha],
        changed_status_lines=status,
    )
    assert errors == []
    assert resolved == {
        "candidate_sha": CANDIDATE_SHA,
        "confirmation": verifier.PRODUCTION_RELEASE_CONFIRMATION,
        "legacy_baseline_config_sha256": "",
        "legacy_baseline_confirmation": "",
        "legacy_baseline_image_digest": "",
        "release_parent_sha": parent_sha,
    }

    invalid_cases = (
        ({**payload, "unexpected": True}, [parent_sha], status),
        ({**payload, "release_parent_sha": "c" * 40}, [parent_sha], status),
        (payload, [parent_sha, "c" * 40], status),
        (payload, [parent_sha], [*status, "M\tfly.toml"]),
        (payload, [parent_sha], [f"D\t{verifier.PRODUCTION_RELEASE_TRIGGER_PATH}"]),
        ({**payload, "confirmation": "DEPLOY SOMETHING ELSE"}, [parent_sha], status),
        (
            {
                **payload,
                "legacy_baseline_image_digest": IMAGE_DIGEST,
            },
            [parent_sha],
            status,
        ),
    )
    for invalid_payload, invalid_parents, invalid_status in invalid_cases:
        invalid_errors, _ = verifier.production_release_trigger_errors(
            invalid_payload,
            head_sha=CANDIDATE_SHA,
            parent_shas=invalid_parents,
            changed_status_lines=invalid_status,
        )
        assert invalid_errors

    reviewed_legacy = {
        **payload,
        "legacy_baseline_image_digest": IMAGE_DIGEST,
        "legacy_baseline_config_sha256": FLY_CONFIG_SHA,
        "legacy_baseline_confirmation": verifier.LEGACY_BASELINE_CONFIRMATION,
    }
    legacy_errors, legacy_resolved = verifier.production_release_trigger_errors(
        reviewed_legacy,
        head_sha=CANDIDATE_SHA,
        parent_shas=[parent_sha],
        changed_status_lines=[f"M\t{verifier.PRODUCTION_RELEASE_TRIGGER_PATH}"],
    )
    assert legacy_errors == []
    assert legacy_resolved["legacy_baseline_image_digest"] == IMAGE_DIGEST
    assert legacy_resolved["legacy_baseline_config_sha256"] == FLY_CONFIG_SHA


def test_production_workflow_is_exact_candidate_and_never_creates_or_retargets_app() -> None:
    workflow = (
        ROOT / ".github/workflows/fly_api_deploy.yml"
    ).read_text(encoding="utf-8")

    assert workflow.startswith("name: Deploy FastAPI production to Fly\n")
    assert "environment: production" in workflow
    assert "FLY_APP_NAME: juprleagues-api" in workflow
    assert (
        "EXPECTED_SUPABASE_PROJECT_REF: dnoockbwfenunhcibwfn" in workflow
    )
    assert "vars.PRODUCTION_SUPABASE_PROJECT_REF" not in workflow
    assert "SUPABASE_PROD_DATABASE_URL" not in workflow
    assert workflow.count("SUPABASE_PROD_DATABASE_READ_TOKEN") == 8
    assert workflow.count(
        "https://api.supabase.com/v1/projects/"
        "$EXPECTED_SUPABASE_PROJECT_REF/database/query/read-only"
    ) == 2
    assert workflow.count('--write-out "%{http_code}"') == 2
    assert workflow.count('if [ "$attestation_status" != "201" ]; then') == 2
    assert workflow.count("--proto '=https'") == 2
    assert workflow.count("--max-filesize 1048576") == 2
    assert workflow.count("--retry 2") == 2
    assert workflow.count("--retry-delay 2") == 2
    assert workflow.count("--retry-max-time 20") == 2
    assert workflow.count("wc -c <") == 2
    assert workflow.count('--header "Accept: application/json"') == 2
    assert workflow.count('"parameters":[]') == 1
    assert "--location" not in workflow
    assert "/database/query\"" not in workflow
    assert workflow.count(
        "${{ secrets.FLY_SSH_TOKEN || secrets.FLY_API_TOKEN }}"
    ) == 4
    assert "PRODUCTION_SOURCE_BRANCH: rollback-feb8" in workflow
    assert "ref: rollback-feb8" in workflow
    assert "github.event.repository.default_branch" not in workflow
    assert (
        "EXPECTED_MIGRATION_HEAD: "
        "${{ vars.PRODUCTION_MIGRATION_LEDGER_HEAD || "
        "'20261022000000' }}"
    ) in workflow
    assert 'if [ "$GITHUB_REF" != "refs/heads/$PRODUCTION_SOURCE_BRANCH" ]; then' in workflow
    assert '"$HEAD_SHA" != "$GITHUB_SHA"' in workflow
    assert '"$HEAD_SHA" != "$PRODUCTION_SHA"' in workflow
    assert '"$CANDIDATE_SHA" != "$HEAD_SHA"' in workflow
    assert "push:" in workflow
    assert "- .github/production-api-release.trigger" in workflow
    assert 'if [ "$EVENT_NAME" = "push" ]; then' in workflow
    assert 'elif [ "$EVENT_NAME" = "workflow_dispatch" ]; then' in workflow
    assert "deployment_verifier.py release-trigger" in workflow
    assert 'parent_args+=(--parent-sha "$parent_sha")' in workflow
    assert "git diff-tree --no-commit-id --name-status" in workflow
    assert '--changed-status "$trigger_status_path"' in workflow
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

    assert "PRODUCTION_MIGRATION_ATTESTATION_QUERY" in workflow
    assert workflow.count("--management-query-response") == 2
    assert "psql" not in workflow
    assert "config/production_migration_contract.json" in workflow
    assert "deployment_verifier.py preflight" in workflow
    assert "deployment_verifier.py migrations" in workflow
    assert "deployment_verifier.py snapshot" in workflow
    assert "deployment_verifier.py secrets" in workflow
    assert "deployment_verifier.py runtime" in workflow
    assert "deployment_verifier.py final" in workflow
    assert "deployment_verifier.py public-read" in workflow
    assert "deployment_verifier.py cors" in workflow
    assert '"JUPR_PRODUCTION_WRITE_POLICY=enabled"' in workflow
    assert '"JUPR_STAGING_WRITE_WAVE=none"' in workflow
    assert '--build-arg "JUPR_DEPLOYMENT_GIT_SHA=$GITHUB_SHA"' in workflow
    assert ".git_commit_sha == $sha and .image_build_git_sha == $sha" in workflow
    assert "flyctl machines list" in workflow
    assert "$PRODUCTION_FLY_ORIGIN/openapi.json" in workflow
    assert "$PRODUCTION_API_ORIGIN/health" in workflow
    assert 'api_origins=(' in workflow
    assert '"$PRODUCTION_FLY_ORIGIN"' in workflow
    assert '"$PRODUCTION_API_ORIGIN"' in workflow
    assert "https://not-allowed.invalid" in workflow
    assert "Preserve and verify current live production runtime" in workflow
    assert "Activate and verify reviewed candidate runtime" in workflow
    assert "Restore and attest final approved production state" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "production-rollback-snapshot.json" in workflow
    assert "steps.baseline_projection.outcome != 'skipped'" in workflow
    assert "PROMOTION_ACCEPTED:" in workflow
    assert "steps.deploy.outcome == 'success'" in workflow
    assert "steps.candidate_activation.outcome == 'success'" in workflow
    assert "steps.cors_verify.outcome == 'success'" in workflow
    assert "--promotion-accepted \"$PROMOTION_ACCEPTED\"" in workflow
    assert workflow.index("deployment_verifier.py preflight") < workflow.index(
        "flyctl deploy"
    )
    assert workflow.index("deployment_verifier.py migrations") < workflow.index(
        "flyctl deploy"
    )
    assert workflow.index("deployment_verifier.py snapshot") < workflow.index(
        "Preserve and verify current live production runtime"
    )
    assert workflow.index("Preserve and verify current live production runtime") < (
        workflow.index("flyctl deploy")
    )
    assert workflow.index("flyctl deploy") < workflow.index(
        "Activate and verify reviewed candidate runtime"
    )
    assert workflow.index("Activate and verify reviewed candidate runtime") < (
        workflow.index("Wait for exact Fly production health")
    )
    assert workflow.count("flyctl secrets set") == 3
    safe_bundles = re.findall(
        r"runtime_secrets=\(\n(?P<body>.*?)\n\s+\)",
        workflow,
        flags=re.DOTALL,
    )
    assert len(safe_bundles) == 3
    normalized_bundles = [
        [line.strip() for line in bundle.splitlines() if line.strip()]
        for bundle in safe_bundles
    ]
    assert normalized_bundles[1:] == [normalized_bundles[0]] * 2
    assert workflow.index("flyctl secrets set") < workflow.index("flyctl deploy")
    assert workflow.rindex("flyctl secrets set") > workflow.index("flyctl deploy")
    assert '--safe-convergence-only' in workflow
    assert 'if [ "$PROMOTION_ACCEPTED" != "true" ]; then' in workflow
    assert "--stage" not in workflow
    assert workflow.count("JUPR_DEPLOYMENT_GIT_SHA=") == 1
    assert "fly_immutable_image_ref" in workflow
    assert '--image "$PREDEPLOY_IMMUTABLE_IMAGE"' in workflow
    assert "flyctl config save" in workflow
    assert "fly_config_sha256" in workflow
    assert '--config "$PREDEPLOY_CONFIG"' in workflow
    assert "predeploy_fly_config_fingerprint" in workflow
    assert 'sha256sum "$PREDEPLOY_CONFIG"' not in workflow
    assert '&& [ "$DEPLOY_OUTCOME" != "skipped" ]; then' in workflow
    assert workflow.index("--no-pending-only") < workflow.index("flyctl secrets set")
    assert workflow.count("flyctl ssh console") == 3
    assert 'expected_production_feature_flags(profile="release")' in workflow
    assert 'PRODUCTION_FEATURE_PROFILE="$live_feature_profile"' in workflow
    assert 'final_feature_profile="$rollback_feature_profile"' in workflow
    assert "jq -er '.feature_profile'" in workflow
    assert "--capture-unreviewed-legacy-evidence" in workflow
    assert "legacy-unreviewed-evidence" in workflow
    assert "Stop before mutation for legacy baseline review" in workflow
    assert "BOOTSTRAP REVIEWED LEGACY ROLLBACK" in workflow
    assert workflow.index("Preserve pre-deploy baseline evidence") < workflow.index(
        "Stop before mutation for legacy baseline review"
    )
    assert workflow.index("Stop before mutation for legacy baseline review") < (
        workflow.index("flyctl secrets set")
    )


def test_staging_deploy_bakes_the_exact_candidate_sha_into_the_image() -> None:
    workflow = (
        ROOT / ".github/workflows/fly_api_staging_deploy.yml"
    ).read_text(encoding="utf-8")

    assert '--build-arg "JUPR_DEPLOYMENT_GIT_SHA=$GITHUB_SHA"' in workflow


def test_preflight_cli_never_prints_secret_values(tmp_path: Path) -> None:
    github_env = tmp_path / "github-env"
    secret_marker = "sbp_fc_do-not-print-739-secret-value"
    env = {
        **os.environ,
        **_production_env(
            FLY_API_TOKEN=secret_marker,
            SUPABASE_SERVICE_ROLE_KEY=secret_marker,
            SUPABASE_PROD_DATABASE_READ_TOKEN=secret_marker,
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
