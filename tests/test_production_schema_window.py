from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import sys

from scripts import production_schema_window as window


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = ROOT / ".github/workflows/production_schema_window.yml"
CANDIDATE_SHA = "a" * 40
PARENT_SHA = "b" * 40
IMAGE = "registry.fly.io/juprleagues-api@sha256:" + "c" * 64
PRODUCTION_PROJECT_REF = "dnoockbwfenunhcibwfn"


def test_controller_supports_direct_script_execution() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/production_schema_window.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "validate-trigger" in result.stdout


def _fingerprint(flags: dict[str, bool]) -> str:
    return hashlib.sha256(
        "\n".join(
            f"{name}={1 if enabled else 0}"
            for name, enabled in sorted(flags.items())
        ).encode("utf-8")
    ).hexdigest()


def _health(plan: dict[str, object]) -> dict[str, object]:
    feature_flags = dict(plan["feature_flags"])
    controlled_flags = dict(plan["controlled_write_flags"])
    return {
        "ok": True,
        "service": "jupr-api",
        "environment": "production",
        "git_commit_sha": "unknown",
        "image_build_git_sha": "unknown",
        "fly_app_name": "juprleagues-api",
        "fly_image_ref": IMAGE,
        "web_origin": "https://pickleballclubsandwich.com",
        "jwt_verification_mode": "jwks",
        "jwt_verification_project_ref": PRODUCTION_PROJECT_REF,
        "expected_migration_head": "20260802150000",
        "expected_migration_contract": "d" * 64,
        "expected_migration_profile": "staging-candidate-20260802",
        "cors_allowed_origins": list(window.PRODUCTION_ALLOWED_ORIGINS),
        "cors_allowed_origin_regex": None,
        "supabase_project_ref": PRODUCTION_PROJECT_REF,
        "write_prerequisites": {
            "service_role_configured": True,
            "api_audit_required": True,
            "worker_run_log_required": True,
            "email_mode": "dry_run",
            "live_player_update_email_enabled": False,
        },
        "feature_flags": feature_flags,
        "feature_flag_fingerprint": _fingerprint(feature_flags),
        "controlled_write_flags": controlled_flags,
        "controlled_write_flag_fingerprint": _fingerprint(controlled_flags),
        "production_business_write_policy": plan["production_write_policy"],
        "write_wave": "none",
        "business_data_write_wave_active": False,
    }


def _machines(image: str = IMAGE) -> list[dict[str, object]]:
    return [{"state": "started", "config": {"image": image}}]


def test_trigger_contract_is_closed_one_parent_and_trigger_only() -> None:
    payload = {
        "schema_version": 1,
        "action": "quiesce",
        "confirmation": window.PRODUCTION_SCHEMA_WINDOW_CONFIRMATIONS[
            "quiesce"
        ],
        "release_parent_sha": PARENT_SHA,
    }
    status = [f"A\t{window.PRODUCTION_SCHEMA_WINDOW_TRIGGER_PATH}"]
    errors, resolved = window.schema_window_trigger_errors(
        payload,
        head_sha=CANDIDATE_SHA,
        parent_shas=[PARENT_SHA],
        changed_status_lines=status,
    )
    assert errors == []
    assert resolved["action"] == "quiesce"
    assert resolved["head_sha"] == CANDIDATE_SHA
    assert resolved["release_parent_sha"] == PARENT_SHA

    invalid_cases = (
        ({**payload, "unexpected": True}, [PARENT_SHA], status),
        ({key: value for key, value in payload.items() if key != "action"}, [PARENT_SHA], status),
        ({**payload, "schema_version": True}, [PARENT_SHA], status),
        ({**payload, "action": "open"}, [PARENT_SHA], status),
        ({**payload, "confirmation": "QUIESCE"}, [PARENT_SHA], status),
        ({**payload, "release_parent_sha": "e" * 40}, [PARENT_SHA], status),
        (payload, [PARENT_SHA, "e" * 40], status),
        (payload, [PARENT_SHA], [*status, "M\tfly.toml"]),
        (
            payload,
            [PARENT_SHA],
            [f"D\t{window.PRODUCTION_SCHEMA_WINDOW_TRIGGER_PATH}"],
        ),
    )
    for invalid_payload, parents, changed_status in invalid_cases:
        invalid_errors, _ = window.schema_window_trigger_errors(
            invalid_payload,
            head_sha=CANDIDATE_SHA,
            parent_shas=parents,
            changed_status_lines=changed_status,
        )
        assert invalid_errors


def test_restore_trigger_uses_its_distinct_owner_confirmation() -> None:
    payload = {
        "schema_version": 1,
        "action": "restore_baseline",
        "confirmation": window.PRODUCTION_SCHEMA_WINDOW_CONFIRMATIONS[
            "restore_baseline"
        ],
        "release_parent_sha": PARENT_SHA,
    }
    errors, resolved = window.schema_window_trigger_errors(
        payload,
        head_sha=CANDIDATE_SHA,
        parent_shas=[PARENT_SHA],
        changed_status_lines=[
            f"M\t{window.PRODUCTION_SCHEMA_WINDOW_TRIGGER_PATH}"
        ],
    )
    assert errors == []
    assert resolved["action"] == "restore_baseline"


def test_quiesce_plan_is_closed_and_fail_closed() -> None:
    plan = window.transition_plan("quiesce")

    assert plan["production_write_policy"] == "read_only"
    assert plan["settings"]["JUPR_EMAIL_MODE"] == "dry_run"
    assert plan["settings"]["JUPR_STAGING_WRITE_WAVE"] == "none"
    assert plan["settings"]["JUPR_PRODUCTION_WRITE_POLICY"] == "read_only"
    assert set(plan["feature_flags"]) == set(window.PRODUCTION_FEATURE_FLAGS)
    assert not any(plan["feature_flags"].values())
    assert set(plan["controlled_write_flags"]) == set(
        window.ALL_STAGING_WRITE_FLAGS
    )
    assert not any(plan["controlled_write_flags"].values())
    assert "SUPABASE_URL" not in plan["settings"]
    assert "JUPR_ALLOWED_ORIGINS" not in plan["settings"]
    assert "JUPR_ENV" not in plan["settings"]


def test_restore_plan_is_exact_live_tournament_baseline() -> None:
    plan = window.transition_plan("restore_baseline")
    enabled = {
        name for name, is_enabled in plan["feature_flags"].items() if is_enabled
    }

    assert plan["production_write_policy"] == "enabled"
    assert enabled == window.PRODUCTION_LIVE_TOURNAMENT_BASELINE_ENABLED_FLAGS
    assert not (
        enabled & window.PRODUCTION_CANDIDATE_ONLY_LEAGUE_FLAGS
    )
    for name in window.PRODUCTION_CANDIDATE_ONLY_LEAGUE_FLAGS:
        assert plan["settings"][name] == "0"
    assert plan["settings"]["JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL"] == "0"
    assert plan["settings"]["JUPR_EMAIL_MODE"] == "dry_run"


def test_secret_state_rejects_unknown_pending_names() -> None:
    deployed = [
        {"Name": "JUPR_EMAIL_MODE", "DeploymentStatus": "deployed"},
        {
            "Name": "JUPR_PRODUCTION_WRITE_POLICY",
            "DeploymentStatus": "deployed",
        },
    ]
    errors, pending, summary = window.secret_deployment_state(deployed)
    assert errors == []
    assert pending == []
    assert summary["recognized_secret_count"] == 2

    known = [
        {
            "Name": "JUPR_PRODUCTION_WRITE_POLICY",
            "DeploymentStatus": "pending",
        }
    ]
    errors, pending, _ = window.secret_deployment_state(
        known,
        allowed_pending_names={"JUPR_PRODUCTION_WRITE_POLICY"},
    )
    assert errors == []
    assert pending == ["JUPR_PRODUCTION_WRITE_POLICY=pending"]

    unknown = [
        {"Name": "UNRELATED_SECRET", "DeploymentStatus": "pending"}
    ]
    errors, pending, summary = window.secret_deployment_state(
        unknown,
        allowed_pending_names={"JUPR_PRODUCTION_WRITE_POLICY"},
    )
    assert errors
    assert pending == []
    assert summary["unknown_pending"] == ["UNRELATED_SECRET=pending"]


def test_transition_verifier_preserves_identity_cors_supabase_and_image() -> None:
    before_plan = window.transition_plan("restore_baseline")
    after_plan = window.transition_plan("quiesce")
    before = _health(before_plan)
    after = _health(after_plan)

    errors, summary = window.transition_verification_errors(
        before_health=before,
        after_health=after,
        before_machines=_machines(),
        after_machines=_machines(),
        plan=after_plan,
        expected_project_ref=PRODUCTION_PROJECT_REF,
    )
    assert errors == []
    assert summary["fly_image_identity_preserved"] is True
    assert summary["supabase_project_ref"] == PRODUCTION_PROJECT_REF

    changed = dict(after)
    changed["supabase_project_ref"] = "x" * 20
    changed["cors_allowed_origins"] = ["https://not-allowed.invalid"]
    errors, _ = window.transition_verification_errors(
        before_health=before,
        after_health=changed,
        before_machines=_machines(),
        after_machines=_machines("registry.fly.io/juprleagues-api@sha256:" + "f" * 64),
        plan=after_plan,
        expected_project_ref=PRODUCTION_PROJECT_REF,
    )
    assert errors
    assert any("Supabase" in error or "supabase" in error for error in errors)
    assert any("CORS" in error for error in errors)
    assert any("image identity changed" in error for error in errors)


def test_transition_verifier_rejects_a_nonexact_feature_projection() -> None:
    plan = window.transition_plan("restore_baseline")
    before = _health(plan)
    after = _health(plan)
    after_flags = dict(after["feature_flags"])
    after_flags["JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER"] = True
    after["feature_flags"] = after_flags
    after["feature_flag_fingerprint"] = _fingerprint(after_flags)

    errors, _ = window.transition_verification_errors(
        before_health=before,
        after_health=after,
        before_machines=_machines(),
        after_machines=_machines(),
        plan=plan,
        expected_project_ref=PRODUCTION_PROJECT_REF,
    )
    assert any("exact transition plan" in error for error in errors)


def test_transition_verifier_accepts_a_legacy_health_inventory_subset() -> None:
    plan = window.transition_plan("quiesce")
    before = _health(window.transition_plan("restore_baseline"))
    after = _health(plan)
    for name in window.PRODUCTION_CANDIDATE_ONLY_LEAGUE_FLAGS:
        before["feature_flags"].pop(name)
        after["feature_flags"].pop(name)
        before["controlled_write_flags"].pop(name, None)
        after["controlled_write_flags"].pop(name, None)
    before["feature_flag_fingerprint"] = _fingerprint(before["feature_flags"])
    after["feature_flag_fingerprint"] = _fingerprint(after["feature_flags"])
    before["controlled_write_flag_fingerprint"] = _fingerprint(
        before["controlled_write_flags"]
    )
    after["controlled_write_flag_fingerprint"] = _fingerprint(
        after["controlled_write_flags"]
    )

    errors, _ = window.transition_verification_errors(
        before_health=before,
        after_health=after,
        before_machines=_machines(),
        after_machines=_machines(),
        plan=plan,
        expected_project_ref=PRODUCTION_PROJECT_REF,
    )
    assert errors == []


def test_workflow_is_trigger_only_owner_only_and_production_protected() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert workflow.startswith(
        "name: Transition production schema migration window\n"
    )
    assert "  push:\n" in workflow
    assert "      - rollback-feb8" in workflow
    assert "      - .github/production-schema-window.trigger" in workflow
    assert "workflow_dispatch:" not in workflow
    assert "schedule:" not in workflow
    assert "workflow_run:" not in workflow
    assert "environment: production" in workflow
    assert "github.actor_id == 250933369" in workflow
    assert "github.repository == 'pickleballattrespalapas/JUPR'" in workflow
    assert "group: jupr-production-api-deploy" in workflow
    assert "ref: rollback-feb8" in workflow
    assert "fetch-depth: 0" in workflow
    assert "persist-credentials: false" in workflow


def test_workflow_requires_closed_one_parent_trigger_provenance() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert '"$HEAD_SHA" != "$GITHUB_SHA"' in workflow
    assert '"$HEAD_SHA" != "$PRODUCTION_SHA"' in workflow
    assert "git rev-list --parents -n 1" in workflow
    assert "must have exactly one parent" in workflow
    assert "git diff-tree" in workflow
    assert "--name-status" in workflow
    assert "production_schema_window.py validate-trigger" in workflow
    assert "--changed-status-file" in workflow
    assert "--parent-sha" in workflow
    assert not (ROOT / window.PRODUCTION_SCHEMA_WINDOW_TRIGGER_PATH).exists()


def test_workflow_only_transitions_runtime_and_attests_both_sides() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "production_schema_window.py build-plan" in workflow
    assert "flyctl secrets set" in workflow
    assert 'flyctl ssh console \\\n' in workflow
    assert workflow.count("$PRODUCTION_FLY_ORIGIN/health") >= 2
    assert "production_schema_window.py verify-before" in workflow
    assert "production_schema_window.py verify-transition" in workflow
    assert "--allowed-settings" in workflow
    assert "Unknown pending production secret" in workflow
    assert "Verify exact production CORS remains active" in workflow
    assert "production-schema-window-before-${{ github.run_id }}" in workflow
    assert "production-schema-window-evidence-${{ github.run_id }}" in workflow
    assert "schema-window-health-before.json" in workflow
    assert "schema-window-health-after.json" in workflow
    assert "schema-window-machines-before.json" in workflow
    assert "schema-window-machines-after.json" in workflow

    forbidden = (
        "flyctl deploy",
        "flyctl apps create",
        "psql ",
        "SUPABASE_PROD_DATABASE_URL",
        "SUPABASE_SERVICE_ROLE_KEY",
        "supabase/migrations",
        "apply_migration",
        "db_migrate",
        "|| true",
    )
    for value in forbidden:
        assert value not in workflow
