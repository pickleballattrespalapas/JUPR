import ast
import json
import os
import re
import subprocess
from pathlib import Path


CONTROLLER_PATH = ".github/workflows/staging-evidence-automation.yml"
RECOVERY_PATH = ".github/workflows/staging-write-recovery.yml"
HELPER_PATH = "scripts/staging_evidence_automation.py"
PARITY_HELPER_PATH = "scripts/run_parity_staging_wave.py"
WRITE_WAVES_PATH = "scripts/staging_write_waves.py"
FLY_CHILD_PATH = ".github/workflows/fly_api_staging_deploy.yml"
PARITY_CHILD_PATH = ".github/workflows/parity-final-evidence.yml"


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[start_index:end_index]


def _literal_assignment(source: str, name: str) -> object:
    module = ast.parse(source)
    for statement in module.body:
        if (
            isinstance(statement, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in statement.targets
            )
        ):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"Missing literal assignment: {name}")


def _controller_delegation_filter(workflow: str) -> str:
    step = _between(
        workflow,
        "      - name: Verify automated controller delegation\n",
        "\n      - name:",
    )
    match = re.search(
        r"jq -e \\\n\s+'(?P<filter>select\(.+?\) \| \.id)' "
        r'<<<"\$RUN_JSON" >/dev/null',
        step,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group("filter")


def test_controller_is_reopened_issue_only_with_exact_owner_controls() -> None:
    controller = _read(CONTROLLER_PATH)
    helper = _read(HELPER_PATH)
    trigger = _between(controller, "\non:\n", "\npermissions:")

    assert trigger == "\non:\n  issues:\n    types:\n      - reopened\n"
    assert "github.event.repository.id == 1120897513" in controller
    assert "github.event.sender.id == 250933369" in controller
    assert (
        "contains(fromJSON('[1046,1047,1048,1049,1050,1051]'), "
        "github.event.issue.number)"
    ) in controller
    assert 'REPOSITORY = "pickleballattrespalapas/JUPR"' in helper
    assert _literal_assignment(helper, "REPOSITORY_ID") == 1120897513
    assert _literal_assignment(helper, "OWNER_ID") == 250933369
    assert _literal_assignment(helper, "OWNER_LOGIN") == "pickleballattrespalapas"

    authorize = _between(controller, "  authorize:\n", "\n  orchestrate:\n")
    issue_modes = {
        int(number): mode
        for number, mode in re.findall(
            r"--issue-mode (\d+)=([a-z-]+)", authorize
        )
    }
    assert issue_modes == {
        1046: "public-read",
        1047: "public-intake-auth",
        1048: "admin-read-export",
        1049: "match-rating-writes",
        1050: "match-exclusion-recovery",
        1051: "complete-book",
    }
    assert _literal_assignment(helper, "WRITE_WAVE_BY_MODE") == {
        "public-read": "none",
        "public-intake-auth": "public-intake-auth",
        "admin-read-export": "none",
        "match-rating-writes": "tournament-live",
        "match-exclusion-recovery": "match-exclusion-recovery",
        "complete-book": "none",
    }


def test_authorization_is_secretless_and_byte_syncs_the_registry() -> None:
    controller = _read(CONTROLLER_PATH)
    authorize = _between(controller, "  authorize:\n", "\n  orchestrate:\n")

    assert "environment: staging" not in authorize
    assert "${{ secrets." not in authorize
    assert "cmp --silent" in authorize
    assert '<(git show "refs/remotes/origin/staging:$path")' in authorize
    assert "DEFAULT_SHA=" in authorize
    assert '"$GITHUB_SHA" != "$DEFAULT_SHA"' in authorize
    for registry_path in (
        FLY_CHILD_PATH,
        PARITY_CHILD_PATH,
        CONTROLLER_PATH,
        RECOVERY_PATH,
        HELPER_PATH,
        PARITY_HELPER_PATH,
        WRITE_WAVES_PATH,
    ):
        assert registry_path in authorize


def test_orchestrator_owns_global_lock_resolves_vercel_first_and_always_restores() -> None:
    controller = _read(CONTROLLER_PATH)
    orchestrate_header = _between(
        controller, "  orchestrate:\n", "    steps:\n"
    )

    assert "environment: staging" in orchestrate_header
    assert "actions: write" in orchestrate_header
    assert "group: jupr-staging-api-and-parity-evidence" in orchestrate_header
    assert "cancel-in-progress: false" in orchestrate_header

    resolve_index = controller.index(
        "- name: Resolve exact Vercel candidate while writes are disabled"
    )
    activate_index = controller.index(
        "- name: Activate exact least-privilege write wave"
    )
    evidence_index = controller.index("- name: Run exact parity evidence")
    restore_index = controller.index(
        "- name: Always restore and attest no-write mode"
    )
    assert resolve_index < activate_index < evidence_index < restore_index

    restore = _between(
        controller,
        "      - name: Always restore and attest no-write mode\n",
        "\n      - name: Record automation links\n",
    )
    assert "if: always()" in restore
    assert "--workflow fly_api_staging_deploy.yml" in restore
    assert "--input write_wave=none" in restore
    assert "--candidate-sha \"$RESTORE_SHA\"" in restore
    assert "verify-final-none" in restore


def test_controller_dispatches_only_fixed_staging_children_through_helper() -> None:
    controller = _read(CONTROLLER_PATH)
    helper = _read(HELPER_PATH)

    controller_targets = re.findall(
        r"staging_evidence_automation\.py dispatch \\\n"
        r"\s+--workflow ([a-z0-9_-]+\.yml)",
        controller,
    )
    assert controller_targets == [
        "fly_api_staging_deploy.yml",
        "parity-final-evidence.yml",
        "fly_api_staging_deploy.yml",
    ]
    assert _literal_assignment(helper, "STAGING_BRANCH") == "staging"
    assert _literal_assignment(helper, "WORKFLOW_PATHS") == {
        "fly_api_staging_deploy.yml": FLY_CHILD_PATH,
        "parity-final-evidence.yml": PARITY_CHILD_PATH,
    }
    assert '"ref": STAGING_BRANCH' in helper
    assert '"inputs": dict(inputs)' in helper
    assert "return_run_details" not in helper
    assert "Workflow dispatch target is not allowlisted." in helper
    assert (
        'dispatch.add_argument("--workflow", required=True, '
        "choices=tuple(WORKFLOW_PATHS))"
    ) in helper


def test_controller_children_use_verified_dynamic_locks() -> None:
    fly = _read(FLY_CHILD_PATH)
    parity = _read(PARITY_CHILD_PATH)

    for child in (fly, parity):
        assert "orchestration_run_id:" in child
        assert "format('jupr-staging-orchestration-child-{0}'" in child
        assert "|| 'jupr-staging-api-and-parity-evidence'" in child
        assert "cancel-in-progress: false" in child
        assert "Verify automated controller delegation" in child
        assert ".workflow_id == 320947530" in child
        assert '.name == "Automated Staging Evidence"' not in child
        assert (
            '.path == ".github/workflows/staging-evidence-automation.yml'
            '@rollback-feb8"'
            in child
        )
        assert (
            '.path == ".github/workflows/staging-evidence-automation.yml" or'
            in child
        )
        assert ".actor.id == 250933369" in child
        assert ".triggering_actor.id == 250933369" in child
        assert ".run_attempt == 1" in child
        assert ".repository.id == 1120897513" in child

    deploy_job = _between(
        fly,
        "  deploy-staging:\n",
        "    runs-on: ubuntu-latest\n",
    )
    assert "group: jupr-staging-fly-deploy" in deploy_job
    assert "cancel-in-progress: false" in deploy_job


def test_controller_children_accept_live_run_name_but_require_workflow_id() -> None:
    run = {
        "id": 30218492999,
        "workflow_id": 320947530,
        "name": "Staging evidence control #1050",
        "display_title": "Staging evidence control #1050",
        "path": ".github/workflows/staging-evidence-automation.yml",
        "event": "issues",
        "status": "in_progress",
        "run_attempt": 1,
        "head_branch": "rollback-feb8",
        "actor": {"id": 250933369},
        "triggering_actor": {"id": 250933369},
        "repository": {"id": 1120897513},
    }
    environment = {
        **os.environ,
        "ORCHESTRATION_RUN_ID": str(run["id"]),
    }

    for child_path in (FLY_CHILD_PATH, PARITY_CHILD_PATH):
        jq_filter = _controller_delegation_filter(_read(child_path))
        accepted = subprocess.run(
            ["jq", "-e", jq_filter],
            input=json.dumps(run),
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        assert accepted.returncode == 0, accepted.stderr

        wrong_workflow = {**run, "workflow_id": 999}
        rejected = subprocess.run(
            ["jq", "-e", jq_filter],
            input=json.dumps(wrong_workflow),
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        assert rejected.returncode != 0


def test_recovery_is_automatic_and_can_only_dispatch_fixed_none() -> None:
    recovery = _read(RECOVERY_PATH)
    trigger = _between(recovery, "\non:\n", "\npermissions:")
    recovery_job = _between(
        recovery,
        "  restore-if-needed:\n",
        "    steps:\n",
    )

    assert "workflow_run:" in trigger
    assert "      - Automated Staging Evidence" in trigger
    assert "      - completed" in trigger
    assert "schedule:" in trigger
    assert 'cron: "7,37 * * * *"' in trigger
    assert "workflow_dispatch:" not in trigger
    assert "issues:" not in trigger
    assert "\npermissions: {}\n" in recovery
    assert "github.event_name == 'schedule' ||" in recovery_job
    assert "github.event.workflow_run.repository.id == 1120897513" in recovery_job
    assert "github.event.workflow_run.event == 'issues'" in recovery_job
    assert (
        "github.event.workflow_run.head_branch == 'rollback-feb8'"
        in recovery_job
    )
    assert (
        "'.github/workflows/staging-evidence-automation.yml'"
        in recovery_job
    )
    assert (
        "'.github/workflows/staging-evidence-automation.yml@rollback-feb8'"
        in recovery_job
    )
    assert "github.event.workflow_run.run_attempt == 1" in recovery_job
    assert "github.event.workflow_run.actor.id == 250933369" in recovery_job
    assert (
        "github.event.workflow_run.triggering_actor.id == 250933369"
        in recovery_job
    )
    for issue_number in range(1046, 1052):
        assert f'"Staging evidence control #{issue_number}"' in recovery_job
    assert "actions: write" in recovery_job
    assert "contents: read" in recovery_job
    check = _between(
        recovery,
        "      - name: Check whether recovery is required\n",
        "\n      - name: Dispatch canonical no-write recovery\n",
    )
    assert 'if [ "$GITHUB_EVENT_NAME" = "workflow_run" ]; then' in check
    assert 'echo "required=true" >> "$GITHUB_OUTPUT"' in check

    dispatch = _between(
        recovery,
        "      - name: Dispatch canonical no-write recovery\n",
        "\n      - name: Verify final no-write state\n",
    )
    assert "--workflow fly_api_staging_deploy.yml" in dispatch
    assert "--input write_wave=none" in dispatch
    assert "--input \"expected_candidate_sha=$CANDIDATE_SHA\"" in dispatch
    assert "parity-final-evidence.yml" not in dispatch
    assert "verify-final-none" in recovery
    assert "ref: rollback-feb8" in recovery
    assert "cmp --silent" in recovery
    assert '<(git show "refs/remotes/origin/staging:$path")' in recovery
    assert PARITY_HELPER_PATH in recovery
    assert WRITE_WAVES_PATH in recovery


def test_automation_surfaces_have_only_staging_deployment_targets() -> None:
    surfaces = {
        CONTROLLER_PATH: _read(CONTROLLER_PATH),
        RECOVERY_PATH: _read(RECOVERY_PATH),
        HELPER_PATH: _read(HELPER_PATH),
    }

    for path, source in surfaces.items():
        assert "fly_api_deploy.yml" not in source, path
        assert "environment: production" not in source, path
        assert "ref: production" not in source, path
        assert "refs/heads/production" not in source, path

    assert "ref: staging" in surfaces[CONTROLLER_PATH]
    assert "ref: rollback-feb8" in surfaces[RECOVERY_PATH]
    assert 'STAGING_BRANCH = "staging"' in surfaces[HELPER_PATH]
    assert "EXPECTED_STAGING_API_ORIGIN" in surfaces[HELPER_PATH]
    assert "EXPECTED_STAGING_WEB_ORIGIN" in surfaces[HELPER_PATH]
