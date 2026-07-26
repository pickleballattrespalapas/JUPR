import ast
from pathlib import Path

import scripts.staging_write_session as write_session
from scripts.staging_write_waves import NO_WRITE_WAVE, STAGING_WRITE_WAVES


WORKFLOW_PATH = ".github/workflows/staging-write-session.yml"
RECOVERY_PATH = ".github/workflows/staging-write-recovery.yml"
FLY_PATH = ".github/workflows/fly_api_staging_deploy.yml"
EVIDENCE_PATH = ".github/workflows/staging-evidence-automation.yml"
HELPER_PATH = "scripts/staging_write_session.py"


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


def test_controller_has_only_exact_owner_issue_events() -> None:
    workflow = _read(WORKFLOW_PATH)
    helper = _read(HELPER_PATH)
    trigger = _between(workflow, "\non:\n", "\npermissions:")

    assert trigger == (
        "\non:\n"
        "  issues:\n"
        "    types:\n"
        "      - reopened\n"
        "      - edited\n"
    )
    assert "workflow_dispatch:" not in workflow
    assert "github.event.repository.id == 1120897513" in workflow
    assert "github.event.sender.id == 250933369" in workflow
    assert "github.event.issue.number == 1062" in workflow
    assert "github.event.issue.state == 'open'" in workflow
    assert (
        "github.event.issue.title == "
        "'Protected staging write session control'"
    ) in workflow
    assert _literal_assignment(helper, "CONTROL_ISSUE_NUMBER") == 1062
    assert (
        _literal_assignment(helper, "CONTROL_ISSUE_TITLE")
        == "Protected staging write session control"
    )
    assert write_session.OWNER_ID == 250933369
    assert write_session.REPOSITORY_ID == 1120897513


def test_controller_syncs_every_protected_registry_dependency() -> None:
    workflow = _read(WORKFLOW_PATH)
    authorize = _between(workflow, "  authorize:\n", "\n  apply-command:\n")

    assert "environment: staging" not in authorize
    assert "${{ secrets." not in authorize
    assert "ref: rollback-feb8" in authorize
    assert "cmp --silent" in authorize
    assert '<(git show "refs/remotes/origin/staging:$path")' in authorize
    for path in (
        FLY_PATH,
        EVIDENCE_PATH,
        RECOVERY_PATH,
        WORKFLOW_PATH,
        "scripts/staging_evidence_automation.py",
        HELPER_PATH,
        "scripts/staging_write_waves.py",
    ):
        assert path in authorize


def test_all_existing_waves_are_supported_but_never_combined() -> None:
    helper = _read(HELPER_PATH)
    active = tuple(
        wave for wave in STAGING_WRITE_WAVES if wave != NO_WRITE_WAVE
    )

    assert "ACTIVE_WRITE_WAVES" in helper
    assert len(active) == len(STAGING_WRITE_WAVES) - 1
    assert '"all"' not in helper
    assert "enable-all" not in helper.lower()
    assert "write_wave is not allowlisted" in helper
    assert (
        "advance must name two distinct active allowlisted waves"
        in helper
    )


def test_open_advance_close_are_candidate_bound_and_fail_closed() -> None:
    workflow = _read(WORKFLOW_PATH)
    apply_job = _between(
        workflow,
        "  apply-command:\n",
        "\n  wait-for-lease:\n",
    )

    assert "group: jupr-staging-api-and-parity-evidence" in apply_job
    assert "cancel-in-progress: false" in apply_job
    assert "ref: staging" in apply_job
    assert '"$(git rev-parse HEAD)" != "$CANDIDATE_SHA"' in apply_job
    assert "--expected-write-wave \"$EXPECTED_WRITE_WAVE\"" in apply_job
    assert "live-command-before-activation.json" in apply_job
    assert '.session_nonce == $nonce' in apply_job
    assert (
        apply_job.index("- name: Restore none before advance or close")
        < apply_job.index("- name: Activate exactly one allowlisted write wave")
    )
    transition = _between(
        apply_job,
        "      - name: Restore none before advance or close\n",
        "\n      - name: Ensure lease remains live before activation\n",
    )
    assert "--input write_wave=none" in transition
    assert "verify-final-none" in transition
    activation = _between(
        apply_job,
        "      - name: Activate exactly one allowlisted write wave\n",
        "\n      - name: Attest active wave and dry-run email\n",
    )
    assert '--input "write_wave=$WRITE_WAVE"' in activation
    assert '--input "orchestration_run_id=$GITHUB_RUN_ID"' in activation
    failure = _between(
        apply_job,
        "      - name: Restore none after any failed transition\n",
        "\n      - name: Record failed command and close control issue\n",
    )
    assert "if: failure()" in failure
    assert "--input write_wave=none" in failure
    assert "verify-final-none" in failure
    assert "-f state=closed" in apply_job


def test_lease_wait_does_not_hold_lock_and_expiry_rechecks_nonce() -> None:
    workflow = _read(WORKFLOW_PATH)
    waiter = _between(
        workflow,
        "  wait-for-lease:\n",
        "\n  expire-lease:\n",
    )
    expiry = workflow[workflow.index("  expire-lease:\n") :]

    assert "concurrency:" not in waiter
    assert "permissions: {}" in waiter
    assert "timeout-minutes: 65" in waiter
    assert 'date -u -d "$LEASE_EXPIRES_AT" +%s' in waiter
    assert "group: jupr-staging-api-and-parity-evidence" in expiry
    assert "ref: ${{ github.sha }}" in expiry
    assert "ref: rollback-feb8" not in expiry
    assert "staging_write_session.py should-expire" in expiry
    assert '--session-nonce "$SESSION_NONCE"' in expiry
    assert "steps.lease.outputs.expire == 'true'" in expiry
    assert "--input write_wave=none" in expiry
    assert "verify-final-none" in expiry
    assert "A newer owner command superseded this lease" in expiry


def test_safety_recovery_understands_valid_lease_and_failed_controller() -> None:
    recovery = _read(RECOVERY_PATH)
    trigger = _between(recovery, "\non:\n", "\npermissions:")
    check = _between(
        recovery,
        "      - name: Check whether recovery is required\n",
        "\n      - name: Dispatch canonical no-write recovery\n",
    )

    assert "      - Protected Staging Write Session" in trigger
    assert 'cron: "7,37 * * * *"' in trigger
    assert "staging-write-session.yml@rollback-feb8" in recovery
    assert "write_session_controller_failed" in check
    assert "staging_write_session.py inspect" in check
    assert "valid_exact_write_session_lease" in check
    assert "invalid_or_expired_write_session" in check
    assert "lease_inspection_failed" in check
    assert "verify-final-none" in check
    assert "issues: write" in recovery
    assert "issues/1062/comments" in recovery
    assert "issues/1062" in recovery


def test_fly_child_delegation_binds_exact_active_controller_path() -> None:
    fly = _read(FLY_PATH)
    delegation = _between(
        fly,
        "      - name: Verify automated controller delegation\n",
        "\n      - name: Configure one least-privilege staging write wave\n",
    )

    assert (
        "/actions/workflows/staging-write-session.yml" in delegation
    )
    assert (
        '.path == ".github/workflows/staging-write-session.yml"'
        in delegation
    )
    assert (
        '.path == ".github/workflows/staging-write-session.yml@rollback-feb8"'
        in delegation
    )
    assert (
        ".workflow_id == (env.WRITE_SESSION_WORKFLOW_ID | tonumber)"
        in delegation
    )
    assert '.state == "active"' in delegation
    assert ".actor.id == 250933369" in delegation
    assert ".triggering_actor.id == 250933369" in delegation
    assert ".repository.id == 1120897513" in delegation


def test_controller_surfaces_cannot_target_production_or_live_email() -> None:
    surfaces = {
        path: _read(path)
        for path in (
            WORKFLOW_PATH,
            RECOVERY_PATH,
            HELPER_PATH,
        )
    }
    for path, source in surfaces.items():
        assert "fly_api_deploy.yml" not in source, path
        assert "environment: production" not in source, path
        assert "refs/heads/production" not in source, path

    assert "email_mode" in surfaces[HELPER_PATH]
    assert '"dry_run"' in surfaces[HELPER_PATH]
    assert "live_player_update_email_enabled" in surfaces[HELPER_PATH]
    assert "public_live_production_override_enabled" in surfaces[HELPER_PATH]
