import ast
from pathlib import Path


CONTROLLER_PATH = ".github/workflows/staging-evidence-automation.yml"
RECOVERY_PATH = ".github/workflows/staging-write-recovery.yml"
HELPER_PATH = "scripts/staging_evidence_automation.py"
FLY_PATH = ".github/workflows/fly_api_staging_deploy.yml"
PARITY_PATH = ".github/workflows/parity-final-evidence.yml"


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


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


def test_issue_driven_evidence_controller_is_retired() -> None:
    controller = _read(CONTROLLER_PATH)

    assert controller.startswith("name: Retired Automated Staging Evidence\n")
    assert "workflow_dispatch:" in controller
    assert "issues:" not in controller
    assert "schedule:" not in controller
    assert "workflow_run:" not in controller
    assert "Issue-driven staging evidence orchestration is retired." in controller
    assert "Staging remains open for ongoing acceptance testing with write_wave=open." in controller
    assert "flyctl" not in controller


def test_emergency_disable_is_the_only_fixed_none_control() -> None:
    recovery = _read(RECOVERY_PATH)

    assert recovery.startswith("name: Staging Emergency Write Disable\n")
    assert "workflow_dispatch:" in recovery
    assert "github.actor_id == 250933369" in recovery
    assert "DISABLE STAGING WRITES" in recovery
    assert recovery.count("--input write_wave=none") == 1
    assert "schedule:" not in recovery
    assert "workflow_run:" not in recovery
    assert "issues:" not in recovery
    assert "parity-final-evidence.yml" not in recovery


def test_normal_fly_deploy_defaults_to_open_and_remains_staging_only() -> None:
    fly = _read(FLY_PATH)

    assert "|| 'open'" in fly
    assert "|| 'none'" not in fly
    assert 'default: "open"' in fly
    assert "FLY_APP_NAME: juprleagues-api-staging" in fly
    assert 'test "$EXPECTED_SUPABASE_PROJECT_REF" = "sijpxjxvdtrehmqvirfi"' in fly
    assert "Refusing any Supabase target except isolated staging" in fly
    assert '"JUPR_EMAIL_MODE=dry_run"' in fly
    assert '"JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL=0"' in fly
    assert '"JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION=0"' in fly
    assert "environment: production" not in fly
    assert "dnoockbwfenunhcibwfn" not in fly


def test_dispatch_helper_remains_allowlisted_to_staging_children() -> None:
    helper = _read(HELPER_PATH)

    assert _literal_assignment(helper, "STAGING_BRANCH") == "staging"
    assert _literal_assignment(helper, "WORKFLOW_PATHS") == {
        "fly_api_staging_deploy.yml": FLY_PATH,
        "parity-final-evidence.yml": PARITY_PATH,
    }
    assert 'REPOSITORY = "pickleballattrespalapas/JUPR"' in helper
    assert _literal_assignment(helper, "REPOSITORY_ID") == 1120897513
    assert _literal_assignment(helper, "OWNER_ID") == 250933369
    assert '"ref": STAGING_BRANCH' in helper
    assert "Workflow dispatch target is not allowlisted." in helper


def test_automation_surfaces_never_target_production_or_live_email() -> None:
    for path in (CONTROLLER_PATH, RECOVERY_PATH, HELPER_PATH, FLY_PATH, PARITY_PATH):
        source = _read(path)
        assert "fly_api_deploy.yml" not in source, path
        assert "environment: production" not in source, path
        assert "refs/heads/production" not in source, path
        assert "dnoockbwfenunhcibwfn" not in source, path

    fly = _read(FLY_PATH)
    assert '"JUPR_EMAIL_MODE=dry_run"' in fly
    assert '"JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL=0"' in fly
    assert '"JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION=0"' in fly
