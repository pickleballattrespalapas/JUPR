from pathlib import Path


WORKFLOW_PATH = ".github/workflows/staging-write-session.yml"
RECOVERY_PATH = ".github/workflows/staging-write-recovery.yml"
FLY_PATH = ".github/workflows/fly_api_staging_deploy.yml"


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_temporary_write_session_controller_is_retired() -> None:
    workflow = _read(WORKFLOW_PATH)

    assert workflow.startswith("name: Retired Staging Write Session\n")
    assert "workflow_dispatch:" in workflow
    assert "issues:" not in workflow
    assert "schedule:" not in workflow
    assert "workflow_run:" not in workflow
    assert "Temporary staging write leases are retired." in workflow
    assert "Staging is read-only at rest; automatic deployments use write_wave=none." in workflow
    assert "one explicitly approved named wave" in workflow
    assert "flyctl" not in workflow
    assert "SUPABASE" not in workflow


def test_emergency_disable_is_manual_owner_only_and_staging_only() -> None:
    recovery = _read(RECOVERY_PATH)

    assert recovery.startswith("name: Staging Emergency Write Disable\n")
    assert "workflow_dispatch:" in recovery
    assert "schedule:" not in recovery
    assert "workflow_run:" not in recovery
    assert "issues:" not in recovery
    assert "github.actor_id == 250933369" in recovery
    assert "environment: staging" in recovery
    assert "ref: staging" in recovery
    assert "DISABLE STAGING WRITES" in recovery
    assert "--input write_wave=none" in recovery
    assert "--workflow fly_api_staging_deploy.yml" in recovery
    assert "verify-final-none" in recovery
    assert "juprleagues-api" not in recovery
    assert "dnoockbwfenunhcibwfn" not in recovery
    assert "environment: production" not in recovery


def test_normal_staging_deploy_is_fail_closed_and_email_stays_dry_run() -> None:
    fly = _read(FLY_PATH)

    assert "  push:\n    branches:\n      - staging\n" in fly
    assert "|| 'none'" in fly
    assert "|| 'open'" not in fly
    assert 'default: "none"' in fly
    assert "SELECTED_WRITE_WAVE:" in fly
    assert "JUPR_EMAIL_MODE: dry_run" in fly
    assert '"JUPR_EMAIL_MODE=dry_run"' in fly
    assert '"JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL=0"' in fly
    assert '"JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION=0"' in fly
    assert "FLY_APP_NAME: juprleagues-api-staging" in fly
    assert 'test "$EXPECTED_SUPABASE_PROJECT_REF" = "sijpxjxvdtrehmqvirfi"' in fly
    assert "Refusing any Supabase target except isolated staging" in fly
    assert "environment: production" not in fly
    assert "dnoockbwfenunhcibwfn" not in fly
