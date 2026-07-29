#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import re

PATH = Path("tests/test_staging_deployment_config.py")


def replace_function(text: str, name: str, body: str) -> str:
    pattern = rf"def {re.escape(name)}\(.*?(?=\n\ndef |\Z)"
    updated, count = re.subn(pattern, body.rstrip(), text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"Expected one function {name}, found {count}")
    return updated


def main() -> int:
    text = PATH.read_text(encoding="utf-8")

    text = replace_function(
        text,
        "test_staging_fly_config_is_isolated_and_full_surface",
        '''def test_staging_fly_config_is_isolated_and_full_surface():
    production = _toml("fly.toml")
    staging = _toml("fly.staging.toml")

    assert production["app"] == "juprleagues-api"
    assert staging["app"] == "juprleagues-api-staging"
    assert staging["app"] != production["app"]

    env = staging["env"]
    assert env["JUPR_ENV"] == "staging"
    assert env["JUPR_EMAIL_MODE"] == "dry_run"
    assert env["JUPR_WEB_BASE_URL"] == EXPECTED_STAGING_WEB_ORIGIN
    assert env["JUPR_REQUIRE_API_AUDIT_LOG"] == "1"
    assert env["JUPR_REQUIRE_WORKER_RUN_LOG"] == "1"
    assert env["JUPR_STAGING_WRITE_WAVE"] == "open"
    assert {
        name: env[name] == "1" for name in ALL_STAGING_WRITE_FLAGS
    } == expected_write_flags("open")
    assert all(env.get(name, "0") == "0" for name in ALWAYS_DISABLED_FLAGS)
    assert all(env.get(name) == "1" for name in FULL_NEXT_ADMIN_FLAGS)
    assert env["JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES"] == "1"
    assert env["JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP"] == "1"
    assert env["JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS"] == "1"
    assert production["env"]["JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES"] == "0"
    assert production["env"]["JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP"] == "0"
    assert production["env"]["JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS"] == "0"
    assert env["JUPR_ALLOWED_ORIGIN_REGEX"].startswith("^https://jupr")
    assert env["JUPR_ALLOWED_ORIGIN_REGEX"].endswith("vercel\\.app$")
    assert re.fullmatch(
        env["JUPR_ALLOWED_ORIGIN_REGEX"],
        "https://jupr-git-example-pickleballattrespalapas1.vercel.app",
    )''',
    )

    text = replace_function(
        text,
        "test_staging_deploy_workflow_has_production_and_database_guards",
        '''def test_staging_deploy_workflow_has_production_and_database_guards():
    workflow = (ROOT / ".github/workflows/fly_api_staging_deploy.yml").read_text(encoding="utf-8")

    assert "  push:\n    branches:\n      - staging\n" in workflow
    assert (
        "SELECTED_WRITE_WAVE: ${{ github.event_name == 'workflow_dispatch' && github.event.inputs.write_wave || 'open' }}"
        in workflow
    )
    assert "${{ inputs." not in workflow
    assert "STAGING_SUPABASE_URL" in workflow
    assert "STAGING_SUPABASE_SERVICE_ROLE_KEY" in workflow
    assert "STAGING_SUPABASE_PROJECT_REF" in workflow
    assert "FLY_APP_NAME: juprleagues-api-staging" in workflow
    assert "APP_NAME_INPUT: ${{ github.event_name == 'workflow_dispatch' && github.event.inputs.app_name || 'juprleagues-api-staging' }}" in workflow
    assert "PRIMARY_REGION_INPUT: ${{ github.event_name == 'workflow_dispatch' && github.event.inputs.primary_region || 'dfw' }}" in workflow
    assert 'test "$APP_NAME_INPUT" = "juprleagues-api-staging"' in workflow
    assert 'test "$EXPECTED_SUPABASE_PROJECT_REF" = "sijpxjxvdtrehmqvirfi"' in workflow
    assert "Refusing any Supabase target except isolated staging" in workflow
    assert 'test "$GITHUB_REF" = "refs/heads/staging"' in workflow
    assert 'HEAD_SHA="$(git rev-parse HEAD)"' in workflow
    assert 'STAGING_SHA="$(git rev-parse refs/remotes/origin/staging)"' in workflow
    assert 'test "$HEAD_SHA" = "$GITHUB_SHA"' in workflow
    assert 'test "$HEAD_SHA" = "$STAGING_SHA"' in workflow
    assert workflow.index('test "$GITHUB_REF" = "refs/heads/staging"') < workflow.index(
        "flyctl secrets set --stage"
    )
    assert "--require-supabase-isolation" in workflow
    assert "--expect-full-next-admin" in workflow
    assert '--write-wave "$JUPR_STAGING_WRITE_WAVE"' in workflow
    assert "scripts/staging_write_waves.py" in workflow
    assert "expected_candidate_sha:" in workflow
    assert "orchestration_run_id:" in workflow
    assert "timeout-minutes: 55" in workflow
    for staged_safety_value in (
        '"JUPR_ENV=staging"',
        '"JUPR_EMAIL_MODE=dry_run"',
        '"JUPR_REQUIRE_API_AUDIT_LOG=1"',
        '"JUPR_REQUIRE_WORKER_RUN_LOG=1"',
        f'"JUPR_WEB_BASE_URL={EXPECTED_STAGING_WEB_ORIGIN}"',
        '"JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL=0"',
        '"JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION=0"',
    ):
        assert staged_safety_value in workflow
    assert '"SUPABASE_JWKS_URL=${SUPABASE_URL%/}/auth/v1/.well-known/jwks.json"' in workflow
    assert '"JUPR_SUPABASE_JWT_MODE=jwks"' in workflow
    assert '--build-arg "JUPR_DEPLOYMENT_GIT_SHA=$GITHUB_SHA"' in workflow
    assert f'--expected-web-origin "{EXPECTED_STAGING_WEB_ORIGIN}"' in workflow
    assert "fly.staging.toml" in workflow''',
    )

    text = replace_function(
        text,
        "test_staging_deploy_projects_always_on_read_flags_over_stale_fly_secrets",
        '''def test_staging_deploy_projects_always_on_read_flags_over_stale_fly_secrets():
    workflow = (ROOT / ".github/workflows/fly_api_staging_deploy.yml").read_text(
        encoding="utf-8"
    )

    for name in (
        "JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES",
        "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP",
    ):
        assert f'"{name}=1"' in workflow
        assert name not in ALL_STAGING_WRITE_FLAGS''',
    )

    text = replace_function(
        text,
        "test_staging_deploy_wave_choices_exactly_match_the_code_ledger",
        '''def test_staging_deploy_wave_choices_exactly_match_the_code_ledger():
    workflow = (ROOT / ".github/workflows/fly_api_staging_deploy.yml").read_text(
        encoding="utf-8"
    )
    write_wave = workflow.split("      write_wave:\n", 1)[1].split(
        "\npermissions:", 1
    )[0]
    options = write_wave.split("        options:\n", 1)[1]
    choices = tuple(
        line.strip().removeprefix("- ")
        for line in options.splitlines()
        if line.strip().startswith("- ")
    )

    assert 'default: "open"' in write_wave
    assert choices == tuple(STAGING_WRITE_WAVES)''',
    )

    PATH.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
