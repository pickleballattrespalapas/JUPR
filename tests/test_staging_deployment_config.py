from __future__ import annotations

import re
import tomllib
from pathlib import Path

from scripts.check_staging_environment import FULL_NEXT_ADMIN_FLAGS


ROOT = Path(__file__).resolve().parent.parent


def _toml(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text(encoding="utf-8"))


def test_staging_fly_config_is_isolated_and_full_surface():
    production = _toml("fly.toml")
    staging = _toml("fly.staging.toml")

    assert production["app"] == "juprleagues-api"
    assert staging["app"] == "juprleagues-api-staging"
    assert staging["app"] != production["app"]

    env = staging["env"]
    assert env["JUPR_ENV"] == "staging"
    assert env["JUPR_EMAIL_MODE"] == "dry_run"
    assert env["JUPR_REQUIRE_API_AUDIT_LOG"] == "1"
    assert env["JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT"] == "1"
    assert env["JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS"] == "1"
    assert all(env.get(name) == "1" for name in FULL_NEXT_ADMIN_FLAGS)
    assert env["JUPR_ALLOWED_ORIGIN_REGEX"].startswith("^https://jupr")
    assert env["JUPR_ALLOWED_ORIGIN_REGEX"].endswith("vercel\\.app$")
    assert re.fullmatch(
        env["JUPR_ALLOWED_ORIGIN_REGEX"],
        "https://jupr-git-example-pickleballattrespalapas1.vercel.app",
    )


def test_staging_deploy_workflow_has_production_and_database_guards():
    workflow = (ROOT / ".github/workflows/fly_api_staging_deploy.yml").read_text(encoding="utf-8")

    assert "STAGING_SUPABASE_URL" in workflow
    assert "STAGING_SUPABASE_SERVICE_ROLE_KEY" in workflow
    assert "STAGING_SUPABASE_PROJECT_REF" in workflow
    assert 'if [ "$FLY_APP_NAME" = "juprleagues-api" ]' in workflow
    assert "--require-supabase-isolation" in workflow
    assert "--expect-full-next-admin" in workflow
    assert "fly.staging.toml" in workflow


def test_production_cors_includes_both_public_domains():
    origins = set(_toml("fly.toml")["env"]["JUPR_ALLOWED_ORIGINS"].split(","))
    assert "https://juprleagues.com" in origins
    assert "https://pickleballclubsandwich.com" in origins


def test_only_one_staging_smoke_workflow_remains():
    workflows = list((ROOT / ".github/workflows").glob("*.yml"))
    staging_smokes = [
        path
        for path in workflows
        if path.read_text(encoding="utf-8").startswith("name: Staging Smoke\n")
    ]
    assert [path.name for path in staging_smokes] == ["staging_smoke.yml"]


def test_staging_smoke_validates_exact_isolated_targets_before_requests():
    workflow = (ROOT / ".github/workflows/staging_smoke.yml").read_text(encoding="utf-8")

    validation = workflow.index("- name: Validate staging smoke configuration")
    checkout = workflow.index("- name: Check out repository")
    public_smoke = workflow.index("- name: Run public smoke checks")
    assert validation < checkout < public_smoke
    assert 'JUPR_EXPECTED_STAGING_API_ORIGIN: "https://juprleagues-api-staging.fly.dev"' in workflow
    assert 'JUPR_EXPECTED_STAGING_WEB_ORIGIN: "https://jupr-git-staging-pickleballattrespalapas1.vercel.app"' in workflow
    assert "Missing Vercel automation bypass secret" in workflow
    assert "Unsafe staging API URL" in workflow
    assert "Unsafe staging web URL" in workflow


def test_staging_smoke_scopes_bypass_secret_to_request_steps():
    workflow = (ROOT / ".github/workflows/staging_smoke.yml").read_text(encoding="utf-8")
    job_env = workflow.split("    steps:\n", 1)[0]

    assert "VERCEL_AUTOMATION_BYPASS_SECRET" not in job_env
    assert workflow.count(
        "VERCEL_AUTOMATION_BYPASS_SECRET: ${{ secrets.VERCEL_AUTOMATION_BYPASS_SECRET }}"
    ) == 3


def test_browser_smoke_scopes_bypass_headers_and_disables_protected_traces():
    config = (ROOT / "apps/web/playwright.config.ts").read_text(encoding="utf-8")
    spec = (ROOT / "apps/web/e2e/staging.smoke.spec.ts").read_text(encoding="utf-8")

    assert "extraHTTPHeaders" not in config
    assert 'trace: protectedVercelRun ? "off" : "retain-on-failure"' in config
    assert "context.route(`${vercelBypassOrigin}/**`" in spec
    assert 'parsed.hostname.toLowerCase().endsWith(".vercel.app")' in spec
    assert '"x-vercel-protection-bypass": bypassSecret' in spec
    assert "route.continue" not in spec
    assert "maxRedirects: 0" in spec
    assert "route.fulfill({ response })" in spec


def test_schema_copy_workflow_has_explicit_apply_confirmation():
    workflow = (ROOT / ".github/workflows/supabase-schema-copy.yml").read_text(encoding="utf-8")
    assert "APPLY SCHEMA TO STAGING" in workflow
    assert "EXPECTED_SUPABASE_PROJECT_REF" in workflow
    assert 'if [ "$SUPABASE_PROD_DATABASE_URL" = "$SUPABASE_TEST_DATABASE_URL" ]' in workflow
