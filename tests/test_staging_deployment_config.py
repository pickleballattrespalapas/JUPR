from __future__ import annotations

import os
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

from scripts.check_staging_environment import FULL_NEXT_ADMIN_FLAGS


ROOT = Path(__file__).resolve().parent.parent
EXPECTED_STAGING_API_ORIGIN = "https://juprleagues-api-staging.fly.dev"
EXPECTED_STAGING_WEB_ORIGIN = "https://jupr-git-staging-pickleballattrespalapas1.vercel.app"


def _toml(path: str) -> dict:
    return tomllib.loads((ROOT / path).read_text(encoding="utf-8"))


def _staging_validation_script() -> str:
    workflow = (ROOT / ".github/workflows/staging_smoke.yml").read_text(encoding="utf-8")
    step = workflow.split("      - name: Validate staging smoke configuration\n", 1)[1]
    block = step.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0]
    return textwrap.dedent(block)


def _run_staging_validation(
    tmp_path: Path,
    *,
    api: str,
    web: str,
    bypass_secret: str = "test-bypass-secret",
) -> tuple[subprocess.CompletedProcess[str], str]:
    github_env = tmp_path / "github-env"
    env = {
        **os.environ,
        "STAGING_JUPR_API_BASE_URL": api,
        "STAGING_WEB_BASE_URL": web,
        "JUPR_EXPECTED_STAGING_API_ORIGIN": EXPECTED_STAGING_API_ORIGIN,
        "JUPR_EXPECTED_STAGING_WEB_ORIGIN": EXPECTED_STAGING_WEB_ORIGIN,
        "VERCEL_AUTOMATION_BYPASS_SECRET": bypass_secret,
        "GITHUB_ENV": str(github_env),
    }
    result = subprocess.run(
        [sys.executable, "-c", _staging_validation_script()],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    output = github_env.read_text(encoding="utf-8") if github_env.exists() else ""
    return result, output


def test_staging_fly_config_is_isolated_and_full_surface():
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
    assert env["JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT"] == "1"
    assert env["JUPR_ENABLE_AUTO_PLAYER_UPDATE_EMAILS"] == "1"
    assert env["JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL"] == "0"
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
    assert 'JUPR_REQUIRE_WORKER_RUN_LOG: "1"' in workflow
    assert "fly.staging.toml" in workflow


def test_production_cors_includes_both_public_domains():
    production_env = _toml("fly.toml")["env"]
    origins = set(production_env["JUPR_ALLOWED_ORIGINS"].split(","))
    assert "https://juprleagues.com" in origins
    assert "https://pickleballclubsandwich.com" in origins
    assert production_env["JUPR_WEB_BASE_URL"] == "https://pickleballclubsandwich.com"


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
    assert 'api = os.environ.get("STAGING_JUPR_API_BASE_URL", "").strip()' in workflow
    assert 'web = os.environ.get("STAGING_WEB_BASE_URL", "").strip()' in workflow
    assert 'api not in {expected_api, f"{expected_api}/"}' in workflow
    assert 'web not in {expected_web, f"{expected_web}/"}' in workflow
    assert 'print(f"STAGING_JUPR_API_BASE_URL={expected_api}", file=env_file)' in workflow
    assert 'print(f"STAGING_WEB_BASE_URL={expected_web}", file=env_file)' in workflow


def test_staging_smoke_scopes_bypass_secret_to_request_steps():
    workflow = (ROOT / ".github/workflows/staging_smoke.yml").read_text(encoding="utf-8")
    job_env = workflow.split("    steps:\n", 1)[0]

    assert "VERCEL_AUTOMATION_BYPASS_SECRET" not in job_env
    assert workflow.count(
        "VERCEL_AUTOMATION_BYPASS_SECRET: ${{ secrets.VERCEL_AUTOMATION_BYPASS_SECRET }}"
    ) == 3


@pytest.mark.parametrize(
    ("api", "web"),
    [
        (EXPECTED_STAGING_API_ORIGIN, EXPECTED_STAGING_WEB_ORIGIN),
        (f"  {EXPECTED_STAGING_API_ORIGIN}\t", f"\t{EXPECTED_STAGING_WEB_ORIGIN}  "),
        (f"\r\n{EXPECTED_STAGING_API_ORIGIN}/\r\n", f"\n{EXPECTED_STAGING_WEB_ORIGIN}/\n"),
    ],
)
def test_staging_smoke_normalizes_only_surrounding_whitespace_and_single_slash(
    tmp_path: Path,
    api: str,
    web: str,
):
    result, github_env = _run_staging_validation(tmp_path, api=api, web=web)

    assert result.returncode == 0, result.stdout + result.stderr
    assert github_env == (
        f"STAGING_JUPR_API_BASE_URL={EXPECTED_STAGING_API_ORIGIN}\n"
        f"STAGING_WEB_BASE_URL={EXPECTED_STAGING_WEB_ORIGIN}\n"
    )


@pytest.mark.parametrize(
    ("api", "web"),
    [
        (f"{EXPECTED_STAGING_API_ORIGIN}//", EXPECTED_STAGING_WEB_ORIGIN),
        (f"{EXPECTED_STAGING_API_ORIGIN}/health", EXPECTED_STAGING_WEB_ORIGIN),
        (f"{EXPECTED_STAGING_API_ORIGIN}?target=staging", EXPECTED_STAGING_WEB_ORIGIN),
        (
            EXPECTED_STAGING_API_ORIGIN,
            f"https://user@{EXPECTED_STAGING_WEB_ORIGIN.removeprefix('https://')}",
        ),
        (EXPECTED_STAGING_API_ORIGIN, f"{EXPECTED_STAGING_WEB_ORIGIN}/ admin"),
        ("https://juprleagues-api-\nstaging.fly.dev", EXPECTED_STAGING_WEB_ORIGIN),
        ("https://api.juprleagues.com", EXPECTED_STAGING_WEB_ORIGIN),
    ],
)
def test_staging_smoke_rejects_every_non_allowlisted_target(
    tmp_path: Path,
    api: str,
    web: str,
):
    result, github_env = _run_staging_validation(tmp_path, api=api, web=web)

    assert result.returncode == 1
    assert github_env == ""


def test_staging_smoke_rejects_whitespace_only_secret_without_printing_it(tmp_path: Path):
    result, github_env = _run_staging_validation(
        tmp_path,
        api=EXPECTED_STAGING_API_ORIGIN,
        web=EXPECTED_STAGING_WEB_ORIGIN,
        bypass_secret=" \t\r\n ",
    )

    assert result.returncode == 1
    assert "Missing Vercel automation bypass secret" in result.stdout
    assert github_env == ""


def test_staging_smoke_does_not_print_raw_invalid_input_or_bypass_secret(tmp_path: Path):
    invalid_api = "https://attacker.example/private-path"
    bypass_secret = "top-secret-do-not-print"
    result, github_env = _run_staging_validation(
        tmp_path,
        api=invalid_api,
        web=EXPECTED_STAGING_WEB_ORIGIN,
        bypass_secret=bypass_secret,
    )

    combined_output = result.stdout + result.stderr
    assert result.returncode == 1
    assert invalid_api not in combined_output
    assert bypass_secret not in combined_output
    assert github_env == ""


def test_browser_smoke_bootstraps_an_origin_scoped_cookie_without_routing():
    config = (ROOT / "apps/web/playwright.config.ts").read_text(encoding="utf-8")
    spec = (ROOT / "apps/web/e2e/staging.smoke.spec.ts").read_text(encoding="utf-8")
    helper = (ROOT / "apps/web/e2e/support/staging.ts").read_text(encoding="utf-8")

    assert "extraHTTPHeaders" not in config
    assert 'trace: protectedVercelRun ? "off" : "retain-on-failure"' in config
    assert '"https://jupr-git-staging-pickleballattrespalapas1.vercel.app"' in helper
    assert "remoteBaseUrl !== expectedStagingWebOrigin" in helper
    assert helper.count("context.request.get(bootstrapUrl") == 2
    assert "`${vercelBypassOrigin}/api/environment`" in helper
    assert helper.count('"x-vercel-protection-bypass": bypassSecret') == 1
    assert helper.count('"x-vercel-set-bypass-cookie": "true"') == 1
    assert helper.count("maxRedirects: 0") == 2
    assert helper.count("failOnStatusCode: false") == 2
    assert "headersArray().some" in helper
    assert 'name.toLowerCase() === "set-cookie"' in helper
    assert 'verification.status(), "Vercel bypass cookie was not accepted"' in helper
    assert 'verification.headers()["content-type"]' in helper
    assert "await bootstrap.dispose()" in helper
    assert "await verification.dispose()" in helper
    assert "bootstrapStagingContext(context)" in spec
    combined = spec + helper
    assert "context.route" not in combined
    assert "route.fetch" not in combined
    assert "route.fulfill" not in combined
    assert "route.continue" not in combined
    assert "fetched.body" not in combined
    assert "?x-vercel-protection-bypass" not in combined
    assert "storageState" not in combined


def test_schema_copy_workflow_has_explicit_apply_confirmation():
    workflow = (ROOT / ".github/workflows/supabase-schema-copy.yml").read_text(encoding="utf-8")
    assert "APPLY SCHEMA TO STAGING" in workflow
    assert "EXPECTED_SUPABASE_PROJECT_REF" in workflow
    assert 'if [ "$SUPABASE_PROD_DATABASE_URL" = "$SUPABASE_TEST_DATABASE_URL" ]' in workflow
