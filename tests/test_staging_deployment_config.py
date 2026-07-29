from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

from scripts.check_staging_environment import FULL_NEXT_ADMIN_FLAGS
from scripts.run_parity_staging_wave import WAVES
from scripts.staging_write_waves import (
    ALL_STAGING_WRITE_FLAGS,
    ALWAYS_DISABLED_FLAGS,
    OPEN_WRITE_WAVE,
    STAGING_WRITE_WAVES,
    configure_fly_staging,
    expected_write_flags,
)


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


def _browser_evidence_script() -> str:
    workflow = (ROOT / ".github/workflows/staging_smoke.yml").read_text(encoding="utf-8")
    step = workflow.split(
        "      - name: Reject incomplete browser public-read evidence\n", 1
    )[1]
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


def _run_browser_evidence_validation(
    tmp_path: Path, report: object | None
) -> subprocess.CompletedProcess[str]:
    report_path = tmp_path / "apps/web/test-results/public-read-report.json"
    if report is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report), encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-c", _browser_evidence_script()],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "EXPECTED_PUBLIC_READ_TESTS": "78",
            "PYTHONPATH": str(ROOT),
        },
    )


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
    )

def test_staging_wave_configurator_opens_only_the_selected_wave(tmp_path: Path):
    config = tmp_path / "fly.staging.toml"
    config.write_text((ROOT / "fly.staging.toml").read_text(encoding="utf-8"), encoding="utf-8")

    configure_fly_staging(config, wave="public-live")

    env = tomllib.loads(config.read_text(encoding="utf-8"))["env"]
    expected = expected_write_flags("public-live")
    assert env["JUPR_STAGING_WRITE_WAVE"] == "public-live"
    assert {name: env[name] == "1" for name in ALL_STAGING_WRITE_FLAGS} == expected
    assert expected["JUPR_ENABLE_PUBLIC_LIVE_WRITES"] is True


def test_communications_wave_opens_mutations_without_toggling_read_surfaces(
    tmp_path: Path,
):
    config = tmp_path / "fly.staging.toml"
    config.write_text(
        (ROOT / "fly.staging.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    configure_fly_staging(config, wave="communications")

    env = tomllib.loads(config.read_text(encoding="utf-8"))["env"]
    expected = expected_write_flags("communications")
    enabled = {name for name, value in expected.items() if value}
    assert enabled == {
        "JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT",
        "JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS",
    }
    assert "JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES" not in expected
    assert "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP" not in expected
    assert env["JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES"] == "1"
    assert env["JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP"] == "1"


def test_staging_deploy_workflow_has_production_and_database_guards():
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
    assert "fly.staging.toml" in workflow

def test_staging_deploy_projects_always_on_read_flags_over_stale_fly_secrets():
    workflow = (ROOT / ".github/workflows/fly_api_staging_deploy.yml").read_text(
        encoding="utf-8"
    )

    for name in (
        "JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES",
        "JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP",
    ):
        assert f'"{name}=1"' in workflow
        assert name not in ALL_STAGING_WRITE_FLAGS

def test_staging_deploy_wave_choices_exactly_match_the_code_ledger():
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
    assert choices[0] == "open"
    assert set(choices) == {OPEN_WRITE_WAVE, *STAGING_WRITE_WAVES}
    assert len(choices) == len(STAGING_WRITE_WAVES) + 1

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


def test_staging_smoke_shares_the_deploy_and_evidence_lock():
    workflow = (ROOT / ".github/workflows/staging_smoke.yml").read_text(encoding="utf-8")
    assert "group: jupr-staging-api-and-parity-evidence" in workflow
    assert "cancel-in-progress: false" in workflow


def test_staging_smoke_runs_only_the_strict_public_read_manifest():
    workflow = (ROOT / ".github/workflows/staging_smoke.yml").read_text(encoding="utf-8")
    package = json.loads((ROOT / "apps/web/package.json").read_text(encoding="utf-8"))
    command = package["scripts"]["test:e2e:public-read"].split()
    expected_specs = tuple(WAVES["public-read"][0]["specs"])

    assert f'default: "{EXPECTED_STAGING_API_ORIGIN}"' in workflow
    assert f'default: "{EXPECTED_STAGING_WEB_ORIGIN}"' in workflow
    assert "default: false" in workflow.split("      allow_live_unconfigured:\n", 1)[1].split(
        "\npermissions:", 1
    )[0]
    assert "run: npm run test:e2e:public-read -- --reporter=list,json" in workflow
    assert "run: npm run test:e2e:staging" not in workflow
    assert command[:2] == ["playwright", "test"]
    assert tuple(token for token in command[2:] if not token.startswith("--")) == expected_specs
    assert {"--retries=0", "--forbid-only"}.issubset(command)
    assert "PLAYWRIGHT_JSON_OUTPUT_FILE: test-results/public-read-report.json" in workflow
    assert 'EXPECTED_PUBLIC_READ_TESTS: "78"' in workflow
    assert "from scripts.run_parity_staging_wave import report_errors" in workflow
    assert "Reject incomplete browser public-read evidence" in workflow


def test_staging_smoke_browser_evidence_gate_requires_all_78_clean_tests(
    tmp_path: Path,
):
    valid = _run_browser_evidence_validation(
        tmp_path,
        {"stats": {"expected": 78, "skipped": 0, "unexpected": 0, "flaky": 0}},
    )
    assert valid.returncode == 0, valid.stdout + valid.stderr

    incomplete = _run_browser_evidence_validation(
        tmp_path,
        {"stats": {"expected": 65, "skipped": 1, "unexpected": 0, "flaky": 0}},
    )
    assert incomplete.returncode == 1
    assert "skipped 1 test" in incomplete.stdout
    assert "requires exactly 78" in incomplete.stdout


def test_staging_smoke_browser_evidence_gate_fails_when_report_is_missing(
    tmp_path: Path,
):
    result = _run_browser_evidence_validation(tmp_path, None)

    assert result.returncode == 1
    assert "Missing browser evidence" in result.stdout


def test_staging_smoke_validates_exact_isolated_targets_before_requests():
    workflow = (ROOT / ".github/workflows/staging_smoke.yml").read_text(encoding="utf-8")

    validation = workflow.index("- name: Validate staging smoke configuration")
    checkout = workflow.index("- name: Check out repository")
    provenance = workflow.index("- name: Verify canonical staging workflow provenance")
    public_smoke = workflow.index("- name: Run public smoke checks")
    assert checkout < provenance < validation < public_smoke
    assert "${{ secrets." not in workflow[:provenance]
    assert 'if [ "$GITHUB_REF" != "refs/heads/staging" ]; then' in workflow
    assert '[ "$HEAD_SHA" != "$GITHUB_SHA" ] || [ "$HEAD_SHA" != "$STAGING_SHA" ]' in workflow
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
    ) == 5


def test_staging_smoke_attests_exact_sha_and_disabled_write_projection():
    workflow = (ROOT / ".github/workflows/staging_smoke.yml").read_text(encoding="utf-8")
    identity = workflow.split("      - name: Attest exact read-only deployment identity\n", 1)[
        1
    ].split("\n      - name:", 1)[0]

    assert "deployment_identity_errors(" in identity
    assert 'candidate_sha=os.environ["GITHUB_SHA"]' in identity
    assert 'expected_write_wave="none"' in identity
    assert "expected_web_origin=web_origin" in identity
    assert "_immutable_vercel_origin" in identity
    assert "juprleagues-api-staging:deployment-" in identity
    assert "x-vercel-protection-bypass" in identity


def test_staging_smoke_repo_import_steps_set_workspace_pythonpath():
    workflow = (ROOT / ".github/workflows/staging_smoke.yml").read_text(encoding="utf-8")

    for step_name in (
        "Attest exact read-only deployment identity",
        "Reject incomplete browser public-read evidence",
    ):
        step = workflow.split(f"      - name: {step_name}\n", 1)[1].split(
            "\n      - name:", 1
        )[0]
        assert "shell: python" in step
        assert "from scripts.run_parity_staging_wave import" in step
        assert "PYTHONPATH: ${{ github.workspace }}" in step


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
