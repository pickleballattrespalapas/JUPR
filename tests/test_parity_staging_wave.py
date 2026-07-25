from pathlib import Path
import hashlib
import re

import scripts.run_parity_staging_wave as staging_wave
from scripts.run_parity_staging_wave import (
    EXPECTED_STAGING_API_ORIGIN,
    EXPECTED_STAGING_AUTH_ORIGIN,
    EXPECTED_STAGING_PROJECT_REF,
    EXPECTED_STAGING_WEB_ORIGIN,
    EXPECTED_WRITE_WAVE_BY_EVIDENCE_MODE,
    MUTATING_WAVES,
    MUTATION_CONFIRMATION,
    REQUIRED_REAL_SPECS,
    WAVES,
    candidate_errors,
    deployment_identity_errors,
    environment_errors,
    integrated_manifest_errors,
    manifest_errors,
    report_errors,
)
from scripts.staging_write_waves import expected_write_flags


def _base_env() -> dict[str, str]:
    return {
        "STAGING_WEB_BASE_URL": EXPECTED_STAGING_WEB_ORIGIN,
        "STAGING_API_BASE_URL": EXPECTED_STAGING_API_ORIGIN,
        "NEXT_PUBLIC_JUPR_API_BASE_URL": EXPECTED_STAGING_API_ORIGIN,
        "VERCEL_AUTOMATION_BYPASS_SECRET": "secret",
        "JUPR_EXPECTED_STAGING_API_ORIGIN": EXPECTED_STAGING_API_ORIGIN,
        "JUPR_EXPECTED_STAGING_AUTH_ORIGIN": EXPECTED_STAGING_AUTH_ORIGIN,
        "STAGING_SUPABASE_URL": EXPECTED_STAGING_AUTH_ORIGIN,
        "STAGING_SUPABASE_PROJECT_REF": EXPECTED_STAGING_PROJECT_REF,
    }


def _match_write_env() -> dict[str, str]:
    return {
        **_base_env(),
        "STAGING_ADMIN_BEARER_TOKEN": "token",
        "JUPR_TOURNAMENT_LIVE_TOURNAMENT_ID": "tournament-1",
        "JUPR_TOURNAMENT_LIVE_DRAW_ID": "draw-1",
        "JUPR_TOURNAMENT_LIVE_GAME_ID": "game-1",
        "JUPR_TOURNAMENT_LIVE_ORIGINAL_SCORE_A": "11",
        "JUPR_TOURNAMENT_LIVE_ORIGINAL_SCORE_B": "7",
        "JUPR_TOURNAMENT_LIVE_EXERCISE_SCORE_A": "11",
        "JUPR_TOURNAMENT_LIVE_EXERCISE_SCORE_B": "8",
        "JUPR_TOURNAMENT_LIVE_ALLOW_MUTATION_E2E": "1",
        "JUPR_PARITY_MUTATION_CONFIRMATION": MUTATION_CONFIRMATION,
    }


def _api_identity(sha: str, fly_image: str, *, write_wave: str = "none") -> dict[str, object]:
    flags = expected_write_flags(write_wave)
    fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}={1 if enabled else 0}" for name, enabled in sorted(flags.items())
        ).encode("utf-8")
    ).hexdigest()
    return {
        "ok": True,
        "environment": "staging",
        "git_commit_sha": sha,
        "fly_app_name": "juprleagues-api-staging",
        "fly_image_ref": fly_image,
        "web_origin": EXPECTED_STAGING_WEB_ORIGIN,
        "supabase_project_ref": EXPECTED_STAGING_PROJECT_REF,
        "jwt_verification_configured": True,
        "jwt_verification_mode": "jwks",
        "jwt_verification_project_ref": EXPECTED_STAGING_PROJECT_REF,
        "staging_write_wave": write_wave,
        "business_data_write_wave_active": write_wave != "none",
        "security_denial_audit_logging_required": True,
        "public_live_writes_enabled": flags["JUPR_ENABLE_PUBLIC_LIVE_WRITES"],
        "public_live_production_override_enabled": False,
        "controlled_write_flags": flags,
        "controlled_write_flag_fingerprint": fingerprint,
        "registration_edit_secret_configured": write_wave == "public-intake-auth",
        "registration_confirmation_secret_configured": write_wave == "public-intake-auth",
        "write_prerequisites": {
            "service_role_configured": True,
            "api_audit_required": True,
            "worker_run_log_required": True,
            "email_mode": "dry_run",
            "live_player_update_email_enabled": False,
        },
    }


def test_report_guard_requires_tests_and_rejects_skips_flakes_and_failures() -> None:
    assert report_errors(
        {"stats": {"expected": 2, "skipped": 1, "unexpected": 1, "flaky": 1}}
    ) == [
        "Playwright invocation skipped 1 test(s).",
        "Playwright invocation had 1 unexpected result(s).",
        "Playwright invocation had 1 flaky result(s).",
    ]
    assert report_errors(
        {"stats": {"expected": 0, "skipped": 0, "unexpected": 0, "flaky": 0}}
    ) == ["Playwright invocation executed zero passing/expected tests."]
    assert report_errors(
        {"stats": {"expected": 3, "skipped": 0, "unexpected": 0, "flaky": 0}}
    ) == []


def test_environment_guard_uses_exact_origins_and_only_one_mutating_wave() -> None:
    env = _base_env()
    assert environment_errors("public-read", env) == []
    assert MUTATING_WAVES == {"match-rating-writes"}

    wrong_ref = dict(env, STAGING_SUPABASE_PROJECT_REF="production-ref")
    assert any("Refusing project" in error for error in environment_errors("public-read", wrong_ref))

    production_api = dict(
        env,
        STAGING_API_BASE_URL="https://api.juprleagues.com",
        NEXT_PUBLIC_JUPR_API_BASE_URL="https://api.juprleagues.com",
    )
    errors = environment_errors("public-read", production_api)
    assert sum("non-allowlisted staging origin" in error for error in errors) == 2

    credentialed_web = dict(
        env,
        STAGING_WEB_BASE_URL=(
            "https://user@jupr-git-staging-pickleballattrespalapas1.vercel.app"
        ),
    )
    assert any(
        "STAGING_WEB_BASE_URL" in error
        for error in environment_errors("public-read", credentialed_web)
    )

    assert environment_errors("match-rating-writes", _match_write_env()) == []
    missing_confirmation = _match_write_env()
    missing_confirmation.pop("JUPR_PARITY_MUTATION_CONFIRMATION")
    assert any(
        MUTATION_CONFIRMATION in error
        for error in environment_errors("match-rating-writes", missing_confirmation)
    )
    read_with_confirmation = dict(env, JUPR_PARITY_MUTATION_CONFIRMATION=MUTATION_CONFIRMATION)
    assert any(
        "must not receive" in error
        for error in environment_errors("public-read", read_with_confirmation)
    )


def test_candidate_guard_requires_exact_full_sha() -> None:
    sha = "a" * 40
    assert candidate_errors(sha, sha) == []
    assert candidate_errors("abc", sha)
    assert candidate_errors("b" * 40, sha)


def test_deployment_identity_requires_immutable_web_and_exact_live_services() -> None:
    sha = "a" * 40
    vercel_id = "dpl_staging123"
    fly_image = "registry.fly.io/juprleagues-api-staging:deployment-1234567890"
    immutable_origin = "https://jupr-a1b2c3d4-pickleballattrespalapas1.vercel.app"
    assert staging_wave._immutable_vercel_origin(immutable_origin) == immutable_origin
    assert staging_wave._immutable_vercel_origin(EXPECTED_STAGING_WEB_ORIGIN) is None
    assert (
        staging_wave._immutable_vercel_origin(
            "https://jupr-pickleballattrespalapas1.vercel.app"
        )
        is None
    )
    web = {
        "environment": "staging",
        "vercel_environment": "preview",
        "api_origin": EXPECTED_STAGING_API_ORIGIN,
        "auth_origin": EXPECTED_STAGING_AUTH_ORIGIN,
        "preview_isolation_active": True,
        "preview_auth_isolation_active": True,
        "git_commit_sha": sha,
        "vercel_deployment_id": vercel_id,
        "vercel_deployment_origin": immutable_origin,
    }
    api = _api_identity(sha, fly_image)
    assert deployment_identity_errors(
        web,
        api,
        candidate_sha=sha,
        vercel_deployment_id=vercel_id,
        fly_image_ref=fly_image,
        expected_web_origin=immutable_origin,
    ) == []

    stale_alias = dict(web, vercel_deployment_origin=EXPECTED_STAGING_WEB_ORIGIN)
    assert any(
        "immutable deployment origin" in error
        for error in deployment_identity_errors(
            stale_alias,
            api,
            candidate_sha=sha,
            vercel_deployment_id=vercel_id,
            fly_image_ref=fly_image,
        )
    )
    wrong_api = dict(api, fly_app_name="production", supabase_project_ref="prod")
    errors = deployment_identity_errors(
        web,
        wrong_api,
        candidate_sha=sha,
        vercel_deployment_id=vercel_id,
        fly_image_ref=fly_image,
    )
    assert any("fly_app_name" in error for error in errors)
    assert any("supabase_project_ref" in error for error in errors)


def test_deployment_identity_binds_each_evidence_mode_to_exact_write_projection() -> None:
    sha = "a" * 40
    vercel_id = "dpl_staging123"
    fly_image = "registry.fly.io/juprleagues-api-staging:deployment-1234567890"
    immutable_origin = "https://jupr-a1b2c3d4-pickleballattrespalapas1.vercel.app"
    web = {
        "environment": "staging",
        "vercel_environment": "preview",
        "api_origin": EXPECTED_STAGING_API_ORIGIN,
        "auth_origin": EXPECTED_STAGING_AUTH_ORIGIN,
        "preview_isolation_active": True,
        "preview_auth_isolation_active": True,
        "git_commit_sha": sha,
        "vercel_deployment_id": vercel_id,
        "vercel_deployment_origin": immutable_origin,
    }
    for mode, write_wave in EXPECTED_WRITE_WAVE_BY_EVIDENCE_MODE.items():
        api = _api_identity(sha, fly_image, write_wave=write_wave)
        assert deployment_identity_errors(
            web,
            api,
            candidate_sha=sha,
            vercel_deployment_id=vercel_id,
            fly_image_ref=fly_image,
            expected_write_wave=write_wave,
            expected_web_origin=immutable_origin,
        ) == [], mode

        drifted = dict(api, staging_write_wave="none" if write_wave != "none" else "public-live")
        assert any(
            "staging_write_wave" in error
            for error in deployment_identity_errors(
                web,
                drifted,
                candidate_sha=sha,
                vercel_deployment_id=vercel_id,
                fly_image_ref=fly_image,
                expected_write_wave=write_wave,
                expected_web_origin=immutable_origin,
            )
        )

        drifted_web_origin = dict(
            api, web_origin="https://pickleballclubsandwich.com"
        )
        assert any(
            "web_origin" in error
            for error in deployment_identity_errors(
                web,
                drifted_web_origin,
                candidate_sha=sha,
                vercel_deployment_id=vercel_id,
                fly_image_ref=fly_image,
                expected_write_wave=write_wave,
                expected_web_origin=immutable_origin,
            )
        )

    unsafe = _api_identity(sha, fly_image)
    unsafe["write_prerequisites"] = {
        **unsafe["write_prerequisites"],
        "email_mode": "live",
        "live_player_update_email_enabled": True,
    }
    errors = deployment_identity_errors(
        web,
        unsafe,
        candidate_sha=sha,
        vercel_deployment_id=vercel_id,
        fly_image_ref=fly_image,
        expected_web_origin=immutable_origin,
    )
    assert any("email_mode" in error for error in errors)
    assert any("player-update email" in error for error in errors)


def test_identity_preflight_queries_the_supplied_immutable_candidate_origin(
    monkeypatch,
) -> None:
    sha = "a" * 40
    vercel_id = "dpl_staging123"
    fly_image = "registry.fly.io/juprleagues-api-staging:deployment-1234567890"
    immutable_origin = "https://jupr-a1b2c3d4-pickleballattrespalapas1.vercel.app"
    requested_urls: list[str] = []
    requested_headers: list[dict[str, str]] = []

    def fake_get_json(url: str, headers=None):
        requested_urls.append(url)
        requested_headers.append(dict(headers or {}))
        if url.endswith("/api/environment"):
            return {
                "environment": "staging",
                "vercel_environment": "preview",
                "api_origin": EXPECTED_STAGING_API_ORIGIN,
                "auth_origin": EXPECTED_STAGING_AUTH_ORIGIN,
                "preview_isolation_active": True,
                "preview_auth_isolation_active": True,
                "git_commit_sha": sha,
                "vercel_deployment_id": vercel_id,
                "vercel_deployment_origin": immutable_origin,
            }
        return _api_identity(sha, fly_image)

    monkeypatch.setattr(staging_wave, "_get_json", fake_get_json)
    _identity, errors = staging_wave._deployment_identity(
        {"VERCEL_AUTOMATION_BYPASS_SECRET": "secret"},
        candidate_sha=sha,
        vercel_deployment_id=vercel_id,
        fly_image_ref=fly_image,
        web_origin=immutable_origin,
        expected_web_origin=immutable_origin,
    )
    assert errors == []
    assert requested_urls[0] == f"{immutable_origin}/api/environment"
    assert EXPECTED_STAGING_WEB_ORIGIN not in requested_urls[0]
    assert requested_headers[0] == {"x-vercel-protection-bypass": "secret"}
    assert requested_headers[1] == {}


def test_manifest_is_route_specific_and_generic_mutation_modes_are_absent(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/parity-final-evidence.yml").read_text(
        encoding="utf-8"
    )
    assert "WORKFLOW_SHA=\"$(git rev-parse HEAD)\"" in workflow
    assert 'refs/heads/staging)' in workflow
    assert 'refs/heads/rollback-feb8)' in workflow
    assert '[ "$GITHUB_SHA" != "$STAGING_SHA" ] || [ "$WORKFLOW_SHA" != "$STAGING_SHA" ]' in workflow
    assert 'DEFAULT_SHA="$(git rev-parse refs/remotes/origin/rollback-feb8)"' in workflow
    assert '[ "$GITHUB_SHA" != "$DEFAULT_SHA" ] || [ "$WORKFLOW_SHA" != "$STAGING_SHA" ]' in workflow
    assert (
        '<(git show "$GITHUB_SHA:.github/workflows/parity-final-evidence.yml")'
        in workflow
    )
    assert (
        "default-branch workflow registry must be byte-identical to canonical staging"
        in workflow
    )
    assert "reversible-admin-writes" not in WAVES
    assert "recovery" not in WAVES
    assert REQUIRED_REAL_SPECS["match-rating-writes"] == (
        "e2e/tournament-live.staging.spec.ts",
    )
    assert not any(
        "writes.staging.spec.ts" in spec
        for wave in WAVES.values()
        for invocation in wave
        for spec in invocation["specs"]
    )
    for removed in (
        "e2e/public-intake-writes.staging.spec.ts",
        "e2e/communications-writes.staging.spec.ts",
        "e2e/league-admin-writes.staging.spec.ts",
        "e2e/social-moderation-writes.staging.spec.ts",
        "e2e/match-player-writes.staging.spec.ts",
        "e2e/league-live-writes.staging.spec.ts",
        "e2e/recovery-reconciliation.staging.spec.ts",
        "e2e/support/real-staging.ts",
    ):
        assert not (root / "apps" / "web" / removed).exists()
    partner_invocation = next(
        invocation
        for invocation in WAVES["public-intake-auth"]
        if invocation["name"] == "tournament-partner-board-read-only"
    )
    assert partner_invocation["grep"] == "partner board renders the explicit privacy boundary"
    support_invocation = next(
        invocation
        for invocation in WAVES["public-intake-auth"]
        if invocation["name"] == "static-support-read-only"
    )
    assert support_invocation["grep"] == (
        "rating rules, FAQ, and policy request links are complete"
    )
    tournament_ops_invocation = next(
        invocation
        for invocation in WAVES["admin-read-export"]
        if invocation["name"] == "tournament-operations-read-only"
    )
    assert tournament_ops_invocation["grep"] == (
        "route-specific operations surfaces|read-only ops snapshot|"
        "DUPR preview is blocked while write wave is none"
    )
    tournament_ops_source = (
        root / "apps/web/e2e/tournament-operations.staging.spec.ts"
    ).read_text(encoding="utf-8")
    tournament_ops_titles = re.findall(r'test\("([^"]+)"', tournament_ops_source)
    selected_titles = [
        title
        for title in tournament_ops_titles
        if re.search(str(tournament_ops_invocation["grep"]), title)
    ]
    assert selected_titles == [
        "route-specific operations surfaces remain independently addressable",
        "read-only ops snapshot resolves the exact staging draw",
        "DUPR preview is blocked while write wave is none",
    ]
    assert "DUPR preview is authenticated and writes zero rows" not in selected_titles
    assert "stale score CAS is refused before mutation" not in selected_titles

    missing_root = tmp_path / "web"
    missing_root.mkdir()
    assert any(
        "blocked until real staging spec" in error
        for error in manifest_errors("match-rating-writes", missing_root)
    )


def test_integrated_manifest_validates_every_wave_spec_and_grep(tmp_path: Path) -> None:
    web_root = tmp_path / "web"
    for wave in WAVES.values():
        for invocation in wave:
            grep = str(invocation.get("grep", ""))
            for spec in invocation["specs"]:
                path = web_root / str(spec)
                path.parent.mkdir(parents=True, exist_ok=True)
                source = path.read_text(encoding="utf-8") if path.exists() else ""
                if grep:
                    source += f"\n{grep}\n"
                else:
                    source += "\ntest('manifest fixture', () => {});\n"
                path.write_text(source, encoding="utf-8")
    assert integrated_manifest_errors(web_root) == []

    broken_root = tmp_path / "broken"
    broken_root.mkdir()
    errors = integrated_manifest_errors(broken_root)
    assert errors
    assert any("tournament-live.staging.spec.ts" in error for error in errors)


def test_remote_public_wave_waits_for_hydration_and_uses_stable_live_selectors() -> None:
    root = Path(__file__).resolve().parents[1]
    explorer = (root / "apps/web/e2e/public-explorer-recap.spec.ts").read_text(
        encoding="utf-8"
    )
    badges = (root / "apps/web/e2e/staging.smoke.spec.ts").read_text(
        encoding="utf-8"
    )
    partner_board = (
        root / "apps/web/e2e/tournament-partner-board.parity.spec.ts"
    ).read_text(encoding="utf-8")

    hydration_wait = 'getByTestId("match-explorer-summary")'
    first_controlled_change = "await me.selectOption(playerIds[0])"
    assert explorer.index(hydration_wait) < explorer.index(first_controlled_change)
    assert "badge-codex?bucket=all" in badges
    assert "bootstrapStagingContext(context)" in partner_board


def test_manual_workflow_is_dispatch_only_exact_staging_candidate_and_least_scope() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/parity-final-evidence.yml").read_text(
        encoding="utf-8"
    )
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    runner = (root / "scripts/run_parity_staging_wave.py").read_text(encoding="utf-8")
    staging_support = (root / "apps/web/e2e/support/staging.ts").read_text(
        encoding="utf-8"
    )
    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "permissions:\n  contents: read" in workflow
    assert "environment: staging" in workflow
    assert "ref: staging" in workflow
    assert 'refs/heads/staging)' in workflow
    assert 'refs/heads/rollback-feb8)' in workflow
    assert (
        'echo "parity evidence workflow must be dispatched from refs/heads/staging '
        'or refs/heads/rollback-feb8"'
    ) in workflow
    assert '[ "$GITHUB_SHA" != "$STAGING_SHA" ] || [ "$WORKFLOW_SHA" != "$STAGING_SHA" ]' in workflow
    assert '[ "$GITHUB_SHA" != "$DEFAULT_SHA" ] || [ "$WORKFLOW_SHA" != "$STAGING_SHA" ]' in workflow
    assert (
        '<(git show "$GITHUB_SHA:.github/workflows/parity-final-evidence.yml")'
        in workflow
    )
    assert "--require-complete" in workflow
    assert "--identity-only" in workflow
    assert MUTATION_CONFIRMATION in workflow
    assert 'elif [ "$CANDIDATE_SHA" != "$STAGING_SHA" ]' in workflow
    assert 'git diff --name-only "$CANDIDATE_SHA..$STAGING_SHA"' in workflow
    assert "docs/next_parity_manual_staging_book.md" in workflow
    assert "vercel_deployment_origin:" in workflow
    assert "not the mutable staging alias" in workflow
    assert workflow.count('--vercel-deployment-origin "$VERCEL_DEPLOYMENT_ORIGIN"') == 3
    assert 'test -n "$(VERCEL_DEPLOYMENT_ORIGIN)"' in makefile
    assert '--vercel-deployment-origin "$(VERCEL_DEPLOYMENT_ORIGIN)"' in makefile
    assert "reversible-admin-writes" not in workflow
    assert "JUPR_RECOVERY_READBACKS_JSON" not in workflow
    assert "JUPR_PUBLIC_INTAKE_WRITE_CASES_JSON" not in workflow
    assert "JUPR_TOURNAMENT_OPS_ALLOW_MUTATION_E2E" not in workflow
    assert "JUPR_RUN_LIVE_LADDER_MUTATION_E2E" not in workflow
    assert "JUPR_RUN_PUBLIC_LIVE_WRITE_E2E" not in workflow
    assert "JUPR_TOURNAMENT_LIVE_ALLOW_MUTATION_E2E" in workflow
    assert 'STAGING_SUPABASE_PROJECT_REF: "sijpxjxvdtrehmqvirfi"' in workflow
    assert (
        'STAGING_SUPABASE_URL: "https://sijpxjxvdtrehmqvirfi.supabase.co"'
        in workflow
    )
    assert (
        'JUPR_EXPECTED_STAGING_AUTH_ORIGIN: '
        '"https://sijpxjxvdtrehmqvirfi.supabase.co"'
    ) in workflow
    assert (
        'STAGING_API_BASE_URL: "https://juprleagues-api-staging.fly.dev"'
        in workflow
    )
    assert (
        'NEXT_PUBLIC_JUPR_API_BASE_URL: '
        '"https://juprleagues-api-staging.fly.dev"'
    ) in workflow
    assert (
        'STAGING_WEB_BASE_URL: '
        '"https://jupr-git-staging-pickleballattrespalapas1.vercel.app"'
    ) in workflow
    assert "Prepare authenticated parity staging session" in workflow
    assert (
        "if: needs.evidence-contract.outputs.mode == 'admin-read-export' || "
        "needs.evidence-contract.outputs.mode == 'match-rating-writes'"
    ) in workflow
    assert 'python scripts/prepare_parity_staging_session.py "$PARITY_MODE"' in workflow
    assert 'test -n "$GITHUB_ENV"' in workflow
    assert "STAGING_ADMIN_EMAIL: ${{ vars.STAGING_ADMIN_EMAIL }}" in workflow
    assert "STAGING_SUPABASE_ANON_KEY: ${{ secrets.STAGING_SUPABASE_ANON_KEY }}" in workflow
    assert "STAGING_ADMIN_BEARER_TOKEN: ${{" not in workflow
    assert "JUPR_STAGING_ADMIN_ACCESS_TOKEN: ${{" not in workflow
    assert (
        "STAGING_ADMIN_PASSWORD: "
        "${{ needs.evidence-contract.outputs.mode == 'public-intake-auth' "
        "&& secrets.STAGING_ADMIN_PASSWORD || '' }}"
    ) in workflow
    assert "make check-parity-final-evidence-integrated" in workflow
    assert 'playwright_env["STAGING_WEB_BASE_URL"] = attested_web_origin' in runner
    assert (
        'playwright_env["JUPR_ATTESTED_VERCEL_DEPLOYMENT_ORIGIN"] = '
        "attested_web_origin"
    ) in runner
    assert "post-wave re-attestation" in runner
    assert "web_origin=candidate_web_origin" in runner
    assert "expected_web_origin=candidate_web_origin" in runner
    assert "if (!attestedDeploymentOrigin)" in staging_support
    assert "remoteBaseUrl !== expectedStagingWebOrigin" in staging_support
    assert "remoteBaseUrl !== attestedDeploymentOrigin" in staging_support
