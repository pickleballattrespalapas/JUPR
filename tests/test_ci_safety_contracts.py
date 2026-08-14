from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_api_contract_runs_new_deployment_and_admin_ux_regressions() -> None:
    workflow = _read(".github/workflows/api-contract.yml")
    required_tests = (
        "tests/test_production_deployment_hardening.py",
        "tests/test_admin_action_token_guards_static.py",
        "tests/test_authenticated_auto_load_contract.py",
        "tests/test_tournament_setup_request_safety_static.py",
        "tests/test_tournament_setup_selection_ux.py",
        "tests/test_ci_safety_contracts.py",
        "tests/test_staging_evidence_automation.py",
        "tests/test_staging_evidence_automation_workflow.py",
        "tests/test_staging_write_session.py",
        "tests/test_staging_write_session_workflow.py",
        "tests/test_staging_write_wave_guards.py",
    )
    for test_path in required_tests:
        assert test_path in workflow


def test_next_build_runs_builder_contract_on_its_node_20_runtime() -> None:
    workflow = _read(".github/workflows/next-web-build.yml")
    builder_test = _read(
        "apps/web/app/admin/tournament-setup/tournamentSetupBuilder.test.mjs"
    )
    publication_status_test = _read(
        "apps/web/app/admin/tournaments/setup/tournamentSetupPublicationStatus.test.mjs"
    )
    storage_test = _read("apps/web/lib/adminAuthClient.storage.test.mjs")

    assert "node-version: \"20\"" in workflow
    assert "npx tsc" in workflow
    assert "app/admin/tournament-setup/tournamentSetupBuilder.ts" in workflow
    assert "app/admin/tournaments/setup/tournamentSetupPublicationStatus.ts" in workflow
    assert "JUPR_TOURNAMENT_SETUP_BUILDER_MODULE=" in workflow
    assert 'JUPR_TOURNAMENT_SETUP_BUILDER_MODULE="file://' not in workflow
    assert "node app/admin/tournament-setup/tournamentSetupBuilder.test.mjs" in workflow
    assert "JUPR_TOURNAMENT_SETUP_BUILDER_MODULE" in builder_test
    assert "JUPR_TOURNAMENT_SETUP_PUBLICATION_STATUS_MODULE=" in workflow
    assert "node app/admin/tournaments/setup/tournamentSetupPublicationStatus.test.mjs" in workflow
    assert "JUPR_TOURNAMENT_SETUP_PUBLICATION_STATUS_MODULE" in publication_status_test
    assert "npx tsc lib/adminAuthClient.ts" in workflow
    assert "JUPR_ADMIN_AUTH_CLIENT_MODULE=" in workflow
    assert "node lib/adminAuthClient.storage.test.mjs" in workflow
    assert "JUPR_ADMIN_AUTH_CLIENT_MODULE" in storage_test


def test_canonical_smoke_runs_all_mocked_tournament_setup_browser_tests() -> None:
    workflow = _read(".github/workflows/staging_smoke.yml")
    spec = _read("apps/web/e2e/tournament-setup.builder.staging.spec.ts")

    assert "e2e/tournament-setup.builder.staging.spec.ts" in workflow
    assert 'EXPECTED_TOURNAMENT_SETUP_TESTS: "6"' in workflow
    assert "tournament-setup-report.json" in workflow
    assert "report_errors(report)" in workflow
    assert "JUPR_RUN_TOURNAMENT_SETUP_UI_E2E" not in spec
    assert "test.skip(" not in spec
