from pathlib import Path
import re

from scripts.staging_write_waves import (
    ALL_STAGING_WRITE_FLAGS,
    DORMANT_STAGING_WRITE_FLAGS,
    NO_WRITE_WAVE,
    OPEN_WRITE_FLAGS,
    OPEN_WRITE_ROUTES,
    OPEN_WRITE_WAVE,
    STAGING_WRITE_WAVE_ROUTES,
    STAGING_WRITE_WAVES,
    expected_write_flags,
    wave_allows_request,
)


def test_open_wave_enables_every_reviewed_staging_gate() -> None:
    expected = {
        flag
        for flags in STAGING_WRITE_WAVES.values()
        for flag in flags
    }
    assert OPEN_WRITE_WAVE not in STAGING_WRITE_WAVES
    assert set(OPEN_WRITE_FLAGS) == expected
    projection = expected_write_flags(OPEN_WRITE_WAVE)
    assert all(projection[flag] for flag in expected)
    assert all(not projection[flag] for flag in DORMANT_STAGING_WRITE_FLAGS)
    assert set(projection) == set(ALL_STAGING_WRITE_FLAGS)


def test_open_wave_allows_every_reviewed_staging_route() -> None:
    expected = {
        route
        for routes in STAGING_WRITE_WAVE_ROUTES.values()
        for route in routes
    }
    assert OPEN_WRITE_WAVE not in STAGING_WRITE_WAVE_ROUTES
    assert set(OPEN_WRITE_ROUTES) == expected
    for method, template in expected:
        concrete = re.sub(r"\{[^{}]+\}", "1", template)
        assert "{" not in concrete, concrete
        assert wave_allows_request(OPEN_WRITE_WAVE, method, concrete)
    assert not wave_allows_request(NO_WRITE_WAVE, "POST", "/clubs/tres-palapas/support/intake")


def test_source_fly_config_remains_safe_until_the_staging_deploy_projects_its_posture() -> None:
    text = Path("fly.staging.toml").read_text(encoding="utf-8")
    assert 'JUPR_STAGING_WRITE_WAVE = "none"' in text
    assert 'JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL = "0"' in text
    assert 'JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF = "0"' in text
    for flag in expected_write_flags(NO_WRITE_WAVE):
        assert f'{flag} = "0"' in text


def test_staging_deploy_defaults_to_persistent_open_and_retains_explicit_close() -> None:
    text = Path(".github/workflows/fly_api_staging_deploy.yml").read_text(encoding="utf-8")
    assert "|| 'open'" in text
    assert "|| 'none'" not in text
    assert 'default: "open"' in text
    assert "- open" in text
    assert "- none" in text
    assert 'JUPR_EMAIL_MODE: dry_run' in text
    assert '"JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL=0"' in text
    assert '"JUPR_ENABLE_PUBLIC_LIVE_WRITES_PRODUCTION=0"' in text


def test_automatic_lease_and_recovery_guards_are_retired() -> None:
    session = Path(".github/workflows/staging-write-session.yml").read_text(encoding="utf-8")
    recovery = Path(".github/workflows/staging-write-recovery.yml").read_text(encoding="utf-8")
    evidence = Path(".github/workflows/staging-evidence-automation.yml").read_text(encoding="utf-8")
    assert "issues:" not in session
    assert "issues:" not in evidence
    assert "schedule:" not in recovery
    assert "workflow_run:" not in recovery
    assert "Staging Emergency Write Disable" in recovery


def test_admin_status_fetches_do_not_cache_guard_state() -> None:
    uploader = Path("apps/web/lib/adminMatchUploaderApi.ts").read_text(encoding="utf-8")
    assert 'fetch(url, { cache: "no-store" })' in uploader
    for path in Path("apps/web").rglob("*.ts*"):
        if "/admin" not in path.as_posix() and not path.name.startswith("admin"):
            continue
        assert "revalidate: 30" not in path.read_text(encoding="utf-8"), path
