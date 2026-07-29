from pathlib import Path

from scripts.staging_write_waves import (
    ALL_STAGING_WRITE_FLAGS,
    DORMANT_STAGING_WRITE_FLAGS,
    NO_WRITE_WAVE,
    OPEN_WRITE_WAVE,
    STAGING_WRITE_WAVE_ROUTES,
    STAGING_WRITE_WAVES,
    expected_write_flags,
)


def test_open_wave_enables_every_reviewed_staging_gate() -> None:
    expected = {
        flag
        for wave, flags in STAGING_WRITE_WAVES.items()
        if wave not in {NO_WRITE_WAVE, OPEN_WRITE_WAVE}
        for flag in flags
    }
    assert set(STAGING_WRITE_WAVES[OPEN_WRITE_WAVE]) == expected
    projection = expected_write_flags(OPEN_WRITE_WAVE)
    assert all(projection[flag] for flag in expected)
    assert all(not projection[flag] for flag in DORMANT_STAGING_WRITE_FLAGS)
    assert set(projection) == set(ALL_STAGING_WRITE_FLAGS)


def test_open_wave_allows_every_reviewed_staging_route() -> None:
    expected = {
        route
        for wave, routes in STAGING_WRITE_WAVE_ROUTES.items()
        if wave not in {NO_WRITE_WAVE, OPEN_WRITE_WAVE}
        for route in routes
    }
    assert set(STAGING_WRITE_WAVE_ROUTES[OPEN_WRITE_WAVE]) == expected


def test_fly_staging_defaults_to_permanent_open_mode() -> None:
    text = Path("fly.staging.toml").read_text(encoding="utf-8")
    assert 'JUPR_STAGING_WRITE_WAVE = "open"' in text
    assert 'JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL = "0"' in text
    assert 'JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF = "0"' in text
    projection = expected_write_flags(OPEN_WRITE_WAVE)
    for flag, enabled in projection.items():
        assert f'{flag} = "{1 if enabled else 0}"' in text


def test_staging_deploy_defaults_to_open_and_retains_emergency_none() -> None:
    text = Path(".github/workflows/fly_api_staging_deploy.yml").read_text(encoding="utf-8")
    assert "|| 'open'" in text
    assert 'default: "open"' in text
    assert "- open" in text
    assert "- none" in text
    assert "Push-triggered staging deploys must use write_wave=none" not in text
    assert 'JUPR_EMAIL_MODE: dry_run' in text


def test_automatic_lease_and_recovery_guards_are_retired() -> None:
    session = Path(".github/workflows/staging-write-session.yml").read_text(encoding="utf-8")
    recovery = Path(".github/workflows/staging-write-recovery.yml").read_text(encoding="utf-8")
    assert "issues:" not in session
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
