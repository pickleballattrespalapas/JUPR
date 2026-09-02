from __future__ import annotations

import pytest

from jupr_app.services import admin_tournament_live_service as tournament_live
from jupr_app.services.admin_league_live_service import (
    is_admin_league_live_submit_enabled,
)


LEAGUE_FLAGS = {
    "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER": "1",
    "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN": "1",
    "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT": "1",
}
TOURNAMENT_FLAGS = {
    "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS": "1",
    "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES": "1",
    "JUPR_ENABLE_TOURNAMENT_WRITES_PRODUCTION": "1",
    "SUPABASE_SERVICE_ROLE_KEY": "server-only-test-key",
}


def _set_env(monkeypatch, values: dict[str, str]) -> None:
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _enable_production_league_live(monkeypatch) -> None:
    _set_env(
        monkeypatch,
        {
            "JUPR_ENV": "production",
            "JUPR_PRODUCTION_WRITE_POLICY": "enabled",
            **LEAGUE_FLAGS,
        },
    )


def _enable_production_tournament_live(monkeypatch) -> None:
    _set_env(
        monkeypatch,
        {
            "JUPR_ENV": "production",
            "JUPR_PRODUCTION_WRITE_POLICY": "enabled",
            **TOURNAMENT_FLAGS,
        },
    )


def _ready_tournament_status(monkeypatch) -> dict:
    monkeypatch.setattr(
        tournament_live, "_operation_store_ready", lambda _supabase: (True, None)
    )
    monkeypatch.setattr(
        tournament_live, "_audit_store_ready", lambda _supabase: (True, None)
    )
    return tournament_live.build_admin_tournament_live_status(object(), club_id="club")


@pytest.mark.parametrize(
    "missing",
    [
        "JUPR_PRODUCTION_WRITE_POLICY",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_DOMAIN",
        "JUPR_ENABLE_NEXT_ADMIN_LEAGUE_LIVE_SUBMIT",
    ],
)
def test_production_league_live_submit_requires_policy_domain_and_submit_flags(
    monkeypatch, missing: str
) -> None:
    _enable_production_league_live(monkeypatch)
    assert is_admin_league_live_submit_enabled() is True

    monkeypatch.delenv(missing)
    assert is_admin_league_live_submit_enabled() is False


def test_staging_league_live_submit_does_not_require_production_policy(monkeypatch) -> None:
    _set_env(monkeypatch, {"JUPR_ENV": "staging", **LEAGUE_FLAGS})
    monkeypatch.delenv("JUPR_PRODUCTION_WRITE_POLICY", raising=False)

    assert is_admin_league_live_submit_enabled() is True


@pytest.mark.parametrize("environment", ["local", "test", "development", "dev", "preview"])
def test_league_live_submit_has_no_broad_non_hosted_environment_bypass(
    monkeypatch, environment: str
) -> None:
    _set_env(
        monkeypatch,
        {
            "JUPR_ENV": environment,
            "JUPR_PRODUCTION_WRITE_POLICY": "enabled",
            **LEAGUE_FLAGS,
        },
    )

    assert is_admin_league_live_submit_enabled() is False


@pytest.mark.parametrize(
    "missing,expected_exception",
    [
        ("JUPR_PRODUCTION_WRITE_POLICY", PermissionError),
        ("JUPR_ENABLE_TOURNAMENT_WRITES_PRODUCTION", PermissionError),
        ("JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES", PermissionError),
        ("SUPABASE_SERVICE_ROLE_KEY", RuntimeError),
    ],
)
def test_production_tournament_live_requires_the_complete_production_guard(
    monkeypatch, missing: str, expected_exception: type[Exception]
) -> None:
    _enable_production_tournament_live(monkeypatch)
    ready = _ready_tournament_status(monkeypatch)
    assert ready["writes_enabled"] is True
    assert ready["status"] == "write_ready"
    assert ready["staging_only"] is False
    tournament_live.require_tournament_live_write_runtime()

    monkeypatch.delenv(missing)
    assert _ready_tournament_status(monkeypatch)["writes_enabled"] is False
    with pytest.raises(expected_exception):
        tournament_live.require_tournament_live_write_runtime()


def test_tournament_live_status_still_requires_the_admin_domain_flag(monkeypatch) -> None:
    _enable_production_tournament_live(monkeypatch)
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS")

    assert _ready_tournament_status(monkeypatch)["writes_enabled"] is False


def test_staging_tournament_live_remains_enabled_without_production_override(monkeypatch) -> None:
    _set_env(
        monkeypatch,
        {
            "JUPR_ENV": "staging",
            "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS": "1",
            "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES": "1",
            "SUPABASE_SERVICE_ROLE_KEY": "server-only-test-key",
        },
    )
    monkeypatch.delenv("JUPR_PRODUCTION_WRITE_POLICY", raising=False)
    monkeypatch.delenv("JUPR_ENABLE_TOURNAMENT_WRITES_PRODUCTION", raising=False)

    assert _ready_tournament_status(monkeypatch)["writes_enabled"] is True
    tournament_live.require_tournament_live_write_runtime()


@pytest.mark.parametrize("environment", ["local", "test", "development", "dev", "preview"])
def test_tournament_live_has_no_broad_non_hosted_environment_bypass(
    monkeypatch, environment: str
) -> None:
    _enable_production_tournament_live(monkeypatch)
    monkeypatch.setenv("JUPR_ENV", environment)

    assert _ready_tournament_status(monkeypatch)["writes_enabled"] is False
    with pytest.raises(PermissionError, match="explicit staging or production runtime"):
        tournament_live.require_tournament_live_write_runtime()
