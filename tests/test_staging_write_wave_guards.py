from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from jupr_app.services.staging_write_guard import (
    COMMUNICATIONS_MUTATION_FLAG,
    require_staging_communications_mutations,
    require_staging_match_canonical_normalize_writes,
    staging_communications_mutations_enabled,
    staging_match_canonical_normalize_writes_enabled,
)
from scripts.staging_write_waves import (
    ALL_STAGING_WRITE_FLAGS,
    DORMANT_STAGING_WRITE_FLAGS,
    NO_WRITE_WAVE,
    STAGING_WRITE_WAVES,
    STAGING_WRITE_WAVE_ROUTES,
    expected_write_flags,
    wave_allows_request,
)
from services.api.middleware import StagingWriteWaveMiddleware


ROOT = Path(__file__).resolve().parents[1]
UNSAFE_METHODS = {"post", "put", "patch", "delete"}


def _unsafe_fastapi_route_inventory() -> set[tuple[str, str]]:
    inventory: set[tuple[str, str]] = set()
    for source_path in (ROOT / "services" / "api").glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if (
                    not isinstance(decorator, ast.Call)
                    or not isinstance(decorator.func, ast.Attribute)
                    or decorator.func.attr not in UNSAFE_METHODS
                    or not decorator.args
                ):
                    continue
                route = ast.literal_eval(decorator.args[0])
                assert isinstance(route, str), (
                    f"Unsafe route in {source_path} must use a static literal template so the "
                    "staging write-wave inventory can classify it."
                )
                inventory.add((decorator.func.attr.upper(), route))
    return inventory


def test_every_unsafe_fastapi_route_has_an_exact_nonstale_wave_classification() -> None:
    inventory = _unsafe_fastapi_route_inventory()
    manifest = {
        route
        for wave_routes in STAGING_WRITE_WAVE_ROUTES.values()
        for route in wave_routes
    }

    assert inventory == manifest
    assert STAGING_WRITE_WAVE_ROUTES[NO_WRITE_WAVE] == ()
    assert set(STAGING_WRITE_WAVE_ROUTES) == set(STAGING_WRITE_WAVES)
    assert all(
        routes for wave, routes in STAGING_WRITE_WAVE_ROUTES.items() if wave != NO_WRITE_WAVE
    )

    overlaps: dict[tuple[str, str], list[str]] = defaultdict(list)
    for wave, routes in STAGING_WRITE_WAVE_ROUTES.items():
        for route in routes:
            overlaps[route].append(wave)
    actual_overlaps = {
        route: tuple(waves) for route, waves in overlaps.items() if len(waves) > 1
    }
    league_domain_overlap = {
        ("POST", "/admin/clubs/{club_id}/league-manager/live/roster-suggestion"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions"),
        ("PATCH", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/snapshot"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}/plan"),
        ("POST", "/admin/clubs/{club_id}/league-manager/live-sessions/{session_id}/rounds/{round_number}"),
    }
    live_overlap = {
        ("POST", "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/commands"),
        ("POST", "/admin/clubs/{club_id}/tournament-live/tournaments/{tournament_id}/draws/{draw_id}/operations/{operation_key}/reconcile"),
    }
    expected_overlaps = {
        **{
            route: ("league-live-domain", "league-live-submit")
            for route in league_domain_overlap
        },
        **{
            route: ("tournament-live", "tournament-live-official-publish")
            for route in live_overlap
        },
        (
            "POST",
            "/admin/clubs/{club_id}/match-uploader/round-robin/preview",
        ): (
            "match-player",
            "league-live-domain",
            "league-live-submit",
        ),
        (
            "POST",
            "/admin/clubs/{club_id}/tournaments/admin/tournaments/{tournament_id}/draws/{draw_id}/matches/publish",
        ): (
            "tournament-official-publish",
            "tournament-email-handoff",
            "tournament-live-official-publish",
        ),
    }
    assert actual_overlaps == expected_overlaps


def test_get_health_routes_do_not_call_mutating_completion_claim_rpc() -> None:
    source = (ROOT / "services" / "api" / "main.py").read_text(encoding="utf-8")
    health_live_source = source.split('def health_live_sessions()', 1)[1].split(
        '@app.get("/clubs/{club_slug}")', 1
    )[0]
    assert "claim_public_live_completion_executor" not in health_live_source


def test_safe_route_bodies_have_no_direct_non_audit_mutating_sink() -> None:
    mutating_attributes = {"insert", "update", "delete", "upsert", "rpc"}
    violations: list[str] = []
    for source_path in (ROOT / "services" / "api").glob("*.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            methods = {
                decorator.func.attr
                for decorator in node.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
            }
            if not methods.intersection({"get", "head"}):
                continue
            sinks = set()
            for child in ast.walk(node):
                if not isinstance(child, ast.Attribute) or child.attr not in mutating_attributes:
                    continue
                receiver = ast.unparse(child.value)
                if ".table(" in receiver or receiver == "supabase":
                    sinks.add(child.attr)
            if sinks:
                violations.append(f"{source_path.name}:{node.lineno}:{sorted(sinks)}")
    assert violations == []


def test_wave_matching_is_exact_and_unknown_or_none_never_allows_writes() -> None:
    intake_path = "/clubs/tres-palapas/support/intake"
    challenge_path = "/admin/clubs/tres_palapas/challenge-ladder/challenges"
    round_robin_preview_path = (
        "/admin/clubs/tres_palapas/match-uploader/round-robin/preview"
    )
    exclusion_path = "/admin/clubs/fixture/match-log/exclude"
    recovery_path = (
        "/admin/clubs/fixture/match-log/exclusions/"
        "00000000-0000-4000-8000-000000000001/recover"
    )

    assert wave_allows_request("public-intake-auth", "POST", intake_path)
    assert not wave_allows_request("public-intake-auth", "POST", f"{intake_path}/extra")
    assert not wave_allows_request("public-intake-auth", "GET", intake_path)
    assert not wave_allows_request("public-intake-auth", "POST", challenge_path)
    assert wave_allows_request("challenge-ladder", "POST", challenge_path)
    assert wave_allows_request(
        "league-live-domain", "POST", round_robin_preview_path
    )
    assert wave_allows_request(
        "league-live-submit", "POST", round_robin_preview_path
    )
    assert not wave_allows_request(
        "league-live-domain", "POST", f"{round_robin_preview_path}/extra"
    )
    assert not wave_allows_request(
        NO_WRITE_WAVE, "POST", round_robin_preview_path
    )
    assert not wave_allows_request("match-player", "POST", exclusion_path)
    assert wave_allows_request(
        "match-exclusion-recovery", "POST", exclusion_path
    )
    assert wave_allows_request(
        "match-exclusion-recovery", "POST", recovery_path
    )
    assert not wave_allows_request(
        "match-exclusion-recovery", "POST", f"{recovery_path}/extra"
    )
    for uploader_write in ("singles", "batch", "players"):
        uploader_write_path = (
            f"/admin/clubs/tres_palapas/match-uploader/{uploader_write}"
        )
        assert not wave_allows_request(
            "league-live-domain", "POST", uploader_write_path
        )
        assert not wave_allows_request(
            "league-live-submit", "POST", uploader_write_path
        )
    assert not wave_allows_request(NO_WRITE_WAVE, "POST", intake_path)
    assert not wave_allows_request("unknown", "POST", intake_path)


def test_league_live_waves_open_preview_only_uploader_capability() -> None:
    for wave in ("league-live-domain", "league-live-submit"):
        flags = set(STAGING_WRITE_WAVES[wave])
        assert "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_PREVIEW" in flags
        assert "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER" not in flags


def test_direct_singles_uploader_gate_opens_only_in_atomic_match_wave() -> None:
    flag = "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES"

    assert flag not in DORMANT_STAGING_WRITE_FLAGS
    assert {
        wave for wave, flags in STAGING_WRITE_WAVES.items() if flag in flags
    } == {"match-player"}
    assert expected_write_flags("match-player")[flag] is True
    assert f'{flag} = "0"' in (ROOT / "fly.staging.toml").read_text(
        encoding="utf-8"
    )
    assert f'{flag} = "0"' in (ROOT / "fly.toml").read_text(encoding="utf-8")
    assert f'{flag}: "0"' in (
        ROOT / ".github/workflows/fly_api_staging_deploy.yml"
    ).read_text(encoding="utf-8")


def test_match_log_destructive_gate_opens_only_in_atomic_recovery_wave() -> None:
    flag = "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE"

    assert len(ALL_STAGING_WRITE_FLAGS) == 32
    assert flag not in DORMANT_STAGING_WRITE_FLAGS
    assert {
        wave for wave, flags in STAGING_WRITE_WAVES.items() if flag in flags
    } == {"match-exclusion-recovery"}
    assert expected_write_flags("match-player")[flag] is False
    assert expected_write_flags("match-exclusion-recovery") == {
        name: name
        in {
            "JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT",
            "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY",
            "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE",
            "JUPR_ENABLE_NEXT_ADMIN_REPLAY",
        }
        for name in ALL_STAGING_WRITE_FLAGS
    }
    assert f'{flag} = "0"' in (ROOT / "fly.staging.toml").read_text(
        encoding="utf-8"
    )
    assert f'{flag} = "0"' in (ROOT / "fly.toml").read_text(encoding="utf-8")
    assert f'{flag}: "0"' in (
        ROOT / ".github/workflows/fly_api_staging_deploy.yml"
    ).read_text(encoding="utf-8")


def _middleware_client() -> TestClient:
    app = FastAPI()
    app.add_middleware(StagingWriteWaveMiddleware)

    @app.post("/clubs/{club_slug}/support/intake")
    def mutate(club_slug: str) -> dict[str, str]:
        return {"club_slug": club_slug}

    @app.get("/clubs/{club_slug}/support/intake")
    def read(club_slug: str) -> dict[str, str]:
        return {"club_slug": club_slug}

    return TestClient(app)


def test_middleware_denies_unknown_runtime_and_read_only_staging(monkeypatch) -> None:
    client = _middleware_client()
    path = "/clubs/tres-palapas/support/intake"

    monkeypatch.delenv("JUPR_ENV", raising=False)
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "public-intake-auth")
    assert client.post(path).status_code == 403
    assert client.get(path).status_code == 200

    monkeypatch.setenv("JUPR_ENV", "stagin")
    assert client.post(path).status_code == 403

    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    assert client.post(path).status_code == 403


def test_middleware_opens_only_the_selected_exact_staging_route(monkeypatch) -> None:
    client = _middleware_client()
    path = "/clubs/tres-palapas/support/intake"
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "public-intake-auth")

    assert client.post(path).status_code == 200
    assert client.post(f"{path}/extra").status_code == 403


def test_canonical_normalize_status_and_guard_are_staging_only(monkeypatch) -> None:
    monkeypatch.setenv(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_MATCH_CANONICAL_NORMALIZE_WRITES", "1"
    )
    for environment in ("", "test", "local", "production", "stagin"):
        if environment:
            monkeypatch.setenv("JUPR_ENV", environment)
        else:
            monkeypatch.delenv("JUPR_ENV", raising=False)
        assert staging_match_canonical_normalize_writes_enabled() is False
        with pytest.raises(PermissionError):
            require_staging_match_canonical_normalize_writes()

    monkeypatch.setenv("JUPR_ENV", "staging")
    assert staging_match_canonical_normalize_writes_enabled() is True
    require_staging_match_canonical_normalize_writes()


def test_communications_mutation_guard_is_local_friendly_and_staging_exact(
    monkeypatch,
) -> None:
    monkeypatch.setenv(COMMUNICATIONS_MUTATION_FLAG, "1")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "communications")

    for environment in ("local", "test", "development", "dev"):
        monkeypatch.setenv("JUPR_ENV", environment)
        assert staging_communications_mutations_enabled() is True
        require_staging_communications_mutations()

    for environment in ("production", "prod", "stagin", "unknown"):
        monkeypatch.setenv("JUPR_ENV", environment)
        assert staging_communications_mutations_enabled() is False
        with pytest.raises(PermissionError, match="Communications mutations"):
            require_staging_communications_mutations()

    monkeypatch.delenv("JUPR_ENV", raising=False)
    assert staging_communications_mutations_enabled() is False

    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    assert staging_communications_mutations_enabled() is False

    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "communications")
    monkeypatch.setenv(COMMUNICATIONS_MUTATION_FLAG, "0")
    assert staging_communications_mutations_enabled() is False

    monkeypatch.setenv(COMMUNICATIONS_MUTATION_FLAG, "yes")
    assert staging_communications_mutations_enabled() is True
    require_staging_communications_mutations()


def test_every_communications_route_has_an_independent_service_guard() -> None:
    expected_routes = set(STAGING_WRITE_WAVE_ROUTES["communications"])
    guarded_routes: set[tuple[str, str]] = set()

    for source_name in (
        "admin_player_updates_routes.py",
        "admin_verified_updates_routes.py",
        "admin_weekly_recap_routes.py",
    ):
        source_path = ROOT / "services" / "api" / source_name
        tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = {
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            }
            if not calls.intersection(
                {
                    "_require_communications_mutations",
                    "require_staging_communications_mutations",
                }
            ):
                continue
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr in UNSAFE_METHODS
                    and decorator.args
                ):
                    guarded_routes.add(
                        (
                            decorator.func.attr.upper(),
                            ast.literal_eval(decorator.args[0]),
                        )
                    )

    assert guarded_routes == expected_routes
