from __future__ import annotations

from pathlib import Path
import urllib.request

import pytest

from scripts import smoke_public_web as smoke


def _check(url: str, *, allow_vercel_bypass: bool = True) -> smoke.SmokeCheck:
    return smoke.SmokeCheck(
        "test",
        url,
        (200,),
        allow_vercel_bypass=allow_vercel_bypass,
    )


def test_vercel_bypass_headers_are_limited_to_explicit_https_web_checks(monkeypatch):
    monkeypatch.setenv(smoke.VERCEL_AUTOMATION_BYPASS_SECRET_ENV, "  staging-secret  ")

    assert smoke._vercel_bypass_headers(_check("https://jupr-git-staging-team.vercel.app/page")) == {
        "x-vercel-protection-bypass": "staging-secret",
    }


@pytest.mark.parametrize(
    ("url", "allow_vercel_bypass"),
    [
        ("https://juprleagues-api-staging.fly.dev/health", True),
        ("https://example.com/page", True),
        ("https://preview.vercel.app.attacker.example/page", True),
        ("http://preview.vercel.app/page", True),
        ("https://user@preview.vercel.app/page", True),
        ("https://preview.vercel.app:8443/page", True),
        ("https://preview.vercel.app/page", False),
    ],
)
def test_vercel_bypass_headers_are_not_sent_to_untrusted_requests(
    monkeypatch,
    url: str,
    allow_vercel_bypass: bool,
):
    monkeypatch.setenv(smoke.VERCEL_AUTOMATION_BYPASS_SECRET_ENV, "staging-secret")

    assert smoke._vercel_bypass_headers(
        _check(url, allow_vercel_bypass=allow_vercel_bypass)
    ) == {}


def test_cross_origin_redirect_strips_vercel_bypass_headers():
    request = urllib.request.Request(
        "https://preview.vercel.app/page",
        headers={
            "x-vercel-protection-bypass": "staging-secret",
            "x-vercel-set-bypass-cookie": "true",
        },
    )

    redirected = smoke._SameOriginRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://attacker.example/redirected",
    )

    assert redirected is not None
    assert redirected.get_header("X-vercel-protection-bypass") is None
    assert redirected.get_header("X-vercel-set-bypass-cookie") is None


def test_same_origin_redirect_keeps_vercel_bypass_headers():
    request = urllib.request.Request(
        "https://preview.vercel.app/page",
        headers={
            "x-vercel-protection-bypass": "staging-secret",
            "x-vercel-set-bypass-cookie": "true",
        },
    )

    redirected = smoke._SameOriginRedirectHandler().redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://preview.vercel.app/redirected",
    )

    assert redirected is not None
    assert redirected.get_header("X-vercel-protection-bypass") == "staging-secret"
    assert redirected.get_header("X-vercel-set-bypass-cookie") == "true"


def test_request_wires_bypass_headers_without_putting_secret_in_result(monkeypatch):
    captured: dict[str, urllib.request.Request] = {}

    class FakeResponse:
        status = 200
        headers = {"content-type": "text/html"}

        def read(self, _size: int = -1) -> bytes:
            return b"ok"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeOpener:
        def open(self, request: urllib.request.Request, *, timeout: float):
            captured["request"] = request
            assert timeout == 1.0
            return FakeResponse()

    monkeypatch.setenv(smoke.VERCEL_AUTOMATION_BYPASS_SECRET_ENV, "staging-secret")
    monkeypatch.setattr(
        smoke.urllib.request,
        "build_opener",
        lambda *_handlers: FakeOpener(),
    )

    result = smoke._request(
        _check("https://preview.vercel.app/page"),
        timeout_seconds=1.0,
    )

    assert result.ok is True
    assert captured["request"].get_header("X-vercel-protection-bypass") == "staging-secret"
    assert captured["request"].get_header("X-vercel-set-bypass-cookie") is None
    assert "staging-secret" not in repr(result)


def test_api_check_table_tracks_current_auth_and_tournament_contracts():
    checks = {
        check.name: check
        for check in smoke._api_get_checks(
            "https://juprleagues-api-staging.fly.dev",
            "tres-palapas",
            "tres_palapas",
            allow_live_unconfigured=True,
        )
    }

    match_log = checks["api: unauthenticated admin match log read blocked"]
    assert match_log.url.endswith("/admin/clubs/tres_palapas/match-log")
    assert match_log.expected_statuses == (401,)

    operations = checks[
        "api: unauthenticated admin operations status blocked"
    ]
    assert operations.url.endswith("/admin/operations/status")
    assert operations.expected_statuses == (401,)

    tournament = checks["api: admin tournament"]
    assert tournament.url.endswith("/admin/clubs/tres_palapas/tournaments/admin/status")
    assert tournament.expected_statuses == (200,)

    commerce = checks["api: admin tournament commerce"]
    assert commerce.method == "GET"
    assert commerce.url.endswith(
        "/admin/clubs/tres_palapas/tournaments/commerce/status"
    )
    assert commerce.expected_statuses == (200, 401)

    team_tournament = checks["api: admin team tournament competition"]
    assert team_tournament.method == "GET"
    assert team_tournament.url.endswith(
        "/admin/clubs/tres_palapas/tournaments/team-competition/status"
    )
    assert team_tournament.expected_statuses == (401, 403)

    published_team_tournament = checks["api: team tournament results"]
    assert published_team_tournament.method == "GET"
    assert published_team_tournament.url.endswith(
        "/clubs/tres-palapas/tournament-team-results"
    )
    assert published_team_tournament.expected_statuses == (200, 404)

    admin_team_leagues = checks["api: admin team leagues"]
    assert admin_team_leagues.method == "GET"
    assert admin_team_leagues.url.endswith(
        "/admin/clubs/tres_palapas/league-manager/team-leagues"
    )
    assert admin_team_leagues.expected_statuses == (401, 403)

    published_team_leagues = checks["api: team leagues"]
    assert published_team_leagues.method == "GET"
    assert published_team_leagues.url.endswith("/clubs/tres-palapas/team-leagues")
    assert published_team_leagues.expected_statuses == (200, 403)


def test_web_checks_opt_in_to_vercel_bypass_without_exposing_secret(monkeypatch):
    monkeypatch.setenv(smoke.VERCEL_AUTOMATION_BYPASS_SECRET_ENV, "staging-secret")
    checks = smoke._web_get_checks(
        "https://jupr-git-staging-team.vercel.app",
        "tres-palapas",
    )

    assert checks
    assert all(check.allow_vercel_bypass for check in checks)
    assert all("staging-secret" not in check.url for check in checks)
    assert all(check.method == "GET" for check in checks)
    urls = {check.url for check in checks}
    assert {
        "https://jupr-git-staging-team.vercel.app/admin/tournaments/commerce",
        "https://jupr-git-staging-team.vercel.app/admin/tournaments/team-competition",
        "https://jupr-git-staging-team.vercel.app/admin/league-manager/teams",
        "https://jupr-git-staging-team.vercel.app/admin/league-manager/awards",
        "https://jupr-git-staging-team.vercel.app/clubs/tres-palapas/tournament-team-results",
        "https://jupr-git-staging-team.vercel.app/clubs/tres-palapas/team-leagues",
    }.issubset(urls)


def test_new_feature_browser_smoke_is_strictly_read_only():
    spec = (
        Path("apps/web/e2e/staging.smoke.spec.ts")
        .read_text(encoding="utf-8")
    )
    feature_block = spec.split(
        "const readOnlyFeatureAreas: ReadOnlyFeatureArea[] = [", 1
    )[1].split("test.beforeEach", 1)[0]
    test_block = spec.split(
        "for (const featureArea of readOnlyFeatureAreas)", 1
    )[1].split("for (const surface of publicSurfaces)", 1)[0]

    assert "/admin/tournaments/commerce" in feature_block
    assert "/admin/tournaments/team-competition" in feature_block
    assert "/admin/league-manager/teams" in feature_block
    assert "/admin/league-manager/awards" in feature_block
    assert "/tournament-team-results" in feature_block
    assert "/team-leagues" in feature_block
    assert "page.request.get(" in test_block
    for unsafe_method in ("post", "put", "patch", "delete"):
        assert f"page.request.{unsafe_method}(" not in test_block
