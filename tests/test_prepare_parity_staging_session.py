from __future__ import annotations

import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from scripts import prepare_parity_staging_session as session


TOKEN = f"{'a' * 40}.{'b' * 80}.{'c' * 40}"
EMAIL = "staging-admin@example.invalid"
PASSWORD = "never-log-this-password"
ANON_KEY = "never-log-this-anon-key"


def _env(github_env: Path) -> dict[str, str]:
    return {
        "STAGING_API_BASE_URL": session.EXPECTED_API_ORIGIN,
        "STAGING_SUPABASE_URL": session.EXPECTED_SUPABASE_ORIGIN,
        "STAGING_SUPABASE_ANON_KEY": ANON_KEY,
        "STAGING_ADMIN_EMAIL": EMAIL,
        "STAGING_ADMIN_PASSWORD": PASSWORD,
        "GITHUB_ENV": str(github_env),
    }


def _auth_payload(*, email: str = EMAIL, token: str = TOKEN) -> dict:
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": {"email": email},
    }


def _capabilities(*, permissions: list[str] | None = None) -> dict:
    return {
        "authorized": True,
        "user": {"email": EMAIL},
        "requested_club_id": session.EXPECTED_CLUB_ID,
        "assignments": [
            {
                "club_id": session.EXPECTED_CLUB_ID,
                "role": "super_admin",
                "permissions": permissions
                or sorted(session.REQUIRED_PERMISSIONS | {"view_audit_log"}),
            }
        ],
    }


def _recaps() -> dict:
    return {
        "ok": True,
        "mode": "weekly_recap_list",
        "count": 1,
        "recaps": [
            {
                "id": session.EXPECTED_RECAP_ID,
                "week_start": session.EXPECTED_RECAP_WEEK_START,
                "status": "draft",
            }
        ],
    }


def _tournaments() -> dict:
    return {
        "ok": True,
        "mode": "tournament_admin_list",
        "count": 1,
        "tournaments": [{"id": session.EXPECTED_TOURNAMENT_ID}],
    }


def _snapshot(*, draw_id: str = session.EXPECTED_DRAW_ID) -> dict:
    return {
        "ok": True,
        "mode": "tournament_ops_snapshot",
        "tournament": {"id": session.EXPECTED_TOURNAMENT_ID},
        "draw_id": draw_id,
        "draws": [
            {
                "id": draw_id,
                "tournament_id": session.EXPECTED_TOURNAMENT_ID,
            }
        ],
        "state_ready": True,
    }


class FakeTransport:
    def __init__(self, responses: list[dict | Exception]):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return 200, json.dumps(response).encode("utf-8")


def test_admin_read_export_mints_token_validates_fixtures_and_appends_env(
    tmp_path, capsys
):
    github_env = tmp_path / "github-env"
    github_env.write_text("EXISTING=value\n", encoding="utf-8")
    transport = FakeTransport(
        [_auth_payload(), _capabilities(), _recaps(), _tournaments(), _snapshot()]
    )

    prepared = session.prepare_staging_session(
        "admin-read-export",
        env=_env(github_env),
        transport=transport,
    )

    assert prepared == {
        "STAGING_ADMIN_BEARER_TOKEN": TOKEN,
        "JUPR_STAGING_ADMIN_ACCESS_TOKEN": TOKEN,
        "JUPR_COMMUNICATIONS_DRAFT_WEEK_START": session.EXPECTED_RECAP_WEEK_START,
        "JUPR_TOURNAMENT_OPS_TOURNAMENT_ID": session.EXPECTED_TOURNAMENT_ID,
        "JUPR_TOURNAMENT_OPS_DRAW_ID": session.EXPECTED_DRAW_ID,
    }
    assert github_env.read_text(encoding="utf-8").splitlines() == [
        "EXISTING=value",
        f"STAGING_ADMIN_BEARER_TOKEN={TOKEN}",
        f"JUPR_STAGING_ADMIN_ACCESS_TOKEN={TOKEN}",
        f"JUPR_COMMUNICATIONS_DRAFT_WEEK_START={session.EXPECTED_RECAP_WEEK_START}",
        f"JUPR_TOURNAMENT_OPS_TOURNAMENT_ID={session.EXPECTED_TOURNAMENT_ID}",
        f"JUPR_TOURNAMENT_OPS_DRAW_ID={session.EXPECTED_DRAW_ID}",
    ]
    output = capsys.readouterr()
    assert output.out == f"::add-mask::{TOKEN}\n"
    assert PASSWORD not in output.out + output.err
    assert ANON_KEY not in output.out + output.err

    auth_request = transport.requests[0]
    assert auth_request.get_method() == "POST"
    assert (
        auth_request.full_url
        == f"{session.EXPECTED_SUPABASE_ORIGIN}/auth/v1/token?grant_type=password"
    )
    assert json.loads(auth_request.data) == {"email": EMAIL, "password": PASSWORD}
    assert auth_request.get_header("Apikey") == ANON_KEY
    assert transport.requests[1].get_header("Authorization") == f"Bearer {TOKEN}"
    assert (
        transport.requests[-1].full_url
        == f"{session.EXPECTED_API_ORIGIN}/admin/clubs/tres_palapas/tournaments/admin/"
        f"tournaments/{session.EXPECTED_TOURNAMENT_ID}/ops?"
        f"draw_id={session.EXPECTED_DRAW_ID}"
    )
    assert transport.responses == []


def test_match_rating_writes_only_requires_authenticated_capabilities(tmp_path, capsys):
    github_env = tmp_path / "github-env"
    transport = FakeTransport([_auth_payload(), _capabilities()])

    prepared = session.prepare_staging_session(
        "match-rating-writes",
        env=_env(github_env),
        transport=transport,
    )

    assert prepared == {
        "STAGING_ADMIN_BEARER_TOKEN": TOKEN,
        "JUPR_STAGING_ADMIN_ACCESS_TOKEN": TOKEN,
    }
    assert len(transport.requests) == 2
    assert github_env.read_text(encoding="utf-8").splitlines() == [
        f"STAGING_ADMIN_BEARER_TOKEN={TOKEN}",
        f"JUPR_STAGING_ADMIN_ACCESS_TOKEN={TOKEN}",
    ]
    assert capsys.readouterr().out == f"::add-mask::{TOKEN}\n"


@pytest.mark.parametrize(
    ("auth_payload", "expected_error"),
    [
        (_auth_payload(email="somebody-else@example.invalid"), "unexpected user identity"),
        (_auth_payload(token="not-a-jwt"), "invalid access token"),
    ],
)
def test_auth_identity_and_token_validation_fail_before_masking_or_env_write(
    tmp_path, capsys, auth_payload, expected_error
):
    github_env = tmp_path / "github-env"
    transport = FakeTransport([auth_payload])

    with pytest.raises(session.SessionPreparationError, match=expected_error):
        session.prepare_staging_session(
            "admin-read-export",
            env=_env(github_env),
            transport=transport,
        )

    output = capsys.readouterr()
    assert output.out == ""
    assert PASSWORD not in output.out + output.err
    assert not github_env.exists()


def test_capability_validation_requires_all_expected_permissions(tmp_path, capsys):
    github_env = tmp_path / "github-env"
    transport = FakeTransport(
        [
            _auth_payload(),
            _capabilities(permissions=["manage_matches", "manage_tournaments"]),
        ]
    )

    with pytest.raises(
        session.SessionPreparationError, match="manage_subscriptions"
    ):
        session.prepare_staging_session(
            "match-rating-writes",
            env=_env(github_env),
            transport=transport,
        )

    assert capsys.readouterr().out == ""
    assert not github_env.exists()


@pytest.mark.parametrize(
    "responses",
    [
        [
            _auth_payload(),
            _capabilities(),
            {
                "ok": True,
                "mode": "weekly_recap_list",
                "recaps": [
                    {
                        "id": session.EXPECTED_RECAP_ID,
                        "week_start": "2099-02-01",
                        "status": "draft",
                    }
                ],
            },
        ],
        [
            _auth_payload(),
            _capabilities(),
            _recaps(),
            _tournaments(),
            _snapshot(draw_id="wrong-draw"),
        ],
    ],
)
def test_admin_fixture_mismatch_fails_before_masking_or_env_write(
    tmp_path, capsys, responses
):
    github_env = tmp_path / "github-env"

    with pytest.raises(session.SessionPreparationError, match="fixture|snapshot"):
        session.prepare_staging_session(
            "admin-read-export",
            env=_env(github_env),
            transport=FakeTransport(responses),
        )

    assert capsys.readouterr().out == ""
    assert not github_env.exists()


def test_http_error_never_exposes_sensitive_response_body(tmp_path, capsys):
    github_env = tmp_path / "github-env"
    secret_body = f"{PASSWORD} {TOKEN}".encode()
    error = HTTPError(
        "https://example.invalid",
        401,
        "unauthorized",
        {},
        None,
    )
    error.read = lambda: secret_body

    with pytest.raises(
        session.SessionPreparationError,
        match=r"Supabase staging authentication failed with HTTP 401",
    ) as caught:
        session.prepare_staging_session(
            "admin-read-export",
            env=_env(github_env),
            transport=FakeTransport([error]),
        )

    output = capsys.readouterr()
    observed = f"{caught.value}{output.out}{output.err}"
    assert PASSWORD not in observed
    assert TOKEN not in observed
    assert not github_env.exists()


def test_default_transport_redirect_handler_never_forwards_credentials():
    request = Request(
        f"{session.EXPECTED_SUPABASE_ORIGIN}/auth/v1/token",
        headers={"Authorization": f"Bearer {ANON_KEY}", "apikey": ANON_KEY},
    )
    redirected = session._RefuseRedirects().redirect_request(
        request,
        None,
        302,
        "Found",
        {"Location": "https://attacker.example.invalid/capture"},
        "https://attacker.example.invalid/capture",
    )

    assert redirected is None


def test_refuses_any_supabase_project_other_than_exact_staging(tmp_path):
    github_env = tmp_path / "github-env"
    env = _env(github_env)
    env["STAGING_SUPABASE_URL"] = "https://production.example.invalid"
    transport = FakeTransport([])

    with pytest.raises(
        session.SessionPreparationError, match="permitted staging project"
    ):
        session.prepare_staging_session(
            "admin-read-export",
            env=env,
            transport=transport,
        )

    assert transport.requests == []
    assert not github_env.exists()


def test_refuses_any_api_other_than_exact_staging(tmp_path):
    github_env = tmp_path / "github-env"
    env = _env(github_env)
    env["STAGING_API_BASE_URL"] = "https://production.example.invalid"
    transport = FakeTransport([])

    with pytest.raises(
        session.SessionPreparationError, match="permitted staging API"
    ):
        session.prepare_staging_session(
            "admin-read-export",
            env=env,
            transport=transport,
        )

    assert transport.requests == []
    assert not github_env.exists()
