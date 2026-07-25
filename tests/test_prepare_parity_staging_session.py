from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request

import pytest

from scripts import prepare_parity_staging_session as session

TOKEN_HASH = "d" * 64
USER_ID = "c69be83f-b1db-4b1b-bc4c-99c8df45f623"
OTHER_USER_ID = "9170a273-a8ed-469e-8f25-30f3f53e9abb"
SESSION_ID = "07f80228-38f9-4cbc-899c-420e30aa4483"
EMAIL = "staging-admin@example.invalid"
ROLE = "super_admin"
ANON_KEY = "never-log-this-anon-key"
SERVICE_ROLE_KEY = "never-log-this-service-role-key"
ISSUED_AT = int(time.time())


def _base64url(value: object) -> str:
    raw = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _token(*, claims: dict[str, object] | None = None) -> str:
    token_claims: dict[str, object] = {
        "sub": USER_ID,
        "iss": session.EXPECTED_SUPABASE_ISSUER,
        "aud": session.EXPECTED_AUTH_AUDIENCE,
        "email": EMAIL,
        "session_id": SESSION_ID,
        "iat": ISSUED_AT,
        "exp": ISSUED_AT + session.MAX_ACCESS_JWT_LIFETIME_SECONDS,
    }
    if claims is not None:
        token_claims.update(claims)
    return ".".join(
        (
            _base64url({"alg": "HS256", "typ": "JWT"}),
            _base64url(token_claims),
            base64.urlsafe_b64encode(b"test-signature-not-for-verification")
            .rstrip(b"=")
            .decode("ascii"),
        )
    )


def _token_with_raw_payload(raw_payload: bytes) -> str:
    return ".".join(
        (
            _base64url({"alg": "HS256", "typ": "JWT"}),
            base64.urlsafe_b64encode(raw_payload).rstrip(b"=").decode("ascii"),
            base64.urlsafe_b64encode(b"test-signature-not-for-verification")
            .rstrip(b"=")
            .decode("ascii"),
        )
    )


TOKEN = _token()


def _env(github_env: Path) -> dict[str, str]:
    return {
        "STAGING_API_BASE_URL": session.EXPECTED_API_ORIGIN,
        "STAGING_SUPABASE_URL": session.EXPECTED_SUPABASE_ORIGIN,
        "STAGING_SUPABASE_ANON_KEY": ANON_KEY,
        "STAGING_SUPABASE_SERVICE_ROLE_KEY": SERVICE_ROLE_KEY,
        "GITHUB_ENV": str(github_env),
    }


def _assignment(
    *,
    user_id: object = USER_ID,
    email: object = EMAIL,
    role: object = ROLE,
    assignment_id: object = 17,
) -> list[dict]:
    return [
        {
            "id": assignment_id,
            "club_id": session.EXPECTED_CLUB_ID,
            "email": email,
            "role": role,
            "user_id": user_id,
        }
    ]


def _auth_admin_user(
    *,
    user_id: object = USER_ID,
    email: object = EMAIL,
    confirmed: bool = True,
) -> dict:
    return {
        "id": user_id,
        "email": email,
        "email_confirmed_at": "2026-07-01T12:00:00Z" if confirmed else None,
        "deleted_at": None,
        "is_anonymous": False,
    }


def _generated_link(
    *,
    user_id: object = USER_ID,
    email: object = EMAIL,
    token_hash: object = TOKEN_HASH,
    verification_type: object = "magiclink",
) -> dict:
    return {
        "action_link": "https://do-not-record.example.invalid/link",
        "hashed_token": token_hash,
        "verification_type": verification_type,
        "id": user_id,
        "email": email,
    }


def _auth_payload(
    *,
    user_id: object = USER_ID,
    email: object = EMAIL,
    token: object = TOKEN,
    token_type: object = "bearer",
    expires_in: object = session.MAX_ACCESS_JWT_LIFETIME_SECONDS,
    expires_at: object = ISSUED_AT + session.MAX_ACCESS_JWT_LIFETIME_SECONDS,
) -> dict:
    return {
        "access_token": token,
        "token_type": token_type,
        "expires_in": expires_in,
        "expires_at": expires_at,
        "user": {"id": user_id, "email": email},
    }


def _without(payload: dict, key: str) -> dict:
    result = dict(payload)
    result.pop(key)
    return result


def _capabilities(
    *,
    email: object = EMAIL,
    role: object = ROLE,
    permissions: list[str] | None = None,
) -> dict:
    return {
        "authorized": True,
        "user": {"email": email},
        "requested_club_id": session.EXPECTED_CLUB_ID,
        "assignments": [
            {
                "club_id": session.EXPECTED_CLUB_ID,
                "role": role,
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


def _authenticated_responses() -> list[dict]:
    return [
        _assignment(),
        _auth_admin_user(),
        _generated_link(),
        _auth_payload(),
        _capabilities(),
    ]


class FakeTransport:
    def __init__(self, responses: list[object]):
        self.responses = list(responses)
        self.requests: list[Request] = []

    def __call__(self, request: Request) -> tuple[int, bytes]:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("Unexpected extra HTTP request.")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        status = 200
        payload = response
        if (
            isinstance(response, tuple)
            and len(response) == 2
            and isinstance(response[0], int)
        ):
            status, payload = response
        if isinstance(payload, bytes):
            raw = payload
        elif payload is None:
            raw = b""
        else:
            raw = json.dumps(payload).encode("utf-8")
        return status, raw


def _read_report(report_dir: Path, filename: str) -> tuple[dict, str]:
    report_text = (report_dir / filename).read_text(encoding="utf-8")
    return json.loads(report_text), report_text


def test_admin_read_export_avoids_role_dml_and_user_lifecycle_writes(tmp_path, capsys):
    github_env = tmp_path / "github-env"
    github_env.write_text("EXISTING=value\n", encoding="utf-8")
    report_dir = tmp_path / "reports"
    transport = FakeTransport(
        [*_authenticated_responses(), _recaps(), _tournaments(), _snapshot()]
    )

    prepared = session.prepare_staging_session(
        "admin-read-export",
        env=_env(github_env),
        transport=transport,
        report_dir=report_dir,
    )

    assert prepared == {
        "STAGING_ADMIN_BEARER_TOKEN": TOKEN,
        "JUPR_STAGING_ADMIN_ACCESS_TOKEN": TOKEN,
        "STAGING_ADMIN_EMAIL": EMAIL,
        "JUPR_STAGING_ADMIN_EMAIL": EMAIL,
        "JUPR_REAL_AUTH_EXPECTED_ROLE": ROLE,
        "JUPR_COMMUNICATIONS_DRAFT_WEEK_START": session.EXPECTED_RECAP_WEEK_START,
        "JUPR_TOURNAMENT_OPS_TOURNAMENT_ID": session.EXPECTED_TOURNAMENT_ID,
        "JUPR_TOURNAMENT_OPS_DRAW_ID": session.EXPECTED_DRAW_ID,
    }
    assert github_env.read_text(encoding="utf-8").splitlines() == [
        "EXISTING=value",
        f"STAGING_ADMIN_BEARER_TOKEN={TOKEN}",
        f"JUPR_STAGING_ADMIN_ACCESS_TOKEN={TOKEN}",
        f"STAGING_ADMIN_EMAIL={EMAIL}",
        f"JUPR_STAGING_ADMIN_EMAIL={EMAIL}",
        f"JUPR_REAL_AUTH_EXPECTED_ROLE={ROLE}",
        f"JUPR_COMMUNICATIONS_DRAFT_WEEK_START={session.EXPECTED_RECAP_WEEK_START}",
        f"JUPR_TOURNAMENT_OPS_TOURNAMENT_ID={session.EXPECTED_TOURNAMENT_ID}",
        f"JUPR_TOURNAMENT_OPS_DRAW_ID={session.EXPECTED_DRAW_ID}",
    ]
    output = capsys.readouterr()
    assert output.out == f"::add-mask::{TOKEN}\n::add-mask::{EMAIL}\n"
    assert SERVICE_ROLE_KEY not in output.out + output.err
    assert ANON_KEY not in output.out + output.err
    assert TOKEN_HASH not in output.out + output.err

    assert [request.get_method() for request in transport.requests] == [
        "GET",
        "GET",
        "POST",
        "POST",
        "GET",
        "GET",
        "GET",
        "GET",
    ]
    role_lookup = transport.requests[0]
    assert role_lookup.full_url.startswith(
        f"{session.EXPECTED_SUPABASE_ORIGIN}/rest/v1/admin_role_assignments?"
    )
    role_query = parse_qs(urlsplit(role_lookup.full_url).query)
    assert role_query == {
        "select": ["id,club_id,email,role,user_id"],
        "club_id": [f"eq.{session.EXPECTED_CLUB_ID}"],
        "role": ["in.(club_owner,super_admin)"],
        "user_id": ["not.is.null"],
        "order": ["id.asc"],
        "limit": ["2"],
    }
    assert role_lookup.get_header("Authorization") == f"Bearer {SERVICE_ROLE_KEY}"
    assert (
        transport.requests[1].full_url
        == f"{session.EXPECTED_SUPABASE_ORIGIN}/auth/v1/admin/users/{USER_ID}"
    )
    generate_request = transport.requests[2]
    assert generate_request.full_url == (
        f"{session.EXPECTED_SUPABASE_ORIGIN}/auth/v1/admin/generate_link"
    )
    assert json.loads(generate_request.data) == {
        "type": "magiclink",
        "email": EMAIL,
    }
    verify_request = transport.requests[3]
    assert verify_request.full_url == (
        f"{session.EXPECTED_SUPABASE_ORIGIN}/auth/v1/verify"
    )
    assert json.loads(verify_request.data) == {
        "token_hash": TOKEN_HASH,
        "type": "magiclink",
    }
    assert verify_request.get_header("Apikey") == ANON_KEY
    assert transport.requests[4].get_header("Authorization") == f"Bearer {TOKEN}"
    assert not any(
        request.get_method() in {"DELETE", "PATCH", "PUT"}
        or (
            request.get_method() == "POST"
            and (
                "/auth/v1/admin/users" in request.full_url
                or "/rest/v1/admin_role_assignments" in request.full_url
            )
        )
        for request in transport.requests
    )
    assert transport.responses == []

    report, report_text = _read_report(
        report_dir,
        session.PREPARATION_REPORT_NAME,
    )
    assert report["status"] == "passed"
    assert report["mode"] == "admin-read-export"
    assert report["identity"] == {
        "email_sha256": hashlib.sha256(EMAIL.encode("utf-8")).hexdigest(),
        "role": ROLE,
    }
    assert report["validated_fixture_sets"] == [
        "weekly_recap",
        "tournament",
        "tournament_draw",
    ]
    for sensitive in (
        EMAIL,
        TOKEN,
        TOKEN_HASH,
        ANON_KEY,
        SERVICE_ROLE_KEY,
        USER_ID,
        "action_link",
    ):
        assert sensitive not in report_text


@pytest.mark.parametrize("mode", session.SUPPORTED_MODES)
def test_every_authenticated_mode_uses_the_same_bootstrap(mode, tmp_path):
    github_env = tmp_path / "github-env"
    responses = _authenticated_responses()
    if mode == "admin-read-export":
        responses.extend([_recaps(), _tournaments(), _snapshot()])
    transport = FakeTransport(responses)

    prepared = session.prepare_staging_session(
        mode,
        env=_env(github_env),
        transport=transport,
    )

    assert prepared["STAGING_ADMIN_BEARER_TOKEN"] == TOKEN
    assert prepared["STAGING_ADMIN_EMAIL"] == EMAIL
    assert prepared["JUPR_REAL_AUTH_EXPECTED_ROLE"] == ROLE
    assert len(transport.requests) == (8 if mode == "admin-read-export" else 5)
    assert transport.responses == []


@pytest.mark.parametrize(
    ("assignment_payload", "expected_error"),
    [
        ([], "exactly one eligible bound admin assignment"),
        (
            [*_assignment(), {**_assignment()[0], "id": 18}],
            "exactly one eligible bound admin assignment",
        ),
        (_assignment(user_id=None), "invalid response"),
        (_assignment(user_id="not-a-uuid"), "invalid response"),
        (_assignment(role="read_only"), "invalid response"),
        (_assignment(email=" Staging-Admin@example.invalid "), "invalid response"),
        (_assignment(assignment_id=0), "invalid response"),
    ],
)
def test_assignment_lookup_fails_closed(assignment_payload, expected_error, tmp_path):
    github_env = tmp_path / "github-env"
    transport = FakeTransport([assignment_payload])

    with pytest.raises(session.SessionPreparationError, match=expected_error):
        session.prepare_staging_session(
            "public-intake-auth",
            env=_env(github_env),
            transport=transport,
        )

    assert len(transport.requests) == 1
    assert not github_env.exists()


@pytest.mark.parametrize(
    "auth_user",
    [
        _auth_admin_user(user_id=OTHER_USER_ID),
        _auth_admin_user(email="somebody-else@example.invalid"),
        _auth_admin_user(confirmed=False),
        {**_auth_admin_user(), "deleted_at": "2026-07-25T00:00:00Z"},
        {**_auth_admin_user(), "is_anonymous": True},
    ],
)
def test_auth_admin_user_must_match_confirmed_bound_identity(auth_user, tmp_path):
    github_env = tmp_path / "github-env"
    transport = FakeTransport([_assignment(), auth_user])

    with pytest.raises(
        session.SessionPreparationError,
        match="did not match the bound assignment",
    ):
        session.prepare_staging_session(
            "match-rating-writes",
            env=_env(github_env),
            transport=transport,
        )

    assert len(transport.requests) == 2
    assert not github_env.exists()


@pytest.mark.parametrize(
    ("link_payload", "expected_error"),
    [
        (_generated_link(token_hash="short"), "invalid response"),
        (_generated_link(verification_type="recovery"), "invalid response"),
        (_generated_link(user_id=OTHER_USER_ID), "unexpected user identity"),
        (
            _generated_link(email="somebody-else@example.invalid"),
            "unexpected user identity",
        ),
        (
            {
                "hashed_token": TOKEN_HASH,
                "verification_type": "magiclink",
                "user": {"id": USER_ID, "email": EMAIL},
            },
            "invalid response",
        ),
    ],
)
def test_generate_link_response_must_be_candidate_identity_bound(
    link_payload, expected_error, tmp_path
):
    github_env = tmp_path / "github-env"
    transport = FakeTransport([_assignment(), _auth_admin_user(), link_payload])

    with pytest.raises(session.SessionPreparationError, match=expected_error):
        session.prepare_staging_session(
            "match-rating-writes",
            env=_env(github_env),
            transport=transport,
        )

    assert len(transport.requests) == 3
    assert not github_env.exists()


@pytest.mark.parametrize(
    ("auth_payload", "expected_error", "expects_cleanup"),
    [
        (
            _auth_payload(user_id=OTHER_USER_ID),
            "unexpected user identity",
            True,
        ),
        (
            _auth_payload(email="somebody-else@example.invalid"),
            "unexpected user identity",
            True,
        ),
        (_auth_payload(token="not-a-jwt"), "invalid access token", False),
        ({**_auth_payload(), "token_type": "mac"}, "invalid token type", True),
    ],
)
def test_verified_session_must_match_identity_and_token_contract(
    auth_payload, expected_error, expects_cleanup, tmp_path, capsys
):
    github_env = tmp_path / "github-env"
    responses: list[object] = [
        _assignment(),
        _auth_admin_user(),
        _generated_link(),
        auth_payload,
    ]
    if expects_cleanup:
        responses.append((204, None))
    transport = FakeTransport(responses)

    with pytest.raises(session.SessionPreparationError, match=expected_error):
        session.prepare_staging_session(
            "match-rating-writes",
            env=_env(github_env),
            transport=transport,
        )

    assert len(transport.requests) == (5 if expects_cleanup else 4)
    if expects_cleanup:
        assert transport.requests[-1].full_url == (
            f"{session.EXPECTED_SUPABASE_ORIGIN}/auth/v1/logout?scope=local"
        )
        env_lines = github_env.read_text(encoding="utf-8").splitlines()
        assert env_lines[0] == f"STAGING_ADMIN_BEARER_TOKEN={TOKEN}"
        assert env_lines[-5:] == [
            "STAGING_ADMIN_BEARER_TOKEN=",
            "JUPR_STAGING_ADMIN_ACCESS_TOKEN=",
            "STAGING_ADMIN_EMAIL=",
            "JUPR_STAGING_ADMIN_EMAIL=",
            "JUPR_REAL_AUTH_EXPECTED_ROLE=",
        ]
        assert capsys.readouterr().out == f"::add-mask::{TOKEN}\n"
    else:
        assert not github_env.exists()
        assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    ("auth_payload", "expected_error"),
    [
        (_without(_auth_payload(), "token_type"), "invalid token type"),
        (_auth_payload(token_type="Bearer"), "invalid token type"),
        (_auth_payload(token_type=" bearer"), "invalid token type"),
        (_without(_auth_payload(), "expires_in"), "invalid access-token lifetime"),
        (_auth_payload(expires_in=True), "invalid access-token lifetime"),
        (_auth_payload(expires_in="3600"), "invalid access-token lifetime"),
        (_auth_payload(expires_in=0), "invalid access-token lifetime"),
        (
            _auth_payload(
                expires_in=session.MAX_ACCESS_JWT_LIFETIME_SECONDS + 1,
            ),
            "invalid access-token lifetime",
        ),
        (
            _auth_payload(token=_token_with_raw_payload(b"x" * 80)),
            "invalid access-token claims",
        ),
        (
            _auth_payload(token=_token(claims={"sub": OTHER_USER_ID})),
            "invalid access-token claims",
        ),
        (
            _auth_payload(
                token=_token(
                    claims={"iss": "https://production.example.invalid/auth/v1"}
                )
            ),
            "invalid access-token claims",
        ),
        (
            _auth_payload(token=_token(claims={"aud": ["other"]})),
            "invalid access-token claims",
        ),
        (
            _auth_payload(token=_token(claims={"aud": ["authenticated", 17]})),
            "invalid access-token claims",
        ),
        (
            _auth_payload(
                token=_token(claims={"email": "somebody-else@example.invalid"})
            ),
            "invalid access-token claims",
        ),
        (
            _auth_payload(token=_token(claims={"session_id": "not-a-uuid"})),
            "invalid access-token claims",
        ),
        (
            _auth_payload(token=_token(claims={"iat": True})),
            "invalid access-token lifetime",
        ),
        (
            _auth_payload(token=_token(claims={"exp": "never"})),
            "invalid access-token lifetime",
        ),
        (
            _auth_payload(
                token=_token(
                    claims={
                        "exp": ISSUED_AT + session.MAX_ACCESS_JWT_LIFETIME_SECONDS + 1
                    }
                )
            ),
            "invalid access-token lifetime",
        ),
        (
            _auth_payload(
                token=_token(
                    claims={
                        "iat": ISSUED_AT - 4000,
                        "exp": ISSUED_AT - 400,
                    }
                )
            ),
            "invalid access-token lifetime",
        ),
        (
            _auth_payload(
                token=_token(
                    claims={
                        "iat": ISSUED_AT + session.ACCESS_JWT_CLOCK_SKEW_SECONDS + 1,
                        "exp": ISSUED_AT
                        + session.ACCESS_JWT_CLOCK_SKEW_SECONDS
                        + 1
                        + session.MAX_ACCESS_JWT_LIFETIME_SECONDS,
                    }
                )
            ),
            "invalid access-token lifetime",
        ),
        (
            _auth_payload(expires_in=600),
            "invalid access-token lifetime",
        ),
        (
            _auth_payload(expires_at="soon"),
            "invalid access-token lifetime",
        ),
        (
            _auth_payload(
                expires_at=ISSUED_AT + session.MAX_ACCESS_JWT_LIFETIME_SECONDS - 1
            ),
            "invalid access-token lifetime",
        ),
    ],
)
def test_verified_jwt_preflight_fails_closed_and_ends_refresh_session(
    auth_payload, expected_error, tmp_path, capsys
):
    github_env = tmp_path / "github-env"
    candidate_token = str(auth_payload["access_token"])
    transport = FakeTransport(
        [
            _assignment(),
            _auth_admin_user(),
            _generated_link(),
            auth_payload,
            (204, None),
        ]
    )

    with pytest.raises(session.SessionPreparationError, match=expected_error):
        session.prepare_staging_session(
            "match-rating-writes",
            env=_env(github_env),
            transport=transport,
        )

    assert transport.requests[-1].full_url == (
        f"{session.EXPECTED_SUPABASE_ORIGIN}/auth/v1/logout?scope=local"
    )
    assert (
        transport.requests[-1].get_header("Authorization")
        == f"Bearer {candidate_token}"
    )
    env_lines = github_env.read_text(encoding="utf-8").splitlines()
    assert env_lines[0] == f"STAGING_ADMIN_BEARER_TOKEN={candidate_token}"
    assert env_lines[-5:] == [
        "STAGING_ADMIN_BEARER_TOKEN=",
        "JUPR_STAGING_ADMIN_ACCESS_TOKEN=",
        "STAGING_ADMIN_EMAIL=",
        "JUPR_STAGING_ADMIN_EMAIL=",
        "JUPR_REAL_AUTH_EXPECTED_ROLE=",
    ]
    assert capsys.readouterr().out == f"::add-mask::{candidate_token}\n"


def test_verified_jwt_accepts_standard_audience_array_and_optional_expires_at(
    tmp_path,
):
    github_env = tmp_path / "github-env"
    array_audience_token = _token(
        claims={"aud": ["secondary-audience", session.EXPECTED_AUTH_AUDIENCE]}
    )
    auth_payload = _without(
        _auth_payload(token=array_audience_token),
        "expires_at",
    )
    transport = FakeTransport(
        [
            _assignment(),
            _auth_admin_user(),
            _generated_link(),
            auth_payload,
            _capabilities(),
        ]
    )

    prepared = session.prepare_staging_session(
        "match-rating-writes",
        env=_env(github_env),
        transport=transport,
    )

    assert prepared["STAGING_ADMIN_BEARER_TOKEN"] == array_audience_token
    assert transport.responses == []


def test_invalid_verified_identity_cleanup_failure_exports_token_for_retry(tmp_path):
    github_env = tmp_path / "github-env"
    transport = FakeTransport(
        [
            _assignment(),
            _auth_admin_user(),
            _generated_link(),
            _auth_payload(user_id=OTHER_USER_ID),
            (500, {"private": TOKEN}),
        ]
    )

    with pytest.raises(
        session.SessionPreparationError,
        match="unexpected user identity.*cleanup also failed",
    ):
        session.prepare_staging_session(
            "match-rating-writes",
            env=_env(github_env),
            transport=transport,
        )

    assert github_env.read_text(encoding="utf-8").splitlines() == [
        f"STAGING_ADMIN_BEARER_TOKEN={TOKEN}"
    ]


@pytest.mark.parametrize(
    ("capabilities", "expected_error"),
    [
        (
            _capabilities(
                permissions=["manage_matches", "manage_tournaments"],
            ),
            "manage_subscriptions",
        ),
        (_capabilities(role="club_owner"), "unexpected assignment"),
        (
            {**_capabilities(), "requested_club_id": "other-club"},
            "unexpected club",
        ),
        (
            {**_capabilities(), "assignments": [*_capabilities()["assignments"]] * 2},
            "unexpected assignments",
        ),
    ],
)
def test_capabilities_fail_closed_but_leave_token_for_cleanup(
    capabilities, expected_error, tmp_path, capsys
):
    github_env = tmp_path / "github-env"
    report_dir = tmp_path / "reports"
    transport = FakeTransport(
        [
            _assignment(),
            _auth_admin_user(),
            _generated_link(),
            _auth_payload(),
            capabilities,
        ]
    )

    with pytest.raises(session.SessionPreparationError, match=expected_error):
        session.prepare_staging_session(
            "match-rating-writes",
            env=_env(github_env),
            transport=transport,
            report_dir=report_dir,
        )

    env_lines = github_env.read_text(encoding="utf-8").splitlines()
    assert f"STAGING_ADMIN_BEARER_TOKEN={TOKEN}" in env_lines
    assert f"STAGING_ADMIN_EMAIL={EMAIL}" in env_lines
    assert capsys.readouterr().out == f"::add-mask::{TOKEN}\n::add-mask::{EMAIL}\n"
    report, report_text = _read_report(
        report_dir,
        session.PREPARATION_REPORT_NAME,
    )
    assert report["status"] == "failed"
    assert EMAIL not in report_text
    assert TOKEN not in report_text


@pytest.mark.parametrize(
    "responses",
    [
        [
            *_authenticated_responses(),
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
            *_authenticated_responses(),
            _recaps(),
            _tournaments(),
            _snapshot(draw_id="00000000-0000-4000-8000-000000000000"),
        ],
    ],
)
def test_admin_fixture_mismatch_fails_with_cleanup_eligible_session_exported(
    tmp_path, responses
):
    github_env = tmp_path / "github-env"

    with pytest.raises(session.SessionPreparationError, match="fixture|snapshot"):
        session.prepare_staging_session(
            "admin-read-export",
            env=_env(github_env),
            transport=FakeTransport(responses),
        )

    assert f"STAGING_ADMIN_BEARER_TOKEN={TOKEN}" in github_env.read_text(
        encoding="utf-8"
    )


def test_http_error_and_failure_report_never_expose_sensitive_response(
    tmp_path, capsys
):
    github_env = tmp_path / "github-env"
    report_dir = tmp_path / "reports"
    secret_body = f"{EMAIL} {TOKEN} {TOKEN_HASH} {SERVICE_ROLE_KEY}".encode()
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
        match=r"session link generation failed with HTTP 401",
    ) as caught:
        session.prepare_staging_session(
            "admin-read-export",
            env=_env(github_env),
            transport=FakeTransport([_assignment(), _auth_admin_user(), error]),
            report_dir=report_dir,
        )

    output = capsys.readouterr()
    _, report_text = _read_report(report_dir, session.PREPARATION_REPORT_NAME)
    observed = f"{caught.value}{output.out}{output.err}{report_text}"
    for sensitive in (EMAIL, TOKEN, TOKEN_HASH, SERVICE_ROLE_KEY, ANON_KEY, USER_ID):
        assert sensitive not in observed
    assert not github_env.exists()


def test_env_export_failure_immediately_ends_refresh_session(tmp_path):
    github_env = tmp_path / "github-env-is-a-directory"
    github_env.mkdir()
    transport = FakeTransport(
        [
            _assignment(),
            _auth_admin_user(),
            _generated_link(),
            _auth_payload(),
            (204, None),
        ]
    )

    with pytest.raises(
        session.SessionPreparationError,
        match="Could not append the prepared values to GITHUB_ENV",
    ):
        session.prepare_staging_session(
            "match-rating-writes",
            env=_env(github_env),
            transport=transport,
        )

    assert [request.get_method() for request in transport.requests] == [
        "GET",
        "GET",
        "POST",
        "POST",
        "POST",
    ]
    assert transport.requests[-1].full_url == (
        f"{session.EXPECTED_SUPABASE_ORIGIN}/auth/v1/logout?scope=local"
    )
    assert transport.responses == []


def test_default_transport_redirect_handler_never_forwards_credentials():
    request = Request(
        f"{session.EXPECTED_SUPABASE_ORIGIN}/auth/v1/admin/generate_link",
        headers={
            "Authorization": f"Bearer {SERVICE_ROLE_KEY}",
            "apikey": SERVICE_ROLE_KEY,
        },
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


@pytest.mark.parametrize(
    ("name", "value", "expected_error"),
    [
        (
            "STAGING_SUPABASE_URL",
            "https://production.example.invalid",
            "permitted staging project",
        ),
        (
            "STAGING_API_BASE_URL",
            "https://production.example.invalid",
            "permitted staging API",
        ),
    ],
)
def test_refuses_any_runtime_other_than_exact_staging(
    name, value, expected_error, tmp_path
):
    github_env = tmp_path / "github-env"
    env = _env(github_env)
    env[name] = value
    transport = FakeTransport([])

    with pytest.raises(session.SessionPreparationError, match=expected_error):
        session.prepare_staging_session(
            "admin-read-export",
            env=env,
            transport=transport,
        )

    assert transport.requests == []
    assert not github_env.exists()


@pytest.mark.parametrize(
    ("response", "expected_status"),
    [
        ((204, None), "ended"),
        ((200, {}), "ended"),
        (
            (
                401,
                {
                    "code": 401,
                    "error_code": "session_not_found",
                },
            ),
            "already_inactive",
        ),
        ((403, {"code": "session_expired"}), "already_inactive"),
    ],
)
def test_cleanup_ends_or_accepts_already_inactive_refresh_session(
    response, expected_status, tmp_path
):
    github_env = tmp_path / "github-env"
    report_dir = tmp_path / "reports"
    env = _env(github_env)
    env["STAGING_ADMIN_BEARER_TOKEN"] = TOKEN
    transport = FakeTransport([response])

    cleaned = session.cleanup_staging_session(
        env=env,
        transport=transport,
        report_dir=report_dir,
    )

    assert cleaned == {
        "refresh_session_status": expected_status,
        "access_jwt_status": "may_remain_valid_until_exp_claim",
    }
    request = transport.requests[0]
    assert request.get_method() == "POST"
    assert request.full_url == (
        f"{session.EXPECTED_SUPABASE_ORIGIN}/auth/v1/logout?scope=local"
    )
    assert request.get_header("Apikey") == ANON_KEY
    assert request.get_header("Authorization") == f"Bearer {TOKEN}"
    assert request.data is None
    assert github_env.read_text(encoding="utf-8").splitlines() == [
        "STAGING_ADMIN_BEARER_TOKEN=",
        "JUPR_STAGING_ADMIN_ACCESS_TOKEN=",
        "STAGING_ADMIN_EMAIL=",
        "JUPR_STAGING_ADMIN_EMAIL=",
        "JUPR_REAL_AUTH_EXPECTED_ROLE=",
    ]
    report, report_text = _read_report(report_dir, session.CLEANUP_REPORT_NAME)
    assert report["status"] == "passed"
    assert report["schema_version"] == 2
    assert report["refresh_session_status"] == expected_status
    assert report["access_jwt_status"] == "may_remain_valid_until_exp_claim"
    for sensitive in (TOKEN, EMAIL, ANON_KEY, SERVICE_ROLE_KEY):
        assert sensitive not in report_text


def test_cleanup_after_browser_signout_accepts_only_session_not_found(tmp_path):
    github_env = tmp_path / "github-env"
    env = _env(github_env)
    env["STAGING_ADMIN_BEARER_TOKEN"] = TOKEN
    error = HTTPError(
        "https://example.invalid",
        403,
        "forbidden",
        {},
        None,
    )
    error.read = lambda _limit: json.dumps(
        {
            "code": 403,
            "error_code": "session_not_found",
            "message": "Session from session_id claim in JWT does not exist",
        }
    ).encode("utf-8")

    cleaned = session.cleanup_staging_session(
        env=env,
        transport=FakeTransport([error]),
    )

    assert cleaned == {
        "refresh_session_status": "already_inactive",
        "access_jwt_status": "may_remain_valid_until_exp_claim",
    }


def test_cleanup_without_prepared_token_is_sanitized_noop(tmp_path):
    github_env = tmp_path / "github-env"
    report_dir = tmp_path / "reports"
    env = {
        "STAGING_SUPABASE_URL": session.EXPECTED_SUPABASE_ORIGIN,
        "GITHUB_ENV": str(github_env),
    }
    transport = FakeTransport([])

    cleaned = session.cleanup_staging_session(
        env=env,
        transport=transport,
        report_dir=report_dir,
    )

    assert cleaned == {
        "refresh_session_status": "not_started",
        "access_jwt_status": "not_exported",
    }
    assert transport.requests == []
    assert "STAGING_ADMIN_BEARER_TOKEN=" in github_env.read_text(encoding="utf-8")
    report, _ = _read_report(report_dir, session.CLEANUP_REPORT_NAME)
    assert report["schema_version"] == 2
    assert report["refresh_session_status"] == "not_started"
    assert report["access_jwt_status"] == "not_exported"


@pytest.mark.parametrize(
    ("response", "expected_error"),
    [
        ((500, {"private": TOKEN}), "failed with HTTP 500"),
        ((403, {"code": "not_admin"}), "failed with HTTP 403"),
    ],
)
def test_genuine_cleanup_failure_is_reported_and_clears_future_env(
    response, expected_error, tmp_path
):
    github_env = tmp_path / "github-env"
    report_dir = tmp_path / "reports"
    env = _env(github_env)
    env["STAGING_ADMIN_BEARER_TOKEN"] = TOKEN

    with pytest.raises(session.SessionPreparationError, match=expected_error):
        session.cleanup_staging_session(
            env=env,
            transport=FakeTransport([response]),
            report_dir=report_dir,
        )

    assert "STAGING_ADMIN_BEARER_TOKEN=" in github_env.read_text(encoding="utf-8")
    report, report_text = _read_report(report_dir, session.CLEANUP_REPORT_NAME)
    assert report["status"] == "failed"
    assert TOKEN not in report_text


def test_cleanup_refuses_other_project_before_using_token(tmp_path):
    github_env = tmp_path / "github-env"
    env = _env(github_env)
    env["STAGING_SUPABASE_URL"] = "https://production.example.invalid"
    env["STAGING_ADMIN_BEARER_TOKEN"] = TOKEN
    transport = FakeTransport([])

    with pytest.raises(
        session.SessionPreparationError,
        match="permitted staging project",
    ):
        session.cleanup_staging_session(env=env, transport=transport)

    assert transport.requests == []


def test_cleanup_cli_mode_is_supported():
    args = session._parser().parse_args(
        ["cleanup", "--report-dir", "parity-staging-artifacts"]
    )

    assert args.mode == "cleanup"
    assert args.report_dir == Path("parity-staging-artifacts")
