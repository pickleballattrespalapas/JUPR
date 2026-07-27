from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
from uuid import uuid4

import pytest

import scripts.staging_write_session as session
from scripts.run_parity_staging_wave import (
    EXPECTED_FLY_APP_NAME,
    EXPECTED_STAGING_PROJECT_REF,
    EXPECTED_STAGING_WEB_ORIGIN,
)
from scripts.staging_evidence_automation import ContractError
from scripts.staging_write_waves import (
    NO_WRITE_WAVE,
    STAGING_WRITE_WAVES,
    expected_write_flags,
)


CANDIDATE_SHA = "a" * 40
NOW = datetime(2026, 7, 26, 18, 0, 0, tzinfo=timezone.utc)


def _stamp(value: datetime) -> str:
    return value.strftime(session.TIMESTAMP_FORMAT)


def _body(
    *,
    command: str = "open",
    candidate_sha: str = CANDIDATE_SHA,
    expected_write_wave: str = "none",
    write_wave: str = "league-manager",
    nonce: str | None = None,
    started: datetime = NOW,
    expires: datetime | None = None,
    extra: str = "",
) -> str:
    values = [
        f"command: {command}",
        f"candidate_sha: {candidate_sha}",
        f"expected_write_wave: {expected_write_wave}",
        f"write_wave: {write_wave}",
    ]
    if command in {"open", "advance"}:
        values.extend(
            (
                f"lease_started_at: {_stamp(started)}",
                f"lease_expires_at: {_stamp(expires or started + timedelta(minutes=20))}",
            )
        )
    values.append(f"session_nonce: {nonce or uuid4()}")
    if extra:
        values.append(extra)
    return (
        "# Protected staging write session control\n\n"
        "Only the controller interprets this exact block.\n\n"
        "```yaml\n"
        + "\n".join(values)
        + "\n```\n"
    )


def _issue(*, body: str, state: str = "open") -> dict[str, object]:
    return {
        "number": session.CONTROL_ISSUE_NUMBER,
        "title": session.CONTROL_ISSUE_TITLE,
        "state": state,
        "locked": True,
        "author_association": "OWNER",
        "user": {
            "id": session.OWNER_ID,
            "login": session.OWNER_LOGIN,
        },
        "body": body,
        "labels": [],
    }


def _event(
    *,
    action: str,
    body: str,
    previous_body: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "action": action,
        "issue": _issue(body=body),
        "repository": {
            "id": session.REPOSITORY_ID,
            "full_name": session.REPOSITORY,
            "owner": {
                "id": session.OWNER_ID,
                "login": session.OWNER_LOGIN,
            },
        },
        "sender": {
            "id": session.OWNER_ID,
            "login": session.OWNER_LOGIN,
        },
    }
    if action == "edited":
        parsed = session.parse_session_command(body)
        prior = previous_body or _body(
            command="open",
            candidate_sha=parsed.candidate_sha,
            expected_write_wave="none",
            write_wave=parsed.expected_write_wave,
            nonce=parsed.session_nonce,
        )
        payload["changes"] = {"body": {"from": prior}}
    return payload


def _request(live_issue: dict[str, object], *, candidate_sha: str = CANDIDATE_SHA):
    calls: list[tuple[str, str, object | None]] = []

    def request(method: str, path: str, payload: object | None) -> object:
        calls.append((method, path, payload))
        if path == (
            f"/repos/{session.REPOSITORY}/issues/"
            f"{session.CONTROL_ISSUE_NUMBER}"
        ):
            return copy.deepcopy(live_issue)
        if path == (
            f"/repos/{session.REPOSITORY}/git/ref/heads/"
            f"{session.STAGING_BRANCH}"
        ):
            return {"object": {"sha": candidate_sha}}
        raise AssertionError(f"Unexpected request: {method} {path}")

    return request, calls


def _fly_health(
    *,
    command: session.SessionCommand,
    wave: str | None = None,
) -> dict[str, object]:
    active_wave = wave or command.write_wave
    flags = expected_write_flags(active_wave)
    fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}={1 if enabled else 0}"
            for name, enabled in sorted(flags.items())
        ).encode("utf-8")
    ).hexdigest()
    return {
        "ok": True,
        "environment": "staging",
        "git_commit_sha": command.candidate_sha,
        "fly_app_name": EXPECTED_FLY_APP_NAME,
        "fly_image_ref": (
            "registry.fly.io/juprleagues-api-staging:"
            "deployment-01KYGWRITESESSION000000000"
        ),
        "web_origin": EXPECTED_STAGING_WEB_ORIGIN,
        "supabase_project_ref": EXPECTED_STAGING_PROJECT_REF,
        "staging_write_wave": active_wave,
        "business_data_write_wave_active": active_wave != NO_WRITE_WAVE,
        "controlled_write_flags": flags,
        "controlled_write_flag_fingerprint": fingerprint,
        "public_live_writes_enabled": flags["JUPR_ENABLE_PUBLIC_LIVE_WRITES"],
        "public_live_production_override_enabled": False,
        "jwt_verification_configured": True,
        "jwt_verification_mode": "jwks",
        "jwt_verification_project_ref": EXPECTED_STAGING_PROJECT_REF,
        "security_denial_audit_logging_required": True,
        "registration_edit_secret_configured": active_wave
        == "public-intake-auth",
        "registration_confirmation_secret_configured": active_wave
        == "public-intake-auth",
        "write_prerequisites": {
            "service_role_configured": True,
            "api_audit_required": True,
            "worker_run_log_required": True,
            "email_mode": "dry_run",
            "live_player_update_email_enabled": False,
        },
    }


def test_every_existing_active_wave_is_supported_without_enable_all() -> None:
    assert session.ACTIVE_WRITE_WAVES == tuple(
        wave for wave in STAGING_WRITE_WAVES if wave != NO_WRITE_WAVE
    )
    assert "all" not in session.ACTIVE_WRITE_WAVES
    for wave in session.ACTIVE_WRITE_WAVES:
        parsed = session.parse_session_command(_body(write_wave=wave))
        assert parsed.write_wave == wave


def test_open_advance_and_close_are_exact_one_wave_transitions() -> None:
    opened = session.parse_session_command(_body())
    assert opened.command == "open"
    assert opened.expected_write_wave == "none"
    assert opened.write_wave == "league-manager"

    advanced = session.parse_session_command(
        _body(
            command="advance",
            expected_write_wave="league-manager",
            write_wave="league-awards",
        )
    )
    assert advanced.command == "advance"
    assert advanced.expected_write_wave == "league-manager"
    assert advanced.write_wave == "league-awards"

    closed = session.parse_session_command(
        _body(
            command="close",
            expected_write_wave="league-awards",
            write_wave="none",
        )
    )
    assert closed.command == "close"
    assert closed.write_wave == "none"
    assert closed.lease_expires_at == ""


@pytest.mark.parametrize(
    "body",
    [
        _body(write_wave="all"),
        _body(expected_write_wave="match-player"),
        _body(
            command="advance",
            expected_write_wave="league-manager",
            write_wave="league-manager",
        ),
        _body(command="close", write_wave="league-manager"),
        _body(extra="unexpected_field: nope"),
        _body(nonce="not-a-uuid"),
        _body(candidate_sha="ABC"),
    ],
)
def test_invalid_or_broad_commands_fail_closed(body: str) -> None:
    with pytest.raises(ContractError):
        session.parse_session_command(body)


def test_lease_is_self_contained_fresh_and_bounded() -> None:
    command = session.parse_session_command(_body())
    started, expires = session.validate_lease(
        command,
        now=NOW,
        require_fresh_start=True,
    )
    assert expires - started == timedelta(minutes=20)

    too_long = session.parse_session_command(
        _body(expires=NOW + timedelta(minutes=61))
    )
    with pytest.raises(ContractError, match="between 5 and 60"):
        session.validate_lease(
            too_long,
            now=NOW,
            require_fresh_start=True,
        )

    stale = session.parse_session_command(
        _body(
            started=NOW - timedelta(minutes=4),
            expires=NOW + timedelta(minutes=16),
        )
    )
    with pytest.raises(ContractError, match="not fresh"):
        session.validate_lease(stale, now=NOW, require_fresh_start=True)


@pytest.mark.parametrize(
    ("action", "body"),
    [
        ("reopened", _body()),
        (
            "edited",
            _body(
                command="advance",
                expected_write_wave="league-manager",
                write_wave="league-awards",
            ),
        ),
        (
            "edited",
            _body(
                command="close",
                expected_write_wave="league-awards",
                write_wave="none",
            ),
        ),
    ],
)
def test_authorization_binds_owner_issue_body_and_current_candidate(
    action: str,
    body: str,
) -> None:
    event = _event(action=action, body=body)
    request, calls = _request(_issue(body=body))

    result = session.authorize_event(
        event,
        run_attempt=1,
        request_json=request,
        now=NOW,
    )

    command = session.parse_session_command(body)
    assert result == {
        "authorized": True,
        "superseded": False,
        "issue_number": session.CONTROL_ISSUE_NUMBER,
        **session.asdict(command),
    }
    assert calls[-1] == (
        "GET",
        (
            f"/repos/{session.REPOSITORY}/git/ref/heads/"
            f"{session.STAGING_BRANCH}"
        ),
        None,
    )


def test_authorization_treats_a_newer_live_body_as_superseded_noop() -> None:
    event_body = _body()
    live_body = _body(write_wave="communications")
    request, _ = _request(_issue(body=live_body))

    result = session.authorize_event(
        _event(action="reopened", body=event_body),
        run_attempt=1,
        request_json=request,
        now=NOW,
    )

    assert result == {
        "authorized": False,
        "superseded": True,
        "issue_number": session.CONTROL_ISSUE_NUMBER,
    }


def test_authorization_treats_open_edit_companion_from_close_as_noop() -> None:
    nonce = str(uuid4())
    open_body = _body(nonce=nonce)
    close_body = _body(
        command="close",
        expected_write_wave="league-manager",
        write_wave="none",
        nonce=nonce,
    )
    request, calls = _request(_issue(body=open_body))

    result = session.authorize_event(
        _event(
            action="edited",
            body=open_body,
            previous_body=close_body,
        ),
        run_attempt=1,
        request_json=request,
        now=NOW,
    )

    assert result == {
        "authorized": False,
        "superseded": True,
        "issue_number": session.CONTROL_ISSUE_NUMBER,
    }
    assert calls == []


@pytest.mark.parametrize(
    "previous_body",
    [
        _body(),
        _body(
            command="advance",
            expected_write_wave="league-manager",
            write_wave="league-awards",
        ),
        "not a control command",
    ],
)
def test_open_edit_companion_rejects_active_or_malformed_prior_command(
    previous_body: str,
) -> None:
    open_body = _body()
    request, _ = _request(_issue(body=open_body))

    with pytest.raises(ContractError):
        session.authorize_event(
            _event(
                action="edited",
                body=open_body,
                previous_body=previous_body,
            ),
            run_attempt=1,
            request_json=request,
            now=NOW,
        )


@pytest.mark.parametrize(
    "case",
    [
        "candidate",
        "nonce",
        "expected_wave",
        "previous_close",
    ],
)
def test_advance_and_close_require_coherent_issue_ledger_continuity(
    case: str,
) -> None:
    nonce = str(uuid4())
    new_body = _body(
        command="advance",
        expected_write_wave="league-manager",
        write_wave="league-awards",
        nonce=nonce,
    )
    previous_body = _body(
        command="open",
        write_wave="league-manager",
        nonce=nonce,
    )
    if case == "candidate":
        previous_body = _body(
            command="open",
            candidate_sha="b" * 40,
            write_wave="league-manager",
            nonce=nonce,
        )
    elif case == "nonce":
        previous_body = _body(
            command="open",
            write_wave="league-manager",
            nonce=str(uuid4()),
        )
    elif case == "expected_wave":
        previous_body = _body(
            command="open",
            write_wave="communications",
            nonce=nonce,
        )
    elif case == "previous_close":
        previous_body = _body(
            command="close",
            expected_write_wave="league-manager",
            write_wave="none",
            nonce=nonce,
        )
    event = _event(
        action="edited",
        body=new_body,
        previous_body=previous_body,
    )
    request, _ = _request(_issue(body=new_body))

    with pytest.raises(ContractError, match="active issue ledger"):
        session.authorize_event(
            event,
            run_attempt=1,
            request_json=request,
            now=NOW,
        )


def test_close_requires_previous_active_wave_and_same_session_nonce() -> None:
    nonce = str(uuid4())
    close_body = _body(
        command="close",
        expected_write_wave="league-awards",
        write_wave="none",
        nonce=nonce,
    )
    previous_body = _body(
        command="open",
        write_wave="league-manager",
        nonce=nonce,
    )
    event = _event(
        action="edited",
        body=close_body,
        previous_body=previous_body,
    )
    request, _ = _request(_issue(body=close_body))

    with pytest.raises(ContractError, match="active issue ledger"):
        session.authorize_event(
            event,
            run_attempt=1,
            request_json=request,
            now=NOW,
        )


@pytest.mark.parametrize(
    "case",
    [
        "wrong_issue",
        "wrong_title",
        "wrong_sender",
        "unlocked",
        "wrong_action",
        "rerun",
        "candidate_drift",
        "close_reopened",
        "advance_without_body_change",
    ],
)
def test_authorization_rejects_nonexact_control_events(case: str) -> None:
    body = _body()
    action = "reopened"
    event = _event(action=action, body=body)
    live = _issue(body=body)
    run_attempt = 1
    candidate = CANDIDATE_SHA

    if case == "wrong_issue":
        event["issue"]["number"] = 999
    elif case == "wrong_title":
        event["issue"]["title"] = "Almost the protected control"
    elif case == "wrong_sender":
        event["sender"]["id"] = session.OWNER_ID + 1
    elif case == "unlocked":
        event["issue"]["locked"] = False
    elif case == "wrong_action":
        event["action"] = "opened"
    elif case == "rerun":
        run_attempt = 2
    elif case == "candidate_drift":
        candidate = "b" * 40
    elif case == "close_reopened":
        body = _body(
            command="close",
            expected_write_wave="league-manager",
            write_wave="none",
        )
        event = _event(action="reopened", body=body)
        live = _issue(body=body)
    elif case == "advance_without_body_change":
        body = _body(
            command="advance",
            expected_write_wave="league-manager",
            write_wave="league-awards",
        )
        event = _event(action="edited", body=body)
        event["changes"] = {}
        live = _issue(body=body)
    request, _ = _request(live, candidate_sha=candidate)

    with pytest.raises(ContractError):
        session.authorize_event(
            event,
            run_attempt=run_attempt,
            request_json=request,
            now=NOW,
        )


def test_scheduled_recovery_keeps_only_exact_unexpired_dry_run_lease() -> None:
    body = _body()
    command = session.parse_session_command(body)
    health = _fly_health(command=command)

    result = session.inspect_active_lease(
        issue=_issue(body=body),
        current_candidate_sha=CANDIDATE_SHA,
        fly=health,
        now=NOW + timedelta(minutes=1),
    )

    assert result["keep_active"] is True
    assert result["write_wave"] == "league-manager"
    assert result["email_mode"] == "dry_run"


@pytest.mark.parametrize("current_wave", ["none", "league-manager"])
def test_open_lease_accepts_only_exact_none_or_target_transition(
    current_wave: str,
) -> None:
    body = _body()
    command = session.parse_session_command(body)
    result = session.inspect_active_lease(
        issue=_issue(body=body),
        current_candidate_sha=CANDIDATE_SHA,
        fly=_fly_health(command=command, wave=current_wave),
        now=NOW + timedelta(minutes=1),
    )

    assert result["keep_active"] is True
    assert result["current_write_wave"] == current_wave


@pytest.mark.parametrize(
    "current_wave",
    ["league-manager", "none", "league-awards"],
)
def test_advance_lease_accepts_only_expected_none_or_target_transition(
    current_wave: str,
) -> None:
    body = _body(
        command="advance",
        expected_write_wave="league-manager",
        write_wave="league-awards",
    )
    command = session.parse_session_command(body)
    result = session.inspect_active_lease(
        issue=_issue(body=body),
        current_candidate_sha=CANDIDATE_SHA,
        fly=_fly_health(command=command, wave=current_wave),
        now=NOW + timedelta(minutes=1),
    )

    assert result["keep_active"] is True
    assert result["current_write_wave"] == current_wave


def test_transition_state_still_requires_exact_flags_for_current_wave() -> None:
    body = _body(
        command="advance",
        expected_write_wave="league-manager",
        write_wave="league-awards",
    )
    command = session.parse_session_command(body)
    health = _fly_health(command=command, wave="none")
    health["controlled_write_flags"] = expected_write_flags("league-manager")

    result = session.inspect_active_lease(
        issue=_issue(body=body),
        current_candidate_sha=CANDIDATE_SHA,
        fly=health,
        now=NOW + timedelta(minutes=1),
    )

    assert result["keep_active"] is False
    assert "controlled_write_flags" in str(result["reason"])


@pytest.mark.parametrize(
    "case",
    [
        "expired",
        "closed_issue",
        "close_command",
        "candidate_drift",
        "wrong_wave",
        "wrong_flags",
        "live_email",
        "production_override",
    ],
)
def test_recovery_refuses_every_nonexact_active_lease(case: str) -> None:
    body = _body()
    command = session.parse_session_command(body)
    issue = _issue(body=body)
    candidate = CANDIDATE_SHA
    health = _fly_health(command=command)
    now = NOW + timedelta(minutes=1)

    if case == "expired":
        now = NOW + timedelta(minutes=21)
    elif case == "closed_issue":
        issue["state"] = "closed"
    elif case == "close_command":
        body = _body(
            command="close",
            expected_write_wave="league-manager",
            write_wave="none",
        )
        issue["body"] = body
    elif case == "candidate_drift":
        candidate = "b" * 40
    elif case == "wrong_wave":
        health["staging_write_wave"] = "communications"
    elif case == "wrong_flags":
        health["controlled_write_flags"] = expected_write_flags("communications")
    elif case == "live_email":
        health["write_prerequisites"]["live_player_update_email_enabled"] = True
    elif case == "production_override":
        health["public_live_production_override_enabled"] = True

    result = session.inspect_active_lease(
        issue=issue,
        current_candidate_sha=candidate,
        fly=health,
        now=now,
    )

    assert result["keep_active"] is False
    assert result["reason"]


def test_old_expiry_cannot_close_a_superseding_lease() -> None:
    old_body = _body()
    old_command = session.parse_session_command(old_body)
    new_body = _body(write_wave="communications")

    superseded = session.should_expire_lease(
        issue=_issue(body=new_body),
        current_candidate_sha=CANDIDATE_SHA,
        command=old_command,
        now=NOW + timedelta(minutes=21),
    )
    assert superseded["expire"] is False
    assert "superseded" in str(superseded["reason"]).lower()

    current = session.should_expire_lease(
        issue=_issue(body=old_body),
        current_candidate_sha=CANDIDATE_SHA,
        command=old_command,
        now=NOW + timedelta(minutes=21),
    )
    assert current["expire"] is True
    assert current["session_nonce"] == old_command.session_nonce


def test_expiry_returns_a_reopen_safe_close_body() -> None:
    active_body = _body()
    active_command = session.parse_session_command(active_body)
    expired = session.should_expire_lease(
        issue=_issue(body=active_body),
        current_candidate_sha=CANDIDATE_SHA,
        command=active_command,
        now=NOW + timedelta(minutes=21),
    )

    close_body = expired["close_body"]
    close_command = session.parse_session_command(close_body)
    assert close_command.command == "close"
    assert close_command.candidate_sha == active_command.candidate_sha
    assert close_command.expected_write_wave == active_command.write_wave
    assert close_command.write_wave == NO_WRITE_WAVE
    assert close_command.session_nonce == active_command.session_nonce
    assert close_command.lease_started_at == ""
    assert close_command.lease_expires_at == ""

    fresh_open_body = _body()
    request, calls = _request(_issue(body=fresh_open_body))
    companion = session.authorize_event(
        _event(
            action="edited",
            body=fresh_open_body,
            previous_body=close_body,
        ),
        run_attempt=1,
        request_json=request,
        now=NOW,
    )
    assert companion == {
        "authorized": False,
        "superseded": True,
        "issue_number": session.CONTROL_ISSUE_NUMBER,
    }
    assert calls == []

    reopened = session.authorize_event(
        _event(action="reopened", body=fresh_open_body),
        run_attempt=1,
        request_json=request,
        now=NOW,
    )
    assert reopened["authorized"] is True
    assert reopened["superseded"] is False


@pytest.mark.parametrize("case", ["closed_issue", "candidate_drift"])
def test_exact_current_lease_fails_closed_without_waiting_for_cron(
    case: str,
) -> None:
    body = _body()
    command = session.parse_session_command(body)
    issue = _issue(body=body)
    candidate = CANDIDATE_SHA
    if case == "closed_issue":
        issue["state"] = "closed"
    else:
        candidate = "b" * 40

    result = session.should_expire_lease(
        issue=issue,
        current_candidate_sha=candidate,
        command=command,
        now=NOW + timedelta(minutes=1),
    )

    assert result["expire"] is True
    assert result["session_nonce"] == command.session_nonce
    assert result["reason"]
