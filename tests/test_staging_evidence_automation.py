import copy
import hashlib

import pytest

import scripts.staging_evidence_automation as automation
from scripts.run_parity_staging_wave import (
    EXPECTED_FLY_APP_NAME,
    EXPECTED_STAGING_API_ORIGIN,
    EXPECTED_STAGING_AUTH_ORIGIN,
    EXPECTED_STAGING_PROJECT_REF,
    EXPECTED_STAGING_WEB_ORIGIN,
    MUTATION_CONFIRMATION,
)
from scripts.staging_write_waves import expected_write_flags


CANDIDATE_SHA = "a" * 40
OTHER_SHA = "b" * 40
ISSUE_NUMBER = 9001
DEPLOYMENT_ID = "dpl_Abcdefgh12345678"
IMMUTABLE_VERCEL_ORIGIN = (
    "https://jupr-hgwnn727e-pickleballattrespalapas1.vercel.app"
)
FLY_IMAGE = (
    "registry.fly.io/juprleagues-api-staging:"
    "deployment-01KYFWBT0Q2DSM3VMN7JEGR594"
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _issue(*, number: int = ISSUE_NUMBER) -> dict[str, object]:
    return {
        "number": number,
        "state": "open",
        "locked": True,
        "author_association": "OWNER",
        "user": {"id": automation.OWNER_ID, "login": automation.OWNER_LOGIN},
        "title": "This title must not select a mode",
        "body": "match-rating-writes and complete-book are ignored here",
    }


def _event(*, number: int = ISSUE_NUMBER) -> dict[str, object]:
    return {
        "action": "reopened",
        "issue": _issue(number=number),
        "repository": {
            "id": automation.REPOSITORY_ID,
            "full_name": automation.REPOSITORY,
            "owner": {
                "id": automation.OWNER_ID,
                "login": automation.OWNER_LOGIN,
            },
        },
        "sender": {
            "id": automation.OWNER_ID,
            "login": automation.OWNER_LOGIN,
        },
    }


def _authorization_request(
    live_issue: dict[str, object],
    *,
    candidate_sha: str = CANDIDATE_SHA,
):
    calls: list[tuple[str, str, object | None]] = []

    def request(method: str, path: str, payload: object | None) -> object:
        calls.append((method, path, payload))
        if path == f"/repos/{automation.REPOSITORY}/issues/{ISSUE_NUMBER}":
            return copy.deepcopy(live_issue)
        if path == (
            f"/repos/{automation.REPOSITORY}/git/ref/heads/"
            f"{automation.STAGING_BRANCH}"
        ):
            return {"object": {"sha": candidate_sha}}
        raise AssertionError(f"Unexpected GitHub request: {method} {path}")

    return request, calls


def _web_identity(
    *,
    candidate_sha: str = CANDIDATE_SHA,
    deployment_id: str = DEPLOYMENT_ID,
    deployment_origin: str = IMMUTABLE_VERCEL_ORIGIN,
) -> dict[str, object]:
    return {
        "environment": "staging",
        "vercel_environment": "preview",
        "git_commit_sha": candidate_sha,
        "api_origin": EXPECTED_STAGING_API_ORIGIN,
        "auth_origin": EXPECTED_STAGING_AUTH_ORIGIN,
        "preview_isolation_active": True,
        "preview_auth_isolation_active": True,
        "vercel_deployment_id": deployment_id,
        "vercel_deployment_origin": deployment_origin,
    }


def _fly_identity(
    *,
    candidate_sha: str = CANDIDATE_SHA,
    write_wave: str = "none",
) -> dict[str, object]:
    flags = expected_write_flags(write_wave)
    fingerprint = hashlib.sha256(
        "\n".join(
            f"{name}={1 if enabled else 0}"
            for name, enabled in sorted(flags.items())
        ).encode("utf-8")
    ).hexdigest()
    return {
        "ok": True,
        "environment": "staging",
        "git_commit_sha": candidate_sha,
        "fly_app_name": EXPECTED_FLY_APP_NAME,
        "fly_image_ref": FLY_IMAGE,
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
        "registration_edit_secret_configured": write_wave
        == "public-intake-auth",
        "registration_confirmation_secret_configured": write_wave
        == "public-intake-auth",
        "write_prerequisites": {
            "service_role_configured": True,
            "api_audit_required": True,
            "worker_run_log_required": True,
            "email_mode": "dry_run",
            "live_player_update_email_enabled": False,
        },
    }


@pytest.mark.parametrize(
    ("mode", "write_wave", "confirmation"),
    [
        ("public-read", "none", ""),
        ("public-intake-auth", "public-intake-auth", ""),
        ("admin-read-export", "none", ""),
        ("match-rating-writes", "tournament-live", MUTATION_CONFIRMATION),
        (
            "match-exclusion-recovery",
            "match-exclusion-recovery",
            MUTATION_CONFIRMATION,
        ),
        ("complete-book", "none", ""),
    ],
)
def test_authorization_selects_exact_mode_from_locked_issue_number(
    mode: str,
    write_wave: str,
    confirmation: str,
) -> None:
    event = _event()
    event["issue"]["title"] = "wrong mode in the title"
    event["issue"]["body"] = "match-rating-writes"
    live = _issue()
    live["title"] = "another ignored title"
    live["body"] = "public-read"
    request, calls = _authorization_request(live)

    result = automation.authorize_event(
        event,
        issue_modes={ISSUE_NUMBER: mode},
        run_attempt=1,
        request_json=request,
    )

    assert result == {
        "authorized": True,
        "issue_number": ISSUE_NUMBER,
        "mode": mode,
        "write_wave": write_wave,
        "mutation_confirmation": confirmation,
        "candidate_sha": CANDIDATE_SHA,
    }
    assert calls == [
        (
            "GET",
            f"/repos/{automation.REPOSITORY}/issues/{ISSUE_NUMBER}",
            None,
        ),
        (
            "GET",
            (
                f"/repos/{automation.REPOSITORY}/git/ref/heads/"
                f"{automation.STAGING_BRANCH}"
            ),
            None,
        ),
    ]


@pytest.mark.parametrize(
    "case",
    [
        "wrong_action",
        "wrong_actor_id",
        "wrong_actor_login",
        "wrong_repo_name",
        "wrong_repo_id",
        "wrong_issue",
        "wrong_issue_author",
        "unlocked",
        "rerun",
        "pull_request",
        "live_closed",
        "live_unlocked",
    ],
)
def test_authorization_rejects_every_nonexact_controller_event(case: str) -> None:
    event = _event()
    live = _issue()
    run_attempt = 1
    issue_modes = {ISSUE_NUMBER: "complete-book"}

    if case == "wrong_action":
        event["action"] = "opened"
    elif case == "wrong_actor_id":
        event["sender"]["id"] = automation.OWNER_ID + 1
    elif case == "wrong_actor_login":
        event["sender"]["login"] = "someone-else"
    elif case == "wrong_repo_name":
        event["repository"]["full_name"] = "someone-else/JUPR"
    elif case == "wrong_repo_id":
        event["repository"]["id"] = automation.REPOSITORY_ID + 1
    elif case == "wrong_issue":
        event["issue"]["number"] = ISSUE_NUMBER + 1
    elif case == "wrong_issue_author":
        event["issue"]["user"]["id"] = automation.OWNER_ID + 1
    elif case == "unlocked":
        event["issue"]["locked"] = False
    elif case == "rerun":
        run_attempt = 2
    elif case == "pull_request":
        event["issue"]["pull_request"] = {"url": "https://example.invalid/pr"}
    elif case == "live_closed":
        live["state"] = "closed"
    elif case == "live_unlocked":
        live["locked"] = False
    else:  # pragma: no cover
        raise AssertionError(case)

    request, _ = _authorization_request(live)

    with pytest.raises(automation.ContractError, match="authorization rejected"):
        automation.authorize_event(
            event,
            issue_modes=issue_modes,
            run_attempt=run_attempt,
            request_json=request,
        )


def test_vercel_alias_and_immutable_identity_must_attest_the_same_candidate() -> None:
    alias = _web_identity()
    immutable = _web_identity()
    calls: list[tuple[str, dict[str, str] | None]] = []

    def get_json(url: str, headers: dict[str, str] | None) -> object:
        calls.append((url, headers))
        if url == f"{EXPECTED_STAGING_WEB_ORIGIN}/api/environment":
            return copy.deepcopy(alias)
        if url == f"{IMMUTABLE_VERCEL_ORIGIN}/api/environment":
            return copy.deepcopy(immutable)
        raise AssertionError(url)

    result = automation.resolve_vercel_identity(
        candidate_sha=CANDIDATE_SHA,
        bypass_secret="bypass-secret",
        timeout_seconds=0,
        poll_seconds=0,
        get_json=get_json,
    )

    assert result == {
        "candidate_sha": CANDIDATE_SHA,
        "vercel_deployment_id": DEPLOYMENT_ID,
        "vercel_deployment_origin": IMMUTABLE_VERCEL_ORIGIN,
        "web": immutable,
    }
    assert calls == [
        (
            f"{EXPECTED_STAGING_WEB_ORIGIN}/api/environment",
            {"x-vercel-protection-bypass": "bypass-secret"},
        ),
        (
            f"{IMMUTABLE_VERCEL_ORIGIN}/api/environment",
            {"x-vercel-protection-bypass": "bypass-secret"},
        ),
    ]


def test_vercel_immutable_identity_mismatch_is_rejected() -> None:
    responses = iter(
        [
            _web_identity(),
            _web_identity(deployment_id="dpl_Different123456"),
        ]
    )

    with pytest.raises(
        automation.ContractError,
        match="deployment ID changed across origins",
    ):
        automation.resolve_vercel_identity(
            candidate_sha=CANDIDATE_SHA,
            bypass_secret="bypass-secret",
            timeout_seconds=0,
            poll_seconds=0,
            get_json=lambda _url, _headers: next(responses),
            monotonic=lambda: 0,
        )


def test_vercel_alias_resolution_times_out_without_accepting_wrong_candidate() -> None:
    clock = FakeClock()
    calls = 0

    def get_json(_url: str, _headers: dict[str, str] | None) -> object:
        nonlocal calls
        calls += 1
        return _web_identity(candidate_sha=OTHER_SHA)

    with pytest.raises(automation.ContractError, match="did not become ready"):
        automation.resolve_vercel_identity(
            candidate_sha=CANDIDATE_SHA,
            bypass_secret="bypass-secret",
            timeout_seconds=2,
            poll_seconds=1,
            get_json=get_json,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert calls == 3
    assert clock.sleeps == [1, 1]


def test_dispatch_uses_only_allowlisted_active_workflow_staging_ref_and_inputs() -> None:
    calls: list[tuple[str, str, object | None]] = []
    inputs = {
        "write_wave": "match-exclusion-recovery",
        "expected_candidate_sha": CANDIDATE_SHA,
        "orchestration_run_id": "42",
    }

    def request(method: str, path: str, payload: object | None) -> object:
        calls.append((method, path, payload))
        if method == "GET":
            return {
                "id": 55,
                "path": automation.WORKFLOW_PATHS[
                    "fly_api_staging_deploy.yml"
                ],
                "state": "active",
            }
        return {
            "workflow_run_id": 200,
            "html_url": "https://github.example/runs/200",
        }

    result = automation.dispatch_workflow(
        request,
        workflow="fly_api_staging_deploy.yml",
        inputs=inputs,
    )

    assert result == {
        "workflow": "fly_api_staging_deploy.yml",
        "workflow_id": 55,
        "workflow_run_id": 200,
        "html_url": "https://github.example/runs/200",
    }
    assert calls == [
        (
            "GET",
            (
                f"/repos/{automation.REPOSITORY}/actions/workflows/"
                "fly_api_staging_deploy.yml"
            ),
            None,
        ),
        (
            "POST",
            (
                f"/repos/{automation.REPOSITORY}/actions/workflows/"
                "fly_api_staging_deploy.yml/dispatches"
            ),
            {
                "ref": automation.STAGING_BRANCH,
                "inputs": inputs,
            },
        ),
    ]


def test_dispatch_rejects_nonallowlisted_workflow_without_api_request() -> None:
    def request(_method: str, _path: str, _payload: object | None) -> object:
        raise AssertionError("An unallowlisted workflow must not reach GitHub.")

    with pytest.raises(automation.ContractError, match="not allowlisted"):
        automation.dispatch_workflow(
            request,
            workflow="production.yml",
            inputs={},
        )


def _workflow_metadata(
    workflow: str = "parity-final-evidence.yml",
) -> dict[str, object]:
    return {
        "id": 77,
        "path": automation.WORKFLOW_PATHS[workflow],
        "state": "active",
    }


def _workflow_run(
    *,
    status: str,
    conclusion: str | None = None,
    head_sha: str = CANDIDATE_SHA,
) -> dict[str, object]:
    return {
        "id": 200,
        "workflow_id": 77,
        "event": "workflow_dispatch",
        "head_branch": automation.STAGING_BRANCH,
        "head_sha": head_sha,
        "repository": {"id": automation.REPOSITORY_ID},
        "status": status,
        "conclusion": conclusion,
        "html_url": "https://github.example/runs/200",
    }


def test_wait_for_workflow_run_accepts_only_exact_successful_run_identity() -> None:
    runs = iter(
        [
            _workflow_run(status="queued"),
            _workflow_run(status="completed", conclusion="success"),
        ]
    )
    calls: list[tuple[str, str, object | None]] = []
    clock = FakeClock()

    def request(method: str, path: str, payload: object | None) -> object:
        calls.append((method, path, payload))
        if "/actions/workflows/" in path:
            return _workflow_metadata()
        return next(runs)

    result = automation.wait_for_workflow_run(
        request,
        workflow="parity-final-evidence.yml",
        run_id=200,
        candidate_sha=CANDIDATE_SHA,
        timeout_seconds=5,
        poll_seconds=1,
        sleep=clock.sleep,
        monotonic=clock.monotonic,
    )

    assert result == {
        "workflow": "parity-final-evidence.yml",
        "workflow_run_id": 200,
        "candidate_sha": CANDIDATE_SHA,
        "status": "completed",
        "conclusion": "success",
        "html_url": "https://github.example/runs/200",
    }
    assert clock.sleeps == [1]
    assert calls[0] == (
        "GET",
        (
            f"/repos/{automation.REPOSITORY}/actions/workflows/"
            "parity-final-evidence.yml"
        ),
        None,
    )


def test_wait_for_workflow_run_rejects_wrong_run_identity() -> None:
    def request(_method: str, path: str, _payload: object | None) -> object:
        if "/actions/workflows/" in path:
            return _workflow_metadata()
        return _workflow_run(
            status="completed",
            conclusion="success",
            head_sha=OTHER_SHA,
        )

    with pytest.raises(automation.ContractError, match="head SHA"):
        automation.wait_for_workflow_run(
            request,
            workflow="parity-final-evidence.yml",
            run_id=200,
            candidate_sha=CANDIDATE_SHA,
        )


def test_wait_for_workflow_run_rejects_failed_conclusion() -> None:
    def request(_method: str, path: str, _payload: object | None) -> object:
        if "/actions/workflows/" in path:
            return _workflow_metadata()
        return _workflow_run(status="completed", conclusion="failure")

    with pytest.raises(automation.ContractError, match="concluded failure"):
        automation.wait_for_workflow_run(
            request,
            workflow="parity-final-evidence.yml",
            run_id=200,
            candidate_sha=CANDIDATE_SHA,
        )


def test_wait_for_workflow_run_times_out() -> None:
    clock = FakeClock()

    def request(_method: str, path: str, _payload: object | None) -> object:
        if "/actions/workflows/" in path:
            return _workflow_metadata()
        return _workflow_run(status="in_progress")

    with pytest.raises(automation.ContractError, match="timed out"):
        automation.wait_for_workflow_run(
            request,
            workflow="parity-final-evidence.yml",
            run_id=200,
            candidate_sha=CANDIDATE_SHA,
            timeout_seconds=2,
            poll_seconds=1,
            sleep=clock.sleep,
            monotonic=clock.monotonic,
        )

    assert clock.sleeps == [1, 1]


def test_deployment_identity_requires_exact_vercel_fly_candidate_and_write_wave() -> None:
    vercel = {
        "candidate_sha": CANDIDATE_SHA,
        "vercel_deployment_id": DEPLOYMENT_ID,
        "vercel_deployment_origin": IMMUTABLE_VERCEL_ORIGIN,
        "web": _web_identity(),
    }
    fly = _fly_identity(write_wave="match-exclusion-recovery")

    result = automation.verify_deployment_identity(
        candidate_sha=CANDIDATE_SHA,
        vercel=vercel,
        fly=fly,
        expected_write_wave="match-exclusion-recovery",
    )

    assert result == {
        "candidate_sha": CANDIDATE_SHA,
        "vercel_deployment_id": DEPLOYMENT_ID,
        "vercel_deployment_origin": IMMUTABLE_VERCEL_ORIGIN,
        "fly_image_ref": FLY_IMAGE,
        "write_wave": "match-exclusion-recovery",
    }

    wrong_fly = _fly_identity(write_wave="none")
    with pytest.raises(automation.ContractError, match="Fly deployment identity"):
        automation.verify_deployment_identity(
            candidate_sha=CANDIDATE_SHA,
            vercel=vercel,
            fly=wrong_fly,
            expected_write_wave="match-exclusion-recovery",
        )


def test_final_none_accepts_exact_safe_health_and_rejects_active_write_wave() -> None:
    safe = _fly_identity(write_wave="none")

    assert automation.verify_final_none(safe) == {
        "safe": True,
        "candidate_sha": CANDIDATE_SHA,
        "fly_image_ref": FLY_IMAGE,
        "write_wave": "none",
    }

    active = _fly_identity(write_wave="match-exclusion-recovery")
    with pytest.raises(
        automation.ContractError,
        match="Final Fly no-write verification rejected",
    ):
        automation.verify_final_none(active)
