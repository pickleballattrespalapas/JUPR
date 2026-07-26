from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

import pytest
from fastapi.testclient import TestClient

from jupr_app.services.direct_match_entry_service import (
    DirectMatchRecoveryRequiredError,
)
from services.api.main import app


VALID_MATCH = {"t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 7}


@pytest.fixture
def client():
    return TestClient(app)


def _enable(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", "true")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role")


def _mock_common(monkeypatch):
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: object())
    monkeypatch.setattr(
        "services.api.main.load_data",
        lambda _supabase, _club_id: (None, None, None, None, None, None, None, None, None, False, ""),
    )


def _atomic_result():
    return {
        "ok": True,
        "match_write_committed": True,
        "submitted_count": 1,
        "result": {"inserted": 1},
        "feedback": {},
        "operation": {"idempotent": False},
        "warnings": [],
    }


def test_successful_scorekeeper_write_passes_atomic_audit_identity(
    client,
    monkeypatch,
):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "scorekeeper"})())
    captured = {}

    def _submit_atomic(_supabase, **kwargs):
        captured.update(kwargs)
        return _atomic_result()

    monkeypatch.setattr(
        "services.api.main.submit_atomic_direct_matches",
        _submit_atomic,
    )

    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={
            "matches": [VALID_MATCH],
            "source": "next/admin/score-entry",
            "idempotency_key": "test:audit-identity",
        },
        headers={"Authorization": "Bearer x"},
    )

    assert response.status_code == 200
    assert captured["club_id"] == "club-1"
    assert captured["actor_email"] == "keeper@club.com"
    assert captured["actor_role"] == "scorekeeper"
    assert captured["source"] == "next/admin/score-entry"
    assert captured["idempotency_key"] == "test:audit-identity"


def test_audit_payload_does_not_include_tokens_or_service_role(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "scorekeeper"})())
    captured = {}

    def _submit_atomic(_supabase, **kwargs):
        captured.update(kwargs)
        return _atomic_result()

    monkeypatch.setattr(
        "services.api.main.submit_atomic_direct_matches",
        _submit_atomic,
    )
    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={
            "matches": [
                {
                    **VALID_MATCH,
                    "authorization": "Bearer SECRET_TOKEN",
                    "service_role": "SERVICE_KEY",
                }
            ],
            "idempotency_key": "test:audit-secret-filter",
        },
        headers={"Authorization": "Bearer raw-user-token"},
    )

    assert response.status_code == 200
    payload_dump = str(captured)
    assert "raw-user-token" not in payload_dump
    assert "SECRET_TOKEN" not in payload_dump
    assert "SERVICE_KEY" not in payload_dump


def test_atomic_audit_failure_rolls_back_even_without_strict_env(
    client,
    monkeypatch,
):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.delenv("JUPR_REQUIRE_API_AUDIT_LOG", raising=False)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "scorekeeper"})())
    monkeypatch.setattr(
        "services.api.main.submit_atomic_direct_matches",
        lambda *_a, **_k: (_ for _ in ()).throw(
            DirectMatchRecoveryRequiredError(
                "atomic audit response unavailable"
            )
        ),
    )

    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={
            "matches": [VALID_MATCH],
            "idempotency_key": "test:audit-unavailable",
        },
        headers={"Authorization": "Bearer x"},
    )
    assert response.status_code == 503
    assert "audit" in response.json()["detail"]


def test_strict_mode_fails_if_audit_write_fails(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "scorekeeper"})())
    monkeypatch.setattr(
        "services.api.main.submit_atomic_direct_matches",
        lambda *_a, **_k: (_ for _ in ()).throw(
            DirectMatchRecoveryRequiredError(
                "atomic audit response unavailable"
            )
        ),
    )

    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={
            "matches": [VALID_MATCH],
            "idempotency_key": "test:audit-strict",
        },
        headers={"Authorization": "Bearer x"},
    )
    assert response.status_code == 503
    assert "audit" in response.json()["detail"]


def test_denied_different_club_write_does_not_perform_match_write(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "viewer@club.com", "user_id": "u2"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "read_only"})())

    called = {"match": False, "audit": False}

    def _submit(*_a, **_k):
        called["match"] = True
        return _atomic_result()

    def _audit(*_a, **_k):
        called["audit"] = True
        return type("W", (), {"ok": True, "warning": None})()

    monkeypatch.setattr(
        "services.api.main.submit_atomic_direct_matches",
        _submit,
    )
    monkeypatch.setattr("services.api.main.write_admin_activity_log", _audit)

    response = client.post(
        "/admin/clubs/other-club/matches/batch",
        json={
            "matches": [],
            "idempotency_key": "test:audit-denied",
        },
        headers={"Authorization": "Bearer x"},
    )
    assert response.status_code == 403
    assert called["match"] is False
    assert called["audit"] is True
