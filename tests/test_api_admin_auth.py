from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

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


def test_missing_bearer_token_returns_401(client, monkeypatch):
    _enable(monkeypatch)
    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": [], "idempotency_key": "test:missing-bearer"},
    )
    assert response.status_code == 401
    assert "token" in response.json()["detail"]


def test_invalid_token_returns_401(client, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_args, **_kwargs: (_ for _ in ()).throw(HTTPException(status_code=401, detail="invalid token")))
    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": [], "idempotency_key": "test:invalid-token"},
        headers={"Authorization": "Bearer invalid"},
    )
    assert response.status_code == 401


def test_valid_token_no_role_returns_403(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "a@b.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_k: type("R", (), {"role": "read_only"})())

    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": [], "idempotency_key": "test:no-role"},
        headers={"Authorization": "Bearer x"},
    )
    assert response.status_code == 403


def test_scorekeeper_for_requested_club_reaches_submit(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **kwargs: type("R", (), {"role": "scorekeeper"})())
    called = {"ok": False}
    def _submit(*_a, **_k):
        called["ok"] = True
        return _atomic_result()
    monkeypatch.setattr(
        "services.api.main.submit_atomic_direct_matches",
        _submit,
    )
    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={
            "matches": [VALID_MATCH],
            "idempotency_key": "test:scorekeeper-submit",
        },
        headers={"Authorization": "Bearer x"},
    )
    assert response.status_code == 200
    assert called["ok"] is True
    assert response.json()["match_write_committed"] is True
    assert response.json()["recovery"]["match_log_route"] == "/admin/match-log"


def test_scorekeeper_cannot_send_multiple_matches_through_score_entry(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "scorekeeper"})())

    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={
            "matches": [VALID_MATCH, VALID_MATCH],
            "idempotency_key": "test:multiple",
        },
        headers={"Authorization": "Bearer x"},
    )

    assert response.status_code == 400
    assert "exactly one" in response.json()["detail"]


def test_scorekeeper_cannot_send_fractional_scores_through_score_entry(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "scorekeeper"})())

    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={
            "matches": [{**VALID_MATCH, "score_t1": 10.5}],
            "idempotency_key": "test:fractional",
        },
        headers={"Authorization": "Bearer x"},
    )

    assert response.status_code == 400
    assert "whole-number" in response.json()["detail"]


def test_role_for_different_club_returns_403(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **kwargs: type("R", (), {"role": "read_only"})())
    response = client.post(
        "/admin/clubs/club-2/matches/batch",
        json={"matches": [], "idempotency_key": "test:other-club"},
        headers={"Authorization": "Bearer x"},
    )
    assert response.status_code == 403


def test_read_only_returns_403(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "viewer@club.com", "user_id": "u9"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **kwargs: type("R", (), {"role": "read_only"})())
    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": [], "idempotency_key": "test:read-only"},
        headers={"Authorization": "Bearer x"},
    )
    assert response.status_code == 403


def test_no_secret_in_error_response(client, monkeypatch):
    _enable(monkeypatch)
    secret = "super-secret-value"
    monkeypatch.setenv("SUPABASE_JWT_SECRET", secret)
    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": [], "idempotency_key": "test:no-secret"},
        headers={"Authorization": "Bearer invalid"},
    )
    body = response.text
    assert secret not in body
