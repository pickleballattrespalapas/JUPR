from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

import pytest
from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def _enable(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", "true")


def _mock_common(monkeypatch):
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: object())
    monkeypatch.setattr(
        "services.api.main.load_data",
        lambda _supabase, _club_id: (None, None, None, None, None, None, None, None, None, {}),
    )


class _Result:
    ok = True
    errors = []
    data = {"inserted": 1}


def test_missing_bearer_token_returns_401(client, monkeypatch):
    _enable(monkeypatch)
    response = client.post("/admin/clubs/club-1/matches/batch", json={"matches": []})
    assert response.status_code == 401
    assert "token" in response.json()["detail"]


def test_invalid_token_returns_401(client, monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_args, **_kwargs: (_ for _ in ()).throw(Exception("bad")))
    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": []},
        headers={"Authorization": "Bearer invalid"},
    )
    assert response.status_code == 401


def test_valid_token_no_role_returns_403(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "a@b.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_k: type("R", (), {"role": "read_only"})())

    response = client.post("/admin/clubs/club-1/matches/batch", json={"matches": []}, headers={"Authorization": "Bearer x"})
    assert response.status_code == 403


def test_scorekeeper_for_requested_club_reaches_submit(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **kwargs: type("R", (), {"role": "scorekeeper"})())
    called = {"ok": False}
    def _submit(*_a, **_k):
        called["ok"] = True
        return _Result()
    monkeypatch.setattr("services.api.main.submit_match_batch", _submit)
    response = client.post("/admin/clubs/club-1/matches/batch", json={"matches": [{"a":1}]}, headers={"Authorization": "Bearer x"})
    assert response.status_code == 200
    assert called["ok"] is True


def test_role_for_different_club_returns_403(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **kwargs: type("R", (), {"role": "read_only"})())
    response = client.post("/admin/clubs/club-2/matches/batch", json={"matches": []}, headers={"Authorization": "Bearer x"})
    assert response.status_code == 403


def test_read_only_returns_403(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "viewer@club.com", "user_id": "u9"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **kwargs: type("R", (), {"role": "read_only"})())
    response = client.post("/admin/clubs/club-1/matches/batch", json={"matches": []}, headers={"Authorization": "Bearer x"})
    assert response.status_code == 403


def test_no_secret_in_error_response(client, monkeypatch):
    _enable(monkeypatch)
    secret = "super-secret-value"
    monkeypatch.setenv("SUPABASE_JWT_SECRET", secret)
    response = client.post("/admin/clubs/club-1/matches/batch", json={"matches": []}, headers={"Authorization": "Bearer invalid"})
    body = response.text
    assert secret not in body
