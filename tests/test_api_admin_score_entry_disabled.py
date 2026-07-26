import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_admin_batch_endpoint_disabled_by_default_returns_403_before_auth_or_writes(client, monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", raising=False)

    called = {"auth": False, "write": False}

    def _auth(*_args, **_kwargs):
        called["auth"] = True
        raise AssertionError("auth should not run")

    def _write(*_args, **_kwargs):
        called["write"] = True
        raise AssertionError("writes should not run")

    monkeypatch.setattr("services.api.main.authenticate_bearer", _auth)
    monkeypatch.setattr("services.api.main.submit_atomic_direct_matches", _write)

    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": [], "idempotency_key": "test:score-disabled"},
    )

    assert response.status_code == 403
    assert called == {"auth": False, "write": False}


def test_score_entry_status_requires_flag_and_service_role(client, monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", "1")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = client.get("/admin/clubs/club-1/score-entry/status")

    assert response.status_code == 200
    assert response.json()["ready"] is False
    assert response.json()["submit_endpoint"] is None
    assert response.json()["fallback"]["match_uploader_route"] == "/admin/match-uploader"


def test_score_entry_write_fails_closed_without_service_role(client, monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", "1")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": [], "idempotency_key": "test:score-no-role"},
    )

    assert response.status_code == 503
    assert "SUPABASE_SERVICE_ROLE_KEY" in response.json()["detail"]
