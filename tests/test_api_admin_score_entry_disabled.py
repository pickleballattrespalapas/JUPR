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
    monkeypatch.setattr("services.api.main.submit_match_batch", _write)

    response = client.post("/admin/clubs/club-1/matches/batch", json={"matches": []})

    assert response.status_code == 403
    assert called == {"auth": False, "write": False}
