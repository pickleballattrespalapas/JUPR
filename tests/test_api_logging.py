import logging

from fastapi.testclient import TestClient

from services.api.main import app


def test_request_logging_no_auth_or_secrets(caplog):
    caplog.set_level(logging.INFO, logger="jupr.api.request")
    client = TestClient(app)
    response = client.get("/health", headers={"Authorization": "Bearer secret-token", "x-request-id": "req-1"})
    assert response.status_code == 200
    rec = next(r for r in caplog.records if r.name == "jupr.api.request")
    assert rec.method == "GET"
    assert rec.path == "/health"
    assert rec.status_code == 200
    assert rec.request_id == "req-1"
    assert "Authorization" not in rec.getMessage()
    assert "secret" not in rec.getMessage().lower()
