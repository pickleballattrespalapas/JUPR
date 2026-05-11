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


def test_successful_scorekeeper_write_records_audit_payload(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "scorekeeper"})())
    monkeypatch.setattr("services.api.main.submit_match_batch", lambda *_a, **_k: _Result())

    captured = {}

    def _write_audit(_supabase, payload):
        captured["payload"] = payload
        return type("W", (), {"ok": True, "warning": None})()

    monkeypatch.setattr("services.api.main.write_admin_activity_log", _write_audit)

    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": [{"a": 1}], "source": "next/admin/score-entry"},
        headers={"Authorization": "Bearer x"},
    )

    assert response.status_code == 200
    payload = captured["payload"]
    assert payload["club_id"] == "club-1"
    assert payload["actor_email"] == "keeper@club.com"
    assert payload["actor_role"] == "scorekeeper"
    assert payload["source_page"] == "next/admin/score-entry"
    assert payload["after_json"]["source_client"] == "fastapi/nextjs"


def test_audit_payload_does_not_include_tokens_or_service_role(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "scorekeeper"})())
    monkeypatch.setattr("services.api.main.submit_match_batch", lambda *_a, **_k: _Result())

    captured = {}

    def _write_audit(_supabase, payload):
        captured["payload"] = payload
        return type("W", (), {"ok": True, "warning": None})()

    monkeypatch.setattr("services.api.main.write_admin_activity_log", _write_audit)
    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": [{"authorization": "Bearer SECRET_TOKEN", "service_role": "SERVICE_KEY"}]},
        headers={"Authorization": "Bearer raw-user-token"},
    )

    assert response.status_code == 200
    payload_dump = str(captured["payload"])
    assert "raw-user-token" not in payload_dump
    assert "SECRET_TOKEN" not in payload_dump
    assert "SERVICE_KEY" not in payload_dump


def test_missing_audit_table_degrades_gracefully_by_default(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.delenv("JUPR_REQUIRE_API_AUDIT_LOG", raising=False)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "scorekeeper"})())
    monkeypatch.setattr("services.api.main.submit_match_batch", lambda *_a, **_k: _Result())
    monkeypatch.setattr("services.api.main.write_admin_activity_log", lambda *_a, **_k: type("W", (), {"ok": False, "warning": "missing table"})())

    response = client.post("/admin/clubs/club-1/matches/batch", json={"matches": []}, headers={"Authorization": "Bearer x"})
    assert response.status_code == 200


def test_strict_mode_fails_if_audit_write_fails(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "keeper@club.com", "user_id": "u1"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "scorekeeper"})())
    monkeypatch.setattr("services.api.main.submit_match_batch", lambda *_a, **_k: _Result())
    monkeypatch.setattr("services.api.main.write_admin_activity_log", lambda *_a, **_k: type("W", (), {"ok": False, "warning": "boom"})())

    response = client.post("/admin/clubs/club-1/matches/batch", json={"matches": []}, headers={"Authorization": "Bearer x"})
    assert response.status_code == 500
    assert "audit" in response.json()["detail"]


def test_denied_different_club_write_does_not_perform_match_write(client, monkeypatch):
    _enable(monkeypatch)
    _mock_common(monkeypatch)
    monkeypatch.setattr("services.api.main.authenticate_bearer", lambda *_a, **_k: type("U", (), {"email": "viewer@club.com", "user_id": "u2"})())
    monkeypatch.setattr("services.api.main.resolve_admin_role", lambda **_kwargs: type("R", (), {"role": "read_only"})())

    called = {"match": False, "audit": False}

    def _submit(*_a, **_k):
        called["match"] = True
        return _Result()

    def _audit(*_a, **_k):
        called["audit"] = True
        return type("W", (), {"ok": True, "warning": None})()

    monkeypatch.setattr("services.api.main.submit_match_batch", _submit)
    monkeypatch.setattr("services.api.main.write_admin_activity_log", _audit)

    response = client.post("/admin/clubs/other-club/matches/batch", json={"matches": []}, headers={"Authorization": "Bearer x"})
    assert response.status_code == 403
    assert called["match"] is False
    assert called["audit"] is True
