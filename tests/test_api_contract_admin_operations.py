from __future__ import annotations

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def test_admin_operations_status_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")

    response = TestClient(app).get("/admin/operations/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "jupr-api"
    assert payload["environment"] == "staging"
    assert payload["write_pilot_enabled"] is True
    assert payload["strict_audit_required"] is True
    assert "match_log" in payload["enabled_workflows"]
    assert payload["pilot_gates"]
    assert payload["permanent_guardrails"]

    workflows = {workflow["key"]: workflow for workflow in payload["workflows"]}
    assert workflows["match_log"]["enabled"] is True
    assert workflows["match_log"]["apply_enabled"] is True
    assert workflows["match_log"]["apply_env_flag"] == "JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY"
    assert workflows["score_entry"]["env_flag"] == "JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY"
    assert workflows["admin_tools"]["risk"] == "critical"
