from __future__ import annotations

from jupr_app.services.admin_operations_service import build_admin_operations_status


def test_admin_operations_status_defaults_to_guarded_mode(monkeypatch) -> None:
    for name in [
        "JUPR_ENV",
        "JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT",
        "JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY",
        "JUPR_REQUIRE_API_AUDIT_LOG",
        "SUPABASE_SERVICE_ROLE_KEY",
    ]:
        monkeypatch.delenv(name, raising=False)

    payload = build_admin_operations_status()

    assert payload["service"] == "jupr-api"
    assert payload["environment"] == "local"
    assert payload["write_pilot_enabled"] is False
    assert payload["enabled_workflows"] == []
    assert payload["strict_audit_required"] is False
    assert payload["service_role_configured"] is False
    assert payload["workflows"]
    assert any(workflow["key"] == "match_log" for workflow in payload["workflows"])
    assert any("No Supabase service-role" in item for item in payload["permanent_guardrails"])


def test_admin_operations_status_exposes_enabled_pilot_flags(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "true")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", "yes")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "configured")
    monkeypatch.setenv("JUPR_STREAMLIT_FALLBACK_URL", "https://streamlit.example.test")

    payload = build_admin_operations_status()

    assert payload["environment"] == "production"
    assert payload["mode"] == "closed_club_production_write_pilot"
    assert payload["write_pilot_enabled"] is True
    assert payload["strict_audit_required"] is True
    assert payload["service_role_configured"] is True
    assert payload["streamlit_fallback_url"] == "https://streamlit.example.test"
    assert set(payload["enabled_workflows"]) >= {"match_log", "score_entry"}

    workflows = {workflow["key"]: workflow for workflow in payload["workflows"]}
    assert workflows["match_log"]["enabled"] is True
    assert workflows["match_log"]["effective_status"] == "enabled_for_pilot"
    assert workflows["score_entry"]["enabled"] is True
    assert workflows["admin_tools"]["requires_review_before_enablement"] is True
