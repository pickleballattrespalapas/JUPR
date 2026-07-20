from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from scripts.staging_write_waves import ALL_STAGING_WRITE_FLAGS, expected_write_flags
from services.api.main import app


def test_api_app_imports():
    assert app is not None


def test_api_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"ok": True, "service": "jupr-api"}


def test_staging_health_attests_deployment_and_project_identity(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_DEPLOYMENT_GIT_SHA", "A" * 40)
    monkeypatch.setenv("FLY_APP_NAME", "juprleagues-api-staging")
    monkeypatch.setenv("FLY_IMAGE_REF", "registry.fly.io/juprleagues-api-staging@sha256:123")
    monkeypatch.setenv("FLY_MACHINE_VERSION", "42")
    monkeypatch.setenv(
        "JUPR_WEB_BASE_URL",
        "https://jupr-git-staging-pickleballattrespalapas1.vercel.app/",
    )
    monkeypatch.setenv("SUPABASE_URL", "https://sijpxjxvdtrehmqvirfi.supabase.co")
    monkeypatch.setenv("SUPABASE_JWKS_URL", "https://sijpxjxvdtrehmqvirfi.supabase.co/auth/v1/.well-known/jwks.json")
    monkeypatch.setenv("JUPR_SUPABASE_JWT_MODE", "jwks")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    for name in ALL_STAGING_WRITE_FLAGS:
        monkeypatch.setenv(name, "0")

    resp = TestClient(app).get("/health")

    assert resp.status_code == 200
    payload = resp.json()
    assert {
        "ok": True,
        "service": "jupr-api",
        "environment": "staging",
        "git_commit_sha": "a" * 40,
        "fly_app_name": "juprleagues-api-staging",
        "fly_image_ref": "registry.fly.io/juprleagues-api-staging@sha256:123",
        "fly_machine_version": "42",
        "web_origin": "https://jupr-git-staging-pickleballattrespalapas1.vercel.app",
        "supabase_project_ref": "sijpxjxvdtrehmqvirfi",
    }.items() <= payload.items()
    assert payload["staging_write_wave"] == "none"
    assert payload["business_data_write_wave_active"] is False
    assert payload["security_denial_audit_logging_required"] is False
    assert payload["controlled_write_flags"] == expected_write_flags("none")
    assert payload["jwt_verification_configured"] is True
    assert payload["jwt_verification_mode"] == "jwks"
    assert payload["jwt_verification_project_ref"] == "sijpxjxvdtrehmqvirfi"


def test_staging_health_does_not_project_non_origin_web_url(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "staging")
    sensitive_marker = "leak-me-739"
    monkeypatch.setenv(
        "JUPR_WEB_BASE_URL",
        f"https://user:{sensitive_marker}@example.test/private?token={sensitive_marker}",
    )

    payload = TestClient(app).get("/health").json()

    assert payload["web_origin"] is None
    assert sensitive_marker not in str(payload)
