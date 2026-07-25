import logging

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from scripts.deployment_verifier import (
    PRODUCTION_ALLOWED_ORIGINS,
    PRODUCTION_FEATURE_FLAGS,
)
from scripts.staging_write_waves import ALL_STAGING_WRITE_FLAGS, expected_write_flags
from services.api.middleware import (
    StagingWriteWaveMiddleware,
    StructuredRequestLoggingMiddleware,
)
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
    monkeypatch.setenv("JUPR_IMAGE_BUILD_GIT_SHA", "A" * 40)
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
        "image_build_git_sha": "a" * 40,
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


def test_production_health_attests_exact_read_only_deployment_identity(monkeypatch):
    sha = "b" * 40
    project_ref = "abcdefghijklmnopqrst"
    monkeypatch.setenv("JUPR_ENV", "production")
    # Code identity is immutable image metadata. A stale legacy runtime secret
    # must never override the SHA baked into the deployed image.
    monkeypatch.setenv("JUPR_DEPLOYMENT_GIT_SHA", "c" * 40)
    monkeypatch.setenv("JUPR_IMAGE_BUILD_GIT_SHA", sha)
    monkeypatch.setenv("FLY_APP_NAME", "juprleagues-api")
    monkeypatch.setenv("FLY_IMAGE_REF", "registry.fly.io/juprleagues-api:deployment-123")
    monkeypatch.setenv("FLY_MACHINE_VERSION", "43")
    monkeypatch.setenv("JUPR_WEB_BASE_URL", "https://pickleballclubsandwich.com")
    monkeypatch.setenv("SUPABASE_URL", f"https://{project_ref}.supabase.co")
    monkeypatch.setenv(
        "SUPABASE_JWKS_URL",
        f"https://{project_ref}.supabase.co/auth/v1/.well-known/jwks.json",
    )
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "configured")
    monkeypatch.setenv("JUPR_SUPABASE_JWT_MODE", "jwks")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    monkeypatch.setenv("JUPR_PRODUCTION_WRITE_POLICY", "read_only")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    monkeypatch.setenv("JUPR_REQUIRE_WORKER_RUN_LOG", "1")
    monkeypatch.setenv("JUPR_EMAIL_MODE", "dry_run")
    monkeypatch.setenv("JUPR_EXPECTED_MIGRATION_HEAD", "20260720123402")
    monkeypatch.setenv("JUPR_EXPECTED_MIGRATION_CONTRACT", "c" * 64)
    monkeypatch.setenv(
        "JUPR_EXPECTED_MIGRATION_PROFILE",
        "next-fastapi-readonly-2026-07-25",
    )
    monkeypatch.setenv(
        "JUPR_ALLOWED_ORIGINS", ",".join(PRODUCTION_ALLOWED_ORIGINS)
    )
    monkeypatch.delenv("JUPR_ALLOWED_ORIGIN_REGEX", raising=False)
    for name in PRODUCTION_FEATURE_FLAGS:
        monkeypatch.setenv(name, "0")

    payload = TestClient(app).get("/health").json()

    assert payload["environment"] == "production"
    assert payload["git_commit_sha"] == sha
    assert payload["image_build_git_sha"] == sha
    assert payload["fly_app_name"] == "juprleagues-api"
    assert payload["supabase_project_ref"] == project_ref
    assert payload["jwt_verification_project_ref"] == project_ref
    assert payload["write_wave"] == "none"
    assert payload["staging_write_wave"] == "none"
    assert payload["business_data_write_wave_active"] is False
    assert payload["production_business_write_policy"] == "read_only"
    assert payload["expected_migration_head"] == "20260720123402"
    assert payload["expected_migration_contract"] == "c" * 64
    assert (
        payload["expected_migration_profile"]
        == "next-fastapi-readonly-2026-07-25"
    )
    assert payload["cors_allowed_origins"] == list(PRODUCTION_ALLOWED_ORIGINS)
    assert payload["cors_allowed_origin_regex"] is None
    assert payload["feature_flags"] == {
        name: False for name in PRODUCTION_FEATURE_FLAGS
    }
    assert payload["controlled_write_flags"] == expected_write_flags("none")


def test_production_unsafe_requests_fail_closed_until_policy_is_enabled(monkeypatch):
    guarded = FastAPI()
    guarded.add_middleware(StagingWriteWaveMiddleware)

    @guarded.post("/write")
    def write() -> dict[str, bool]:
        return {"written": True}

    client = TestClient(guarded)
    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.delenv("JUPR_PRODUCTION_WRITE_POLICY", raising=False)
    assert client.post("/write").status_code == 403

    monkeypatch.setenv("JUPR_PRODUCTION_WRITE_POLICY", "read_only")
    assert client.post("/write").status_code == 403

    monkeypatch.setenv("JUPR_PRODUCTION_WRITE_POLICY", "enabled")
    assert client.post("/write").status_code == 200


def test_production_policy_denials_are_structurally_logged(monkeypatch, caplog):
    guarded = FastAPI()
    guarded.add_middleware(StagingWriteWaveMiddleware)
    guarded.add_middleware(StructuredRequestLoggingMiddleware)

    @guarded.post("/write")
    def write() -> dict[str, bool]:
        return {"written": True}

    monkeypatch.setenv("JUPR_ENV", "production")
    monkeypatch.setenv("JUPR_PRODUCTION_WRITE_POLICY", "read_only")
    caplog.set_level(logging.INFO, logger="jupr.api.request")

    response = TestClient(guarded).post("/write")

    assert response.status_code == 403
    record = next(
        record
        for record in caplog.records
        if record.name == "jupr.api.request"
        and getattr(record, "path", None) == "/write"
    )
    assert record.method == "POST"
    assert record.status_code == 403
    assert response.headers["x-request-id"] == record.request_id
