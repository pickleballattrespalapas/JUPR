from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

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
    monkeypatch.setenv("SUPABASE_URL", "https://sijpxjxvdtrehmqvirfi.supabase.co")

    resp = TestClient(app).get("/health")

    assert resp.status_code == 200
    assert resp.json() == {
        "ok": True,
        "service": "jupr-api",
        "environment": "staging",
        "git_commit_sha": "a" * 40,
        "fly_app_name": "juprleagues-api-staging",
        "fly_image_ref": "registry.fly.io/juprleagues-api-staging@sha256:123",
        "fly_machine_version": "42",
        "supabase_project_ref": "sijpxjxvdtrehmqvirfi",
    }
