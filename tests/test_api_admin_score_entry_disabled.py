import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_admin_batch_endpoint_disabled_by_default(client, monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", raising=False)

    response = client.post("/admin/clubs/club-1/matches/batch", json={"matches": []})

    assert response.status_code == 403
    assert (
        response.json()["detail"]
        == "Next admin score entry is disabled. Use Streamlit admin until Supabase JWT role auth is implemented."
    )


def test_admin_batch_endpoint_enabled_still_requires_token_guard(client, monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", "true")
    monkeypatch.setenv("JUPR_ADMIN_API_TOKEN", "expected-token")

    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": []},
        headers={"x-admin-permission": "enter_scores", "x-admin-token": "wrong-token"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "invalid admin token"


def test_admin_batch_endpoint_enabled_with_valid_token_reaches_service(client, monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY", "yes")
    monkeypatch.setenv("JUPR_ADMIN_API_TOKEN", "expected-token")

    monkeypatch.setattr(
        "services.api.main.get_supabase_client",
        lambda: object(),
    )
    monkeypatch.setattr(
        "services.api.main.load_data",
        lambda _supabase, _club_id: (None, None, None, None, None, None, None, None, None, {}),
    )

    class _Result:
        ok = True
        errors = []
        data = {"inserted": 1, "skipped_incomplete": 0}

    monkeypatch.setattr("services.api.main.submit_match_batch", lambda *_args, **_kwargs: _Result())

    response = client.post(
        "/admin/clubs/club-1/matches/batch",
        json={"matches": [{"league": "A"}], "source": "next_admin_score_entry"},
        headers={"x-admin-permission": "enter_scores", "x-admin-token": "expected-token"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["auth_mode"] == "token_guard_placeholder"
    assert payload["result"]["inserted"] == 1
