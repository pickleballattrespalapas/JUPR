from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from services.api import admin_starting_rating_review_routes as routes
from tests.test_starting_rating_review_service import DB


@pytest.fixture
def client(monkeypatch):
    db = DB()
    app = FastAPI()
    routes.install_admin_starting_rating_review_routes(app, get_supabase_client=lambda: db)
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setattr(routes, "authenticate_bearer", lambda authorization: SimpleNamespace(email="staff@test",user_id="staff"))
    monkeypatch.setattr(routes, "resolve_admin_role", lambda **kwargs: SimpleNamespace(role="administrator"))
    return TestClient(app), db


def test_preview_uses_private_authorized_endpoint(client):
    response = client[0].post('/admin/clubs/club/starting-rating-review/preview',json={"player_ids":[1],"requested_ratings":{"1":"3.5"}})
    assert response.status_code == 200
    assert response.json()["players"][0]["review_required"]


def test_insufficient_role_cannot_read_or_confirm_private_estimates(client,monkeypatch):
    monkeypatch.setattr(routes,"resolve_admin_role",lambda **kwargs:SimpleNamespace(role="organizer"))
    assert client[0].post('/admin/clubs/club/starting-rating-review/preview',json={"player_ids":[1]}).status_code == 403
    assert not client[1].calls


def test_confirmation_requires_staging_write_gate(client,monkeypatch):
    monkeypatch.setenv("JUPR_ENV","staging")
    monkeypatch.delenv("JUPR_ENABLE_STAGING_NEXT_ADMIN_LEAGUE_MANAGER_WRITES",raising=False)
    response=client[0].post('/admin/clubs/club/starting-rating-review/confirm',json={"reviews":[{"player_id":1,"rating_jupr":3.25,"expected_state_fingerprint":"a"*64}]})
    assert response.status_code == 403
    assert not client[1].calls


def test_confirmation_conflict_is_409(client):
    client[1].error=RuntimeError("STARTING_REVIEW_HISTORY_EXISTS")
    response=client[0].post('/admin/clubs/club/starting-rating-review/confirm',json={"reviews":[{"player_id":1,"rating_jupr":3.25,"expected_state_fingerprint":"a"*64}]})
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "STARTING_RATING_REVIEW_CONFLICT"
