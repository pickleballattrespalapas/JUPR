from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from services.api.main import app
from jupr_app.services.admin_tournament_broadcast_service import CONFIRM_SEND
from tests.test_admin_tournament_broadcast_service import fixture, prepare
from tests.test_api_contract_admin_tournament import _install_auth

ROOT = "/admin/clubs/club/tournaments/admin/tournaments/tour_1/registrations/broadcasts"


@pytest.fixture
def api(fixture, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda *_args: fixture)
    _install_auth(monkeypatch)
    return fixture, TestClient(app), {"Authorization": "Bearer fixture"}


def payload(db):
    return {key: value for key, value in prepare(db).items() if key not in {"club_id", "tournament_id", "actor_email", "actor_role"}}


def test_complete_api_flow_is_dry_run_and_recoverable(api, monkeypatch):
    db, client, headers = api
    monkeypatch.setattr("jupr_app.domain.notifications.tournament_registrant_broadcast_email.send_email_with_inline_chart",
        lambda **_: pytest.fail("Dry-run flow must never use SMTP"))
    preview = client.post(ROOT.rsplit("/", 1)[0]+"/broadcast-preview", headers=headers,
        json={"registration_ids": ["registration_1"], "subject": "Update", "message": "Hello"})
    assert preview.status_code == 200
    assert preview.json()["send_available"] is True
    assert preview.json()["delivery_mode"] == "dry_run"
    body = payload(db)
    created = client.post(ROOT, json=body, headers=headers)
    assert created.status_code == 200
    assert created.headers["cache-control"] == "private, no-store, max-age=0"
    key = created.json()["operation_key"]
    assert created.json()["recipients"][0]["status"] == "pending"
    sent = client.post(f"{ROOT}/{key}/recipients/0/send", headers=headers, json={"confirmation_text": CONFIRM_SEND})
    assert sent.status_code == 200
    assert sent.json()["status"] == "dry_run"
    repeat = client.post(f"{ROOT}/{key}/recipients/0/send", headers=headers, json={"confirmation_text": CONFIRM_SEND})
    assert repeat.json()["status"] == "dry_run"
    result = client.get(f"{ROOT}/{key}", headers=headers).json()
    assert result["pending_count"] == 0
    assert client.get(ROOT, headers=headers).json()["broadcasts"][0]["operation_key"] == key
    assert client.post(ROOT, json=body, headers=headers).json()["pending_count"] == 0


@pytest.mark.parametrize("patch", [{"registration_ids": []}, {"registration_ids": None}, {"message": "x"*10001}, {"subject": "x"*201}])
def test_send_api_requires_bounded_explicit_selection(api, patch):
    db, client, headers = api
    assert client.post(ROOT, json={**payload(db), **patch}, headers=headers).status_code == 422
    assert not db.tables.get("communications_admin_operations")


@pytest.mark.parametrize("method,path,body", [
    ("post", ROOT, "create"), ("get", ROOT, None),
    ("get", ROOT+"/00000000-0000-0000-0000-000000000000", None),
    ("post", ROOT+"/00000000-0000-0000-0000-000000000000/recipients/0/send", "send")])
def test_broadcast_endpoints_require_tournament_permission(api, monkeypatch, method, path, body):
    db, client, headers = api
    monkeypatch.setattr("services.api.admin_tournament_routes.resolve_admin_role", lambda **_: SimpleNamespace(role="viewer"))
    options = {"json": payload(db) if body == "create" else {"confirmation_text": CONFIRM_SEND}} if body else {}
    assert getattr(client, method)(path, headers=headers, **options).status_code == 403
    assert not db.tables.get("communications_admin_operations")


def test_broadcast_requires_authentication(api, monkeypatch):
    db, client, _ = api
    def reject(_):
        raise HTTPException(status_code=401, detail="Sign in required")
    monkeypatch.setattr("services.api.admin_tournament_routes.authenticate_bearer", reject)
    assert client.post(ROOT, json=payload(db)).status_code == 401


def test_broadcast_api_rejects_confirmation_and_cross_tournament(api):
    db, client, headers = api
    assert client.post(ROOT, json={**payload(db), "confirmation_text": ""}, headers=headers).status_code == 400
    assert client.post(ROOT.replace("tour_1", "missing"), json=payload(db), headers=headers).status_code == 400
    assert not db.tables.get("communications_admin_operations")


def test_broadcast_api_does_not_expose_database_errors(api, monkeypatch):
    db, client, headers = api
    def fail(*_args, **_kwargs):
        raise RuntimeError("provider-password-sensitive-diagnostic")
    monkeypatch.setattr("services.api.admin_tournament_routes.list_tournament_broadcasts", fail)
    response = client.get(ROOT, headers=headers)
    assert response.status_code == 500
    assert "sensitive" not in response.text
