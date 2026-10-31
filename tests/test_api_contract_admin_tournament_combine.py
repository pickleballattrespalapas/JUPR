from types import SimpleNamespace

import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi import HTTPException
from fastapi.testclient import TestClient

from jupr_app.services.admin_tournament_ops_service import get_admin_tournament_ops_state_fingerprint
from services.api import admin_tournament_routes as routes
from services.api.main import app
from tests.test_admin_tournament_combine_service import CombineSupabase, combine_tables


URL = "/admin/clubs/club/tournaments/admin/tournaments/tournament/draws/draw-40/combine-divisions"


@pytest.fixture
def harness(monkeypatch):
    db = CombineSupabase(combine_tables())
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: db)
    monkeypatch.setattr(routes, "authenticate_bearer", lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="admin"))
    monkeypatch.setattr(routes, "resolve_admin_role", lambda **_kwargs: SimpleNamespace(role="club_owner"))
    calls = []
    def guarded(_db, **kwargs):
        calls.append(kwargs)
        kwargs["preflight"]()
        return kwargs["mutate"]()
    monkeypatch.setattr(routes, "_guarded_ops_mutation", guarded)
    payload = {"event_option_ids": ["event-40", "event-45"], "name": "Mixed 4.0 / 4.5", "expected_state_fingerprint": get_admin_tournament_ops_state_fingerprint(db, club_id="club", tournament_id="tournament"), "confirmation_text": "COMBINE DIVISIONS", "source": "next_tournament_ops_combine_divisions"}
    return TestClient(app), db, payload, calls


def test_combine_route_carries_reviewed_state_and_scoped_payload_into_guard(harness):
    client, db, payload, calls = harness
    response = client.post(URL, headers={"Authorization": "Bearer local"}, json=payload)
    assert response.status_code == 200
    assert response.json()["draw"]["source_event_option_ids"] == ["event-40", "event-45"]
    assert len(calls) == 1
    guarded = calls[0]
    assert guarded["action"] == "ops_divisions_combine"
    assert guarded["entity_type"] == "tournament_event_draw"
    assert guarded["entity_id"] == "draw-40"
    assert guarded["club_id"] == "club"
    assert guarded["expected_state"] == payload["expected_state_fingerprint"]
    assert guarded["payload"] == {"draw_id": "draw-40", "event_option_ids": payload["event_option_ids"], "name": payload["name"]}
    assert guarded["actor_email"] == "admin@example.com"
    assert len(db.rpc_calls) == 1


@pytest.mark.parametrize("patch,status", [({"expected_state_fingerprint": "0" * 64}, 409), ({"confirmation_text": ""}, 400), ({"event_option_ids": ["event-40"]}, 422), ({"expected_state_fingerprint": "missing"}, 422), ({"name": ""}, 422)])
def test_invalid_or_stale_combine_request_has_no_write(harness, patch, status):
    client, db, payload, _calls = harness
    response = client.post(URL, headers={"Authorization": "Bearer local"}, json={**payload, **patch})
    assert response.status_code == status
    assert db.rpc_calls == []


def test_unavailable_admin_feature_blocks_combine(harness, monkeypatch):
    client, db, payload, calls = harness
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "0")
    response = client.post(URL, headers={"Authorization": "Bearer local"}, json=payload)
    assert response.status_code == 403
    assert calls == []
    assert db.rpc_calls == []


def test_score_only_staff_cannot_combine_divisions(harness, monkeypatch):
    client, db, payload, calls = harness
    monkeypatch.setattr(routes, "resolve_admin_role", lambda **_kwargs: SimpleNamespace(role="scorekeeper"))
    response = client.post(URL, headers={"Authorization": "Bearer local"}, json=payload)
    assert response.status_code == 403
    assert calls == []
    assert db.rpc_calls == []


def test_unauthenticated_combine_is_rejected_before_review_or_mutation(harness, monkeypatch):
    client, db, payload, calls = harness
    def unauthenticated(_authorization):
        raise HTTPException(status_code=401, detail="Authentication required")
    monkeypatch.setattr(routes, "authenticate_bearer", unauthenticated)
    response = client.post(URL, json=payload)
    assert response.status_code == 401
    assert calls == []
    assert db.rpc_calls == []


def test_cross_club_combine_cannot_reach_rpc(harness):
    client, db, payload, _calls = harness
    response = client.post(URL.replace("/clubs/club/", "/clubs/other-club/"), headers={"Authorization": "Bearer local"}, json=payload)
    assert response.status_code == 400
    assert db.rpc_calls == []
