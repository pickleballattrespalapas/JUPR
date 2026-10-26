from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api import admin_league_manager_routes as routes
from jupr_app.services.league_weekly_highlights_service import HighlightsConflict


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-only")
    monkeypatch.setattr(routes, "authenticate_bearer", lambda _: SimpleNamespace(email="admin@example.com", user_id="a"))
    monkeypatch.setattr(routes, "resolve_admin_role", lambda **_: SimpleNamespace(role="administrator"))
    monkeypatch.setattr(routes, "require_league_manager_write_or_403", lambda: None)
    app = FastAPI()
    routes.install_admin_league_manager_routes(app, get_supabase_client=lambda: object())
    return TestClient(app)


PATH = "/admin/clubs/club/league-manager/leagues/Open/weeks/2/highlights"


def test_read_returns_exact_week_and_denies_operator(client, monkeypatch):
    monkeypatch.setattr(routes, "build_admin_league_results", lambda *_a, **k: {"selected_week": 2, "weekly_highlight_selection": {"week_num": k["week_num"], "categories": []}})
    assert client.get(PATH).json() == {"ok": True, "week_num": 2, "categories": []}
    assert client.get(PATH.replace("weeks/2", "weeks/99")).status_code == 400
    monkeypatch.setattr(routes, "resolve_admin_role", lambda **_: SimpleNamespace(role="operator"))
    assert client.get(PATH).status_code == 403


def test_write_contract_conflict_and_service_key_gate(client, monkeypatch):
    calls = []
    monkeypatch.setattr(routes, "save_weekly_highlights", lambda db, **kw: calls.append(kw) or {"ok": True, "revision": 1})
    payload = {"mode": "curated", "selected_ids": [], "expected_revision": 0}
    assert client.put(PATH, json=payload).status_code == 200
    assert calls[0]["club_id"] == "club" and calls[0]["week_num"] == 2
    assert calls[0]["actor_role"] == "administrator"
    assert client.put(PATH, json={**payload, "mode": "invented"}).status_code == 422
    assert client.put(PATH, json={**payload, "selected_ids": ["x"] * 7}).status_code == 422
    def conflict(*_a, **_k):
        raise HighlightsConflict("Reload")
    monkeypatch.setattr(routes, "save_weekly_highlights", conflict)
    assert client.put(PATH, json=payload).status_code == 409
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY")
    assert client.put(PATH, json=payload).status_code == 503
