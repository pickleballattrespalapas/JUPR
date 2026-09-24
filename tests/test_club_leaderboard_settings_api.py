from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from postgrest.exceptions import APIError
import pytest

from services.api import club_leaderboard_settings_routes as routes


def client(monkeypatch, role="administrator"):
    state = SimpleNamespace(calls=[], error=None)
    class Database:
        def rpc(self, name, params):
            state.calls.append((name, params))
            if state.error:
                raise APIError({"code": state.error, "message": "private database details", "details": "", "hint": ""})
            return SimpleNamespace(execute=lambda: SimpleNamespace(data={"revision": params["p_expected_revision"] + 1, "draft": params["p_settings"] or {}}))
        def table(self, name): return self
        def select(self, *_): return self
        def eq(self, *_): return self
        def execute(self): return SimpleNamespace(data=[])
    monkeypatch.setattr(routes, "authenticate_bearer", lambda _: SimpleNamespace(email="verified@example.invalid", user_id="verified-id"))
    monkeypatch.setattr(routes, "resolve_admin_role", lambda **_: SimpleNamespace(role=role, assigned=True))
    app = FastAPI()
    routes.install_club_leaderboard_settings_routes(app, get_supabase_client=Database)
    return TestClient(app), state


PATH = "/admin/clubs/tres_palapas/leaderboard-settings"


def test_defaults_preserve_all_time_and_save_uses_verified_actor(monkeypatch):
    c, state = client(monkeypatch)
    result = c.get(PATH).json()
    assert result["revision"] == 0 and result["published"] is None
    assert result["draft"]["default_season_id"] is None and result["draft"]["seasons"] == []
    assert c.put(PATH, json={"revision": 0, "settings": result["draft"]}).status_code == 200
    name, params = state.calls[0]
    assert name == "save_club_leaderboard_settings"
    assert params["p_actor_id"] == "verified-id" and params["p_club_id"] == "tres_palapas"


@pytest.mark.parametrize("role", ["operator", "organizer", "scorekeeper", "read_only"])
def test_non_administrators_cannot_read_or_change_settings(monkeypatch, role):
    c, state = client(monkeypatch, role)
    assert c.get(PATH).status_code == 403
    assert c.post(PATH + "/publish", json={"revision": 0}).status_code == 403
    assert not state.calls


@pytest.mark.parametrize("code,status", [("40001", 409), ("42501", 403), ("08006", 503)])
def test_stale_revoked_and_uncertain_saves_are_not_retried(monkeypatch, code, status):
    c, state = client(monkeypatch)
    state.error = code
    result = c.put(PATH, json={"revision": 4, "settings": {}})
    assert result.status_code == status and len(state.calls) == 1
    assert "private database" not in result.text


def test_invalid_season_and_caller_supplied_actor_are_rejected(monkeypatch):
    c, state = client(monkeypatch)
    for payload in [{"revision": 0, "settings": {"timezone": "Not/AZone"}},
                    {"revision": 0, "settings": {}, "actor_id": "forged"}]:
        assert c.put(PATH, json=payload).status_code == 422
    assert not state.calls
