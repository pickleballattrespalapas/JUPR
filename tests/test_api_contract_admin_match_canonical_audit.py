from types import SimpleNamespace

import pandas as pd

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


class FakeSupabase:
    def __init__(self):
        self.storage = {"admin_activity_log": []}

    def table(self, name):
        return FakeQuery(self.storage, name)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.payload = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def insert(self, payload):
        self.payload = dict(payload)
        return self

    def update(self, payload):
        self.payload = dict(payload)
        return self

    def execute(self):
        rows = self.storage.setdefault(self.table_name, [])
        if self.payload is not None:
            rows.append(dict(self.payload))
            return SimpleNamespace(data=[dict(self.payload)])
        return SimpleNamespace(data=list(rows))


def fake_load_data(_supabase, club_id, match_limit=5000):
    df_players = pd.DataFrame([
        {"club_id": club_id, "id": 1, "name": "Alex", "rating": 1500, "active": True},
        {"club_id": club_id, "id": 2, "name": "Blair", "rating": 1400, "active": True},
        {"club_id": club_id, "id": 3, "name": "Casey", "rating": 1300, "active": True},
        {"club_id": club_id, "id": 4, "name": "Devon", "rating": 1200, "active": True},
    ])
    df_matches = pd.DataFrame([
        {
            "club_id": club_id,
            "id": 10,
            "date": "2026-01-01T00:00:00Z",
            "league": "Open",
            "match_type": "League",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "score_t1": 11,
            "score_t2": 8,
        }
    ])
    return (
        df_players,
        df_players,
        pd.DataFrame(),
        df_matches,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {"Alex": 1, "Blair": 2, "Casey": 3, "Devon": 4},
        {1: "Alex", 2: "Blair", 3: "Casey", 4: "Devon"},
        False,
        None,
    )


def install_env(monkeypatch, supabase=None):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_CANONICAL_AUDIT", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase or FakeSupabase())
    monkeypatch.setattr("services.api.admin_match_canonical_audit_routes.authenticate_bearer", lambda _authorization: SimpleNamespace(email="owner@example.com", user_id="user-1"))
    monkeypatch.setattr(
        "services.api.admin_match_canonical_audit_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner", assigned=True, source="admin_role_assignments"),
    )
    monkeypatch.setattr("jupr_app.services.admin_match_canonical_audit_service.load_data", fake_load_data)


def test_match_canonical_status_disabled(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_CANONICAL_AUDIT", raising=False)
    response = TestClient(app).get("/admin/clubs/club/match-canonical-audit/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["audit_endpoint"] is None


def test_match_canonical_options_and_audit(monkeypatch):
    install_env(monkeypatch)
    client = TestClient(app)
    options = client.get("/admin/clubs/club/match-canonical-audit/options", headers={"Authorization": "Bearer local"})
    assert options.status_code == 200
    assert options.json()["players"][0]["player_name"] == "Alex"

    audit = client.post(
        "/admin/clubs/club/match-canonical-audit/run",
        headers={"Authorization": "Bearer local"},
        json={"player_id": 1, "league_id": "Open", "limit": 100},
    )
    assert audit.status_code == 200
    assert audit.json()["report"]["counts"]["profile_visible"] == 1


def test_match_canonical_apply_requires_confirmation(monkeypatch):
    install_env(monkeypatch)
    response = TestClient(app).post(
        "/admin/clubs/club/match-canonical-audit/normalize",
        headers={"Authorization": "Bearer local"},
        json={"player_id": 1, "dry_run": False, "confirmation_text": "APPLY"},
    )
    assert response.status_code == 400
    assert "APPLY NORMALIZE" in response.json()["detail"]
