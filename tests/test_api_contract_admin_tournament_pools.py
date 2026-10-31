from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")
from fastapi.testclient import TestClient
from services.api.main import app
from tests.test_admin_tournament_pool_service import PoolSupabase, VERSION, fixture_tables, pool_config


def test_pool_route_requires_admin_and_preserves_guarded_request(monkeypatch):
    tables = fixture_tables()
    db = PoolSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda *_args: db)
    monkeypatch.setattr("services.api.admin_tournament_routes.authenticate_bearer", lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"))
    monkeypatch.setattr("services.api.admin_tournament_routes.resolve_admin_role", lambda **_kwargs: SimpleNamespace(role="club_owner"))
    guarded = []
    def guard(_supabase, **kwargs):
        guarded.append(kwargs)
        kwargs["preflight"]()
        return kwargs["mutate"]()
    monkeypatch.setattr("services.api.admin_tournament_routes._guarded_ops_mutation", guard)
    request = {"pool_config": pool_config(), "expected_state_fingerprint": "reviewed-state", "expected_draw_updated_at": VERSION,
               "expected_team_versions": [{"id": f"team-{n}", "updated_at": VERSION} for n in range(1,7)], "confirmation_text": "SAVE POOLS"}
    response = TestClient(app).put("/admin/clubs/club/tournaments/admin/tournaments/tournament/draws/draw/pools",headers={"Authorization":"Bearer local"},json=request)
    assert response.status_code == 200, response.text
    assert response.json()["draw"]["pool_config"] == pool_config()
    assert guarded[0]["action"] == "ops_pools_save"
    assert guarded[0]["expected_state"] == "reviewed-state"
    assert guarded[0]["payload"]["expected_team_versions"] == request["expected_team_versions"]

    monkeypatch.setattr("services.api.admin_tournament_routes.resolve_admin_role", lambda **_kwargs: SimpleNamespace(role="viewer"))
    denied = TestClient(app).put("/admin/clubs/club/tournaments/admin/tournaments/tournament/draws/draw/pools",headers={"Authorization":"Bearer local"},json=request)
    assert denied.status_code == 403
    assert len(db.rpc_calls) == 1
