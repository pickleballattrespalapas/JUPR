from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


class ExistingOperationSupabase(FakeSupabase):
    def rpc(self, *_args, **_kwargs):
        raise AssertionError("idempotent lookup must return before invoking the merge RPC")


def merge_tables() -> dict[str, list[dict]]:
    return {
        "players": [
            {"club_id": "club", "id": 1, "name": "Source Player", "rating": 1200, "active": True},
            {"club_id": "club", "id": 2, "name": "Target Player", "rating": 1300, "active": True},
            {"club_id": "club", "id": 3, "name": "Other A", "rating": 1200, "active": True},
            {"club_id": "club", "id": 4, "name": "Other B", "rating": 1200, "active": True},
        ],
        "matches": [
            {"club_id": "club", "id": 11, "t1_p1": 1, "t1_p2": 3, "t2_p1": 4, "t2_p2": 3, "score_t1": 11, "score_t2": 9},
            {"club_id": "club", "id": 12, "match_format": "singles", "t1_p1": 3, "t1_p2": None, "t2_p1": 1, "t2_p2": None, "score_t1": 6, "score_t2": 11},
        ],
        "league_ratings": [
            {"club_id": "club", "id": 101, "player_id": 1, "league_name": "Tuesday", "rating": 1200},
            {"club_id": "club", "id": 102, "player_id": 1, "league_name": "Thursday", "rating": 1220},
            {"club_id": "club", "id": 201, "player_id": 2, "league_name": "Tuesday", "rating": 1300},
        ],
        "club_people": [
            {"club_id": "club", "id": "person-source", "display_name": "Source Social", "normalized_name": "source social", "linked_player_id": 1},
        ],
        "admin_activity_log": [],
    }


def _install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr("services.api.admin_player_editor_routes.authenticate_bearer", lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"))
    monkeypatch.setattr("services.api.admin_player_editor_routes.resolve_admin_role", lambda **_kwargs: SimpleNamespace(role="club_owner"))


def test_admin_player_merge_preview_contract(monkeypatch):
    supabase = FakeSupabase(merge_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/players/editor/merge/preview",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "player_merge_preview"
    assert payload["can_merge"] is True
    assert len(payload["preview_fingerprint"]) == 64
    assert payload["match_reference_counts"]["total"] == 2
    assert payload["league_rating_plan"]["move_ids"] == [102]
    assert payload["league_rating_plan"]["delete_ids"] == [101]


def test_admin_player_merge_execute_contract(monkeypatch):
    tables = merge_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    preview_response = TestClient(app).post(
        "/admin/clubs/club/players/editor/merge/preview",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2},
    )
    fingerprint = preview_response.json()["preview_fingerprint"]

    response = TestClient(app).post(
        "/admin/clubs/club/players/editor/merge",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2, "preview_fingerprint": fingerprint, "confirmation_text": "MERGE"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "player_merge_execute"
    assert payload["operation_status"] == "merged_pending_replay"
    assert payload["recovery"]["replay_required"] is True
    assert all(row.get("t1_p1") != 1 and row.get("t2_p1") != 1 for row in tables["matches"])
    assert tables["matches"][0]["t1_p1"] == 2
    assert tables["matches"][1]["t2_p1"] == 2
    assert {row["id"] for row in tables["league_ratings"]} == {102, 201}
    moved = [row for row in tables["league_ratings"] if row["id"] == 102][0]
    assert moved["player_id"] == 2
    source_player = [row for row in tables["players"] if row["id"] == 1][0]
    assert source_player["active"] is False
    assert "MERGED into Target Player" in source_player["name"]
    assert tables["admin_activity_log"][0]["action_type"] == "merge_player_editor_players_admin"
    assert tables["admin_activity_log"][0]["flagged_for_review"] is True


def test_admin_player_merge_requires_confirmation(monkeypatch):
    supabase = FakeSupabase(merge_tables())
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/players/editor/merge",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2, "preview_fingerprint": "0" * 64, "confirmation_text": "M"},
    )

    assert response.status_code == 400
    assert "MERGE" in response.json()["detail"]


def test_admin_player_merge_rejects_stale_preview(monkeypatch):
    tables = merge_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    client = TestClient(app)
    preview = client.post(
        "/admin/clubs/club/players/editor/merge/preview",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2},
    ).json()
    tables["matches"].append(
        {"club_id": "club", "id": 13, "t1_p1": 1, "t1_p2": 3, "t2_p1": 4, "t2_p2": 3}
    )

    response = client.post(
        "/admin/clubs/club/players/editor/merge",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2, "preview_fingerprint": preview["preview_fingerprint"], "confirmation_text": "MERGE"},
    )

    assert response.status_code == 409
    assert "stale" in response.json()["detail"].lower()


def test_admin_player_merge_rejects_source_league_change_after_preview(monkeypatch):
    tables = merge_tables()
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)
    client = TestClient(app)
    preview = client.post(
        "/admin/clubs/club/players/editor/merge/preview",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2},
    ).json()
    tables["league_ratings"][0]["rating"] = 1250

    response = client.post(
        "/admin/clubs/club/players/editor/merge",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2, "preview_fingerprint": preview["preview_fingerprint"], "confirmation_text": "MERGE"},
    )

    assert response.status_code == 409
    assert "stale" in response.json()["detail"].lower()


def test_admin_player_merge_preview_blocks_match_collision(monkeypatch):
    tables = merge_tables()
    tables["matches"][0]["t2_p2"] = 2
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/players/editor/merge/preview",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2},
    )

    assert response.status_code == 200
    assert response.json()["can_merge"] is False
    assert response.json()["collision_match_ids"] == [11]


def test_admin_player_merge_preview_blocks_immutable_tournament_projection(monkeypatch):
    tables = merge_tables()
    tables["matches"][0]["tournament_game_id"] = "game-official-1"
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/players/editor/merge/preview",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["can_merge"] is False
    assert payload["official_tournament_match_ids"] == [11]
    assert "authoritative tournament participant" in payload["warnings"][0]


def test_admin_player_merge_preview_treats_inactive_at_as_inactive(monkeypatch):
    tables = merge_tables()
    tables["players"][0]["inactive_at"] = "2026-07-01T00:00:00+00:00"
    supabase = FakeSupabase(tables)
    _install_env(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/players/editor/merge/preview",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2},
    )

    assert response.status_code == 200
    assert response.json()["can_merge"] is False


def test_admin_player_merge_compensation_route_contract(monkeypatch):
    supabase = FakeSupabase(merge_tables())
    _install_env(monkeypatch, supabase)
    operation_id = "11111111-1111-4111-8111-111111111111"
    captured = {}

    def fake_compensate(_supabase, **kwargs):
        captured.update(kwargs)
        return {"ok": True, "mode": "player_merge_compensated", "operation_id": operation_id, "operation_status": "compensated"}

    monkeypatch.setattr("services.api.admin_player_editor_routes.compensate_admin_player_merge", fake_compensate)
    response = TestClient(app).post(
        f"/admin/clubs/club/players/editor/merge/{operation_id}/compensate",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "COMPENSATE MERGE"},
    )

    assert response.status_code == 200
    assert response.json()["operation_status"] == "compensated"
    assert captured["club_id"] == "club"
    assert captured["operation_id"] == operation_id


def test_admin_player_merge_replay_evidence_route_contract(monkeypatch):
    supabase = FakeSupabase(merge_tables())
    _install_env(monkeypatch, supabase)
    operation_id = "11111111-1111-4111-8111-111111111111"
    replay_job_id = "22222222-2222-4222-8222-222222222222"
    captured = {}

    def fake_verify(_supabase, **kwargs):
        captured.update(kwargs)
        return {"ok": True, "mode": "player_merge_replay_verified", "operation_id": operation_id, "operation_status": "replay_verified", "replay_job_id": replay_job_id}

    monkeypatch.setattr("services.api.admin_player_editor_routes.verify_admin_player_merge_replay", fake_verify)
    response = TestClient(app).post(
        f"/admin/clubs/club/players/editor/merge/{operation_id}/replay-evidence",
        headers={"Authorization": "Bearer local"},
        json={"replay_job_id": replay_job_id, "confirmation_text": "CONFIRM REPLAY RECOVERY"},
    )

    assert response.status_code == 200
    assert response.json()["operation_status"] == "replay_verified"
    assert captured["replay_job_id"] == replay_job_id


def test_admin_player_merge_operation_id_is_idempotent(monkeypatch):
    tables = merge_tables()
    operation_id = "11111111-1111-4111-8111-111111111111"
    preview_supabase = FakeSupabase(tables)
    _install_env(monkeypatch, preview_supabase)
    client = TestClient(app)
    fingerprint = client.post(
        "/admin/clubs/club/players/editor/merge/preview",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2},
    ).json()["preview_fingerprint"]
    tables["admin_player_merge_operations"] = [{
        "id": operation_id,
        "club_id": "club",
        "source_player_id": 1,
        "target_player_id": 2,
        "status": "merged_pending_replay",
        "preview_fingerprint": fingerprint,
        "result_json": {"ok": True, "mode": "player_merge_execute", "operation_id": operation_id, "operation_status": "merged_pending_replay"},
    }]
    supabase = ExistingOperationSupabase(tables)
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)

    response = client.post(
        "/admin/clubs/club/players/editor/merge",
        headers={"Authorization": "Bearer local"},
        json={"source_player_id": 1, "target_player_id": 2, "preview_fingerprint": fingerprint, "operation_id": operation_id, "confirmation_text": "MERGE"},
    )

    assert response.status_code == 200
    assert response.json()["idempotent_replay"] is True
    assert response.json()["operation_id"] == operation_id
