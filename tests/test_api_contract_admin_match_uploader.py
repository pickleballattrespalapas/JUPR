from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_uploader_service import FakeSupabase, fake_load_data, fake_storage
from jupr_app.services.admin_match_uploader_service import match_uploader_player_batch_fingerprint
from jupr_app.services.admin_guarded_write_service import GuardedWriteRecoveryRequired

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def _install_auth(monkeypatch) -> None:
    monkeypatch.setattr(
        "services.api.admin_match_uploader_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_match_uploader_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="scorekeeper"),
    )


def test_match_uploader_status_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", raising=False)
    monkeypatch.delenv(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES",
        raising=False,
    )
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = TestClient(app).get("/admin/clubs/club/match-uploader/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["singles_write_enabled"] is False
    assert payload["singles_submit_endpoint"] is None
    assert payload["submit_endpoint"] is None
    assert "4-Player" in payload["round_robin_format_options"]


def test_match_uploader_submit_disabled_before_auth(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", raising=False)
    called = {"auth": False}

    def fake_auth(*_args, **_kwargs):
        called["auth"] = True
        raise AssertionError("auth should not run while uploader flag is disabled")

    monkeypatch.setattr("services.api.admin_match_uploader_routes.authenticate_bearer", fake_auth)

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/batch",
        json={"matches": [], "idempotency_key": "test:disabled-1"},
    )

    assert response.status_code == 403
    assert called == {"auth": False}


def test_match_uploader_status_enabled_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.delenv(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES",
        raising=False,
    )
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(fake_storage()))

    response = TestClient(app).get("/admin/clubs/club/match-uploader/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["singles_write_enabled"] is False
    assert payload["singles_submit_endpoint"] is None
    assert payload["submit_endpoint"] == "/admin/clubs/{club_id}/match-uploader/batch"
    assert payload["round_robin_preview_endpoint"] == "/admin/clubs/{club_id}/match-uploader/round-robin/preview"
    assert payload["player_create_endpoint"] == "/admin/clubs/{club_id}/match-uploader/players"
    assert "Open" in payload["league_options"]


def test_match_uploader_round_robin_preview_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(fake_storage()))
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/round-robin/preview",
        headers={"Authorization": "Bearer local"},
        json={
            "source": "test",
            "courts": [{"court": 1, "format_type": "4-Player", "player_names": ["Alex", "Blair", "Casey", "Devon"]}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["missing_players"] == []
    assert payload["match_count"] == 3
    assert payload["courts"][0]["matches"][0]["t1_p1"]


def test_match_uploader_preview_gate_does_not_open_uploader_writes(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", raising=False)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_PREVIEW", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr(
        "services.api.main.create_client",
        lambda _url, _credential: FakeSupabase(fake_storage()),
    )
    _install_auth(monkeypatch)
    client = TestClient(app)

    status = client.get("/admin/clubs/club/match-uploader/status")
    assert status.status_code == 200
    assert status.json()["enabled"] is False
    assert status.json()["submit_endpoint"] is None

    preview = client.post(
        "/admin/clubs/club/match-uploader/round-robin/preview",
        headers={"Authorization": "Bearer local"},
        json={
            "source": "test_preview_only",
            "courts": [
                {
                    "court": 1,
                    "format_type": "4-Player",
                    "player_names": ["Alex", "Blair", "Casey", "Devon"],
                }
            ],
        },
    )
    assert preview.status_code == 200
    assert preview.json()["match_count"] == 3

    assert client.post(
        "/admin/clubs/club/match-uploader/batch",
        headers={"Authorization": "Bearer local"},
        json={"matches": [], "idempotency_key": "test:preview-batch"},
    ).status_code == 403
    assert client.post(
        "/admin/clubs/club/match-uploader/singles",
        headers={"Authorization": "Bearer local"},
        json={
            "t1_p1": 1,
            "t2_p1": 2,
            "score_t1": 11,
            "score_t2": 7,
            "idempotency_key": "test:preview-singles",
        },
    ).status_code == 403
    assert client.post(
        "/admin/clubs/club/match-uploader/players",
        headers={"Authorization": "Bearer local"},
        json={"players": [{"name": "Test", "starting_jupr": 3.5}], "reviewed_fingerprint": match_uploader_player_batch_fingerprint([{"name": "Test", "starting_jupr": 3.5}]), "idempotency_key": "preview-player-create", "confirmation_text": "CREATE PLAYERS"},
    ).status_code == 403


def test_match_uploader_create_players_contract(monkeypatch):
    storage = fake_storage()
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    _install_auth(monkeypatch)

    reviewed_players = [{"name": "New Person", "starting_jupr": 3.5}]
    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/players",
        headers={"Authorization": "Bearer local"},
        json={"source": "test", "players": reviewed_players, "reviewed_fingerprint": match_uploader_player_batch_fingerprint(reviewed_players), "idempotency_key": "create-player-batch", "confirmation_text": "CREATE PLAYERS"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["accepted_count"] == 1
    assert payload["players"][0]["name"] == "New Person"
    assert any(row["action_type"] == "create_match_uploader_players" for row in storage["admin_activity_log"])

    operation = TestClient(app).get(
        "/admin/clubs/club/match-uploader/player-operations/create-player-batch",
        headers={"Authorization": "Bearer local"},
    )
    assert operation.status_code == 200
    assert operation.json()["operation_key"] == "create-player-batch"
    assert operation.json()["status"] == "completed"
    assert operation.json()["result_json"]["accepted_count"] == 1
    assert operation.json()["error_text"] is None


def test_match_uploader_player_operation_requires_authentication(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(fake_storage()))

    response = TestClient(app).get(
        "/admin/clubs/club/match-uploader/player-operations/create-player-batch"
    )

    assert response.status_code == 401


def test_match_uploader_ambiguous_player_batch_returns_structured_recovery(monkeypatch):
    storage = fake_storage()
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    monkeypatch.setattr(
        "services.api.admin_match_uploader_routes.create_admin_match_uploader_players",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            GuardedWriteRecoveryRequired("create-player-timeout", "Player batch outcome is uncertain.")
        ),
    )
    _install_auth(monkeypatch)
    reviewed = [{"name": "José Núñez", "starting_jupr": 3.5}]

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/players",
        headers={"Authorization": "Bearer local"},
        json={
            "source": "test",
            "players": reviewed,
            "reviewed_fingerprint": match_uploader_player_batch_fingerprint(reviewed),
            "idempotency_key": "create-player-timeout",
            "confirmation_text": "CREATE PLAYERS",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == {
        "code": "RECOVERY_REQUIRED",
        "kind": "uncertain",
        "message": "Player batch outcome is uncertain.",
        "operation_key": "create-player-timeout",
        "recovery_required": True,
    }


def test_match_uploader_submit_contract(monkeypatch):
    storage = fake_storage()
    calls = []

    def fake_submit_atomic_direct_matches(supabase, **kwargs):
        calls.append({"supabase": supabase, **kwargs})
        return {
            "ok": True,
            "match_write_committed": True,
            "submitted_count": len(kwargs["matches"]),
            "result": {
                "inserted": len(kwargs["matches"]),
                "skipped_incomplete": 0,
                "skipped_empty": 0,
                "skipped_unrated": 0,
            },
            "feedback": {},
            "operation": {"idempotent": False},
            "warnings": [],
        }

    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.load_data", fake_load_data)
    monkeypatch.setattr(
        "jupr_app.services.admin_match_uploader_service.submit_atomic_direct_matches",
        fake_submit_atomic_direct_matches,
    )
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/batch",
        headers={"Authorization": "Bearer local"},
        json={
            "source": "test",
            "idempotency_key": "test:uploader-batch-1",
            "matches": [
                {
                    "date": "2026-03-01",
                    "league": "Open",
                    "week_tag": "Week 1",
                    "match_type": "Live Match",
                    "t1_p1": 1,
                    "t1_p2": 2,
                    "t2_p1": 3,
                    "t2_p2": 4,
                    "score_t1": 11,
                    "score_t2": 7,
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["submitted_count"] == 1
    assert payload["result"]["inserted"] == 1
    assert payload["match_write_committed"] is True
    assert payload["recovery"]["match_log_route"] == "/admin/match-log"
    assert payload["auto_player_updates"]["mode"] in {"disabled", "skipped", "auto_sent"}
    assert calls[0]["matches"][0]["score_t1"] == 11
    assert calls[0]["idempotency_key"] == "test:uploader-batch-1"


def test_match_uploader_email_failure_preserves_committed_write(monkeypatch):
    storage = fake_storage()

    def fake_submit_atomic_direct_matches(_supabase, **kwargs):
        return {
            "ok": True,
            "match_write_committed": True,
            "submitted_count": len(kwargs["matches"]),
            "result": {
                "inserted": len(kwargs["matches"]),
                "skipped_incomplete": 0,
                "skipped_empty": 0,
                "skipped_unrated": 0,
            },
            "feedback": {},
            "operation": {"idempotent": False},
            "warnings": [],
        }

    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: FakeSupabase(storage))
    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.load_data", fake_load_data)
    monkeypatch.setattr(
        "jupr_app.services.admin_match_uploader_service.submit_atomic_direct_matches",
        fake_submit_atomic_direct_matches,
    )
    monkeypatch.setattr(
        "services.api.admin_match_uploader_routes.auto_send_player_updates_for_match_payloads",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("mail provider unavailable")),
    )
    _install_auth(monkeypatch)

    response = TestClient(app).post(
        "/admin/clubs/club/match-uploader/batch",
        headers={"Authorization": "Bearer local"},
        json={
            "source": "test_email_failure",
            "idempotency_key": "test:uploader-email-1",
            "matches": [{"date": "2026-03-01", "league": "Open", "week_tag": "Week 1", "match_type": "Live Match", "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4, "score_t1": 11, "score_t2": 7}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["match_write_committed"] is True
    assert payload["result"]["inserted"] == 1
    assert payload["auto_player_updates"]["mode"] == "error"
    assert "Do not resubmit" in payload["warnings"][-1]
