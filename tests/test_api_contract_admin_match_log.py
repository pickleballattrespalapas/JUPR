from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase, fake_supabase, fake_tables

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def _patch_admin_auth(monkeypatch, supabase: FakeSupabase, role: str = "club_owner") -> None:
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_match_log_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_match_log_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role=role, assigned=True, source="admin_role_assignments"),
    )


def test_admin_match_log_disabled_contract(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", raising=False)
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", raising=False)
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_ANON_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    response = TestClient(app).get("/admin/clubs/club/match-log")

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is False
    assert payload["matches"] == []
    assert payload["duplicate_groups"] == []
    assert payload["resolved_duplicate_groups"] == []


def test_admin_match_log_enabled_contract(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", raising=False)
    supabase = fake_supabase()
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/match-log?filter=League&limit=20",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["enabled"] is True
    assert payload["apply_enabled"] is False
    assert payload["status"] == "planning_only"
    assert payload["summary"]["duplicate_groups"] == 1
    assert payload["summary"]["resolved_duplicate_groups"] == 0
    assert payload["duplicate_groups"][0]["delete_ids"] == [2]
    assert payload["duplicate_delete_preview"]["delete_count"] == 1
    assert payload["correction_plan"]["apply_endpoint"] is None
    assert "notes" not in payload["matches"][0]


def test_admin_match_log_enabled_read_requires_authentication(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: fake_supabase())

    response = TestClient(app).get("/admin/clubs/club/match-log")

    assert response.status_code == 401
    assert response.json()["detail"] == "missing bearer token"


def test_admin_match_log_read_allows_assigned_scorekeeper(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    supabase = fake_supabase()
    _patch_admin_auth(monkeypatch, supabase, role="scorekeeper")

    response = TestClient(app).get(
        "/admin/clubs/club/match-log?limit=20",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    assert response.json()["enabled"] is True


def test_admin_match_log_read_rejects_unassigned_authenticated_user(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    supabase = fake_supabase()
    _patch_admin_auth(monkeypatch, supabase)
    monkeypatch.setattr(
        "services.api.admin_match_log_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="__unassigned__", assigned=False, source="default"),
    )

    response = TestClient(app).get(
        "/admin/clubs/club/match-log",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "insufficient permission"


def test_admin_match_log_player_options_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/match-log/player-options",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "match_log_player_options"
    assert payload["count"] == 4
    assert [player["id"] for player in payload["players"]] == [1, 2, 3, 4]
    assert payload["players"][0]["label"] == "Alex (#1)"


def test_admin_match_log_social_list_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).get(
        "/admin/clubs/club/match-log/social",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "social_match_log_rows"
    assert payload["count"] == 1
    assert payload["rows"][0]["social_match_id"] == "social-1"
    assert payload["rows"][0]["event_name"] == "Friday Social"
    assert payload["rows"][0]["t1_p1"] == "Social Alex"


def test_admin_match_log_social_update_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/social/social-1",
        headers={"Authorization": "Bearer local"},
        json={"event_name": "Friday Social Updated", "score_t1": 8, "score_t2": 11},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "social_match_updated"
    assert tables["live_event_matches"][0]["score_t1"] == 8
    assert tables["live_events"][0]["name"] == "Friday Social Updated"
    assert tables["admin_activity_log"][0]["action_type"] == "social_match_log_update"


def test_admin_match_log_social_delete_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/match-log/social/delete",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "DELETE", "social_match_ids": ["social-1"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "social_matches_deleted"
    assert payload["deleted_count"] == 1
    assert tables["live_event_matches"] == []
    assert tables["admin_activity_log"][0]["action_type"] == "social_match_log_delete"


def test_admin_match_log_apply_disabled_before_auth(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", raising=False)
    called = {"auth": False}

    def fake_auth(*_args, **_kwargs):
        called["auth"] = True
        raise AssertionError("auth should not run while apply flag is disabled")

    monkeypatch.setattr("services.api.admin_match_log_routes.authenticate_bearer", fake_auth)
    response = TestClient(app).patch("/admin/clubs/club/match-log/edits", json={"patches": []})

    assert response.status_code == 403
    assert called == {"auth": False}


def test_admin_match_log_apply_edits_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/edits",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "APPLY", "patches": [{"id": 1, "week_tag": "Week 2"}], "correction_note": "Fix week"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["updated_count"] == 1
    assert tables["matches"][0]["week_tag"] == "Week 2"


def test_admin_match_log_duplicate_cleanup_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/match-log/duplicates/cleanup",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "DELETE", "delete_ids": [2]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["deleted_count"] == 1
    assert [row["id"] for row in tables["matches"]] == [1, 3]


def test_admin_match_log_bulk_exclude_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    called = {}
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    _patch_admin_auth(monkeypatch, supabase)

    def fake_delete_rated_matches_with_replay(**kwargs):
        called.update(kwargs)
        return {
            "deleted_count": len(kwargs["match_ids"]),
            "deleted_ids": list(kwargs["match_ids"]),
            "affected_player_ids": [1, 2],
            "replay_result": {"matches_scanned_total": 3, "matches_rewritten": 2, "league_ratings_rows": 4, "skipped_incomplete": 0},
            "warning": None,
            "error": None,
            "replay_error": None,
            "actor": kwargs["actor"],
        }

    monkeypatch.setattr("services.api.admin_match_log_routes.delete_rated_matches_with_replay", fake_delete_rated_matches_with_replay)

    response = TestClient(app).post(
        "/admin/clubs/club/match-log/exclude",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "DELETE", "match_ids": [3], "note": "wrong test row"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "matches_excluded"
    assert payload["deleted_ids"] == [3]
    assert called["actor"] == "admin@example.com"
    assert called["actor_role"] == "club_owner"
    assert called["note"] == "wrong test row"
    assert called["source"] == "next_match_log_bulk_exclude"


def test_admin_match_log_duplicate_no_issue_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/match-log/duplicates/resolve",
        headers={"Authorization": "Bearer local"},
        json={
            "confirmation_text": "NO ISSUE",
            "match_ids": [1, 2],
            "reason": "Legitimate repeated matchup with same score.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "duplicate_no_issue"
    assert payload["match_ids"] == [1, 2]
    assert [row["id"] for row in tables["matches"]] == [1, 2, 3]
    assert tables["admin_match_log_duplicate_resolutions"][0]["match_id_key"] == "1,2"
    assert tables["admin_activity_log"][0]["action_type"] == "match_duplicate_false_positive_resolved"
