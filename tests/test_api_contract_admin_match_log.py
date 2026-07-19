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
    assert payload["matches"][0]["notes"] == "operator correction context"
    assert payload["summary"]["scanned_matches"] == 3
    assert all(match["id"] != 99 for match in payload["matches"])
    assert payload["warnings"] == []


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
    monkeypatch.delenv("JUPR_REQUIRE_API_AUDIT_LOG", raising=False)
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/social/social-1",
        headers={"Authorization": "Bearer local"},
        json={"event_name": "  Friday   Social  ", "score_t1": 8, "score_t2": 11},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mode"] == "social_match_updated"
    assert tables["live_event_matches"][0]["score_t1"] == 8
    assert tables["live_events"][0]["name"] == "Friday Social"
    assert payload["result"]["patch"] == {"score_t1": 8}
    assert payload["result"]["before"] == {"score_t1": 7}
    assert payload["result"]["after"] == {"score_t1": 8}
    audit = tables["admin_activity_log"][0]
    assert audit["action_type"] == "social_match_log_update"
    assert audit["actor_email"] == "admin@example.com"
    assert audit["actor_role"] == "club_owner"
    assert audit["entity_type"] == "live_event_match"
    assert audit["entity_id"] == "social-1"
    assert audit["source_page"] == "next_match_log_social_editor"
    assert audit["flagged_for_review"] is True
    assert audit["before_json"] == {"score_t1": 7}
    assert audit["after_json"]["source_client"] == "fastapi/nextjs"
    assert audit["after_json"]["source_page"] == "next_match_log_social_editor"
    assert audit["after_json"]["patch"] == {"score_t1": 8}
    assert audit["after_json"]["result"] == payload["result"]


def test_admin_match_log_social_update_rejects_blank_event_name(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/social/social-1",
        headers={"Authorization": "Bearer local"},
        json={"event_name": " \u00a0 "},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Club Social event name cannot be blank."
    assert tables["live_events"][0]["name"] == "Friday Social"
    assert tables["admin_activity_log"] == []
    assert supabase.operations == []


def test_admin_match_log_social_update_rejects_mixed_table_delta(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/social/social-1",
        headers={"Authorization": "Bearer local"},
        json={"event_name": "Friday Social Updated", "score_t1": 8},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Update the Club Social event name separately from match fields."
    assert tables["live_event_matches"][0]["score_t1"] == 7
    assert tables["live_events"][0]["name"] == "Friday Social"
    assert tables["admin_activity_log"] == []
    assert supabase.operations == []


def test_admin_match_log_social_update_strict_audit_failure_rolls_back(monkeypatch):
    tables = fake_tables()
    tables["__failed_insert_tables__"] = {"admin_activity_log"}
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/social/social-1",
        headers={"Authorization": "Bearer local"},
        json={"score_t1": 8},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Audit log write required but unavailable; the Club Social update was rolled back."
    )
    assert tables["live_event_matches"][0]["score_t1"] == 7
    assert tables["admin_activity_log"] == []
    assert [operation["payload"] for operation in supabase.operations] == [
        {"score_t1": 8},
        {"score_t1": 7},
    ]


def test_admin_match_log_social_update_non_strict_audit_failure_warns(monkeypatch):
    tables = fake_tables()
    tables["__failed_insert_tables__"] = {"admin_activity_log"}
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.delenv("JUPR_REQUIRE_API_AUDIT_LOG", raising=False)
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/social/social-1",
        headers={"Authorization": "Bearer local"},
        json={"score_t1": 8},
    )

    assert response.status_code == 200
    assert response.json()["warnings"] == ["Admin activity log write failed."]
    assert tables["live_event_matches"][0]["score_t1"] == 8
    assert tables["admin_activity_log"] == []


def test_admin_match_log_social_update_reports_critical_rollback_failure(monkeypatch):
    from services.api import admin_match_log_routes

    tables = fake_tables()
    tables["__failed_insert_tables__"] = {"admin_activity_log"}
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    _patch_admin_auth(monkeypatch, supabase)
    real_update = admin_match_log_routes.update_social_match_row
    calls = {"count": 0}

    def update_then_fail_rollback(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("rollback unavailable")
        return real_update(*args, **kwargs)

    monkeypatch.setattr(admin_match_log_routes, "update_social_match_row", update_then_fail_rollback)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/social/social-1",
        headers={"Authorization": "Bearer local"},
        json={"score_t1": 8},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Critical: audit log write failed and the Club Social update could not be rolled back. "
        "Manual review is required."
    )
    assert tables["live_event_matches"][0]["score_t1"] == 8
    assert tables["admin_activity_log"] == []


def test_admin_match_log_social_update_rollback_preserves_concurrent_newer_value(monkeypatch):
    from services.api import admin_match_log_routes

    tables = fake_tables()
    tables["__failed_insert_tables__"] = {"admin_activity_log"}
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    _patch_admin_auth(monkeypatch, supabase)
    real_update = admin_match_log_routes.update_social_match_row
    calls = {"count": 0}

    def update_with_concurrent_change(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            tables["live_event_matches"][0]["score_t1"] = 9
        return real_update(*args, **kwargs)

    monkeypatch.setattr(admin_match_log_routes, "update_social_match_row", update_with_concurrent_change)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/social/social-1",
        headers={"Authorization": "Bearer local"},
        json={"score_t1": 8},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Critical: audit log write failed and the Club Social update could not be rolled back. "
        "Manual review is required."
    )
    assert tables["live_event_matches"][0]["score_t1"] == 9
    assert tables["admin_activity_log"] == []
    assert [operation["payload"] for operation in supabase.operations] == [{"score_t1": 8}]


def test_admin_match_log_social_update_rejects_true_noop_without_audit(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/social/social-1",
        headers={"Authorization": "Bearer local"},
        json={
            "event_name": "  Friday   Social  ",
            "score_t1": 7,
            "score_t2": 11,
            "round_number": 1,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "No Club Social changes detected."
    assert tables["live_event_matches"][0]["score_t1"] == 7
    assert tables["live_events"][0]["name"] == "Friday Social"
    assert tables["admin_activity_log"] == []
    assert supabase.operations == []


def test_admin_match_log_social_delete_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.delenv("JUPR_REQUIRE_API_AUDIT_LOG", raising=False)
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
    audit = tables["admin_activity_log"][0]
    assert audit["action_type"] == "social_match_log_delete"
    assert audit["before_json"][0]["id"] == "social-1"
    assert audit["after_json"]["deleted_count"] == 1


def test_admin_match_log_social_delete_strict_audit_failure_restores_rows(monkeypatch):
    tables = fake_tables()
    original_row = dict(tables["live_event_matches"][0])
    tables["__failed_insert_tables__"] = {"admin_activity_log"}
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/match-log/social/delete",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "DELETE", "social_match_ids": ["social-1"]},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Audit log write required but unavailable; the Club Social delete was rolled back."
    )
    assert tables["live_event_matches"] == [original_row]
    assert tables["admin_activity_log"] == []


def test_admin_match_log_social_delete_reports_critical_restore_failure(monkeypatch):
    tables = fake_tables()
    tables["__failed_insert_tables__"] = {"admin_activity_log", "live_event_matches"}
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/match-log/social/delete",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "DELETE", "social_match_ids": ["social-1"]},
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Critical: audit log write failed and the Club Social delete could not be rolled back. "
        "Manual review is required."
    )
    assert tables["live_event_matches"] == []
    assert tables["admin_activity_log"] == []


def test_admin_match_log_social_delete_non_strict_audit_failure_warns(monkeypatch):
    tables = fake_tables()
    tables["__failed_insert_tables__"] = {"admin_activity_log"}
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    monkeypatch.delenv("JUPR_REQUIRE_API_AUDIT_LOG", raising=False)
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).post(
        "/admin/clubs/club/match-log/social/delete",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "DELETE", "social_match_ids": ["social-1"]},
    )

    assert response.status_code == 200
    assert response.json()["deleted_count"] == 1
    assert response.json()["warnings"] == ["Admin activity log write failed."]
    assert tables["live_event_matches"] == []
    assert tables["admin_activity_log"] == []


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

    def fake_atomic(_supabase, **kwargs):
        tables["matches"][0]["week_tag"] = kwargs["patches"][0]["week_tag"]
        return {
            "ok": True,
            "mode": "applied",
            "atomic": True,
            "operation_id": "operation-1",
            "operation_status": "succeeded",
            "updated_count": 1,
            "updated_ids": [1],
            "recompute_scope": {"standings": True, "ratings": False},
            "warnings": [],
        }

    monkeypatch.setattr("jupr_app.services.admin_match_log_service.apply_atomic_match_edits", fake_atomic)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/edits",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "APPLY", "patches": [{"id": 1, "week_tag": "Week 2"}], "correction_note": "Fix week", "idempotency_key": "api-edit-1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["updated_count"] == 1
    assert payload["atomic"] is True
    assert tables["matches"][0]["week_tag"] == "Week 2"


def test_admin_match_log_apply_rejects_activity_edits_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    _patch_admin_auth(monkeypatch, supabase)

    response = TestClient(app).patch(
        "/admin/clubs/club/match-log/edits",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "APPLY", "patches": [{"id": 1, "is_active": False}]},
    )

    assert response.status_code == 400
    assert "guarded rated-match exclude workflow" in response.json()["detail"]


def test_admin_match_log_recovery_contract(monkeypatch):
    tables = fake_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_APPLY", "1")
    _patch_admin_auth(monkeypatch, supabase)
    monkeypatch.setattr(
        "services.api.admin_match_log_routes.recover_atomic_match_edit",
        lambda *_args, **kwargs: {
            "ok": True,
            "mode": "recovered",
            "operation_id": kwargs["operation_id"],
            "operation_status": "succeeded",
            "replay_job_id": "job-1",
            "warnings": [],
        },
    )

    response = TestClient(app).post(
        "/admin/clubs/club/match-log/edits/operation-1/recover",
        headers={"Authorization": "Bearer local"},
        json={"confirmation_text": "RECOVER"},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "recovered"
    assert response.json()["operation_id"] == "operation-1"


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
    assert [row["id"] for row in tables["matches"]] == [1, 3, 99]


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
    assert [row["id"] for row in tables["matches"]] == [1, 2, 3, 99]
    assert tables["admin_match_log_duplicate_resolutions"][0]["match_id_key"] == "1,2"
    assert tables["admin_activity_log"][0]["action_type"] == "match_duplicate_false_positive_resolved"
