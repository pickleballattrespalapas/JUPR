from copy import deepcopy
from types import SimpleNamespace

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api.admin_tools_routes import install_admin_tools_routes
from services.api.main import app


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.limit_value = None
        self.insert_payload = None
        self.upsert_payload = None
        self.update_payload = None
        self.delete_flag = False
        self.is_filter = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def is_(self, key, value):
        self.is_filter = (key, value)
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = dict(payload)
        return self

    def upsert(self, payload, **_kwargs):
        self.upsert_payload = dict(payload)
        return self

    def update(self, payload):
        self.update_payload = dict(payload)
        return self

    def delete(self):
        self.delete_flag = True
        return self

    def execute(self):
        rows = self.storage.setdefault(self.table_name, [])
        scoped = list(rows)
        for key, expected in self.filters:
            scoped = [row for row in scoped if str(row.get(key)) == str(expected)]
        if self.is_filter is not None:
            key, expected = self.is_filter
            scoped = [row for row in scoped if row.get(key) is expected]
        if self.insert_payload is not None:
            row = {
                "id": f"row-{len(rows) + 1}",
                "created_at": f"created-{len(rows) + 1}",
                "updated_at": f"inserted-{id(self)}",
                **self.insert_payload,
            }
            rows.append(row)
            return SimpleNamespace(data=[row])
        if self.upsert_payload is not None:
            email = self.upsert_payload.get("email")
            club_id = self.upsert_payload.get("club_id")
            existing = next((row for row in rows if row.get("email") == email and row.get("club_id") == club_id), None)
            if existing:
                existing.update(self.upsert_payload)
                existing["updated_at"] = f"upserted-{id(self)}"
                row = existing
            else:
                row = {
                    "id": f"row-{len(rows) + 1}",
                    "created_at": "2026-01-01T00:00:00Z",
                    "updated_at": f"upserted-{id(self)}",
                    **self.upsert_payload,
                }
                rows.append(row)
            return SimpleNamespace(data=[row])
        if self.update_payload is not None:
            updated = []
            for row in rows:
                if row in scoped:
                    row.update(self.update_payload)
                    row["updated_at"] = f"updated-{id(self)}"
                    updated.append(dict(row))
            return SimpleNamespace(data=updated)
        if self.delete_flag:
            self.storage[self.table_name] = [row for row in rows if row not in scoped]
            return SimpleNamespace(data=scoped)
        if self.limit_value is not None:
            scoped = scoped[: self.limit_value]
        return SimpleNamespace(data=scoped)


class FakeSupabase:
    def __init__(self):
        self.storage = {
            "admin_role_assignments": [
                {"id": "role-owner", "club_id": "club", "email": "owner@example.com", "role": "super_admin", "user_id": "user-1", "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"}
            ],
            "admin_activity_log": [],
            "admin_guarded_operations": [],
            "matches": [{"club_id": "club", "id": 1, "t1_p1_r": 1200, "t1_p2_r": 1200, "t2_p1_r": 1200, "t2_p2_r": 1200, "t1_p1_r_end": 1210, "t1_p2_r_end": 1210, "t2_p1_r_end": 1190, "t2_p2_r_end": 1190}],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


def install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr("services.api.admin_tools_routes.authenticate_bearer", lambda _authorization: SimpleNamespace(email="owner@example.com", user_id="user-1"))
    monkeypatch.setattr(
        "services.api.admin_tools_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="super_admin", assigned=True, source="admin_role_assignments"),
    )


def test_admin_tools_status_disabled(monkeypatch):
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", raising=False)
    response = TestClient(app).get("/admin/clubs/club/tools/status")
    assert response.status_code == 200
    assert response.json()["enabled"] is False


def test_tournament_backfill_preview_route_requires_authentication(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    supabase = FakeSupabase()
    local_app = FastAPI()
    install_admin_tools_routes(local_app, get_supabase_client=lambda: supabase)
    contract_path = "/admin/clubs/{club_id}/tools/backfills/tournament-matches/preview"
    assert set(local_app.openapi()["paths"][contract_path]) == {"get"}
    apply_path = "/admin/clubs/{club_id}/tools/backfills/tournament-matches/apply"
    assert set(local_app.openapi()["paths"][apply_path]) == {"post"}
    report_path = "/admin/clubs/{club_id}/tools/reports/ratings"
    assert set(local_app.openapi()["paths"][report_path]) == {"get"}
    social_review_path = "/admin/clubs/{club_id}/tools/social-submissions"
    assert set(local_app.openapi()["paths"][social_review_path]) == {"get"}
    social_moderation_path = "/admin/clubs/{club_id}/tools/social-submissions/{event_id}/moderate"
    assert set(local_app.openapi()["paths"][social_moderation_path]) == {"post"}
    before = deepcopy(supabase.storage)

    response = TestClient(local_app).get("/admin/clubs/club/tools/backfills/tournament-matches/preview")
    report_response = TestClient(local_app).get("/admin/clubs/club/tools/reports/ratings")
    social_review_response = TestClient(local_app).get("/admin/clubs/club/tools/social-submissions?status=pending")
    social_moderation_response = TestClient(local_app).post(
        "/admin/clubs/club/tools/social-submissions/harmless/moderate",
        json={
            "action": "approve",
            "expected_status": "pending",
            "confirmation_text": "APPROVE SOCIAL SUBMISSION",
        },
    )
    apply_response = TestClient(local_app).post(
        "/admin/clubs/club/tools/backfills/tournament-matches/apply",
        json={
            "game_ids": ["harmless"],
            "preview_fingerprint": "not-used-without-auth",
            "preview_limit": 500,
            "confirmation_text": "BACKFILL TOURNAMENT MATCHES",
        },
    )

    assert response.status_code == 401
    assert report_response.status_code == 401
    assert social_review_response.status_code == 401
    assert social_moderation_response.status_code == 401
    assert apply_response.status_code == 401
    assert supabase.storage == before


def test_social_review_allows_read_only_role_but_moderation_requires_manage_matches(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    monkeypatch.setattr("services.api.admin_tools_routes.authenticate_bearer", lambda _authorization: SimpleNamespace(email="reader@example.com", user_id="reader-1"))
    monkeypatch.setattr(
        "services.api.admin_tools_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="read_only", assigned=True, source="admin_role_assignments"),
    )
    supabase = FakeSupabase()
    local_app = FastAPI()
    install_admin_tools_routes(local_app, get_supabase_client=lambda: supabase)
    client = TestClient(local_app)

    review = client.get(
        "/admin/clubs/club/tools/social-submissions?status=pending",
        headers={"Authorization": "Bearer local"},
    )
    moderate = client.post(
        "/admin/clubs/club/tools/social-submissions/harmless/moderate",
        headers={"Authorization": "Bearer local"},
        json={
            "action": "approve",
            "expected_status": "pending",
            "confirmation_text": "APPROVE SOCIAL SUBMISSION",
        },
    )

    assert review.status_code == 200
    assert review.json()["read_only"] is True
    assert moderate.status_code == 403
    assert moderate.json()["detail"] == "insufficient permission"
    assert not supabase.storage.get("live_events")
    assert supabase.storage["admin_activity_log"][-1]["after_json"]["required_permission"] == "manage_matches"


def test_admin_tools_read_rejects_authenticated_user_without_club_assignment(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    monkeypatch.setattr(
        "services.api.admin_tools_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="owner@example.com", user_id="user-1"),
    )
    supabase = FakeSupabase()
    supabase.storage["admin_role_assignments"] = []
    local_app = FastAPI()
    install_admin_tools_routes(local_app, get_supabase_client=lambda: supabase)

    response = TestClient(local_app).get(
        "/admin/clubs/club/tools/overview",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "insufficient permission"
    assert supabase.storage["admin_activity_log"][-1]["after_json"]["reason"] == "missing_club_assignment"


def test_admin_tools_read_rejects_assignment_for_other_club(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    monkeypatch.setattr(
        "services.api.admin_tools_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="owner@example.com", user_id="user-1"),
    )
    supabase = FakeSupabase()
    supabase.storage["admin_role_assignments"][0]["club_id"] = "other-club"
    local_app = FastAPI()
    install_admin_tools_routes(local_app, get_supabase_client=lambda: supabase)

    response = TestClient(local_app).get(
        "/admin/clubs/club/tools/overview",
        headers={"Authorization": "Bearer local"},
    )

    assert response.status_code == 403
    assert response.json()["detail"] == "insufficient permission"
    assert supabase.storage["admin_activity_log"][-1]["after_json"]["reason"] == "missing_club_assignment"


def test_admin_tools_overview_and_role_update(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    client = TestClient(app)
    overview = client.get("/admin/clubs/club/tools/overview", headers={"Authorization": "Bearer local"})
    assert overview.status_code == 200
    assert overview.json()["roles"][0]["email"] == "owner@example.com"
    assert overview.json()["health"]["match_schema"]["snapshot_columns_present"] is True
    report = client.get("/admin/clubs/club/tools/reports/ratings", headers={"Authorization": "Bearer local"})
    assert report.status_code == 200
    assert report.json()["read_only"] is True
    assert report.json()["scope"] == "OVERALL"

    rejected = client.patch(
        "/admin/clubs/club/tools/roles",
        headers={"Authorization": "Bearer local"},
        json={"email": "score@example.com", "role": "scorekeeper", "action": "upsert", "confirmation_text": "SAVE"},
    )
    assert rejected.status_code == 400
    assert "SAVE ROLE" in rejected.json()["detail"]

    saved = client.patch(
        "/admin/clubs/club/tools/roles",
        headers={"Authorization": "Bearer local"},
        json={"email": "score@example.com", "role": "scorekeeper", "action": "upsert", "confirmation_text": "SAVE ROLE", "operation_key": "role-overview-save"},
    )
    assert saved.status_code == 200
    emails = {row["email"] for row in saved.json()["roles"]}
    assert "score@example.com" in emails
    assert supabase.storage["admin_activity_log"]


def test_admin_tools_badge_recompute_permission_is_mode_specific(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    monkeypatch.setattr(
        "services.api.admin_tools_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="read_only", assigned=True, source="admin_role_assignments"),
    )
    monkeypatch.setattr(
        "services.api.admin_tools_routes.run_admin_badge_recompute_job",
        lambda *_args, **_kwargs: {"ok": True, "mode": "dry-run", "read_only": True, "summary": {}},
    )
    client = TestClient(app)

    preview = client.post(
        "/admin/clubs/club/tools/workers/badge-recompute",
        headers={"Authorization": "Bearer local"},
        json={"mode": "dry-run"},
    )
    apply = client.post(
        "/admin/clubs/club/tools/workers/badge-recompute",
        headers={"Authorization": "Bearer local"},
        json={"mode": "append-only", "confirmation_text": "RUN BADGE RECOMPUTE", "operation_key": "tools-badge-apply"},
    )

    assert preview.status_code == 200
    assert apply.status_code == 403
    assert supabase.storage["admin_activity_log"][-1]["after_json"]["required_permission"] == "run_replay"


def _patch_role(client, *, email="score@example.com", role="scorekeeper", action="upsert"):
    return client.patch(
        "/admin/clubs/club/tools/roles",
        headers={"Authorization": "Bearer local"},
        json={
            "email": email,
            "role": role,
            "action": action,
            "confirmation_text": "REVOKE ROLE" if action == "revoke" else "SAVE ROLE",
            "operation_key": "role-operation-test",
        },
    )


def test_role_upsert_strict_preflight_audit_failure_does_not_mutate(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    before = deepcopy(supabase.storage["admin_role_assignments"])
    monkeypatch.setattr(
        "jupr_app.services.admin_guarded_write_service.write_admin_activity_log",
        lambda *_args, **_kwargs: SimpleNamespace(ok=False, warning="boom"),
    )

    response = _patch_role(TestClient(app))

    assert response.status_code == 500
    assert "Required audit intent" in response.json()["detail"]
    assert supabase.storage["admin_role_assignments"] == before


def test_role_upsert_strict_completion_audit_failure_removes_new_assignment(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    writes = iter(
        [
            SimpleNamespace(ok=True, warning=None),
            SimpleNamespace(ok=False, warning="boom"),
        ]
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_guarded_write_service.write_admin_activity_log",
        lambda *_args, **_kwargs: next(writes),
    )

    response = _patch_role(TestClient(app))

    assert response.status_code == 500
    assert "prior role assignment was restored" in response.json()["detail"]
    assert {row["email"] for row in supabase.storage["admin_role_assignments"]} == {
        "owner@example.com"
    }


def test_role_upsert_strict_completion_audit_failure_restores_existing_assignment(monkeypatch):
    supabase = FakeSupabase()
    supabase.storage["admin_role_assignments"].append(
        {
            "id": "role-score",
            "club_id": "club",
            "email": "score@example.com",
            "role": "scorekeeper",
            "user_id": "score-user",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    writes = iter(
        [
            SimpleNamespace(ok=True, warning=None),
            SimpleNamespace(ok=False, warning="boom"),
        ]
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_guarded_write_service.write_admin_activity_log",
        lambda *_args, **_kwargs: next(writes),
    )

    response = _patch_role(TestClient(app), role="organizer")

    assert response.status_code == 500
    restored = next(
        row
        for row in supabase.storage["admin_role_assignments"]
        if row["email"] == "score@example.com"
    )
    assert restored["role"] == "scorekeeper"
    assert restored["user_id"] == "score-user"


def test_role_revoke_strict_completion_audit_failure_restores_assignment(monkeypatch):
    supabase = FakeSupabase()
    supabase.storage["admin_role_assignments"].append(
        {
            "id": "role-organizer",
            "club_id": "club",
            "email": "organizer@example.com",
            "role": "organizer",
            "user_id": "organizer-user",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    writes = iter(
        [
            SimpleNamespace(ok=True, warning=None),
            SimpleNamespace(ok=False, warning="boom"),
        ]
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_guarded_write_service.write_admin_activity_log",
        lambda *_args, **_kwargs: next(writes),
    )

    response = _patch_role(
        TestClient(app),
        email="organizer@example.com",
        role="organizer",
        action="revoke",
    )

    assert response.status_code == 500
    restored = next(
        row
        for row in supabase.storage["admin_role_assignments"]
        if row["email"] == "organizer@example.com"
    )
    assert restored["role"] == "organizer"
    assert restored["user_id"] == "organizer-user"


def test_role_change_audit_failure_is_always_strict(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    monkeypatch.delenv("JUPR_REQUIRE_API_AUDIT_LOG", raising=False)
    monkeypatch.setattr(
        "jupr_app.services.admin_guarded_write_service.write_admin_activity_log",
        lambda *_args, **_kwargs: SimpleNamespace(ok=False, warning="audit unavailable"),
    )

    response = _patch_role(TestClient(app))

    assert response.status_code == 500
    assert "Required audit intent" in response.json()["detail"]
    assert "score@example.com" not in {
        row["email"] for row in supabase.storage["admin_role_assignments"]
    }


def test_role_change_strict_success_records_intent_and_completion(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")

    response = _patch_role(TestClient(app))

    assert response.status_code == 200
    logs = supabase.storage["admin_activity_log"]
    intent, completion = logs[-2:]
    assert intent["action_type"] == "role_assignment_upsert_intent"
    assert intent["before_json"] is None
    assert intent["after_json"]["workflow"] == "admin_role_assignment"
    assert intent["after_json"]["request_fingerprint"]
    assert intent["entity_id"] == "role-operation-test"
    assert completion["action_type"] == "role_assignment_upsert"
    assert completion["actor_email"] == "owner@example.com"
    assert completion["actor_role"] == "super_admin"
    assert completion["entity_id"] == "score@example.com"
    assert completion["source_page"] == "next_admin_tools_roles"
    assert completion["flagged_for_review"] is True
    assert completion["after_json"]["operation"] == "completion"
    assert completion["after_json"]["operation_id"] == response.json()["operation_id"]
    assert completion["after_json"]["assignment"] == {
        "email": "score@example.com",
        "role": "scorekeeper",
        "user_id": None,
    }
    assert response.json()["operation_key"] == "role-operation-test"


def test_role_change_compensation_preserves_newer_concurrent_assignment(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    call_count = 0

    def fail_completion_after_concurrent_change(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return SimpleNamespace(ok=True, warning=None)
        changed = next(
            row
            for row in supabase.storage["admin_role_assignments"]
            if row["email"] == "score@example.com"
        )
        changed["updated_at"] = "concurrent-same-value-write"
        return SimpleNamespace(ok=False, warning="boom")

    monkeypatch.setattr(
        "jupr_app.services.admin_guarded_write_service.write_admin_activity_log",
        fail_completion_after_concurrent_change,
    )

    response = _patch_role(TestClient(app))

    assert response.status_code == 409
    assert response.json()["detail"]["message"].startswith(
        "Critical: audit log write failed and the role assignment change could not be rolled back."
    )
    preserved = next(
        row
        for row in supabase.storage["admin_role_assignments"]
        if row["email"] == "score@example.com"
    )
    assert preserved["role"] == "scorekeeper"
    assert preserved["updated_at"] == "concurrent-same-value-write"


def test_role_upsert_unknown_commit_outcome_requires_manual_review(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")

    def write_then_fail(_supabase, *, club_id, action, before_record, after):
        _supabase.table("admin_role_assignments").insert(
            {
                "club_id": club_id,
                "email": after["email"],
                "role": after["role"],
                "user_id": after["user_id"],
            }
        ).execute()
        raise RuntimeError("write response lost")

    monkeypatch.setattr(
        "jupr_app.services.admin_tools_service._apply_role_assignment_change",
        write_then_fail,
    )

    response = _patch_role(TestClient(app))

    assert response.status_code == 409
    assert response.json()["detail"]["message"].startswith(
        "Critical: the role assignment write outcome is unknown."
    )
    assert "score@example.com" in {
        row["email"] for row in supabase.storage["admin_role_assignments"]
    }


def test_role_upsert_verification_failure_rolls_back_with_write_token(monkeypatch):
    from jupr_app.domain.admin.role_assignments import (
        list_role_assignments as list_role_assignments_from_db,
    )

    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    list_calls = 0

    def hide_persisted_row_once(_supabase, club_id):
        nonlocal list_calls
        list_calls += 1
        rows = list_role_assignments_from_db(_supabase, club_id)
        if list_calls == 2:
            return [row for row in rows if row["email"] != "score@example.com"]
        return rows

    monkeypatch.setattr(
        "jupr_app.services.admin_tools_service.list_role_assignments",
        hide_persisted_row_once,
    )

    response = _patch_role(TestClient(app))

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Role assignment verification failed; the prior assignment was restored."
    )
    assert {row["email"] for row in supabase.storage["admin_role_assignments"]} == {
        "owner@example.com"
    }


def test_role_revoke_postcheck_restores_final_super_admin_after_concurrent_revoke(monkeypatch):
    from jupr_app.services import admin_tools_service

    supabase = FakeSupabase()
    supabase.storage["admin_role_assignments"].append(
        {
            "id": "role-backup",
            "club_id": "club",
            "email": "backup@example.com",
            "role": "super_admin",
            "user_id": "backup-user",
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
        }
    )
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    original_apply = admin_tools_service._apply_role_assignment_change

    def apply_after_other_super_admin_disappears(*args, **kwargs):
        supabase.storage["admin_role_assignments"] = [
            row
            for row in supabase.storage["admin_role_assignments"]
            if row["email"] != "owner@example.com"
        ]
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(
        "jupr_app.services.admin_tools_service._apply_role_assignment_change",
        apply_after_other_super_admin_disappears,
    )

    response = _patch_role(
        TestClient(app),
        email="backup@example.com",
        role="super_admin",
        action="revoke",
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Unsafe concurrent change blocked: this would remove the final super_admin access."
    )
    remaining = supabase.storage["admin_role_assignments"]
    assert [(row["email"], row["role"]) for row in remaining] == [
        ("backup@example.com", "super_admin")
    ]


def test_role_strict_completion_audit_failure_reports_compensation_exception(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    writes = iter(
        [
            SimpleNamespace(ok=True, warning=None),
            SimpleNamespace(ok=False, warning="boom"),
        ]
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_guarded_write_service.write_admin_activity_log",
        lambda *_args, **_kwargs: next(writes),
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tools_service._compensate_role_assignment_change",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("rollback offline")),
    )

    response = _patch_role(TestClient(app))

    assert response.status_code == 409
    assert response.json()["detail"]["message"].startswith(
        "Critical: audit log write failed and the role assignment change could not be rolled back."
    )
