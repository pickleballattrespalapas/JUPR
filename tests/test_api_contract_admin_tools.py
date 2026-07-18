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
            row = {"id": f"row-{len(rows) + 1}", **self.insert_payload}
            rows.append(row)
            return SimpleNamespace(data=[row])
        if self.upsert_payload is not None:
            email = self.upsert_payload.get("email")
            club_id = self.upsert_payload.get("club_id")
            existing = next((row for row in rows if row.get("email") == email and row.get("club_id") == club_id), None)
            if existing:
                existing.update(self.upsert_payload)
                row = existing
            else:
                row = {"created_at": "2026-01-01T00:00:00Z", **self.upsert_payload}
                rows.append(row)
            return SimpleNamespace(data=[row])
        if self.update_payload is not None:
            updated = []
            for row in rows:
                if row in scoped:
                    row.update(self.update_payload)
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
                {"club_id": "club", "email": "owner@example.com", "role": "super_admin", "user_id": "user-1", "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"}
            ],
            "admin_activity_log": [],
            "matches": [{"club_id": "club", "id": 1, "t1_p1_r": 1200, "t1_p2_r": 1200, "t2_p1_r": 1200, "t2_p2_r": 1200, "t1_p1_r_end": 1210, "t1_p2_r_end": 1210, "t2_p1_r_end": 1190, "t2_p2_r_end": 1190}],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


def install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOOLS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr("services.api.admin_tools_routes.authenticate_bearer", lambda _authorization: SimpleNamespace(email="owner@example.com", user_id="user-1"))
    monkeypatch.setattr("services.api.admin_tools_routes.resolve_admin_role", lambda **_kwargs: SimpleNamespace(role="super_admin"))


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
    monkeypatch.setattr("services.api.admin_tools_routes.resolve_admin_role", lambda **_kwargs: SimpleNamespace(role="read_only"))
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
        json={"email": "score@example.com", "role": "scorekeeper", "action": "upsert", "confirmation_text": "SAVE ROLE"},
    )
    assert saved.status_code == 200
    emails = {row["email"] for row in saved.json()["roles"]}
    assert "score@example.com" in emails
    assert supabase.storage["admin_activity_log"]
