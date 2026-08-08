from __future__ import annotations

from types import SimpleNamespace

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.neq_filters = []
        self.limit_value = None
        self.insert_payload = None
        self.upsert_payload = None
        self.update_payload = None
        self.delete_flag = False

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def neq(self, key, value):
        self.neq_filters.append((key, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def upsert(self, payload, **_kwargs):
        self.upsert_payload = payload
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
        for key, expected in self.neq_filters:
            scoped = [row for row in scoped if str(row.get(key)) != str(expected)]
        if self.delete_flag:
            self.storage[self.table_name] = [row for row in rows if row not in scoped]
            return SimpleNamespace(data=scoped, count=len(scoped))
        if self.update_payload is not None:
            updated = []
            for row in rows:
                if row in scoped:
                    row.update(self.update_payload)
                    updated.append(dict(row))
            return SimpleNamespace(data=updated, count=len(updated))
        if self.upsert_payload is not None:
            payloads = self.upsert_payload if isinstance(self.upsert_payload, list) else [self.upsert_payload]
            out = []
            for payload in payloads:
                row = dict(payload)
                key_fields = ["tournament_id"] if self.table_name == "tournament_registration_settings" else ["id"]
                existing = next((item for item in rows if all(str(item.get(key)) == str(row.get(key)) for key in key_fields if row.get(key) is not None)), None)
                if existing:
                    existing.update(row)
                    out.append(dict(existing))
                else:
                    if row.get("id") is None:
                        row["id"] = f"row-{len(rows) + 1}"
                    rows.append(row)
                    out.append(dict(row))
            return SimpleNamespace(data=out, count=len(out))
        if self.insert_payload is not None:
            payloads = self.insert_payload if isinstance(self.insert_payload, list) else [self.insert_payload]
            out = []
            for payload in payloads:
                row = dict(payload)
                if row.get("id") is None:
                    row["id"] = f"row-{len(rows) + 1}"
                rows.append(row)
                out.append(dict(row))
            return SimpleNamespace(data=out, count=len(out))
        if self.limit_value is not None:
            scoped = scoped[: self.limit_value]
        return SimpleNamespace(data=scoped, count=len(scoped))


class FakeSupabase:
    def __init__(self):
        self.storage = {
            "tournaments": [
                {"id": "t1", "club_id": "club", "name": "Fall Classic", "status": "DRAFT", "start_date": "2026-10-01", "end_date": "2026-10-02", "created_at": "2026-01-01T00:00:00Z"}
            ],
            "tournament_registration_settings": [
                {"id": "regset1", "tournament_id": "t1", "registration_slug": "fall-classic", "locale": "en", "registration_status": "draft", "waitlist_enabled": True, "partner_board_enabled": True, "rules_markdown": "", "refund_policy_markdown": "", "sponsor_markdown": ""}
            ],
            "tournament_registration_days": [],
            "tournament_event_options": [],
            "tournament_registrations": [],
            "tournament_registration_selections": [],
            "tournament_event_draws": [],
            "tournament_teams": [],
            "tournament_games": [],
            "admin_activity_log": [],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


def install_env(monkeypatch, supabase):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr("services.api.admin_tournament_setup_routes.authenticate_bearer", lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"))
    monkeypatch.setattr("services.api.admin_tournament_setup_routes.resolve_admin_role", lambda **_kwargs: SimpleNamespace(role="club_owner"))


def test_tournament_setup_status_and_list(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    client = TestClient(app)

    status = client.get("/admin/clubs/club/tournaments/setup/status")
    assert status.status_code == 200
    assert status.json()["status"] == "ready_for_tournament_setup_manager"

    listing = client.get("/admin/clubs/club/tournaments/setup/tournaments", headers={"Authorization": "Bearer local"})
    assert listing.status_code == 200
    assert listing.json()["tournaments"][0]["id"] == "t1"


def test_tournament_setup_creates_a_draft_shell(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_ENV", "test")
    client = TestClient(app)
    tournament_id = "9a5660ae-4c71-4bd9-8c67-57c82b662d87"
    idempotency_key = "675278af-a271-43ac-8f8d-87b36b5886df"

    rejected = client.post(
        "/admin/clubs/club/tournaments/setup/tournaments",
        headers={"Authorization": "Bearer local"},
        json={
            "tournament_id": tournament_id,
            "idempotency_key": idempotency_key,
            "name": "Spring Open",
            "confirmation_text": "CREATE",
        },
    )
    assert rejected.status_code == 400
    assert "CREATE TOURNAMENT" in rejected.json()["detail"]

    created = client.post(
        "/admin/clubs/club/tournaments/setup/tournaments",
        headers={"Authorization": "Bearer local"},
        json={
            "tournament_id": tournament_id,
            "idempotency_key": idempotency_key,
            "name": "Spring Open",
            "start_date": "2027-03-12",
            "end_date": "2027-03-14",
            "confirmation_text": "CREATE TOURNAMENT",
        },
    )

    assert created.status_code == 200
    assert created.json()["tournament"] == {
        "id": tournament_id,
        "club_id": "club",
        "name": "Spring Open",
        "status": "DRAFT",
        "start_date": "2027-03-12",
        "end_date": "2027-03-14",
        "event_tags": {
            "skill_levels": [],
            "date_tags": ["March 2027", "Spring 2027"],
        },
    }
    assert len(supabase.storage["tournaments"]) == 2
    assert supabase.storage["admin_activity_log"][-1]["action_type"] == (
        "tournament_setup_shell_create"
    )


def test_staging_shell_create_retries_by_idempotency_key_without_duplicate(
    monkeypatch,
):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-setup")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_WRITE_PILOT", "1")
    monkeypatch.setenv(
        "JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS",
        "1",
    )
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    client = TestClient(app)
    body = {
        "tournament_id": "b1871068-f73e-4a63-8298-856249457c29",
        "idempotency_key": "f5a71ee0-9931-4dfc-8d9d-c7a0c779b550",
        "name": "Protected Retry Open",
        "start_date": "2027-04-03",
        "end_date": "2027-04-04",
        "confirmation_text": "CREATE TOURNAMENT",
    }

    first = client.post(
        "/admin/clubs/club/tournaments/setup/tournaments",
        headers={"Authorization": "Bearer local"},
        json=body,
    )
    retry = client.post(
        "/admin/clubs/club/tournaments/setup/tournaments",
        headers={"Authorization": "Bearer local"},
        json=body,
    )

    assert first.status_code == 200
    assert retry.status_code == 200
    assert first.json()["idempotent_replay"] is False
    assert retry.json()["idempotent_replay"] is True
    assert retry.json()["operation_key"] == first.json()["operation_key"]
    assert [
        row["id"]
        for row in supabase.storage["tournaments"]
        if row["id"] == body["tournament_id"]
    ] == [body["tournament_id"]]
    assert supabase.storage["tournament_admin_operations"][0]["status"] == (
        "completed"
    )


def test_tournament_setup_settings_confirmation(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    client = TestClient(app)

    rejected = client.patch(
        "/admin/clubs/club/tournaments/setup/tournaments/t1/settings",
        headers={"Authorization": "Bearer local"},
        json={"registration_status": "open", "confirmation_text": "SAVE"},
    )
    assert rejected.status_code == 400
    assert "SAVE SETUP" in rejected.json()["detail"]

    saved = client.patch(
        "/admin/clubs/club/tournaments/setup/tournaments/t1/settings",
        headers={"Authorization": "Bearer local"},
        json={"registration_status": "open", "partner_board_enabled": False, "confirmation_text": "SAVE SETUP"},
    )
    assert saved.status_code == 200
    assert saved.json()["settings"]["registration_status"] == "open"
    assert supabase.storage["admin_activity_log"]


def test_tournament_setup_draft_and_publish(monkeypatch):
    supabase = FakeSupabase()
    install_env(monkeypatch, supabase)
    client = TestClient(app)
    days = [{"id": "day1", "tournament_id": "t1", "label": "Day 1", "date": "2026-10-01", "enabled": True, "sort_order": 1}]
    events = [{"id": "event1", "tournament_id": "t1", "registration_day_id": "day1", "event_family_label": "Mixed Doubles", "division_name": "Mixed Open", "event_type": "MIXED_DOUBLES", "gender_restriction": "MIXED", "skill_label": "Open", "skill_mode": "OPEN", "event_format_default": "ROUND_ROBIN_PLUS_PLAYOFF", "scoring_default": "GAME_TO_15", "status": "open", "enabled": True, "sort_order": 1}]

    draft = client.put(
        "/admin/clubs/club/tournaments/setup/tournaments/t1/draft",
        headers={"Authorization": "Bearer local"},
        json={"days": days, "event_options": events, "confirmation_text": "SAVE SETUP DRAFT"},
    )
    assert draft.status_code == 200
    assert draft.json()["builder_draft"]["days"][0]["id"] == "day1"

    published = client.post(
        "/admin/clubs/club/tournaments/setup/tournaments/t1/publish",
        headers={"Authorization": "Bearer local"},
        json={"days": days, "event_options": events, "confirmation_text": "PUBLISH SETUP"},
    )
    assert published.status_code == 200
    assert published.json()["publish_result"]["mode"] == "replace"
    assert supabase.storage["tournament_registration_days"]
    assert supabase.storage["tournament_event_options"]
