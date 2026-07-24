from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tests.conftest import require_api_dependency

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from jupr_app.services.admin_verified_updates_service import update_admin_verified_update_request
from jupr_app.services.public_verified_updates_service import (
    PUBLIC_VERIFIED_UPDATE_REQUESTS_PER_EMAIL_PER_HOUR,
    create_public_verified_update_request,
    get_public_verified_update_request_status,
    list_public_verified_update_player_options,
)
from services.api.main import app


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.in_filters = []
        self.gte_filters = []
        self.limit_value = None
        self.update_payload = None
        self.insert_payload = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def in_(self, key, values):
        self.in_filters.append((key, {str(v) for v in values}))
        return self

    def gte(self, key, value):
        self.gte_filters.append((key, value))
        return self

    def order(self, *_args, **_kwargs):
        return self

    def range(self, *_args, **_kwargs):
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = dict(payload)
        return self

    def update(self, payload):
        self.update_payload = dict(payload)
        return self

    def execute(self):
        rows = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            row = {"id": f"row-{len(rows) + 1}", **self.insert_payload}
            rows.append(row)
            return SimpleNamespace(data=[row])
        scoped = list(rows)
        for key, expected in self.filters:
            scoped = [row for row in scoped if str(row.get(key)) == str(expected)]
        for key, allowed in self.in_filters:
            scoped = [row for row in scoped if str(row.get(key)) in allowed]
        for key, minimum in self.gte_filters:
            scoped = [row for row in scoped if str(row.get(key) or "") >= str(minimum)]
        if self.update_payload is not None:
            updated = []
            for row in rows:
                if row in scoped:
                    row.update(self.update_payload)
                    updated.append(dict(row))
            return SimpleNamespace(data=updated)
        if self.limit_value is not None:
            scoped = scoped[: self.limit_value]
        return SimpleNamespace(data=scoped)


class FakeSupabase:
    def __init__(self):
        self.storage = {
            "players": [{"club_id": "club", "id": 1, "name": "Alex", "active": True}],
            "player_profile_update_subscriptions": [],
            "admin_activity_log": [],
        }

    def table(self, name):
        return FakeQuery(self.storage, name)


def _api_client(monkeypatch, supabase: FakeSupabase) -> TestClient:
    supabase.storage["clubs"] = [{"id": "club", "slug": "sandwich-club", "name": "Sandwich Club"}]
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    return TestClient(app)


def _admin_api_client(monkeypatch, supabase: FakeSupabase) -> TestClient:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "local")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "local-service")
    monkeypatch.setattr(
        "services.api.main.create_client",
        lambda _url, _credential: supabase,
    )
    monkeypatch.setattr(
        "services.api.admin_verified_updates_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(
            email="admin@example.com",
            user_id="user-1",
        ),
    )
    monkeypatch.setattr(
        "services.api.admin_verified_updates_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )
    return TestClient(app)


def test_public_verified_updates_options_and_request(monkeypatch):
    supabase = FakeSupabase()
    options = list_public_verified_update_player_options(supabase, club_id="club")
    assert options["count"] == 1
    created = create_public_verified_update_request(supabase, club_id="club", player_id=1, email="user@example.com")
    assert created["ok"] is True
    assert created["request_status"] == "pending_admin_review"

    options_after = list_public_verified_update_player_options(supabase, club_id="club")
    assert options_after["players"][0]["already_requested"] is True
    assert options_after["players"][0]["request_status"] == "pending_admin_review"
    status = get_public_verified_update_request_status(supabase, club_id="club", player_id=1)
    assert status["player"]["already_requested"] is True


def test_public_verified_updates_duplicate_same_email_is_idempotent():
    supabase = FakeSupabase()

    first = create_public_verified_update_request(supabase, club_id="club", player_id=1, email="User@Example.com")
    second = create_public_verified_update_request(supabase, club_id="club", player_id=1, email="user@example.com")

    assert first["deduplicated"] is False
    assert second["deduplicated"] is True
    assert second["subscription_id"] == first["subscription_id"]
    assert len(supabase.storage["player_profile_update_subscriptions"]) == 1


@pytest.mark.parametrize(
    "email",
    ["missing-at.example.com", "a@localhost", "a b@example.com", ".dot@example.com", "double..dot@example.com"],
)
def test_public_verified_updates_rejects_invalid_email(email):
    with pytest.raises(ValueError, match="valid email"):
        create_public_verified_update_request(FakeSupabase(), club_id="club", player_id=1, email=email)


def test_public_verified_updates_caps_recent_requests_per_email():
    supabase = FakeSupabase()
    now = datetime.now(timezone.utc).isoformat()
    supabase.storage["player_profile_update_subscriptions"] = [
        {
            "id": f"existing-{index}",
            "club_id": "club",
            "player_id": 100 + index,
            "email": "user@example.com",
            "email_normalized": "user@example.com",
            "request_status": "rejected",
            "created_at": now,
        }
        for index in range(PUBLIC_VERIFIED_UPDATE_REQUESTS_PER_EMAIL_PER_HOUR)
    ]

    with pytest.raises(ValueError, match="Too many recent"):
        create_public_verified_update_request(supabase, club_id="club", player_id=1, email="user@example.com")


def test_public_verified_updates_api_is_club_scoped_and_validates_email(monkeypatch):
    supabase = FakeSupabase()
    supabase.storage["players"].append({"club_id": "other-club", "id": 2, "name": "Other Club Player", "active": True})
    client = _api_client(monkeypatch, supabase)

    options = client.get("/clubs/sandwich-club/verified-updates/options")
    invalid = client.post(
        "/clubs/sandwich-club/verified-updates/request",
        json={"player_id": 1, "email": "invalid"},
    )

    assert options.status_code == 200
    assert options.json()["club"] == {"id": "club", "slug": "sandwich-club", "name": "Sandwich Club"}
    assert [player["id"] for player in options.json()["players"]] == [1]
    assert invalid.status_code == 400
    assert "valid email" in invalid.json()["detail"]


def test_public_verified_updates_api_returns_429_for_durable_email_rate_cap(monkeypatch):
    supabase = FakeSupabase()
    now = datetime.now(timezone.utc).isoformat()
    supabase.storage["player_profile_update_subscriptions"] = [
        {
            "id": f"existing-{index}",
            "club_id": "club",
            "player_id": 100 + index,
            "email": "user@example.com",
            "email_normalized": "user@example.com",
            "request_status": "rejected",
            "created_at": now,
        }
        for index in range(PUBLIC_VERIFIED_UPDATE_REQUESTS_PER_EMAIL_PER_HOUR)
    ]
    client = _api_client(monkeypatch, supabase)

    response = client.post(
        "/clubs/sandwich-club/verified-updates/request",
        json={"player_id": 1, "email": "user@example.com"},
    )

    assert response.status_code == 429
    assert "Too many recent" in response.json()["detail"]


def test_admin_verified_updates_requires_confirmation(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    supabase = FakeSupabase()
    created = create_public_verified_update_request(supabase, club_id="club", player_id=1, email="user@example.com")
    try:
        update_admin_verified_update_request(
            supabase,
            club_id="club",
            subscription_id=created["subscription_id"],
            action="approve",
            admin_note="",
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="SAVE",
        )
    except ValueError as exc:
        assert "SAVE VERIFIED REQUEST" in str(exc)
    else:
        raise AssertionError("expected confirmation error")


def test_staging_verified_updates_reads_stay_open_while_mutations_are_double_guarded(
    monkeypatch,
) -> None:
    supabase = FakeSupabase()
    created = create_public_verified_update_request(
        supabase,
        club_id="club",
        player_id=1,
        email="user@example.com",
    )
    client = _admin_api_client(monkeypatch, supabase)
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS", "0")

    status = client.get("/admin/clubs/club/verified-updates/status")
    requests = client.get(
        "/admin/clubs/club/verified-updates/requests",
        headers={"Authorization": "Bearer local"},
    )
    denied_by_wave = client.patch(
        f"/admin/clubs/club/verified-updates/requests/{created['subscription_id']}",
        headers={"Authorization": "Bearer local"},
        json={
            "action": "approve",
            "confirmation_text": "SAVE",
        },
    )

    assert status.status_code == 200
    assert status.json()["enabled"] is True
    assert status.json()["mutations_enabled"] is False
    assert requests.status_code == 200
    assert requests.json()["count"] == 1
    assert denied_by_wave.status_code == 403

    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "communications")
    denied_by_service = client.patch(
        f"/admin/clubs/club/verified-updates/requests/{created['subscription_id']}",
        headers={"Authorization": "Bearer local"},
        json={
            "action": "approve",
            "confirmation_text": "SAVE",
        },
    )
    assert denied_by_service.status_code == 403
    assert "Communications mutations are disabled" in denied_by_service.json()["detail"]

    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS", "1")
    confirmation_required = client.patch(
        f"/admin/clubs/club/verified-updates/requests/{created['subscription_id']}",
        headers={"Authorization": "Bearer local"},
        json={
            "action": "approve",
            "confirmation_text": "SAVE",
        },
    )
    assert confirmation_required.status_code == 400
    assert "SAVE VERIFIED REQUEST" in confirmation_required.json()["detail"]


def test_admin_verified_updates_approve_writes_audit(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES", "1")
    supabase = FakeSupabase()
    created = create_public_verified_update_request(supabase, club_id="club", player_id=1, email="user@example.com")
    updated = update_admin_verified_update_request(
        supabase,
        club_id="club",
        subscription_id=created["subscription_id"],
        action="approve",
        admin_note="approved",
        actor_email="admin@example.com",
        actor_role="club_owner",
        confirmation_text="SAVE VERIFIED REQUEST",
    )
    assert updated["request"]["request_status"] == "active"
    assert supabase.storage["admin_activity_log"]
