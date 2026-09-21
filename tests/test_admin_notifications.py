from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import pytest

from jupr_app.services import admin_notifications_service as service
from services.api.admin_notifications_routes import install_admin_notifications_routes

NOW = datetime(2026, 9, 21, 18, tzinfo=timezone.utc)
USER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
OTHER_USER = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"


class Query:
    def __init__(self, db, table):
        self.db, self.name, self.filters = db, table, {}
        self.bounds, self.sort, self.payload = None, None, None

    def select(self, _columns):
        return self

    def eq(self, field, value):
        self.filters[field] = value
        return self

    def order(self, field):
        self.sort = field
        return self

    def range(self, start, end):
        self.bounds = (start, end)
        return self

    def upsert(self, payload, *, on_conflict):
        self.payload = copy.deepcopy(payload if isinstance(payload, list) else [payload])
        self.conflict = on_conflict.split(",")
        return self

    def execute(self):
        if self.name in self.db.fail:
            raise RuntimeError("private database detail")
        if self.payload is not None:
            self.db.writes.append((self.name, copy.deepcopy(self.payload)))
            assert self.name in {service.PREFERENCES_TABLE, service.STATES_TABLE}
            for new in self.payload:
                rows = self.db.rows.setdefault(self.name, [])
                current = next((row for row in rows if all(row[k] == new[k] for k in self.conflict)), None)
                if current is None:
                    rows.append(new)
                else:
                    current.update(new)
            return SimpleNamespace(data=copy.deepcopy(self.payload))
        if self.name != "admin_role_assignments":
            assert self.filters.get("club_id") and self.filters.get("user_id")
        rows = [row for row in self.db.rows.get(self.name, []) if all(row.get(k) == v for k, v in self.filters.items())]
        if self.sort:
            rows.sort(key=lambda row: row[self.sort])
        if self.bounds:
            rows = rows[self.bounds[0]:self.bounds[1] + 1]
        return SimpleNamespace(data=copy.deepcopy(rows))


class Database:
    def __init__(self):
        self.rows, self.writes, self.fail = {}, [], set()

    def table(self, name):
        return Query(self, name)


def item(source_id="first", *, category="actions", version="1", age=0):
    return {"category": category, "kind": "action" if category == "actions" else "activity",
        "source_id": source_id, "source_version": version, "title": "Review this result" if category == "actions" else "A player signed up",
        "description": "Open the club workspace to review the record.", "href": "/admin/play-generators/submissions",
        "occurred_at": (NOW - timedelta(days=age)).isoformat()}


class Sources:
    def __init__(self):
        self.items = {"club-a": [item(), item("signup", category="activity")]}
        self.fail, self.resolve_calls = set(), []

    def collect(self, db, *, club_id, assignments, now, history_days, item_limit):
        assert history_days == 30 and item_limit == 200
        authorized = any(row["club_id"] == club_id and row["role"] in {"administrator", "club_owner", "super_admin"} for row in assignments)
        if not authorized:
            return {"categories": [], "items": [], "sources": [], "generated_at": now.isoformat()}
        categories, items, reports = [], [], []
        for key, kind in (("actions", "action"), ("activity", "activity")):
            categories.append({"key": key, "kind": kind, "label": key, "description": "Review club records.", "href": "/admin"})
            records = [row for row in self.items.get(club_id, []) if row["category"] == key and
                (kind == "action" or service._instant(row["occurred_at"]) >= now - timedelta(days=history_days))]
            unavailable = key in self.fail
            reports.append({"key": key, "status": "unavailable" if unavailable else "ready",
                "total_count": None if unavailable else len(records), "truncated": len(records) > item_limit})
            if not unavailable:
                items.extend(records[:item_limit])
        return {"categories": categories, "items": copy.deepcopy(items), "sources": reports, "generated_at": now.isoformat()}

    def resolve(self, db, *, club_id, assignments, category, source_id, source_version, now):
        self.resolve_calls.append((club_id, category, source_id, source_version))
        if category in self.fail:
            raise RuntimeError("source read failed")
        return next((copy.deepcopy(row) for row in self.items.get(club_id, []) if row["category"] == category
            and row["source_id"] == source_id and row["source_version"] == source_version), None)


@pytest.fixture
def fixture(monkeypatch):
    db, sources = Database(), Sources()
    monkeypatch.setattr(service, "collect_admin_notification_sources", sources.collect)
    monkeypatch.setattr(service, "resolve_admin_notification_item", sources.resolve)
    monkeypatch.setenv("JUPR_ENV", "test")
    return db, sources


def args(*, club="club-a", user=USER, role="administrator"):
    return {"club_id": club, "user_id": user, "assignments": [{"club_id": club, "role": role}], "now": NOW}


def feed(fixture, **kwargs):
    return service.get_admin_notifications(fixture[0], **args(**kwargs))


def change(fixture, record, state, **kwargs):
    return service.update_admin_notification_state(fixture[0], **args(**kwargs), key=service.notification_key(record), state=state)


def test_clear_is_personal_and_never_resolves_or_hides_the_club_action(fixture):
    db, sources = fixture
    original = copy.deepcopy(sources.items)
    result = change(fixture, item(), "cleared")
    assert next(row for row in result["items"] if row["category"] == "actions")["state"] == "cleared"
    assert result["categories"][0]["total_count"] == 1
    assert sources.items == original
    assert feed(fixture, user=OTHER_USER)["items"][0]["state"] == "new"
    sources.items["club-b"] = [item()]
    assert feed(fixture, club="club-b")["items"][0]["state"] == "new"
    assert all(name == service.STATES_TABLE for name, _ in db.writes)
    assert db.writes[0][1][0]["club_id"] == "club-a"
    assert db.writes[0][1][0]["user_id"] == USER


def test_new_submission_with_the_same_count_is_new_and_old_clear_does_not_apply(fixture):
    change(fixture, item(), "cleared")
    fixture[1].items["club-a"] = [item(version="2")]
    result = feed(fixture)
    assert len(result["items"]) == 1 and result["items"][0]["state"] == "new"
    assert result["items"][0]["key"] != service.notification_key(item())


def test_flag_disable_reenable_and_restore_preserve_personal_intent(fixture):
    db, _ = fixture
    change(fixture, item(), "flagged")
    result = service.update_admin_notification_preferences(db, **args(), categories={"actions": False})
    assert not any(row["category"] == "actions" for row in result["items"])
    assert next(c for c in result["categories"] if c["key"] == "actions")["total_count"] == 1
    with pytest.raises(service.NotificationConflict):
        change(fixture, item(), "cleared")
    result = service.update_admin_notification_preferences(db, **args(), categories={"actions": True})
    assert next(row for row in result["items"] if row["category"] == "actions")["state"] == "flagged"
    result = change(fixture, item(), "new")
    assert next(row for row in result["items"] if row["category"] == "actions")["state"] == "new"


def test_flagged_activity_survives_30_day_window_and_then_can_be_cleared(fixture):
    signup = item("signup", category="activity")
    change(fixture, signup, "flagged")
    later = {**args(), "now": NOW + timedelta(days=60)}
    result = service.get_admin_notifications(fixture[0], **later)
    assert next(row for row in result["items"] if row["category"] == "activity")["state"] == "flagged"
    result = service.update_admin_notification_state(fixture[0], **later, key=service.notification_key(signup), state="cleared")
    assert not any(row["category"] == "activity" for row in result["items"])


@pytest.mark.parametrize("state", ["flagged", "cleared"])
def test_resolved_or_deleted_action_never_survives_as_a_saved_pending_task(fixture, state):
    change(fixture, item(), state)
    fixture[1].items["club-a"] = []
    assert feed(fixture)["items"] == []
    with pytest.raises(service.NotificationConflict):
        change(fixture, item(), "cleared")


def test_saved_action_outside_bounded_source_page_is_individually_revalidated(fixture):
    db, sources = fixture
    change(fixture, item(), "flagged")
    sources.items["club-a"] = [item(str(n)) for n in range(201)] + [item()]
    result = feed(fixture)
    assert result["truncated"] is True
    assert any(row["key"] == service.notification_key(item()) and row["state"] == "flagged" for row in result["items"])
    assert ("club-a", "actions", "first", "1") in sources.resolve_calls


def test_disabled_busy_category_does_not_truncate_the_enabled_inbox(fixture):
    db, sources = fixture
    sources.items["club-a"] = [item(str(n)) for n in range(201)]
    assert feed(fixture)["truncated"] is True
    result = service.update_admin_notification_preferences(db, **args(), categories={"actions": False})
    assert result["truncated"] is False and result["items"] == []
    assert next(row for row in result["categories"] if row["key"] == "actions")["total_count"] == 201
    assert feed(fixture)["truncated"] is False
    result = service.update_admin_notification_preferences(db, **args(), categories={"actions": True})
    assert result["truncated"] is True


@pytest.mark.parametrize("href", ["/administrator", "//example.invalid/admin", "/admin\\outside", "https://example.invalid/admin"])
def test_notification_snapshot_rejects_non_admin_destinations(href):
    with pytest.raises(ValueError, match="destination"):
        service._safe_item({**item(), "href": href})


def test_source_failure_is_unavailable_and_cannot_clear_unverified_work(fixture):
    change(fixture, item(), "flagged")
    fixture[1].fail.add("actions")
    result = feed(fixture)
    category = next(row for row in result["categories"] if row["key"] == "actions")
    assert category["status"] == "unavailable" and category["total_count"] is None
    writes = len(fixture[0].writes)
    with pytest.raises(service.NotificationUnavailable):
        change(fixture, item(), "cleared")
    assert len(fixture[0].writes) == writes
    assert fixture[0].rows[service.STATES_TABLE][0]["state"] == "flagged"


def test_stale_clear_racing_a_source_change_is_rejected(fixture, monkeypatch):
    monkeypatch.setattr(service, "resolve_admin_notification_item", lambda *a, **k: None)
    with pytest.raises(service.NotificationConflict):
        change(fixture, item(), "cleared")
    assert not fixture[0].writes


@pytest.mark.parametrize("role", ["operator", "read_only", "scorekeeper"])
def test_unauthorized_categories_cannot_be_enabled_and_saved_flags_are_hidden(fixture, role):
    change(fixture, item(), "flagged")
    assert feed(fixture, role=role)["items"] == []
    with pytest.raises(PermissionError):
        service.update_admin_notification_preferences(fixture[0], **args(role=role), categories={"actions": True})
    with pytest.raises(service.NotificationConflict):
        change(fixture, item(), "cleared", role=role)


def test_unknown_categories_and_spoofed_item_hashes_do_not_write(fixture):
    with pytest.raises(PermissionError):
        service.update_admin_notification_preferences(fixture[0], **args(), categories={"unknown": True})
    with pytest.raises(service.NotificationConflict):
        service.update_admin_notification_state(fixture[0], **args(), key="a" * 64, state="cleared")
    assert not fixture[0].writes


@pytest.mark.parametrize("table", [service.PREFERENCES_TABLE, service.STATES_TABLE])
def test_personal_storage_failure_does_not_reset_preferences_or_show_a_false_empty_feed(fixture, table):
    fixture[0].fail.add(table)
    with pytest.raises(service.NotificationUnavailable, match="saved notification settings"):
        feed(fixture)


def test_all_saved_flags_are_read_across_multiple_storage_pages(fixture):
    db, sources = fixture
    sources.items["club-a"] = []
    for n in range(501):
        snapshot = service._safe_item(item(str(n), category="activity", age=60))
        db.rows.setdefault(service.STATES_TABLE, []).append({"club_id": "club-a", "user_id": USER,
            "notification_key": snapshot["key"], "category_key": "activity", "kind": "activity", "state": "flagged", "snapshot": snapshot})
    result = feed(fixture)
    assert len(result["items"]) == 501
    assert all(row["state"] == "flagged" for row in result["items"])


def route_client(fixture, monkeypatch, *, main=False, role="administrator"):
    db, _ = fixture
    db.rows["admin_role_assignments"] = [{"club_id": "club-a", "role": role, "user_id": USER, "email": "admin@example.invalid"}]
    def authenticate(header):
        if header != "Bearer fixture-token":
            raise HTTPException(401, "invalid bearer token")
        return SimpleNamespace(user_id=USER, email="admin@example.invalid")
    monkeypatch.setattr("services.api.admin_auth_routes.authenticate_bearer", authenticate)
    if main:
        from services.api.main import app
        monkeypatch.setenv("SUPABASE_URL", "https://sijpxjxvdtrehmqvirfi.supabase.co")
        monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-key")
        monkeypatch.setattr("services.api.main.create_client", lambda *a, **k: db)
    else:
        app = FastAPI()
        install_admin_notifications_routes(app, get_supabase_client=lambda: db)
    return TestClient(app)


def test_http_contract_requires_session_and_club_binding_and_rejects_identity_spoofing(fixture, monkeypatch):
    client = route_client(fixture, monkeypatch)
    base = "/admin/clubs/club-a/notifications"
    headers = {"Authorization": "Bearer fixture-token"}
    assert client.get(base).status_code == 401
    assert client.get("/admin/clubs/club-b/notifications", headers=headers).status_code == 403
    response = client.get(base, headers=headers)
    assert response.status_code == 200 and response.headers["cache-control"] == "private, no-store"
    assert client.put(base + "/preferences", headers=headers, json={"categories": {"actions": False}, "user_id": OTHER_USER}).status_code == 422
    assert client.put(base + "/preferences", headers=headers, json={"categories": {"actions": "false"}}).status_code == 422
    assert client.put(base + "/items/not-a-key", headers=headers, json={"state": "cleared"}).status_code == 422
    assert client.put(base + "/items/" + service.notification_key(item()), headers=headers, json={"state": "delete"}).status_code == 422
    assert not fixture[0].writes


def test_staging_open_allows_personal_puts_and_emergency_stop_blocks_them(fixture, monkeypatch):
    client = route_client(fixture, monkeypatch, main=True)
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "open")
    headers = {"Authorization": "Bearer fixture-token"}
    base = "/admin/clubs/club-a/notifications"
    response = client.put(base + "/preferences", headers=headers, json={"categories": {"actions": True}})
    assert response.status_code == 200, response.text
    response = client.put(base + "/items/" + service.notification_key(item()), headers=headers, json={"state": "flagged"})
    assert response.status_code == 200, response.text
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "none")
    before = len(fixture[0].writes)
    assert client.put(base + "/preferences", headers=headers, json={"categories": {"actions": False}}).status_code == 403
    assert client.put(base + "/items/" + service.notification_key(item()), headers=headers, json={"state": "cleared"}).status_code == 403
    assert len(fixture[0].writes) == before


def test_migration_limits_public_access_and_safe_cancellation_projection():
    migration = Path("supabase/migrations/20260921184507_admin_personal_notifications.sql").read_text().lower()
    assert "security definer" not in migration
    assert "alter table public.admin_notification_preferences enable row level security" in migration
    assert "alter table public.admin_notification_states enable row level security" in migration
    assert "from public,anon,authenticated" in migration
    assert "primary key(club_id,user_id,notification_key)" in migration
    assert "primary key(club_id,user_id,category_key)" in migration
    function = migration.split("create function public.pcs_admin_tournament_notification_source", 1)[1]
    assert "where t.club_id=p_club_id" in function
    assert "c.before_json#>>'{registration,display_name}'" in function
    assert "actor_email" not in function and "phone" not in function
    assert "count(*) over()" in function
