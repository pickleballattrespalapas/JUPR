from types import SimpleNamespace

from jupr_app.services.admin_verified_updates_service import update_admin_verified_update_request
from jupr_app.services.public_verified_updates_service import create_public_verified_update_request, list_public_verified_update_player_options


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.in_filters = []
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


def test_public_verified_updates_options_and_request(monkeypatch):
    supabase = FakeSupabase()
    options = list_public_verified_update_player_options(supabase, club_id="club")
    assert options["count"] == 1
    created = create_public_verified_update_request(supabase, club_id="club", player_id=1, email="user@example.com")
    assert created["ok"] is True
    assert created["request_status"] == "pending_admin_review"


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
