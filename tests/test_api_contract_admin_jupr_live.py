from types import SimpleNamespace

from jupr_app.services.admin_jupr_live_service import (
    create_admin_jupr_live_session,
    list_admin_jupr_live_sessions,
    update_admin_jupr_live_session_status,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters = []
        self.insert_payload = None
        self.update_payload = None
        self.limit_value = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def lt(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
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
        self.storage = {"live_sessions": [], "admin_activity_log": []}

    def table(self, name):
        return FakeQuery(self.storage, name)


def test_jupr_live_create_requires_confirmation(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE", "1")
    try:
        create_admin_jupr_live_session(
            FakeSupabase(),
            club_id="club",
            title="Night",
            event_type="round_robin",
            participant_names=[],
            actor_email="admin@example.com",
            actor_role="scorekeeper",
            confirmation_text="CREATE",
        )
    except ValueError as exc:
        assert "CREATE LIVE SESSION" in str(exc)
    else:
        raise AssertionError("expected confirmation error")


def test_jupr_live_create_list_and_status_update(monkeypatch):
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_JUPR_LIVE", "1")
    supabase = FakeSupabase()
    created = create_admin_jupr_live_session(
        supabase,
        club_id="club",
        title="Night",
        event_type="round_robin",
        participant_names=["Alex", "Blair"],
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        confirmation_text="CREATE LIVE SESSION",
    )
    assert created["ok"] is True
    assert created["session"]["status"] == "active"
    listed = list_admin_jupr_live_sessions(supabase, club_id="club", status="active")
    assert listed["count"] == 1
    updated = update_admin_jupr_live_session_status(
        supabase,
        club_id="club",
        session_key=created["session"]["session_key"],
        status="completed",
        title=None,
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        confirmation_text="SAVE LIVE SESSION",
    )
    assert updated["session"]["status"] == "completed"
    assert len(supabase.storage["admin_activity_log"]) >= 2
