from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.admin_replay_service import build_admin_replay_status, run_admin_replay_history


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters: list[tuple[str, object]] = []
        self.insert_payload = None

    def select(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def execute(self):
        table = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            row = dict(self.insert_payload)
            table.append(row)
            return SimpleNamespace(data=[row])
        rows = list(table)
        for key, expected in self.filters:
            rows = [row for row in rows if str(row.get(key)) == str(expected)]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, storage):
        self.storage = storage

    def table(self, name):
        return FakeQuery(self.storage, name)


def fake_storage():
    return {
        "leagues_metadata": [
            {"club_id": "club", "league_name": "Open", "k_factor": 32, "is_active": True},
            {"club_id": "club", "league_name": "Advanced", "k_factor": 24, "is_active": False},
        ],
        "admin_activity_log": [],
    }


def test_replay_status_disabled_is_db_free(monkeypatch) -> None:
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", raising=False)

    payload = build_admin_replay_status(None, club_id="club")

    assert payload["enabled"] is False
    assert payload["status"] == "streamlit_fallback"
    assert payload["options"] == ["ALL (Full System Reset)"]
    assert payload["apply_endpoint"] is None


def test_replay_status_enabled_lists_options(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")

    payload = build_admin_replay_status(FakeSupabase(fake_storage()), club_id="club")

    assert payload["enabled"] is True
    assert payload["status"] == "replay_enabled"
    assert payload["options"] == ["ALL (Full System Reset)", "Advanced", "Open"]


def test_run_replay_requires_confirmation(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")

    try:
        run_admin_replay_history(
            FakeSupabase(fake_storage()),
            club_id="club",
            target_reset="Open",
            actor_email="admin@example.com",
            actor_role="super_admin",
            confirmation_text="nope",
        )
    except ValueError as exc:
        assert "REPLAY" in str(exc)
    else:
        raise AssertionError("Expected confirmation failure")


def test_run_replay_calls_domain_and_audits(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_REPLAY", "1")
    storage = fake_storage()
    calls = []

    def fake_tracked_replay(**kwargs):
        calls.append(kwargs)
        return {
            "job_id": "job-1",
            "job_status": "succeeded",
            "idempotent_replay": False,
            "result": {
                "target_reset": kwargs["target_reset"],
                "players_updated": False,
                "skipped_incomplete": 0,
                "matches_rewritten": 3,
                "matches_snapshots_updated_rows": 3,
                "league_ratings_rows": 4,
                "matches_scanned_total": 5,
            },
        }

    monkeypatch.setattr("jupr_app.services.admin_replay_service.run_replay_with_job_tracking", fake_tracked_replay)

    result = run_admin_replay_history(
        FakeSupabase(storage),
        club_id="club",
        target_reset="Open",
        actor_email="admin@example.com",
        actor_role="super_admin",
        confirmation_text="REPLAY",
    )

    assert result["ok"] is True
    assert result["result"]["matches_rewritten"] == 3
    assert calls and calls[0]["target_reset"] == "Open"
    assert result["job_id"] == "job-1"
    assert storage["admin_activity_log"][0]["action_type"] == "replay_history"
