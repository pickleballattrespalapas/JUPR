from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from jupr_app.services.admin_match_uploader_service import (
    build_admin_match_uploader_status,
    submit_admin_match_uploader_batch,
)
from jupr_app.services.result_types import ServiceResult


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters: list[tuple[str, object]] = []
        self.limit_value: int | None = None
        self.order_key: str | None = None
        self.order_desc = False
        self.insert_payload = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, key, desc=False):
        self.order_key = key
        self.order_desc = bool(desc)
        return self

    def limit(self, value):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = payload
        return self

    def execute(self):
        table = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            rows = self.insert_payload if isinstance(self.insert_payload, list) else [self.insert_payload]
            for row in rows:
                table.append(dict(row))
            return SimpleNamespace(data=rows)
        rows = list(table)
        for key, expected in self.filters:
            rows = [row for row in rows if str(row.get(key)) == str(expected)]
        if self.order_key:
            rows = sorted(rows, key=lambda row: str(row.get(self.order_key) or ""), reverse=self.order_desc)
        if self.limit_value is not None:
            rows = rows[: self.limit_value]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, storage):
        self.storage = storage

    def table(self, name):
        return FakeQuery(self.storage, name)


def fake_storage():
    return {
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 2, "name": "Blair", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 3, "name": "Casey", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
            {"club_id": "club", "id": 4, "name": "Devon", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0},
        ],
        "matches": [{"club_id": "club", "id": 99, "date": "2026-03-01T00:00:00Z"}],
        "leagues_metadata": [
            {"club_id": "club", "league_name": "Open", "is_active": True},
            {"club_id": "club", "league_name": "Advanced", "is_active": True},
        ],
        "admin_activity_log": [],
    }


def fake_load_data(_supabase, _club_id):
    players = pd.DataFrame(fake_storage()["players"])
    return (
        players,
        players,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"league_name": "Open", "k_factor": 32}]),
        pd.DataFrame(),
        pd.DataFrame(),
        {"Alex": 1, "Blair": 2, "Casey": 3, "Devon": 4},
        {1: "Alex", 2: "Blair", 3: "Casey", 4: "Devon"},
        False,
        None,
    )


def test_match_uploader_status_disabled_is_db_free(monkeypatch) -> None:
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", raising=False)

    payload = build_admin_match_uploader_status(None, club_id="club")

    assert payload["enabled"] is False
    assert payload["submit_endpoint"] is None
    assert payload["max_batch_rows"] >= 1


def test_match_uploader_status_enabled_lists_leagues(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")

    payload = build_admin_match_uploader_status(FakeSupabase(fake_storage()), club_id="club")

    assert payload["enabled"] is True
    assert payload["submit_endpoint"] == "/admin/clubs/{club_id}/match-uploader/batch"
    assert "Open" in payload["league_options"]


def test_submit_match_uploader_batch(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    calls = []

    def fake_submit_match_batch(ctx, matches, **kwargs):
        calls.append({"ctx": ctx, "matches": matches, "kwargs": kwargs})
        return ServiceResult.success(data={"inserted": len(matches), "skipped_incomplete": 0, "skipped_empty": 0, "skipped_unrated": 0})

    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.load_data", fake_load_data)
    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.submit_match_batch", fake_submit_match_batch)
    storage = fake_storage()

    result = submit_admin_match_uploader_batch(
        FakeSupabase(storage),
        club_id="club",
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        matches=[
            {
                "date": "2026-03-01",
                "league": "Open",
                "week_tag": "Week 1",
                "match_type": "Live Match",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
            }
        ],
        source="test",
    )

    assert result["ok"] is True
    assert result["submitted_count"] == 1
    assert result["result"]["inserted"] == 1
    assert calls[0]["matches"][0]["league"] == "Open"
    assert storage["admin_activity_log"][0]["action_type"] == "submit_match_uploader_batch"


def test_submit_match_uploader_rejects_empty(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.load_data", fake_load_data)

    try:
        submit_admin_match_uploader_batch(
            FakeSupabase(fake_storage()),
            club_id="club",
            actor_email="admin@example.com",
            actor_role="scorekeeper",
            matches=[{"t1_p1": 1}],
        )
    except ValueError as exc:
        assert "No valid match" in str(exc)
    else:
        raise AssertionError("Expected empty batch rejection")
