from __future__ import annotations

from types import SimpleNamespace

from jupr_app.domain import match_pipeline


class DummyQuery:
    def __init__(self, supabase: "DummySupabase", table_name: str) -> None:
        self.supabase = supabase
        self.table_name = table_name
        self._filters: list[tuple[str, object, str]] = []
        self._limit: int | None = None
        self._payload = None
        self._on_conflict = None
        self._ignore_duplicates = False

    def select(self, _fields: str):
        return self

    def eq(self, col: str, value):
        self._filters.append((col, value, "eq"))
        return self

    def in_(self, col: str, values):
        self._filters.append((col, list(values), "in"))
        return self

    def limit(self, value: int):
        self._limit = int(value)
        return self

    def upsert(self, payload, on_conflict=None, ignore_duplicates=False):
        self._payload = dict(payload)
        self._on_conflict = on_conflict
        self._ignore_duplicates = bool(ignore_duplicates)
        return self

    def execute(self):
        if self._payload is not None:
            return self._execute_upsert()
        return self._execute_select()

    def _execute_select(self):
        rows = [dict(row) for row in self.supabase.tables[self.table_name]]
        for col, value, mode in self._filters:
            if mode == "eq":
                rows = [row for row in rows if row.get(col) == value]
            elif mode == "in":
                rows = [row for row in rows if row.get(col) in value]
        if self._limit is not None:
            rows = rows[: self._limit]
        return SimpleNamespace(data=rows)

    def _execute_upsert(self):
        if self.table_name != "matches":
            raise AssertionError("unexpected upsert table")
        key = self._payload.get("idempotency_key")
        existing = None
        for row in self.supabase.tables["matches"]:
            if row.get(self._on_conflict) == key:
                existing = row
                break

        if existing is not None and self._ignore_duplicates:
            return SimpleNamespace(data=[])

        if existing is not None:
            existing.update(self._payload)
            return SimpleNamespace(data=[dict(existing)])

        inserted = dict(self._payload)
        inserted.setdefault("id", f"m{len(self.supabase.tables['matches']) + 1}")
        self.supabase.tables["matches"].append(inserted)
        return SimpleNamespace(data=[dict(inserted)])


class DummySupabase:
    def __init__(self) -> None:
        self.tables = {
            "players": [
                {"id": 1, "club_id": "club-1", "rating": 1200.0, "last_game_at": None},
                {"id": 2, "club_id": "club-1", "rating": 1200.0, "last_game_at": None},
            ],
            "matches": [],
        }

    def table(self, name: str):
        return DummyQuery(self, name)


def test_record_match_replay_is_idempotent(monkeypatch, capsys):
    supabase = DummySupabase()
    player_updates = []

    def fake_sb_update(_supabase, _table, payload, *, filters):
        player_updates.append((dict(filters), dict(payload)))
        return SimpleNamespace(data=[{"ok": True}])

    monkeypatch.setattr(match_pipeline, "sb_update", fake_sb_update)
    monkeypatch.setattr(match_pipeline, "sb_upsert", lambda *args, **kwargs: SimpleNamespace(data=[{"ok": True}]))

    payload = {
        "club_id": "club-1",
        "team_a_player_ids": [1],
        "team_b_player_ids": [2],
        "score_a": 11,
        "score_b": 9,
        "played_at": "2026-01-01T00:00:00+00:00",
    }

    first = match_pipeline.record_match(supabase, **payload)
    second = match_pipeline.record_match(supabase, **payload)

    assert first["status"] == "inserted"
    assert second["status"] == "exists"
    assert len(player_updates) == 2
    assert "[PIPELINE] idempotent hit — skipping delta" in capsys.readouterr().out
