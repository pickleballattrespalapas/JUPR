from __future__ import annotations

from types import SimpleNamespace

from jupr_app.domain.gamification.fact_replay import rebuild_facts_from_history


class FakeTable:
    def __init__(self, storage: dict, name: str):
        self.storage = storage
        self.name = name
        self.filters: list[tuple[str, str, object]] = []
        self.limit_count: int | None = None
        self._result_override = None
        self._order_by: str | None = None
        self._order_desc: bool = False

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def limit(self, count):
        self.limit_count = int(count)
        return self

    def order(self, column, desc=False):
        self._order_by = str(column)
        self._order_desc = bool(desc)
        return self

    def insert(self, payload, **kwargs):
        rows = payload if isinstance(payload, list) else [payload]
        existing = self.storage.setdefault(self.name, [])
        keys = [k.strip() for k in str(kwargs.get("on_conflict") or "").split(",") if k.strip()]
        ignore_duplicates = bool(kwargs.get("ignore_duplicates"))
        inserted = []
        for row in rows:
            row = dict(row)
            match = None
            if keys:
                for current in existing:
                    if all(str(current.get(k)) == str(row.get(k)) for k in keys):
                        match = current
                        break
            if match is not None and ignore_duplicates:
                continue
            if match is None:
                existing.append(row)
                inserted.append(row)
            else:
                match.update(row)
                inserted.append(match)
        self._result_override = inserted
        return self

    def upsert(self, payload, on_conflict=None):
        keys = [k.strip() for k in str(on_conflict or "").split(",") if k.strip()]
        rows = payload if isinstance(payload, list) else [payload]
        existing = self.storage.setdefault(self.name, [])
        for row in rows:
            row = dict(row)
            match = None
            if keys:
                for current in existing:
                    if all(str(current.get(k)) == str(row.get(k)) for k in keys):
                        match = current
                        break
            if match is None:
                existing.append(row)
            else:
                match.update(row)
        return self

    def execute(self):
        if self._result_override is not None:
            rows = list(self._result_override)
            self._result_override = None
            return SimpleNamespace(data=rows)

        rows = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                rows = [r for r in rows if str(r.get(column)) == str(value)]

        if self._order_by is not None:
            rows = sorted(rows, key=lambda row: str(row.get(self._order_by) or ""), reverse=self._order_desc)

        if self.limit_count is not None:
            rows = rows[: self.limit_count]

        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, storage: dict):
        self.storage = storage

    def table(self, name: str):
        return FakeTable(self.storage, name)


def _fact_num(storage: dict, player_id: int, fact_key: str) -> float:
    for row in storage.get("player_badge_facts", []):
        if int(row["player_id"]) == int(player_id) and row["fact_key"] == fact_key:
            return float(row.get("fact_value_num") or 0.0)
    return 0.0


def test_rebuild_facts_from_history_streaks_and_idempotency():
    storage = {
        "players": [
            {"club_id": "club", "id": 1, "rating": 1260.0, "starting_rating": 1200.0},
            {"club_id": "club", "id": 2, "rating": 1240.0, "starting_rating": 1200.0},
            {"club_id": "club", "id": 3, "rating": 1220.0, "starting_rating": 1200.0},
            {"club_id": "club", "id": 4, "rating": 1210.0, "starting_rating": 1200.0},
        ],
        "matches": [
            {
                "id": "m2",
                "club_id": "club",
                "date": "2026-01-02T10:00:00Z",
                "score_t1": 11,
                "score_t2": 7,
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
            },
            {
                "id": "m1",
                "club_id": "club",
                "date": "2026-01-01T10:00:00Z",
                "score_t1": 11,
                "score_t2": 9,
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
            },
            {
                "id": "m3",
                "club_id": "club",
                "date": "2026-01-03T10:00:00Z",
                "score_t1": 5,
                "score_t2": 11,
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
            },
        ],
        "player_badge_facts": [],
        "processed_match_facts": [],
    }
    supabase = FakeSupabase(storage)

    first = rebuild_facts_from_history(supabase, club_id="club")

    assert first["matches_seen"] == 3
    assert first["matches_processed"] == 3
    assert first["players_touched"] == 4

    assert _fact_num(storage, 1, "total_matches") == 3.0
    assert _fact_num(storage, 1, "current_win_streak") == 0.0
    assert _fact_num(storage, 1, "best_win_streak") == 2.0

    assert _fact_num(storage, 3, "total_matches") == 3.0
    assert _fact_num(storage, 3, "current_win_streak") == 1.0
    assert _fact_num(storage, 3, "best_win_streak") == 1.0

    second = rebuild_facts_from_history(supabase, club_id="club")

    assert second["matches_seen"] == 3
    assert second["matches_processed"] == 3
    assert second["players_touched"] == 4

    assert _fact_num(storage, 1, "total_matches") == 3.0
    assert _fact_num(storage, 1, "best_win_streak") == 2.0
    assert _fact_num(storage, 3, "total_matches") == 3.0

    assert len(storage["processed_match_facts"]) == 12
