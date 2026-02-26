from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from jupr_app.domain.recaps.weekly_recap import _load_completed_tournaments


class FakeTable:
    def __init__(self, storage: dict[str, list[dict]], name: str):
        self.storage = storage
        self.name = name
        self.filters: list[tuple[str, str, object]] = []

    def select(self, _columns: str):
        return self

    def eq(self, column: str, value: object):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column: str, values: list[object]):
        self.filters.append(("in", column, values))
        return self

    def gte(self, column: str, value: object):
        self.filters.append(("gte", column, value))
        return self

    def lte(self, column: str, value: object):
        self.filters.append(("lte", column, value))
        return self

    def execute(self):
        rows = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                rows = [row for row in rows if str(row.get(column)) == str(value)]
            elif op == "in":
                rows = [row for row in rows if row.get(column) in value]
            elif op in {"gte", "lte"}:
                rows = [
                    row
                    for row in rows
                    if row.get(column) is not None
                    and (
                        str(row.get(column)) >= str(value)
                        if op == "gte"
                        else str(row.get(column)) <= str(value)
                    )
                ]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, storage: dict[str, list[dict]]):
        self.storage = storage

    def table(self, name: str):
        return FakeTable(self.storage, name)


def test_load_completed_tournaments_uses_matches_and_range_with_scores():
    storage = {
        "matches": [
            {
                "id": "m1",
                "club_id": "club-1",
                "context_type": "TOURNAMENT",
                "tournament_id": "t-1",
                "date": "2025-02-08T12:00:00+00:00",
                "score_t1": 11,
                "score_t2": 7,
            },
            {
                "id": "m2",
                "club_id": "club-1",
                "context_type": "TOURNAMENT",
                "tournament_id": "t-2",
                "date": "2025-02-08T13:00:00+00:00",
                "score_t1": 0,
                "score_t2": 0,
            },
        ],
        "tournaments": [
            {"id": "t-1", "club_id": "club-1", "name": "Cup", "status": "COMPLETE"},
            {"id": "t-2", "club_id": "club-1", "name": "Open", "status": "COMPLETE"},
        ],
    }

    supabase = FakeSupabase(storage)
    tournaments = _load_completed_tournaments(
        supabase,
        "club-1",
        datetime(2025, 2, 8, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 2, 8, 23, 59, tzinfo=timezone.utc),
    )

    assert [row["id"] for row in tournaments] == ["t-1"]


def test_load_completed_tournaments_returns_empty_when_no_range_matches():
    storage = {
        "matches": [
            {
                "id": "m1",
                "club_id": "club-1",
                "context_type": "TOURNAMENT",
                "tournament_id": "t-1",
                "date": "2025-02-01T12:00:00+00:00",
                "score_t1": 11,
                "score_t2": 7,
            }
        ],
        "tournaments": [
            {"id": "t-1", "club_id": "club-1", "name": "Cup", "status": "COMPLETE"},
        ],
    }

    supabase = FakeSupabase(storage)
    tournaments = _load_completed_tournaments(
        supabase,
        "club-1",
        datetime(2025, 2, 8, 0, 0, tzinfo=timezone.utc),
        datetime(2025, 2, 8, 23, 59, tzinfo=timezone.utc),
    )

    assert tournaments == []
