from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from jupr_app.domain.live_social import (
    resolve_or_create_club_person,
    save_social_round_robin,
    social_round_robin_match_rows_from_event,
)


class _Resp:
    def __init__(self, data):
        self.data = data


class _TableQuery:
    def __init__(self, store: dict[str, list[dict]], name: str):
        self.store = store
        self.name = name
        self._filters: list[tuple[str, object]] = []
        self._op = "select"
        self._payload = None

    def select(self, _cols: str):
        self._op = "select"
        return self

    def eq(self, key: str, value: object):
        self._filters.append((key, value))
        return self

    def insert(self, payload):
        self._op = "insert"
        self._payload = payload
        return self

    def upsert(self, payload, on_conflict: str):
        self._op = "upsert"
        self._payload = (payload, on_conflict)
        return self

    def update(self, payload: dict):
        self._op = "update"
        self._payload = payload
        return self

    def delete(self):
        self._op = "delete"
        return self

    def _filtered(self) -> list[dict]:
        rows = list(self.store.setdefault(self.name, []))
        for key, value in self._filters:
            rows = [row for row in rows if row.get(key) == value]
        return rows

    def execute(self):
        rows = self.store.setdefault(self.name, [])
        if self._op == "select":
            return _Resp([dict(r) for r in self._filtered()])
        if self._op == "delete":
            keep = []
            removed = []
            for row in rows:
                if all(row.get(k) == v for k, v in self._filters):
                    removed.append(row)
                else:
                    keep.append(row)
            self.store[self.name] = keep
            return _Resp(removed)
        if self._op == "update":
            updated = []
            for row in rows:
                if all(row.get(k) == v for k, v in self._filters):
                    row.update(dict(self._payload))
                    updated.append(dict(row))
            return _Resp(updated)
        if self._op == "insert":
            payload = self._payload if isinstance(self._payload, list) else [self._payload]
            inserted = []
            for row in payload:
                record = dict(row)
                record.setdefault("id", str(uuid4()))
                rows.append(record)
                inserted.append(dict(record))
            return _Resp(inserted)
        if self._op == "upsert":
            payload, on_conflict = self._payload
            keys = [k.strip() for k in on_conflict.split(",")]
            match = None
            for row in rows:
                if all(row.get(k) == payload.get(k) for k in keys):
                    match = row
                    break
            if match is None:
                record = dict(payload)
                record.setdefault("id", str(uuid4()))
                rows.append(record)
                return _Resp([dict(record)])
            match.update(dict(payload))
            return _Resp([dict(match)])
        raise AssertionError(f"Unsupported op {self._op}")


class _FakeSupabase:
    def __init__(self):
        self.store: dict[str, list[dict]] = {}

    def table(self, name: str):
        return _TableQuery(self.store, name)


@dataclass
class _Ctx:
    supabase: _FakeSupabase
    club_id: str
    name_to_id: dict


def _sample_event() -> dict:
    return {
        "type": "round_robin",
        "name": "Friday Social",
        "sourceEventUid": "rr-fixed-1",
        "eventDate": "2026-03-29",
        "participants": [
            {"id": "p-1", "name": "Alice", "seed": 1},
            {"id": "p-2", "name": "Bob", "seed": 2},
            {"id": "p-3", "name": "Cami", "seed": 3},
            {"id": "p-4", "name": "Drew", "seed": 4},
        ],
        "rounds": [
            {
                "number": 1,
                "matches": [
                    {"id": "m1", "teamA": ["p-1", "p-2"], "teamB": ["p-3", "p-4"], "scoreA": 11, "scoreB": 7},
                    {"id": "m2", "teamA": ["p-1", "p-3"], "teamB": ["p-2", "p-4"], "scoreA": None, "scoreB": None},
                ],
            }
        ],
    }


def test_match_rows_only_emit_scored_matches():
    rows = social_round_robin_match_rows_from_event(_sample_event())

    assert len(rows) == 1
    assert rows[0]["match_key"] == "m1"


def test_resolve_or_create_links_existing_player_row():
    supabase = _FakeSupabase()
    supabase.store["club_people"] = [
        {
            "id": "cp-1",
            "club_id": "club-1",
            "display_name": "Alice",
            "normalized_name": "alice",
            "linked_player_id": 10,
            "first_seen_on": "2026-03-01",
            "last_seen_on": "2026-03-10",
        }
    ]
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={"Alice": 10})

    row, created_new, matched_player = resolve_or_create_club_person(
        ctx,
        display_name="Alice",
        event_date="2026-03-29",
    )

    assert row["id"] == "cp-1"
    assert created_new is False
    assert matched_player is True
    assert supabase.store["club_people"][0]["last_seen_on"] == "2026-03-29"


def test_resolve_or_create_creates_social_person_when_unmatched():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={})

    row, created_new, matched_player = resolve_or_create_club_person(
        ctx,
        display_name="New Person",
        event_date="2026-03-29",
    )

    assert row["linked_player_id"] is None
    assert row["normalized_name"] == "new person"
    assert created_new is True
    assert matched_player is False


def test_resave_replaces_children_instead_of_duplicating():
    supabase = _FakeSupabase()
    ctx = _Ctx(supabase=supabase, club_id="club-1", name_to_id={"Alice": 10})
    event = _sample_event()

    first = save_social_round_robin(ctx, event)
    assert first["match_count"] == 1

    event["rounds"][0]["matches"][0]["scoreA"] = 11
    event["rounds"][0]["matches"][0]["scoreB"] = 2
    second = save_social_round_robin(ctx, event)

    assert first["event_id"] == second["event_id"]
    participants = [
        row for row in supabase.store["live_event_participants"] if row["event_id"] == first["event_id"]
    ]
    matches = [
        row for row in supabase.store["live_event_matches"] if row["event_id"] == first["event_id"]
    ]
    assert len(participants) == 4
    assert len(matches) == 1
    assert matches[0]["score_t2"] == 2
