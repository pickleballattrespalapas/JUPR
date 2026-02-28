from types import SimpleNamespace

import pytest
from postgrest.exceptions import APIError

from jupr_app.domain.gamification.badge_types import BadgeCandidate
from jupr_app.domain.gamification.badges_repo import ensure_player_badges_contract, upsert_player_badges


class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, set(values)))
        return self

    def upsert(self, rows, on_conflict=None):
        existing = self.storage.setdefault(self.name, [])
        keys = [c.strip() for c in str(on_conflict or "").split(",") if c.strip()]
        existing_keys = {tuple(row.get(k) for k in keys) for row in existing} if keys else set()
        for row in rows:
            key = tuple(row.get(k) for k in keys) if keys else None
            if key is not None and key in existing_keys:
                continue
            existing.append(row)
            if key is not None:
                existing_keys.add(key)
        return self

    def execute(self):
        data = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                data = [row for row in data if str(row.get(column)) == str(value)]
            elif op == "in":
                data = [row for row in data if row.get(column) in value]
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else {}

    def table(self, name):
        return FakeTable(self.storage, name)


def _default_indexes():
    return [
        {
            "schemaname": "public",
            "tablename": "player_badges",
            "indexname": "player_badges_unique_context",
            "indexdef": (
                "CREATE UNIQUE INDEX player_badges_unique_context "
                "ON public.player_badges USING btree (club_id, player_id, badge_id, context_id)"
            ),
        }
    ]


def test_player_badges_idempotent_upsert():
    storage = {"pg_indexes": _default_indexes()}
    supabase = FakeSupabase(storage)
    candidate = BadgeCandidate(
        badge_id="participant",
        player_id=1,
        club_id="club",
        context_type="overall",
        context_id="overall",
        match_id=None,
        value_json={"games": 1},
    )
    upsert_player_badges(supabase, "club", [candidate])
    upsert_player_badges(supabase, "club", [candidate])
    assert len(storage["player_badges"]) == 1


def test_player_badges_multi_context_allowed():
    storage = {"pg_indexes": _default_indexes()}
    supabase = FakeSupabase(storage)
    base = dict(
        badge_id="mountain_climber",
        player_id=1,
        club_id="club",
        context_type="league",
        match_id=None,
    )
    upsert_player_badges(
        supabase,
        "club",
        [BadgeCandidate(context_id="league:alpha", value_json={"games": 1}, **base)],
    )
    upsert_player_badges(
        supabase,
        "club",
        [BadgeCandidate(context_id="league:beta", value_json={"games": 2}, **base)],
    )
    assert len(storage["player_badges"]) == 2


def test_player_badges_contract_check_rejects_wrong_index(monkeypatch):
    storage = {
        "pg_indexes": [
            {
                "schemaname": "public",
                "tablename": "player_badges",
                "indexname": "player_badges_unique_context_type",
                "indexdef": (
                    "CREATE UNIQUE INDEX player_badges_unique_context_type "
                    "ON public.player_badges USING btree (club_id, player_id, badge_id, context_type, context_id)"
                ),
            }
        ]
    }
    supabase = FakeSupabase(storage)
    monkeypatch.setenv("BADGE_DEBUG", "1")
    with pytest.raises(RuntimeError):
        ensure_player_badges_contract(supabase)


def test_player_badges_upsert_retries_when_awarded_by_uuid_cast_fails(monkeypatch):
    monkeypatch.setattr("jupr_app.domain.gamification.badges_repo._PLAYER_BADGES_CONTRACT_CHECKED", True)

    calls: list[list[dict[str, object]]] = []

    class _RetryTable:
        def upsert(self, rows, on_conflict=None):
            calls.append(rows)
            if len(calls) == 1:
                raise APIError(
                    {
                        "code": "22P02",
                        "message": 'invalid input syntax for type uuid: "engine"',
                    }
                )
            return self

        def execute(self):
            return SimpleNamespace(data=[])

    class _RetrySupabase:
        def table(self, _name):
            return _RetryTable()

    candidate = BadgeCandidate(
        badge_id="participant",
        player_id=1,
        club_id="club",
        context_type="overall",
        context_id="overall",
        match_id=None,
        value_json={"games": 1},
    )

    upsert_player_badges(_RetrySupabase(), "club", [candidate])

    assert len(calls) == 2
    assert "awarded_by" in calls[0][0]
    assert "awarded_by" not in calls[1][0]
    assert "rule_version" in calls[1][0]
    assert "eval_run_id" in calls[1][0]
