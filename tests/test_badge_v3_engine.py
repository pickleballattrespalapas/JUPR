from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.domain.gamification.badge_queue import enqueue_badge_eval
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue
from jupr_app.domain.gamification.v3_engine import evaluate_badges_v3


class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []
        self.limit_count = None
        self.update_payload = None

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, set(values)))
        return self

    def order(self, _column, desc=False):
        self.sort_desc = desc
        return self

    def limit(self, count):
        self.limit_count = count
        return self

    def update(self, payload):
        self.update_payload = payload
        return self

    def insert(self, payload):
        rows = payload if isinstance(payload, list) else [payload]
        stored = self.storage.setdefault(self.name, [])
        for row in rows:
            stored.append(dict(row))
        return self

    def upsert(self, payload, on_conflict=None):
        rows = payload if isinstance(payload, list) else [payload]
        keys = [c.strip() for c in str(on_conflict or "").split(",") if c.strip()]
        stored = self.storage.setdefault(self.name, [])
        existing = {tuple(row.get(key) for key in keys): row for row in stored} if keys else {}
        for row in rows:
            row_dict = dict(row)
            if keys:
                key = tuple(row_dict.get(key_name) for key_name in keys)
                if key in existing:
                    existing[key].update(row_dict)
                else:
                    stored.append(row_dict)
                    existing[key] = row_dict
            else:
                stored.append(row_dict)
        return self

    def execute(self):
        data = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                data = [row for row in data if str(row.get(column)) == str(value)]
            elif op == "in":
                data = [row for row in data if row.get(column) in value]

        if self.update_payload is not None:
            for row in data:
                row.update(self.update_payload)

        if self.limit_count is not None:
            data = data[: int(self.limit_count)]
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage):
        self.storage = storage

    def table(self, name):
        return FakeTable(self.storage, name)


def test_evaluate_badges_v3_awards_and_locks_badge():
    storage = {
        "badges": [{"badge_id": "grinder", "status": "published", "award_count": 0, "is_locked": False}],
        "badge_rule_conditions": [
            {"badge_id": "grinder", "fact_key": "total_matches", "operator": ">=", "value_numeric": 20},
            {"badge_id": "grinder", "fact_key": "is_league_champion", "operator": "is", "value_boolean": True},
        ],
        "player_badge_facts": [
            {
                "club_id": "club",
                "player_id": 7,
                "context_id": "overall",
                "fact_key": "total_matches",
                "fact_value_num": 24,
                "fact_value_bool": None,
            },
            {
                "club_id": "club",
                "player_id": 7,
                "context_id": "overall",
                "fact_key": "is_league_champion",
                "fact_value_num": None,
                "fact_value_bool": True,
            },
        ],
        "player_badges": [],
    }
    supabase = FakeSupabase(storage)
    ctx = SimpleNamespace(supabase=supabase, club_id="club", context_id="overall")

    awarded = evaluate_badges_v3(7, ctx)

    assert awarded == ["grinder"]
    assert len(storage["player_badges"]) == 1
    assert storage["player_badges"][0]["badge_id"] == "grinder"
    assert storage["badges"][0]["award_count"] == 1
    assert storage["badges"][0]["is_locked"] is True


def test_worker_dispatches_to_v3_when_flag_enabled(monkeypatch):
    storage = {
        "badges": [{"badge_id": "grinder", "status": "published", "award_count": 0, "is_locked": False}],
        "badge_rule_conditions": [],
        "player_badge_facts": [],
        "player_badges": [],
    }
    supabase = FakeSupabase(storage)
    enqueue_badge_eval(
        supabase,
        club_id="club",
        event_type="match_recorded",
        player_ids=[7],
        match_id="m1",
    )

    ctx = SimpleNamespace(
        supabase=supabase,
        club_id="club",
        df_badges=SimpleNamespace(empty=True),
        df_matches=None,
        df_players_all=None,
        df_leagues=None,
        df_meta=None,
    )

    calls: list[int] = []

    def _fake_v3(player_id, _context):
        calls.append(int(player_id))
        return []

    monkeypatch.setattr("jupr_app.domain.gamification.v3_engine.USE_BADGE_ENGINE_V3", True)
    monkeypatch.setattr("jupr_app.domain.gamification.v3_engine.evaluate_badges_v3", _fake_v3)
    monkeypatch.setattr(
        "jupr_app.domain.gamification.badge_worker.compute_candidates_for_player",
        lambda *_args, **_kwargs: pytest.fail("v2 evaluator should not run when v3 flag is enabled"),
    )

    result = process_badge_eval_queue(supabase, max_jobs=1, time_budget_seconds=2, ctx=ctx)

    assert result["processed"] == 1
    assert calls == [7]
