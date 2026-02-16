from __future__ import annotations

from types import SimpleNamespace

from jupr_app.domain.gamification.fact_engine import update_match_facts_for_players


class FakeTable:
    def __init__(self, storage: dict, name: str):
        self.storage = storage
        self.name = name
        self.filters: list[tuple[str, str, object]] = []
        self.limit_count = None

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def limit(self, count):
        self.limit_count = int(count)
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
        rows = list(self.storage.get(self.name, []))
        for op, column, value in self.filters:
            if op == "eq":
                rows = [r for r in rows if str(r.get(column)) == str(value)]
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


def test_total_matches_and_idempotent_processing():
    storage = {
        "players": [
            {"club_id": "club", "id": 1, "rating": 1280.0, "starting_rating": 1200.0},
            {"club_id": "club", "id": 2, "rating": 1240.0, "starting_rating": 1200.0},
            {"club_id": "club", "id": 3, "rating": 1220.0, "starting_rating": 1200.0},
            {"club_id": "club", "id": 4, "rating": 1180.0, "starting_rating": 1200.0},
        ],
        "player_badge_facts": [],
    }
    supabase = FakeSupabase(storage)

    payload = {
        "match_id": "m-1",
        "score_t1": 11,
        "score_t2": 8,
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "t1_p1_r": 1180.0,
        "t1_p2_r": 1160.0,
        "t2_p1_r": 1320.0,
        "t2_p2_r": 1300.0,
    }

    update_match_facts_for_players(supabase, "club", [1, 2, 3, 4], payload)
    update_match_facts_for_players(supabase, "club", [1, 2, 3, 4], payload)

    for pid in [1, 2, 3, 4]:
        assert _fact_num(storage, pid, "total_matches") == 1.0


def test_win_streak_best_streak_and_reset():
    storage = {
        "players": [
            {"club_id": "club", "id": 1, "rating": 1260.0, "starting_rating": 1200.0},
            {"club_id": "club", "id": 2, "rating": 1260.0, "starting_rating": 1200.0},
            {"club_id": "club", "id": 3, "rating": 1260.0, "starting_rating": 1200.0},
            {"club_id": "club", "id": 4, "rating": 1260.0, "starting_rating": 1200.0},
        ],
        "player_badge_facts": [],
    }
    supabase = FakeSupabase(storage)

    win_payload = {
        "match_id": "m-win",
        "score_t1": 11,
        "score_t2": 7,
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
    }
    lose_payload = {
        "match_id": "m-lose",
        "score_t1": 5,
        "score_t2": 11,
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
    }

    update_match_facts_for_players(supabase, "club", [1, 2], win_payload)
    update_match_facts_for_players(supabase, "club", [1, 2], win_payload | {"match_id": "m-win-2"})
    update_match_facts_for_players(supabase, "club", [1, 2], lose_payload)

    assert _fact_num(storage, 1, "current_win_streak") == 0.0
    assert _fact_num(storage, 1, "best_win_streak") == 2.0


def test_rating_delta_and_upset_wins():
    storage = {
        "players": [
            {"club_id": "club", "id": 1, "rating": 1280.0, "starting_rating": 1200.0},
            {"club_id": "club", "id": 2, "rating": 1260.0, "starting_rating": 1200.0},
            {"club_id": "club", "id": 3, "rating": 1330.0, "starting_rating": 1300.0},
            {"club_id": "club", "id": 4, "rating": 1310.0, "starting_rating": 1300.0},
        ],
        "player_badge_facts": [],
    }
    supabase = FakeSupabase(storage)

    upset_payload = {
        "match_id": "m-upset",
        "score_t1": 11,
        "score_t2": 6,
        "t1_p1": 1,
        "t1_p2": 2,
        "t2_p1": 3,
        "t2_p2": 4,
        "t1_p1_r": 1160.0,
        "t1_p2_r": 1180.0,
        "t2_p1_r": 1320.0,
        "t2_p2_r": 1340.0,
    }
    non_upset_payload = {
        "match_id": "m-non-upset",
        "score_t1": 11,
        "score_t2": 9,
        "t1_p1": 3,
        "t1_p2": 4,
        "t2_p1": 1,
        "t2_p2": 2,
        "t1_p1_r": 1320.0,
        "t1_p2_r": 1340.0,
        "t2_p1_r": 1160.0,
        "t2_p2_r": 1180.0,
    }

    update_match_facts_for_players(supabase, "club", [1, 2, 3, 4], upset_payload)
    update_match_facts_for_players(supabase, "club", [1, 2, 3, 4], non_upset_payload)

    assert _fact_num(storage, 1, "rating_delta") == 80.0
    assert _fact_num(storage, 3, "rating_delta") == 30.0
    assert _fact_num(storage, 1, "upset_wins") == 1.0
    assert _fact_num(storage, 3, "upset_wins") == 0.0
