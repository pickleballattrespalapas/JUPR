from __future__ import annotations

from types import SimpleNamespace

from postgrest.exceptions import APIError

from jupr_app.domain.gamification.v3_engine import evaluate_badges_v3_for_player


class FakeTable:
    def __init__(self, storage: dict, name: str):
        self.storage = storage
        self.name = name
        self.filters: list[tuple[str, str, object]] = []
        self.limit_count: int | None = None
        self.update_payload: dict | None = None

    def select(self, _cols: str):
        return self

    def eq(self, column: str, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column: str, values):
        self.filters.append(("in", column, set(values)))
        return self

    def limit(self, count: int):
        self.limit_count = int(count)
        return self

    def insert(self, payload: dict):
        rows = payload if isinstance(payload, list) else [payload]
        target = self.storage.setdefault(self.name, [])
        for row in rows:
            row_dict = dict(row)
            if self.name == "player_badges":
                key = (
                    row_dict.get("club_id"),
                    row_dict.get("player_id"),
                    row_dict.get("badge_id"),
                    row_dict.get("context_id"),
                )
                existing = {
                    (
                        existing_row.get("club_id"),
                        existing_row.get("player_id"),
                        existing_row.get("badge_id"),
                        existing_row.get("context_id"),
                    )
                    for existing_row in target
                }
                if key in existing:
                    raise APIError({"code": "23505", "message": "duplicate key"})
            target.append(row_dict)
        return self

    def update(self, payload: dict):
        self.update_payload = dict(payload)
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
            data = data[: self.limit_count]
        return SimpleNamespace(data=data)


class FakeSupabase:
    def __init__(self, storage: dict):
        self.storage = storage

    def table(self, name: str):
        return FakeTable(self.storage, name)


def _build_base_storage() -> dict:
    return {
        "badges": [
            {"club_id": "club", "badge_id": "grinder", "status": "published", "award_count": 0, "is_locked": False},
        ],
        "badge_rule_conditions": [
            {"badge_id": "grinder", "fact_key": "matches_seen", "operator": ">=", "value_numeric": 1},
        ],
        "player_badge_facts": [
            {"club_id": "club", "player_id": 7, "context_id": "overall", "fact_key": "matches_seen", "fact_value_num": 3},
        ],
        "player_badges": [],
    }


def test_single_condition_pass():
    storage = _build_base_storage()
    awarded = evaluate_badges_v3_for_player(FakeSupabase(storage), "club", 7, "overall")

    assert awarded == ["grinder"]
    assert len(storage["player_badges"]) == 1


def test_multi_condition_and_pass():
    storage = _build_base_storage()
    storage["badge_rule_conditions"].append(
        {"badge_id": "grinder", "fact_key": "wins", "operator": ">=", "value_numeric": 2}
    )
    storage["player_badge_facts"].append(
        {"club_id": "club", "player_id": 7, "context_id": "overall", "fact_key": "wins", "fact_value_num": 2}
    )

    awarded = evaluate_badges_v3_for_player(FakeSupabase(storage), "club", 7, "overall")

    assert awarded == ["grinder"]


def test_condition_fail_prevents_award():
    storage = _build_base_storage()
    storage["badge_rule_conditions"].append(
        {"badge_id": "grinder", "fact_key": "wins", "operator": ">=", "value_numeric": 2}
    )

    awarded = evaluate_badges_v3_for_player(FakeSupabase(storage), "club", 7, "overall")

    assert awarded == []
    assert storage["player_badges"] == []


def test_idempotent_insert():
    storage = _build_base_storage()
    supabase = FakeSupabase(storage)

    first = evaluate_badges_v3_for_player(supabase, "club", 7, "overall")
    second = evaluate_badges_v3_for_player(supabase, "club", 7, "overall")

    assert first == ["grinder"]
    assert second == []
    assert len(storage["player_badges"]) == 1


def test_lock_on_first_award():
    storage = _build_base_storage()
    evaluate_badges_v3_for_player(FakeSupabase(storage), "club", 7, "overall")

    assert storage["badges"][0]["is_locked"] is True
    assert storage["badges"][0]["award_count"] == 1


def test_no_duplicate_award_count_increment():
    storage = _build_base_storage()
    supabase = FakeSupabase(storage)

    evaluate_badges_v3_for_player(supabase, "club", 7, "overall")
    evaluate_badges_v3_for_player(supabase, "club", 7, "overall")

    assert storage["badges"][0]["award_count"] == 1
