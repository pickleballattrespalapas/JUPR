from types import SimpleNamespace

from jupr_app.domain.tournament_podium import (
    award_tournament_trophies_from_podium,
    mint_tournament_podium_badges,
    upsert_tournament_podium,
)

class DummyTable:
    def __init__(self) -> None:
        self.upsert_calls = []

    def upsert(self, payload, on_conflict=None):
        self.upsert_calls.append({"payload": payload, "on_conflict": on_conflict})
        return self

    def execute(self):
        return self

class DummySupabase:
    def __init__(self) -> None:
        self.last_table = None
        self.table_obj = DummyTable()

    def table(self, name: str):
        self.last_table = name
        return self.table_obj

class FakeTable:
    def __init__(self, storage, name):
        self.storage = storage
        self.name = name
        self.filters = []
        self.ordering = None

    def select(self, _cols):
        return self

    def eq(self, column, value):
        self.filters.append(("eq", column, value))
        return self

    def in_(self, column, values):
        self.filters.append(("in", column, set(values)))
        return self

    def like(self, column, pattern):
        self.filters.append(("like", column, pattern))
        return self

    def order(self, column, desc=False):
        self.ordering = (column, desc)
        return self

    def upsert(self, rows, on_conflict=None):
        existing = self.storage.setdefault(self.name, [])
        keys = [c.strip() for c in str(on_conflict or "").split(",") if c.strip()]
        existing_keys = {tuple(row.get(k) for k in keys) for row in existing} if keys else set()
        for row in rows:
            key = tuple(row.get(k) for k in keys) if keys else None
            if key is not None and key in existing_keys:
                continue
            existing.append(dict(row))
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
            elif op == "like":
                needle = str(value).replace("%", "")
                data = [row for row in data if needle in str(row.get(column) or "")]
        if self.ordering:
            column, desc = self.ordering
            data = sorted(data, key=lambda row: row.get(column), reverse=bool(desc))
        return SimpleNamespace(data=data)

class FakeSupabase:
    def __init__(self, storage=None):
        self.storage = storage if storage is not None else {}

    def table(self, name):
        return FakeTable(self.storage, name)

def test_upsert_tournament_podium_is_idempotent():
    supabase = DummySupabase()
    payload = [
        {"tournament_id": "tour1", "placement": 1, "team_id": "t1", "source": "ROUND_ROBIN"},
        {"tournament_id": "tour1", "placement": 2, "team_id": "t2", "source": "ROUND_ROBIN"},
    ]

    upsert_tournament_podium(supabase, "tour1", payload)
    upsert_tournament_podium(supabase, "tour1", payload)

    assert supabase.last_table == "tournament_podium"
    assert len(supabase.table_obj.upsert_calls) == 2
    for call in supabase.table_obj.upsert_calls:
        assert call["payload"] == payload
        assert call["on_conflict"] == "tournament_id,placement"

def test_award_tournament_trophies_from_podium_is_idempotent_and_profile_compatible():
    storage = {
        "pg_indexes": [
            {
                "schemaname": "public",
                "tablename": "player_badges",
                "indexname": "player_badges_unique",
                "indexdef": "CREATE UNIQUE INDEX player_badges_unique ON public.player_badges USING btree (club_id, player_id, badge_id, context_id)",
            }
        ],
        "tournament_podium": [
            {"tournament_id": "tour1", "placement": 1, "team_id": "team1", "source": "PLAYOFF"},
            {"tournament_id": "tour1", "placement": 2, "team_id": "team2", "source": "PLAYOFF"},
            {"tournament_id": "tour1", "placement": 3, "team_id": "team3", "source": "PLAYOFF"},
        ],
        "tournament_teams": [
            {"id": "team1", "team_number": 1, "player1_id": 101, "player2_id": 102},
            {"id": "team2", "team_number": 2, "player1_id": 103, "player2_id": 104},
            {"id": "team3", "team_number": 3, "player1_id": 105, "player2_id": 106},
        ],
        "player_badges": [],
    }
    supabase = FakeSupabase(storage)
    ctx = SimpleNamespace(supabase=supabase, club_id="club1")

    created_once = award_tournament_trophies_from_podium(ctx, "tour1", "Spring Open")
    created_twice = award_tournament_trophies_from_podium(ctx, "tour1", "Spring Open")

    assert len(created_once) == 6
    assert created_twice == []
    rows = storage["player_badges"]
    assert len(rows) == 6
    assert {row["context_type"] for row in rows} == {"tournament"}
    assert all(":podium:" in row["context_id"] for row in rows)

def test_mint_tournament_podium_badges_is_idempotent_and_profile_compatible():
    storage = {
        "pg_indexes": [
            {
                "schemaname": "public",
                "tablename": "player_badges",
                "indexname": "player_badges_unique",
                "indexdef": "CREATE UNIQUE INDEX player_badges_unique ON public.player_badges USING btree (club_id, player_id, badge_id, context_id)",
            }
        ],
        "tournament_podium": [
            {"tournament_id": "tour1", "placement": 1, "team_id": "team1", "source": "PLAYOFF"},
            {"tournament_id": "tour1", "placement": 2, "team_id": "team2", "source": "PLAYOFF"},
            {"tournament_id": "tour1", "placement": 3, "team_id": "team3", "source": "PLAYOFF"},
        ],
        "tournament_teams": [
            {"id": "team1", "team_number": 1, "player1_id": 101, "player2_id": 102},
            {"id": "team2", "team_number": 2, "player1_id": 103, "player2_id": 104},
            {"id": "team3", "team_number": 3, "player1_id": 105, "player2_id": 106},
        ],
        "player_badges": [],
    }
    supabase = FakeSupabase(storage)
    ctx = SimpleNamespace(supabase=supabase, club_id="club1")

    created_once = mint_tournament_podium_badges(ctx, "tour1", "Spring Open")
    created_twice = mint_tournament_podium_badges(ctx, "tour1", "Spring Open")

    assert len(created_once) == 6
    assert created_twice == []

    rows = storage["player_badges"]
    assert len(rows) == 6
    assert {row["context_type"] for row in rows} == {"tournament"}
    assert all(":podium:" in row["context_id"] for row in rows)
    assert {row["badge_id"] for row in rows} == {
        "tournament_champion",
        "tournament_runner_up",
        "tournament_third_place",
    }
    assert all(row.get("earned_at") for row in rows)

    profile_rows = [
        row
        for row in rows
        if str(row.get("context_type")) == "tournament" and ":podium:" in str(row.get("context_id") or "")
    ]
    assert len(profile_rows) == 6

    for row in rows:
        assert row["value_num"] in {1.0, 2.0, 3.0}
        value_json = row.get("value_json") or {}
        assert value_json.get("tournament_id") == "tour1"
        assert value_json.get("tournament_name") == "Spring Open"
        assert value_json.get("placement") in {1, 2, 3}
        assert value_json.get("team_id") in {"team1", "team2", "team3"}
        assert value_json.get("team_number") in {1, 2, 3}
