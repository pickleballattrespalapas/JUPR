from __future__ import annotations

from types import SimpleNamespace

from jupr_app.services.admin_player_editor_service import (
    build_admin_player_editor_status,
    create_admin_player_editor_player,
    get_admin_player_editor_detail,
    list_admin_player_editor_players,
    update_admin_player_editor_player,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters: list[tuple[str, object]] = []
        self.order_key: str | None = None
        self.order_desc = False
        self.limit_value: int | None = None
        self.insert_payload = None
        self.update_payload = None

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

    def update(self, payload):
        self.update_payload = dict(payload or {})
        return self

    def _matching_rows(self, table):
        rows = list(table)
        for key, expected in self.filters:
            rows = [row for row in rows if str(row.get(key)) == str(expected)]
        if self.order_key:
            rows = sorted(rows, key=lambda row: str(row.get(self.order_key) or ""), reverse=self.order_desc)
        if self.limit_value is not None:
            rows = rows[: self.limit_value]
        return rows

    def execute(self):
        table = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            rows = self.insert_payload if isinstance(self.insert_payload, list) else [self.insert_payload]
            inserted = []
            for row in rows:
                stored = dict(row)
                if stored.get("id") is None and self.table_name == "players":
                    ids = []
                    for existing in table:
                        try:
                            ids.append(int(existing.get("id")))
                        except Exception:
                            pass
                    stored["id"] = max(ids or [0]) + 1
                table.append(stored)
                inserted.append(stored)
            return SimpleNamespace(data=inserted)
        matched = self._matching_rows(table)
        if self.update_payload is not None:
            for row in matched:
                row.update(self.update_payload)
            return SimpleNamespace(data=matched)
        return SimpleNamespace(data=matched)


class FakeSupabase:
    def __init__(self, storage):
        self.storage = storage

    def table(self, name):
        return FakeQuery(self.storage, name)


def fake_storage():
    return {
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "rating": 1400, "starting_rating": 1400, "wins": 4, "losses": 2, "matches_played": 6, "active": True, "inactive_at": None},
            {"club_id": "club", "id": 2, "name": "Blair", "rating": 1320, "starting_rating": 1300, "wins": 2, "losses": 3, "matches_played": 5, "active": True, "inactive_at": None},
        ],
        "league_ratings": [
            {"club_id": "club", "id": 10, "player_id": 1, "league_name": "Open", "rating": 1420, "starting_rating": 1400, "wins": 3, "losses": 1, "matches_played": 4, "is_active": True, "inactive_at": None},
        ],
        "matches": [
            {"club_id": "club", "id": 100, "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4},
            {"club_id": "club", "id": 101, "t1_p1": 2, "t1_p2": 1, "t2_p1": 3, "t2_p2": 4},
        ],
        "admin_activity_log": [],
    }


def test_player_editor_status_disabled_is_db_free(monkeypatch) -> None:
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", raising=False)

    payload = build_admin_player_editor_status(None, club_id="club")

    assert payload["enabled"] is False
    assert payload["players_endpoint"] is None


def test_player_editor_status_enabled_counts_players(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")

    payload = build_admin_player_editor_status(FakeSupabase(fake_storage()), club_id="club")

    assert payload["enabled"] is True
    assert payload["status"] == "ready_for_player_create_update_foundation"
    assert payload["player_count"] == 2


def test_list_and_detail_player_editor(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    supabase = FakeSupabase(fake_storage())

    listing = list_admin_player_editor_players(supabase, club_id="club")
    detail = get_admin_player_editor_detail(supabase, club_id="club", player_id=1)

    assert listing["count"] == 2
    assert listing["players"][0]["name"] == "Alex"
    assert detail["player"]["rating_jupr"] == 3.5
    assert detail["league_ratings"][0]["league_name"] == "Open"
    assert detail["match_reference_counts"]["total"] == 2


def test_create_player_editor_player_writes_audit(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()

    result = create_admin_player_editor_player(
        FakeSupabase(storage),
        club_id="club",
        name="Casey",
        starting_jupr=3.25,
        actor_email="owner@example.com",
        actor_role="club_owner",
        source="test",
    )

    assert result["ok"] is True
    assert result["player"]["name"] == "Casey"
    assert storage["players"][-1]["rating"] == 1300.0
    assert storage["admin_activity_log"][0]["action_type"] == "create_player_editor_player"


def test_update_player_editor_player_writes_audit(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()

    result = update_admin_player_editor_player(
        FakeSupabase(storage),
        club_id="club",
        player_id=1,
        patch={"name": "Alex R", "rating_jupr": 3.7, "starting_jupr": 3.4, "active": False},
        actor_email="owner@example.com",
        actor_role="club_owner",
        source="test",
    )

    assert result["ok"] is True
    assert result["player"]["name"] == "Alex R"
    assert result["player"]["rating"] == 1480.0
    assert result["player"]["active"] is False
    assert result["player"]["inactive_at"]
    assert storage["admin_activity_log"][0]["action_type"] == "update_player_editor_player"
