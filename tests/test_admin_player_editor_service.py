from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.services.admin_guarded_write_service import GuardedWriteRecoveryRequired

from jupr_app.services.admin_player_editor_service import (
    PlayerEditorConflictError,
    build_admin_player_editor_status,
    create_admin_player_editor_player,
    get_admin_player_editor_detail,
    list_admin_player_editor_players,
    reconcile_admin_player_editor_operation,
    update_admin_player_editor_player,
)
from jupr_app.services.admin_player_league_rating_service import update_admin_player_editor_league_rating


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

    def is_(self, key, value):
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
                if stored.get("id") is None and self.table_name in {"players", "admin_guarded_operations"}:
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
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "test-service-role")

    payload = build_admin_player_editor_status(FakeSupabase(fake_storage()), club_id="club")

    assert payload["enabled"] is True
    assert payload["status"] == "ready_for_transactional_player_editor_pilot"
    assert payload["transactional_merge_ready"] is True
    assert payload["player_merge_endpoint"] == "/admin/clubs/{club_id}/players/editor/merge"
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
        idempotency_key="player-create-casey",
        source="test",
    )

    assert result["ok"] is True
    assert result["player"]["name"] == "Casey"
    assert storage["players"][-1]["rating"] == 1300.0
    assert any(row["action_type"] == "create_player_editor_player" for row in storage["admin_activity_log"])


def test_update_player_editor_player_writes_audit(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()

    supabase = FakeSupabase(storage)
    before = get_admin_player_editor_detail(supabase, club_id="club", player_id=1)["player"]
    result = update_admin_player_editor_player(
        supabase,
        club_id="club",
        player_id=1,
        patch={"name": "Alex R", "rating_jupr": 3.7, "starting_jupr": 3.4, "active": False},
        actor_email="owner@example.com",
        actor_role="club_owner",
        expected_state_fingerprint=before["state_fingerprint"],
        idempotency_key="player-update-alex",
        source="test",
    )

    assert result["ok"] is True
    assert result["player"]["name"] == "Alex R"
    assert result["player"]["rating"] == 1480.0
    assert result["player"]["active"] is False
    assert result["player"]["inactive_at"]
    assert any(row["action_type"] == "update_player_editor_player" for row in storage["admin_activity_log"])


def test_update_player_editor_league_rating_writes_audit(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()

    supabase = FakeSupabase(storage)
    before = get_admin_player_editor_detail(supabase, club_id="club", player_id=1)["league_ratings"][0]
    result = update_admin_player_editor_league_rating(
        supabase,
        club_id="club",
        player_id=1,
        league_rating_id=10,
        patch={"rating_jupr": 3.8, "starting_jupr": 3.5, "is_active": False},
        actor_email="owner@example.com",
        actor_role="club_owner",
        expected_state_fingerprint=before["state_fingerprint"],
        idempotency_key="league-rating-update",
        confirmation_text="SAVE LEAGUE RATING",
        source="test",
    )

    assert result["ok"] is True
    assert result["mode"] == "player_editor_league_rating_update"
    assert result["league_rating"]["rating"] == 1520.0
    assert result["league_rating"]["starting_rating"] == 1400.0
    assert result["league_rating"]["is_active"] is False
    assert result["league_rating"]["inactive_at"]
    assert any(row["action_type"] == "update_player_editor_league_rating" for row in storage["admin_activity_log"])


def test_player_editor_create_exact_retry_replays_without_duplicate(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    kwargs = {
        "club_id": "club",
        "name": "Casey",
        "starting_jupr": 3.25,
        "actor_email": "owner@example.com",
        "actor_role": "club_owner",
        "idempotency_key": "player-create-replay",
        "source": "test",
    }

    first = create_admin_player_editor_player(supabase, **kwargs)
    replay = create_admin_player_editor_player(supabase, **kwargs)

    assert replay["idempotent"] is True
    assert replay["player"]["id"] == first["player"]["id"]
    assert [row["name"] for row in storage["players"]].count("Casey") == 1


class AmbiguousMutationQuery(FakeQuery):
    def __init__(self, storage, table_name, *, target_table: str, target_kind: str):
        super().__init__(storage, table_name)
        self.target_table = target_table
        self.target_kind = target_kind

    def execute(self):
        is_target = self.table_name == self.target_table and (
            (self.target_kind == "insert" and self.insert_payload is not None)
            or (self.target_kind == "update" and self.update_payload is not None)
        )
        result = super().execute()
        if is_target:
            raise TimeoutError("response lost after commit")
        return result


class AmbiguousMutationSupabase(FakeSupabase):
    def __init__(self, storage, *, target_table: str, target_kind: str):
        super().__init__(storage)
        self.target_table = target_table
        self.target_kind = target_kind

    def table(self, name):
        return AmbiguousMutationQuery(
            self.storage,
            name,
            target_table=self.target_table,
            target_kind=self.target_kind,
        )


def test_player_editor_create_timeout_marks_recovery_with_readback(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()

    with pytest.raises(GuardedWriteRecoveryRequired, match="may have been created"):
        create_admin_player_editor_player(
            AmbiguousMutationSupabase(storage, target_table="players", target_kind="insert"),
            club_id="club",
            name="Casey",
            starting_jupr=3.25,
            actor_email="owner@example.com",
            actor_role="club_owner",
            idempotency_key="player-create-timeout",
            source="test",
        )

    operation = storage["admin_guarded_operations"][0]
    assert operation["status"] == "recovery_required"
    assert operation["result_json"]["readback_verified"] is True
    assert operation["result_json"]["players"][0]["name"] == "Casey"

    reconciled = reconcile_admin_player_editor_operation(
        FakeSupabase(storage),
        club_id="club",
        operation_key="player-create-timeout",
        confirmation_text="RECONCILE PLAYER OPERATION",
        actor_email="owner@example.com",
        actor_role="club_owner",
        source="test",
    )
    assert reconciled["reconciled"] is True
    assert reconciled["player"]["name"] == "Casey"
    assert operation["status"] == "completed"


def test_player_editor_update_timeout_marks_recovery_with_readback(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()
    baseline = FakeSupabase(storage)
    before = get_admin_player_editor_detail(baseline, club_id="club", player_id=1)["player"]

    with pytest.raises(GuardedWriteRecoveryRequired, match="may have committed"):
        update_admin_player_editor_player(
            AmbiguousMutationSupabase(storage, target_table="players", target_kind="update"),
            club_id="club",
            player_id=1,
            patch={"name": "Alex R"},
            actor_email="owner@example.com",
            actor_role="club_owner",
            expected_state_fingerprint=before["state_fingerprint"],
            idempotency_key="player-update-timeout",
            source="test",
        )

    operation = storage["admin_guarded_operations"][0]
    assert operation["status"] == "recovery_required"
    assert operation["result_json"]["readback_verified"] is True
    assert operation["result_json"]["player"]["name"] == "Alex R"

    reconciled = reconcile_admin_player_editor_operation(
        FakeSupabase(storage),
        club_id="club",
        operation_key="player-update-timeout",
        confirmation_text="RECONCILE PLAYER OPERATION",
        actor_email="owner@example.com",
        actor_role="club_owner",
        source="test",
    )
    assert reconciled["reconciled"] is True
    assert reconciled["player"]["name"] == "Alex R"


def test_league_rating_update_timeout_marks_recovery_with_readback(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()
    baseline = FakeSupabase(storage)
    before = get_admin_player_editor_detail(baseline, club_id="club", player_id=1)["league_ratings"][0]

    with pytest.raises(GuardedWriteRecoveryRequired, match="may have committed"):
        update_admin_player_editor_league_rating(
            AmbiguousMutationSupabase(storage, target_table="league_ratings", target_kind="update"),
            club_id="club",
            player_id=1,
            league_rating_id=10,
            patch={"rating_jupr": 3.8},
            actor_email="owner@example.com",
            actor_role="club_owner",
            expected_state_fingerprint=before["state_fingerprint"],
            idempotency_key="rating-update-timeout",
            confirmation_text="SAVE LEAGUE RATING",
            source="test",
        )

    operation = storage["admin_guarded_operations"][0]
    assert operation["status"] == "recovery_required"
    assert operation["result_json"]["readback_verified"] is True
    assert operation["result_json"]["league_rating"]["rating"] == 1520.0

    reconciled = reconcile_admin_player_editor_operation(
        FakeSupabase(storage),
        club_id="club",
        operation_key="rating-update-timeout",
        confirmation_text="RECONCILE PLAYER OPERATION",
        actor_email="owner@example.com",
        actor_role="club_owner",
        source="test",
    )
    assert reconciled["reconciled"] is True
    assert reconciled["league_rating"]["rating"] == 1520.0


def test_player_create_reconcile_proves_absence_and_closes_operation(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()
    storage["admin_guarded_operations"] = [
        {
            "id": 1,
            "club_id": "club",
            "workflow": "player_editor_create",
            "operation_key": "player-create-absent",
            "status": "intent_recorded",
            "before_json": None,
            "result_json": {
                "planned": {
                    "player": {
                        "club_id": "club",
                        "name": "Missing Person",
                        "rating": 1400.0,
                        "starting_rating": 1400.0,
                        "wins": 0,
                        "losses": 0,
                        "matches_played": 0,
                        "active": True,
                        "last_game_at": None,
                        "inactive_at": None,
                    }
                },
                "preexisting_player_ids": [],
            },
        }
    ]

    reconciled = reconcile_admin_player_editor_operation(
        FakeSupabase(storage),
        club_id="club",
        operation_key="player-create-absent",
        confirmation_text="RECONCILE PLAYER OPERATION",
        actor_email="owner@example.com",
        actor_role="club_owner",
        source="test",
    )

    assert reconciled["status"] == "failed"
    assert reconciled["recovery_required"] is False
    assert storage["admin_guarded_operations"][0]["status"] == "failed"


def test_player_editor_update_replays_before_current_state_check(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    before = get_admin_player_editor_detail(supabase, club_id="club", player_id=1)["player"]
    kwargs = {
        "club_id": "club",
        "player_id": 1,
        "patch": {"name": "Alex R", "active": False},
        "actor_email": "owner@example.com",
        "actor_role": "club_owner",
        "expected_state_fingerprint": before["state_fingerprint"],
        "idempotency_key": "player-update-replay",
        "source": "test",
    }

    first = update_admin_player_editor_player(supabase, **kwargs)
    replay = update_admin_player_editor_player(supabase, **kwargs)

    assert replay["idempotent"] is True
    assert replay["player"] == first["player"]
    assert storage["players"][0]["name"] == "Alex R"


def test_player_editor_update_rejects_stale_review(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()
    supabase = FakeSupabase(storage)

    with pytest.raises(PlayerEditorConflictError, match="changed after it was loaded"):
        update_admin_player_editor_player(
            supabase,
            club_id="club",
            player_id=1,
            patch={"name": "Alex R"},
            actor_email="owner@example.com",
            actor_role="club_owner",
            expected_state_fingerprint="0" * 64,
            idempotency_key="player-update-stale",
            source="test",
        )

    assert storage["players"][0]["name"] == "Alex"


def test_league_rating_exact_retry_replays_before_current_state_check(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_PLAYER_EDITOR", "1")
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    before = get_admin_player_editor_detail(supabase, club_id="club", player_id=1)["league_ratings"][0]
    kwargs = {
        "club_id": "club",
        "player_id": 1,
        "league_rating_id": 10,
        "patch": {"rating_jupr": 3.8, "is_active": False},
        "actor_email": "owner@example.com",
        "actor_role": "club_owner",
        "expected_state_fingerprint": before["state_fingerprint"],
        "idempotency_key": "league-rating-replay",
        "confirmation_text": "SAVE LEAGUE RATING",
        "source": "test",
    }

    first = update_admin_player_editor_league_rating(supabase, **kwargs)
    replay = update_admin_player_editor_league_rating(supabase, **kwargs)

    assert replay["idempotent"] is True
    assert replay["league_rating"] == first["league_rating"]
