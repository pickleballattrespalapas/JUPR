from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.services.admin_guarded_write_service import GuardedWriteRecoveryRequired

from jupr_app.services.admin_match_uploader_service import (
    build_admin_match_uploader_round_robin_preview,
    build_admin_match_uploader_status,
    create_admin_match_uploader_players,
    match_uploader_player_batch_fingerprint,
    reconcile_admin_match_uploader_player_operation,
    submit_admin_match_uploader_batch,
)


class FakeQuery:
    def __init__(self, storage, table_name):
        self.storage = storage
        self.table_name = table_name
        self.filters: list[tuple[str, object]] = []
        self.limit_value: int | None = None
        self.order_key: str | None = None
        self.order_desc = False
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

    def execute(self):
        table = self.storage.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            rows = self.insert_payload if isinstance(self.insert_payload, list) else [self.insert_payload]
            inserted = []
            for row in rows:
                stored = dict(row)
                if stored.get("id") is None:
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
        rows = list(table)
        for key, expected in self.filters:
            rows = [row for row in rows if str(row.get(key)) == str(expected)]
        if self.update_payload is not None:
            for row in rows:
                row.update(self.update_payload)
            return SimpleNamespace(data=rows)
        if self.order_key:
            rows = sorted(rows, key=lambda row: str(row.get(self.order_key) or ""), reverse=self.order_desc)
        if self.limit_value is not None:
            rows = rows[: self.limit_value]
        return SimpleNamespace(data=rows)


class FakeSupabase:
    def __init__(self, storage):
        self.storage = storage

    def table(self, name):
        return FakeQuery(self.storage, name)


def fake_storage():
    return {
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0, "active": True},
            {"club_id": "club", "id": 2, "name": "Blair", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0, "active": True},
            {"club_id": "club", "id": 3, "name": "Casey", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0, "active": True},
            {"club_id": "club", "id": 4, "name": "Devon", "rating": 1200, "wins": 0, "losses": 0, "matches_played": 0, "active": True},
        ],
        "matches": [{"club_id": "club", "id": 99, "date": "2026-03-01T00:00:00Z"}],
        "events": [],
        "leagues_metadata": [
            {"club_id": "club", "league_name": "Open", "is_active": True},
            {"club_id": "club", "league_name": "Advanced", "is_active": True},
            {
                "club_id": "club",
                "league_name": "Stale Active",
                "is_active": True,
                "status": "active",
                "ended_at": "2026-01-01T00:00:00Z",
            },
        ],
        "admin_activity_log": [],
    }


def fake_load_data(_supabase, _club_id):
    players = pd.DataFrame(fake_storage()["players"])
    return (
        players,
        players,
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame([{"league_name": "Open", "k_factor": 32}]),
        pd.DataFrame(),
        pd.DataFrame(),
        {"Alex": 1, "Blair": 2, "Casey": 3, "Devon": 4},
        {1: "Alex", 2: "Blair", 3: "Casey", 4: "Devon"},
        False,
        None,
    )


def test_match_uploader_status_disabled_is_db_free(monkeypatch) -> None:
    monkeypatch.delenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", raising=False)
    monkeypatch.delenv(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES",
        raising=False,
    )

    payload = build_admin_match_uploader_status(None, club_id="club")

    assert payload["enabled"] is False
    assert payload["singles_write_enabled"] is False
    assert payload["submit_endpoint"] is None
    assert payload["max_batch_rows"] >= 1
    assert "4-Player" in payload["round_robin_format_options"]


def test_match_uploader_status_enabled_lists_leagues_and_round_robin(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.delenv(
        "JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES",
        raising=False,
    )

    payload = build_admin_match_uploader_status(FakeSupabase(fake_storage()), club_id="club")

    assert payload["enabled"] is True
    assert payload["singles_write_enabled"] is False
    assert payload["singles_submit_endpoint"] is None
    assert payload["status"] == "ready_for_manual_batch_and_round_robin"
    assert payload["submit_endpoint"] == "/admin/clubs/{club_id}/match-uploader/batch"
    assert payload["round_robin_preview_endpoint"] == "/admin/clubs/{club_id}/match-uploader/round-robin/preview"
    assert payload["player_create_endpoint"] == "/admin/clubs/{club_id}/match-uploader/players"
    assert "Open" in payload["league_options"]
    assert "Stale Active" not in payload["league_options"]
    assert payload["round_robin_expected_games"]["4-Player"] == 3


def test_round_robin_preview_reports_missing_players(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")

    payload = build_admin_match_uploader_round_robin_preview(
        FakeSupabase(fake_storage()),
        club_id="club",
        courts=[{"format_type": "4-Player", "player_names": ["Alex", "Blair", "Casey", "New Person"]}],
    )

    assert payload["ok"] is True
    assert payload["missing_players"] == ["New Person"]
    assert payload["match_count"] == 0
    assert payload["courts"] == []


def test_create_players_then_continue_round_robin_preview(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    storage = fake_storage()
    supabase = FakeSupabase(storage)

    reviewed_players = [{"name": "New Person", "starting_jupr": 3.5}]
    created = create_admin_match_uploader_players(
        supabase,
        club_id="club",
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        players=reviewed_players,
        reviewed_fingerprint=match_uploader_player_batch_fingerprint(reviewed_players),
        idempotency_key="create-player-batch",
        confirmation_text="CREATE PLAYERS",
        source="test",
    )

    assert created["ok"] is True
    assert created["accepted_count"] == 1
    assert created["players"][0]["name"] == "New Person"
    assert storage["players"][-1]["rating"] == 1400.0
    assert any(row["action_type"] == "create_match_uploader_players" for row in storage["admin_activity_log"])

    preview = build_admin_match_uploader_round_robin_preview(
        supabase,
        club_id="club",
        courts=[{"format_type": "4-Player", "player_names": ["Alex", "Blair", "Casey", "New Person"]}],
    )

    assert preview["missing_players"] == []
    assert preview["match_count"] == 3
    assert preview["courts"][0]["matches"][0]["t1_p1"] in {1, 2, 3, 5}


def test_match_uploader_new_player_requires_explicit_starting_rating(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    storage = fake_storage()
    reviewed_players = [{"name": "No Default", "starting_jupr": None}]

    with pytest.raises(ValueError, match="Starting JUPR is required"):
        create_admin_match_uploader_players(
            FakeSupabase(storage),
            club_id="club",
            actor_email="admin@example.com",
            actor_role="scorekeeper",
            players=reviewed_players,
            reviewed_fingerprint=match_uploader_player_batch_fingerprint(reviewed_players),
            idempotency_key="create-player-no-default",
            confirmation_text="CREATE PLAYERS",
            source="test",
        )

    assert "No Default" not in {row["name"] for row in storage["players"]}


def test_create_players_exact_retry_replays_without_duplicate(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    storage = fake_storage()
    supabase = FakeSupabase(storage)
    reviewed_players = [{"name": "New Person", "starting_jupr": 3.5}]
    kwargs = {
        "club_id": "club",
        "actor_email": "admin@example.com",
        "actor_role": "scorekeeper",
        "players": reviewed_players,
        "reviewed_fingerprint": match_uploader_player_batch_fingerprint(reviewed_players),
        "idempotency_key": "create-player-replay",
        "confirmation_text": "CREATE PLAYERS",
        "source": "test",
    }

    first = create_admin_match_uploader_players(supabase, **kwargs)
    replay = create_admin_match_uploader_players(supabase, **kwargs)

    assert replay["idempotent"] is True
    assert replay["players"] == first["players"]
    assert [row["name"] for row in storage["players"]].count("New Person") == 1


def test_create_players_rejects_changed_review_fingerprint(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    storage = fake_storage()

    with pytest.raises(ValueError, match="changed after review"):
        create_admin_match_uploader_players(
            FakeSupabase(storage),
            club_id="club",
            actor_email="admin@example.com",
            actor_role="scorekeeper",
            players=[{"name": "New Person", "starting_jupr": 3.5}],
            reviewed_fingerprint="0" * 64,
            idempotency_key="create-player-stale-review",
            confirmation_text="CREATE PLAYERS",
            source="test",
        )

    assert all(row["name"] != "New Person" for row in storage["players"])


def test_player_batch_fingerprint_matches_browser_utf8_and_lower_dedupe() -> None:
    assert match_uploader_player_batch_fingerprint(
        [{"name": "José Núñez", "starting_jupr": 3.5}]
    ) == "5888c5c97ca987a0a0421f184452a3b7488eb69df2b53c1941182adc30e85158"

    # Browser toLocaleLowerCase does not fold ß to ss. Both reviewed rows
    # must remain in the canonical list, matching the browser contract.
    assert match_uploader_player_batch_fingerprint(
        [
            {"name": "Straße", "starting_jupr": 3.5},
            {"name": "STRASSE", "starting_jupr": 4.0},
        ]
    ) != match_uploader_player_batch_fingerprint(
        [{"name": "Straße", "starting_jupr": 3.5}]
    )


def test_create_players_insert_timeout_marks_recovery_with_readback(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    storage = fake_storage()

    class TimeoutAfterInsertQuery(FakeQuery):
        def execute(self):
            if self.table_name == "players" and self.insert_payload is not None:
                super().execute()
                raise TimeoutError("response lost after commit")
            return super().execute()

    class TimeoutAfterInsertSupabase(FakeSupabase):
        def table(self, name):
            return TimeoutAfterInsertQuery(self.storage, name)

    reviewed = [{"name": "José Núñez", "starting_jupr": 3.5}]
    with pytest.raises(GuardedWriteRecoveryRequired, match="may have committed"):
        create_admin_match_uploader_players(
            TimeoutAfterInsertSupabase(storage),
            club_id="club",
            players=reviewed,
            actor_email="admin@example.com",
            actor_role="scorekeeper",
            reviewed_fingerprint=match_uploader_player_batch_fingerprint(reviewed),
            idempotency_key="create-player-timeout",
            confirmation_text="CREATE PLAYERS",
            source="test",
        )

    operation = storage["admin_guarded_operations"][0]
    assert operation["status"] == "recovery_required"
    assert operation["result_json"]["readback_verified"] is True
    assert operation["result_json"]["matched_count"] == 1
    assert [row["name"] for row in storage["players"]].count("José Núñez") == 1

    reconciled = reconcile_admin_match_uploader_player_operation(
        FakeSupabase(storage),
        club_id="club",
        operation_key="create-player-timeout",
        confirmation_text="RECONCILE PLAYER BATCH",
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        source="test",
    )

    assert reconciled["reconciled"] is True
    assert reconciled["players"][0]["name"] == "José Núñez"
    assert storage["admin_guarded_operations"][0]["status"] == "completed"
    assert any(row["action_type"] == "reconcile_match_uploader_player_batch" for row in storage["admin_activity_log"])


def test_player_batch_reconcile_accepts_intent_recorded_when_recovery_marker_was_lost(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    storage = fake_storage()
    reviewed = [{"name": "José Núñez", "starting_jupr": 3.5}]

    class TimeoutAndLostMarkerQuery(FakeQuery):
        def execute(self):
            if self.table_name == "players" and self.insert_payload is not None:
                super().execute()
                raise TimeoutError("response lost after commit")
            if self.table_name == "admin_guarded_operations" and self.update_payload and self.update_payload.get("status") == "recovery_required":
                raise TimeoutError("ledger marker lost")
            return super().execute()

    class TimeoutAndLostMarkerSupabase(FakeSupabase):
        def table(self, name):
            return TimeoutAndLostMarkerQuery(self.storage, name)

    with pytest.raises(GuardedWriteRecoveryRequired):
        create_admin_match_uploader_players(
            TimeoutAndLostMarkerSupabase(storage),
            club_id="club",
            players=reviewed,
            actor_email="admin@example.com",
            actor_role="scorekeeper",
            reviewed_fingerprint=match_uploader_player_batch_fingerprint(reviewed),
            idempotency_key="create-player-lost-marker",
            confirmation_text="CREATE PLAYERS",
            source="test",
        )

    assert storage["admin_guarded_operations"][0]["status"] == "intent_recorded"
    reconciled = reconcile_admin_match_uploader_player_operation(
        FakeSupabase(storage),
        club_id="club",
        operation_key="create-player-lost-marker",
        confirmation_text="RECONCILE PLAYER BATCH",
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        source="test",
    )
    assert reconciled["reconciled"] is True


def test_create_players_uses_browser_lower_membership_for_strasse(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    storage = fake_storage()
    storage["players"].append(
        {"club_id": "club", "id": 5, "name": "STRASSE", "rating": 1400, "wins": 0, "losses": 0, "matches_played": 0, "active": True}
    )
    reviewed = [{"name": "Straße", "starting_jupr": 3.5}]

    result = create_admin_match_uploader_players(
        FakeSupabase(storage),
        club_id="club",
        players=reviewed,
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        reviewed_fingerprint=match_uploader_player_batch_fingerprint(reviewed),
        idempotency_key="create-player-strasse",
        confirmation_text="CREATE PLAYERS",
        source="test",
    )

    assert result["created_count"] == 1
    assert result["accepted_count"] == 1
    assert result["players"][0]["name"] == "Straße"
    assert storage["admin_guarded_operations"][0]["status"] == "completed"


def test_submit_match_uploader_batch(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    calls = []

    def fake_submit_atomic_direct_matches(supabase, **kwargs):
        calls.append({"supabase": supabase, **kwargs})
        return {
            "ok": True,
            "match_write_committed": True,
            "submitted_count": len(kwargs["matches"]),
            "result": {
                "inserted": len(kwargs["matches"]),
                "skipped_incomplete": 0,
                "skipped_empty": 0,
                "skipped_unrated": 0,
            },
            "feedback": {},
            "operation": {"idempotent": False},
            "warnings": [],
        }

    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.load_data", fake_load_data)
    monkeypatch.setattr(
        "jupr_app.services.admin_match_uploader_service.submit_atomic_direct_matches",
        fake_submit_atomic_direct_matches,
    )
    storage = fake_storage()

    result = submit_admin_match_uploader_batch(
        FakeSupabase(storage),
        club_id="club",
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        idempotency_key="test:batch-1",
        match_format="singles",
        matches=[
            {
                "date": "2026-03-01",
                "league": "Open",
                "week_tag": "Week 1",
                "match_type": "Live Match",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
            }
        ],
        source="test",
    )

    assert result["ok"] is True
    assert result["submitted_count"] == 1
    assert result["result"]["inserted"] == 1
    assert calls[0]["matches"][0]["league"] == "Open"
    assert calls[0]["idempotency_key"] == "test:batch-1"
    assert calls[0]["match_format"] == "singles"


def test_submit_match_uploader_popup_context_name_creates_event(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    calls = []

    def fake_submit_atomic_direct_matches(supabase, **kwargs):
        calls.append({"supabase": supabase, **kwargs})
        return {
            "ok": True,
            "match_write_committed": True,
            "submitted_count": len(kwargs["matches"]),
            "result": {"inserted": len(kwargs["matches"])},
            "feedback": {},
            "operation": {"idempotent": False},
            "warnings": [],
        }

    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.load_data", fake_load_data)
    monkeypatch.setattr(
        "jupr_app.services.admin_match_uploader_service.submit_atomic_direct_matches",
        fake_submit_atomic_direct_matches,
    )
    storage = fake_storage()

    result = submit_admin_match_uploader_batch(
        FakeSupabase(storage),
        club_id="club",
        actor_email="admin@example.com",
        actor_role="scorekeeper",
        idempotency_key="test:popup-1",
        matches=[
            {
                "date": "2026-03-01",
                "league": "POPUP",
                "week_tag": "Event",
                "match_type": "PopUp",
                "is_popup": True,
                "context_type": "event",
                "context_name": "Saturday Social",
                "t1_p1": 1,
                "t1_p2": 2,
                "t2_p1": 3,
                "t2_p2": 4,
                "score_t1": 11,
                "score_t2": 7,
            }
        ],
        source="test",
    )

    assert result["ok"] is True
    assert storage["events"][0]["name"] == "Saturday Social"
    assert calls[0]["matches"][0]["context_type"] == "event"
    assert calls[0]["matches"][0]["context_id"] == str(storage["events"][0]["id"])
    assert "context_name" not in calls[0]["matches"][0]


def test_submit_match_uploader_rejects_reserved_league_live_context(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setattr(
        "jupr_app.services.admin_match_uploader_service.load_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("reserved contexts must fail before data loading")
        ),
    )

    with pytest.raises(
        PermissionError,
        match="reserved for the persisted League Live publisher",
    ):
        submit_admin_match_uploader_batch(
            FakeSupabase(fake_storage()),
            club_id="club",
            actor_email="admin@example.com",
            actor_role="scorekeeper",
            idempotency_key="test:reserved-context-1",
            matches=[
                {
                    "date": "2026-08-24",
                    "league": "Tuesday Ladder",
                    "match_type": "League Manager Live",
                    "context_type": "league_live_session",
                    "context_id": "09b05cd0-3b60-5f58-9834-808b4a7dc6a7",
                    "t1_p1": 1,
                    "t1_p2": 2,
                    "t2_p1": 3,
                    "t2_p2": 4,
                    "score_t1": 11,
                    "score_t2": 7,
                }
            ],
        )


def test_submit_match_uploader_rejects_empty(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER", "1")
    monkeypatch.setattr("jupr_app.services.admin_match_uploader_service.load_data", fake_load_data)

    try:
        submit_admin_match_uploader_batch(
            FakeSupabase(fake_storage()),
            club_id="club",
            actor_email="admin@example.com",
            actor_role="scorekeeper",
            idempotency_key="test:empty-1",
            matches=[{"t1_p1": 1}],
        )
    except ValueError as exc:
        assert "No valid match" in str(exc)
    else:
        raise AssertionError("Expected empty batch rejection")
