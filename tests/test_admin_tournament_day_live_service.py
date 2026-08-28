from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace
import uuid

import pytest

from jupr_app.services import admin_tournament_day_live_service as day_live
from jupr_app.services.admin_tournament_day_live_service import (
    build_admin_tournament_day_live_snapshot,
    execute_admin_tournament_day_live_command,
)
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
)


class FakeQuery:
    def __init__(self, client, table_name: str):
        self.client = client
        self.table_name = table_name
        self.filters: list[tuple[str, str, object]] = []
        self.order_key: str | None = None
        self.order_desc = False
        self.limit_value: int | None = None
        self.insert_payload = None
        self.update_payload = None
        self.delete_mode = False
        self.single_mode = False

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key: str, value: object):
        self.filters.append(("eq", key, value))
        return self

    def neq(self, key: str, value: object):
        self.filters.append(("neq", key, value))
        return self

    def in_(self, key: str, values):
        self.filters.append(("in", key, {str(value) for value in values or []}))
        return self

    def is_(self, key: str, value: object):
        self.filters.append(("is", key, value))
        return self

    def contains(self, key: str, value: object):
        self.filters.append(("contains", key, value))
        return self

    def order(self, key: str, desc: bool = False):
        self.order_key = key
        self.order_desc = bool(desc)
        return self

    def limit(self, value: int):
        self.limit_value = int(value)
        return self

    def insert(self, payload):
        self.insert_payload = deepcopy(payload)
        return self

    def update(self, payload):
        self.update_payload = deepcopy(payload)
        return self

    def delete(self):
        self.delete_mode = True
        return self

    def single(self):
        self.single_mode = True
        return self

    def maybe_single(self):
        self.single_mode = True
        return self

    def _matched(self) -> list[dict]:
        rows = list(self.client.tables.setdefault(self.table_name, []))
        for operation, key, expected in self.filters:
            if operation == "eq":
                rows = [row for row in rows if str(row.get(key)) == str(expected)]
            elif operation == "neq":
                rows = [row for row in rows if str(row.get(key)) != str(expected)]
            elif operation == "in":
                rows = [row for row in rows if str(row.get(key)) in expected]
            elif operation == "is":
                rows = [row for row in rows if row.get(key) is expected]
            elif operation == "contains":
                expected_values = set(expected if isinstance(expected, list) else [expected])
                rows = [
                    row
                    for row in rows
                    if expected_values.issubset(
                        set(row.get(key) if isinstance(row.get(key), list) else [])
                    )
                ]
        if self.order_key:
            rows.sort(
                key=lambda row: str(row.get(self.order_key) or ""),
                reverse=self.order_desc,
            )
        if self.limit_value is not None:
            rows = rows[: self.limit_value]
        return rows

    def execute(self):
        table = self.client.tables.setdefault(self.table_name, [])
        if self.insert_payload is not None:
            rows = (
                self.insert_payload
                if isinstance(self.insert_payload, list)
                else [self.insert_payload]
            )
            saved = [deepcopy(row) for row in rows]
            table.extend(saved)
            self.client.table_writes.append(("insert", self.table_name, saved))
            return SimpleNamespace(data=deepcopy(saved))
        matched = self._matched()
        if self.update_payload is not None:
            for row in matched:
                row.update(deepcopy(self.update_payload))
            self.client.table_writes.append(
                ("update", self.table_name, deepcopy(self.update_payload))
            )
            return SimpleNamespace(data=deepcopy(matched))
        if self.delete_mode:
            self.client.tables[self.table_name] = [
                row for row in table if row not in matched
            ]
            self.client.table_writes.append(
                ("delete", self.table_name, deepcopy(matched))
            )
            return SimpleNamespace(data=deepcopy(matched))
        data = deepcopy(matched)
        if self.single_mode:
            return SimpleNamespace(data=data[0] if data else None)
        return SimpleNamespace(data=data)


class FakeRpc:
    def __init__(self, client, name: str, params: dict):
        self.client = client
        self.name = name
        self.params = deepcopy(params)

    def execute(self):
        self.client.rpc_calls.append((self.name, deepcopy(self.params)))
        handler = self.client.rpc_handlers.get(self.name)
        if handler is None:
            raise AssertionError(f"Unexpected RPC {self.name}")
        result = handler(deepcopy(self.params))
        return SimpleNamespace(data=deepcopy(result))


class FakeSupabase:
    def __init__(self, tables: dict[str, list[dict]] | None = None):
        self.tables = deepcopy(tables or {})
        self.rpc_calls: list[tuple[str, dict]] = []
        self.rpc_handlers: dict[str, object] = {}
        self.table_writes: list[tuple[str, str, object]] = []

    def table(self, name: str):
        return FakeQuery(self, name)

    def rpc(self, name: str, params: dict):
        return FakeRpc(self, name, params)


def snapshot_tables() -> dict[str, list[dict]]:
    players = [
        {"id": player_id, "club_id": "club-1", "name": f"Player {player_id}", "active": True}
        for player_id in range(1, 9)
    ]
    registrations = [
        {
            "id": f"registration-{player_id}",
            "tournament_id": "tour-1",
            "player_id": player_id,
            "status": "CONFIRMED",
            "payment_status": "PAID",
            "payment_status": "paid",
            "display_name": f"Player {player_id}",
        }
        for player_id in range(1, 9)
    ]
    check_ins = [
        {
            "id": f"check-in-{player_id}",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "registration_id": f"registration-{player_id}",
            "attendance_status": "CHECKED_IN",
            "checked_in": True,
            "waiver_verified": True,
            "attendee_identity_key": f"player:{player_id}",
            "updated_at": "2026-08-17T08:00:00Z",
        }
        for player_id in range(1, 9)
    ]
    return {
        "tournaments": [
            {
                "id": "tour-1",
                "club_id": "club-1",
                "name": "Summer Classic",
                "status": "PUBLISHED",
                "start_date": "2026-08-20",
                "end_date": "2026-08-20",
                "updated_at": "2026-08-17T07:00:00Z",
            }
        ],
        "tournament_registration_days": [
            {
                "id": "day-1",
                "tournament_id": "tour-1",
                "label": "Thursday",
                "event_date": "2026-08-20",
                "enabled": True,
                "sort_order": 1,
                "court_count": 2,
                "court_labels": ["Center 1", "Center 2"],
                "available_court_ids": ["court-1", "court-2"],
                "court_open_time": "08:00",
                "court_close_time": "18:00",
                "updated_at": "2026-08-17T07:00:00Z",
            }
        ],
        "tournament_registration_settings": [
            {
                "id": "settings-1",
                "tournament_id": "tour-1",
                "timezone": "America/Chicago",
                "venue_courts_json": [
                    {"id": "court-1", "title": "Center 1"},
                    {"id": "court-2", "title": "Center 2"},
                ],
            }
        ],
        "tournament_event_options": [
            {
                "id": "event-a",
                "tournament_id": "tour-1",
                "registration_day_id": "day-1",
                "scheduled_day_ids": ["day-1"],
                "label": "Women's 3.5",
                "status": "ACTIVE",
                "enabled": True,
                "active": True,
                "enabled": True,
            },
            {
                "id": "event-b",
                "tournament_id": "tour-1",
                "registration_day_id": "day-1",
                "scheduled_day_ids": ["day-1"],
                "label": "Mixed 4.0",
                "status": "ACTIVE",
                "enabled": True,
                "active": True,
                "enabled": True,
            },
        ],
        "tournament_event_draws": [
            {
                "id": "draw-a",
                "tournament_id": "tour-1",
                "registration_day_id": "day-1",
                "event_option_id": "event-a",
                "name": "Women's 3.5 Draw",
                "status": "ACTIVE",
                "updated_at": "2026-08-17T08:00:00Z",
            },
            {
                "id": "draw-b",
                "tournament_id": "tour-1",
                "registration_day_id": "day-1",
                "event_option_id": "event-b",
                "name": "Mixed 4.0 Draw",
                "status": "ACTIVE",
                "updated_at": "2026-08-17T08:00:00Z",
            },
        ],
        "tournament_teams": [
            {
                "id": "team-a1",
                "tournament_id": "tour-1",
                "draw_id": "draw-a",
                "registration_day_id": "day-1",
                "event_option_id": "event-a",
                "team_number": 1,
                "player1_id": 1,
                "player2_id": 2,
                "updated_at": "2026-08-17T08:00:00Z",
            },
            {
                "id": "team-a2",
                "tournament_id": "tour-1",
                "draw_id": "draw-a",
                "registration_day_id": "day-1",
                "event_option_id": "event-a",
                "team_number": 2,
                "player1_id": 3,
                "player2_id": 4,
                "updated_at": "2026-08-17T08:00:00Z",
            },
            {
                "id": "team-b1",
                "tournament_id": "tour-1",
                "draw_id": "draw-b",
                "registration_day_id": "day-1",
                "event_option_id": "event-b",
                "team_number": 1,
                "player1_id": 1,
                "player2_id": 5,
                "updated_at": "2026-08-17T08:00:00Z",
            },
            {
                "id": "team-b2",
                "tournament_id": "tour-1",
                "draw_id": "draw-b",
                "registration_day_id": "day-1",
                "event_option_id": "event-b",
                "team_number": 2,
                "player1_id": 6,
                "player2_id": 7,
                "updated_at": "2026-08-17T08:00:00Z",
            },
        ],
        "tournament_games": [
            {
                "id": "game-a",
                "tournament_id": "tour-1",
                "draw_id": "draw-a",
                "registration_day_id": "day-1",
                "event_option_id": "event-a",
                "stage": "ROUND_ROBIN",
                "rr_round_number": 1,
                "rr_slot_number": 1,
                "team_a_id": "team-a1",
                "team_b_id": "team-a2",
                "score_a": None,
                "score_b": None,
                "winner_team_id": None,
                "finalized_at": None,
                "updated_at": "2026-08-17T08:00:00Z",
            },
            {
                "id": "game-b",
                "tournament_id": "tour-1",
                "draw_id": "draw-b",
                "registration_day_id": "day-1",
                "event_option_id": "event-b",
                "stage": "ROUND_ROBIN",
                "rr_round_number": 1,
                "rr_slot_number": 1,
                "team_a_id": "team-b1",
                "team_b_id": "team-b2",
                "score_a": None,
                "score_b": None,
                "winner_team_id": None,
                "finalized_at": None,
                "updated_at": "2026-08-17T08:00:00Z",
            },
        ],
        "players": players,
        "tournament_registrations": registrations,
        "tournament_registration_check_ins": check_ins,
        "tournament_day_live_runs": [],
        "tournament_day_live_draws": [],
        "tournament_day_live_courts": [],
        "tournament_day_live_queue": [],
        "tournament_day_live_participant_claims": [],
        "tournament_admin_operations": [],
        "admin_activity_log": [],
    }


def _workspace_snapshot(*, draw_b_state: str = "ACTIVE") -> dict:
    return {
        "ok": True,
        "mode": "tournament_day_live",
        "tournament": {"id": "tour-1", "name": "Summer Classic", "status": "PUBLISHED"},
        "day_scope": {
            "selected_day_id": "day-1",
            "selected_day": {"id": "day-1", "label": "Thursday"},
            "available_days": [{"id": "day-1", "label": "Thursday"}],
        },
        "day_run": {
            "id": "run-1",
            "registration_day_id": "day-1",
            "state": "ACTIVE",
            "version": "7",
            "updated_at": "2026-08-17T09:00:00Z",
        },
        "state_fingerprint": "a" * 64,
        "queue_version": "11",
        "summary": {
            "courts": 2,
            "available_courts": 2,
            "active_draws": 2,
            "eligible_games": 3,
            "held_games": 0,
            "completed_games": 0,
        },
        "draws": [
            {
                "id": "draw-a",
                "name": "Women's 3.5 Draw",
                "state": "ACTIVE",
                "activation_state": "ACTIVE",
                "version": "3",
                "total_games": 1,
                "finalized_games": 0,
                "queued_games": 1,
                "active_games": 0,
                "held_games": 0,
                "team_versions": [
                    {"id": "team-a1", "updated_at": "team-a1-v1"},
                    {"id": "team-a2", "updated_at": "team-a2-v1"},
                ],
                "source_game_versions": [
                    {"id": "game-a", "updated_at": "game-a-v1"}
                ],
                "readiness": {
                    "activate": {"ready": False, "blockers": []},
                    "pause": {"ready": True, "blockers": []},
                    "resume": {"ready": False, "blockers": []},
                    "generate_playoffs": {
                        "ready": True,
                        "blockers": [],
                        "allowed_advance_counts": [4],
                        "default_advance_count": None,
                    },
                    "podium": {"ready": False, "blockers": []},
                },
            },
            {
                "id": "draw-b",
                "name": "Mixed 4.0 Draw",
                "state": draw_b_state,
                "activation_state": draw_b_state,
                "version": "4",
                "total_games": 2,
                "finalized_games": 0,
                "queued_games": 2,
                "active_games": 0,
                "held_games": 0,
                "team_versions": [],
                "source_game_versions": [],
                "readiness": {
                    "activate": {"ready": draw_b_state == "INACTIVE", "blockers": []},
                    "pause": {"ready": draw_b_state == "ACTIVE", "blockers": []},
                    "resume": {"ready": draw_b_state == "PAUSED", "blockers": []},
                    "generate_playoffs": {
                        "ready": False,
                        "blockers": [],
                        "allowed_advance_counts": [],
                        "default_advance_count": None,
                    },
                    "podium": {"ready": False, "blockers": []},
                },
            },
        ],
        "courts": [
            {
                "id": "court-row-1",
                "label": "Center 1",
                "position": 1,
                "state": "AVAILABLE",
                "version": "2",
                "current_assignment": None,
            },
            {
                "id": "court-row-2",
                "label": "Center 2",
                "position": 2,
                "state": "AVAILABLE",
                "version": "2",
                "current_assignment": None,
            },
        ],
        "games": [],
        "eligible_queue": [],
        "held_games": [],
        "blocked_games": [],
        "operations": [],
        "readiness": {
            "activate_day": {"ready": False, "blockers": []},
            "auto_fill_courts": {"ready": True, "blockers": []},
        },
        "runtime": {"writes_enabled": True, "warnings": []},
        "warnings": [],
    }


def _preactivation_snapshot() -> dict:
    snapshot = _workspace_snapshot(draw_b_state="INACTIVE")
    for draw in snapshot["draws"]:
        draw["state"] = "INACTIVE"
        draw["activation_state"] = "INACTIVE"
        draw["readiness"]["activate"] = {"ready": True, "blockers": []}
        draw["readiness"]["pause"] = {"ready": False, "blockers": []}
    for position, court in enumerate(snapshot["courts"], start=1):
        # A preactivation snapshot projects configured inventory court keys;
        # durable live-court row IDs do not exist until activate_day commits.
        court["id"] = f"court-{position}"
    snapshot["day_run"] = {
        "id": "",
        "registration_day_id": "day-1",
        "state": "DRAFT",
        "version": "0",
        "updated_at": None,
    }
    snapshot["queue_version"] = "0"
    snapshot["readiness"]["activate_day"] = {"ready": True, "blockers": []}
    snapshot["readiness"]["auto_fill_courts"] = {
        "ready": False,
        "blockers": [{"code": "DAY_NOT_ACTIVE", "message": "Activate the day."}],
    }
    return snapshot


def _request(
    action: str,
    snapshot: dict,
    *,
    payload: dict | None = None,
    key: str | None = None,
    **expected_overrides,
) -> dict:
    confirmations = {
        "activate_day": "ACTIVATE DAY",
        "activate_draw": "ACTIVATE DRAW",
        "pause_draw": "PAUSE DRAW",
        "resume_draw": "RESUME DRAW",
        "auto_fill_courts": "AUTO FILL COURTS",
        "assign_next_court": "ASSIGN NEXT OPEN COURT",
        "assign_game_to_court": "ASSIGN GAME TO COURT",
        "requeue_game": "RETURN GAME TO QUEUE",
        "move_game_to_court": "MOVE GAME TO COURT",
        "score_and_release": "SAVE SCORE AND RELEASE COURT",
        "correct_completed_score": "CORRECT COMPLETED SCORE",
        "record_non_played_result": "RECORD NON-PLAYED RESULT",
        "generate_playoffs": "GENERATE PLAYOFFS",
        "close_day": "CLOSE TOURNAMENT DAY",
    }
    expected = {
        "day_run_version": str(snapshot["day_run"]["version"]),
        "state_fingerprint": snapshot["state_fingerprint"],
        "queue_version": snapshot["queue_version"],
        **expected_overrides,
    }
    return {
        "action": action,
        "client_idempotency_key": key or str(uuid.uuid4()),
        "confirmation_text": confirmations[action],
        "expected": expected,
        "payload": dict(payload or {}),
    }


def _execute(supabase, request: dict) -> dict:
    return execute_admin_tournament_day_live_command(
        supabase,
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
        request=request,
        actor_email="admin@example.com",
        actor_role="club_owner",
    )


def _install_command_harness(monkeypatch, supabase, snapshots):
    state = {
        "after_mutation": False,
        "runner_calls": [],
        "snapshots": snapshots if isinstance(snapshots, tuple) else (snapshots, snapshots),
    }

    def snapshot(_supabase, **_scope):
        selected = state["snapshots"][1 if state["after_mutation"] else 0]
        return deepcopy(selected)

    def guarded(**kwargs):
        state["runner_calls"].append(kwargs)
        if str(kwargs["current_state"]()) != str(kwargs["expected_state"]):
            raise StaleTournamentAdminStateError(
                "Tournament day data changed after it was loaded."
            )
        if kwargs.get("preflight") is not None:
            kwargs["preflight"]()
        result = kwargs["mutate"]()
        state["after_mutation"] = True
        return {
            "ok": True,
            "operation_key": "operation-1",
            "request_fingerprint": "request-fingerprint-1",
            "client_idempotency_key": kwargs.get("idempotency_key") or "",
            "status": "completed",
            "idempotent_replay": False,
            **dict(result or {}),
        }

    monkeypatch.setattr(day_live, "build_admin_tournament_day_live_snapshot", snapshot)
    monkeypatch.setattr(day_live, "run_tournament_admin_guarded_operation", guarded)
    monkeypatch.setattr(
        day_live, "require_tournament_admin_mutation_runtime", lambda *_args, **_kwargs: None
    )
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-live")
    monkeypatch.setenv(
        "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES", "1"
    )
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")
    return state


def test_draw_explicit_day_owns_multi_day_event_and_ambiguous_draw_fails_closed() -> None:
    event = {
        "id": "event-a",
        "registration_day_id": "day-1",
        "scheduled_day_ids": ["day-1", "day-2"],
    }
    owned = {
        "id": "draw-a",
        "event_option_id": "event-a",
        "registration_day_id": "day-1",
    }
    ambiguous = {
        "id": "draw-b",
        "event_option_id": "event-a",
        "registration_day_id": None,
    }

    assert day_live._draw_scheduled_for_day(owned, event, "day-1") is True
    assert day_live._draw_scheduled_for_day(owned, event, "day-2") is False
    assert day_live._draw_scheduled_for_day(ambiguous, event, "day-1") is False
    assert day_live._draw_scheduled_for_day(ambiguous, event, "day-2") is False


def test_snapshot_is_day_scoped_multi_draw_human_readable_and_stable() -> None:
    supabase = FakeSupabase(snapshot_tables())

    first = build_admin_tournament_day_live_snapshot(
        supabase,
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )
    second = build_admin_tournament_day_live_snapshot(
        supabase,
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )

    assert first["ok"] is True
    assert first["mode"] == "tournament_day_live"
    assert first["day_scope"]["selected_day_id"] == "day-1"
    assert first["day_run"] == {
        "id": "",
        "registration_day_id": "day-1",
        "state": "DRAFT",
        "version": "0",
        "updated_at": None,
    }
    assert {draw["id"] for draw in first["draws"]} == {"draw-a", "draw-b"}
    assert all(draw["activation_state"] == "INACTIVE" for draw in first["draws"])
    assert [court["label"] for court in first["courts"]] == ["Center 1", "Center 2"]
    assert all(court["state"] == "AVAILABLE" for court in first["courts"])
    assert len(first["games"]) == 2
    game = next(row for row in first["games"] if row["id"] == "game-a")
    assert game["team_a"]["participant_names"] == ["Player 1", "Player 2"]
    assert game["team_b"]["participant_names"] == ["Player 3", "Player 4"]
    assert len(first["state_fingerprint"]) == 64
    assert first["state_fingerprint"] == second["state_fingerprint"]
    assert first["queue_version"] == "0"


def test_check_in_waiver_and_payment_are_informational_to_live_day() -> None:
    baseline_tables = snapshot_tables()
    informational_tables = deepcopy(baseline_tables)
    informational_tables["tournament_registration_check_ins"] = [
        {
            **row,
            "attendance_status": "NOT_CHECKED_IN",
            "checked_in": False,
            "waiver_verified": False,
            "attendee_identity_key": None,
            "updated_at": "2026-08-17T09:00:00Z",
        }
        for row in informational_tables["tournament_registration_check_ins"]
    ]
    for registration in informational_tables["tournament_registrations"]:
        registration["payment_status"] = "UNPAID"
        registration["updated_at"] = "2026-08-17T09:00:00Z"
    informational_tables["tournament_commerce_orders"] = [
        {
            "id": f"commerce-{player_id}",
            "club_id": "club-1",
            "tournament_id": "tour-1",
            "registration_id": f"registration-{player_id}",
            "payment_status": "UNPAID",
            "status": "OPEN",
            "updated_at": "2026-08-17T09:00:00Z",
        }
        for player_id in range(1, 9)
    ]

    baseline = build_admin_tournament_day_live_snapshot(
        FakeSupabase(baseline_tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )
    informational = build_admin_tournament_day_live_snapshot(
        FakeSupabase(informational_tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )

    assert informational["state_fingerprint"] == baseline["state_fingerprint"]
    for draw in informational["draws"]:
        blocker_codes = {
            blocker["code"]
            for blocker in draw["readiness"]["activate"]["blockers"]
        }
        assert "PLAYER_NOT_READY" not in blocker_codes
        assert "PAYMENT_UNRESOLVED" not in blocker_codes


def test_active_queue_ignores_check_in_waiver_and_payment_but_not_registration() -> None:
    tables = snapshot_tables()
    tables["tournament_registration_check_ins"] = []
    for registration in tables["tournament_registrations"]:
        registration["payment_status"] = "UNPAID"
    tables["tournament_commerce_orders"] = []
    tables["tournament_day_live_runs"] = [
        {
            "id": "run-1",
            "club_id": "club-1",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "state": "ACTIVE",
            "version": 2,
            "queue_version": 3,
        }
    ]
    tables["tournament_day_live_draws"] = [
        {
            "id": "day-draw-a",
            "run_id": "run-1",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "draw_id": "draw-a",
            "state": "ACTIVE",
            "source_draw_updated_at": "2026-08-17T08:00:00Z",
            "version": 1,
        }
    ]
    tables["tournament_day_live_courts"] = [
        {
            "id": "court-row-1",
            "run_id": "run-1",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "court_key": "court-1",
            "label": "Center 1",
            "position": 1,
            "state": "OPEN",
            "version": 1,
        }
    ]
    tables["tournament_day_live_queue"] = [
        {
            "id": "queue-a",
            "run_id": "run-1",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "day_draw_id": "day-draw-a",
            "draw_id": "draw-a",
            "game_id": "game-a",
            "team_a_id": "team-a1",
            "team_b_id": "team-a2",
            "state": "WAITING",
            "priority": 1,
            "version": 1,
        }
    ]

    snapshot = build_admin_tournament_day_live_snapshot(
        FakeSupabase(tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )

    assert [row["game_id"] for row in snapshot["eligible_queue"]] == ["game-a"]
    assert snapshot["readiness"]["auto_fill_courts"]["ready"] is True

    structurally_invalid = deepcopy(tables)
    structurally_invalid["tournament_registrations"][0]["status"] = "CANCELLED"
    invalid_snapshot = build_admin_tournament_day_live_snapshot(
        FakeSupabase(structurally_invalid),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )

    assert invalid_snapshot["state_fingerprint"] != snapshot["state_fingerprint"]
    assert all(
        row["game_id"] != "game-a" for row in invalid_snapshot["eligible_queue"]
    )
    blocked = next(
        row for row in invalid_snapshot["blocked_games"] if row["game_id"] == "game-a"
    )
    assert blocked["reason"] == "REGISTRATION_AMBIGUOUS"


def test_activation_readiness_rejects_unsupported_incomplete_or_prebuilt_games() -> None:
    def activation_codes(tables: dict[str, list[dict]]) -> set[str]:
        snapshot = build_admin_tournament_day_live_snapshot(
            FakeSupabase(tables),
            club_id="club-1",
            tournament_id="tour-1",
            registration_day_id="day-1",
        )
        draw = next(row for row in snapshot["draws"] if row["id"] == "draw-a")
        return {
            blocker["code"]
            for blocker in draw["readiness"]["activate"]["blockers"]
        }

    unsupported = snapshot_tables()
    unsupported["tournament_games"][0]["stage"] = "EXHIBITION"
    assert "GAME_STAGE_UNSUPPORTED" in activation_codes(unsupported)

    incomplete_rr = snapshot_tables()
    incomplete_rr["tournament_games"][0]["team_b_id"] = None
    assert "ROUND_ROBIN_TEAMS_REQUIRED" in activation_codes(incomplete_rr)

    wrong_team_scope = snapshot_tables()
    wrong_team_scope["tournament_teams"][0]["event_option_id"] = "event-b"
    wrong_team_scope["tournament_teams"][1]["registration_day_id"] = None
    wrong_scope_snapshot = build_admin_tournament_day_live_snapshot(
        FakeSupabase(wrong_team_scope),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )
    wrong_scope_draw = next(
        row for row in wrong_scope_snapshot["draws"] if row["id"] == "draw-a"
    )
    wrong_scope_codes = {
        blocker["code"]
        for blocker in wrong_scope_draw["readiness"]["activate"]["blockers"]
    }
    assert {"TEAM_SCOPE_INVALID", "TEAM_DAY_SCOPE_INVALID"} <= wrong_scope_codes
    assert wrong_scope_draw["team_rows"] == []
    assert wrong_scope_draw["team_versions"] == []

    prebuilt_playoffs = snapshot_tables()
    source = prebuilt_playoffs["tournament_games"][0]
    source.update(
        {
            "stage": "PLAYOFF",
            "playoff_game_code": "P1",
            "playoff_round": "Semifinal",
            "team_a_source": {"seed": 1},
            "team_b_source": {"seed": 2},
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-a1",
            "loser_team_id": "team-a2",
            "finalized_at": "2026-08-17T10:00:00Z",
        }
    )
    prebuilt_playoffs["tournament_games"].append(
        {
            "id": "game-a-final",
            "tournament_id": "tour-1",
            "draw_id": "draw-a",
            "registration_day_id": "day-1",
            "event_option_id": "event-a",
            "stage": "PLAYOFF",
            "playoff_game_code": "P2",
            "playoff_round": "Final",
            "team_a_id": "team-a1",
            "team_b_id": None,
            "team_a_source": {"seed": 1},
            "team_b_source": {"winnerOf": "P1"},
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
            "updated_at": "2026-08-17T08:00:00Z",
        }
    )
    assert "PLAYOFFS_ALREADY_GENERATED" in activation_codes(prebuilt_playoffs)

    too_small = snapshot_tables()
    assert "PLAYOFF_FORMAT_UNAVAILABLE" in activation_codes(too_small)

    unplayed_teams = snapshot_tables()
    unplayed_teams["tournament_teams"].extend(
        [
            {
                "id": "team-a3",
                "tournament_id": "tour-1",
                "draw_id": "draw-a",
                "registration_day_id": "day-1",
                "event_option_id": "event-a",
                "team_number": 3,
                "player1_id": 5,
                "player2_id": 6,
                "updated_at": "2026-08-17T08:00:00Z",
            },
            {
                "id": "team-a4",
                "tournament_id": "tour-1",
                "draw_id": "draw-a",
                "registration_day_id": "day-1",
                "event_option_id": "event-a",
                "team_number": 4,
                "player1_id": 7,
                "player2_id": 8,
                "updated_at": "2026-08-17T08:00:00Z",
            },
        ]
    )
    unplayed_codes = activation_codes(unplayed_teams)
    assert "PLAYOFF_FORMAT_UNAVAILABLE" not in unplayed_codes
    assert "ROUND_ROBIN_ROSTER_MISMATCH" in unplayed_codes

    duplicate_player = snapshot_tables()
    duplicate_player["tournament_teams"][1]["player1_id"] = 1
    assert "DRAW_ROSTER_PLAYER_DUPLICATE" in activation_codes(duplicate_player)

    cyclic_playoffs = deepcopy(prebuilt_playoffs)
    cyclic_playoffs["tournament_games"][0].update(
        {
            "team_a_id": None,
            "team_b_id": None,
            "team_a_source": {"winnerOf": "P2"},
            "team_b_source": {"loserOf": "P2"},
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
        }
    )
    assert "PLAYOFFS_ALREADY_GENERATED" in activation_codes(cyclic_playoffs)


def test_preactivation_fingerprint_binds_the_exact_reviewed_court_plan() -> None:
    before_tables = snapshot_tables()
    after_tables = deepcopy(before_tables)
    after_tables["tournament_registration_settings"][0]["venue_courts_json"][0][
        "title"
    ] = "Renamed Center Court"

    before = build_admin_tournament_day_live_snapshot(
        FakeSupabase(before_tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )
    after = build_admin_tournament_day_live_snapshot(
        FakeSupabase(after_tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )

    assert before["courts"][0]["label"] == "Center 1"
    assert after["courts"][0]["label"] == "Renamed Center Court"
    assert before["state_fingerprint"] != after["state_fingerprint"]


def test_active_draw_scope_drift_stays_visible_and_blocks_refill_human_readably() -> None:
    tables = snapshot_tables()
    tables["tournament_day_live_runs"] = [
        {
            "id": "run-1",
            "club_id": "club-1",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "state": "ACTIVE",
            "version": 2,
            "queue_version": 3,
        }
    ]
    tables["tournament_day_live_draws"] = [
        {
            "id": "day-draw-a",
            "run_id": "run-1",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "draw_id": "draw-a",
            "state": "ACTIVE",
            "source_draw_updated_at": "2026-08-17T07:59:00Z",
            "version": 1,
        }
    ]
    tables["tournament_day_live_courts"] = [
        {
            "id": "court-row-1",
            "run_id": "run-1",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "court_key": "court-1",
            "label": "Center 1",
            "position": 1,
            "state": "OPEN",
            "version": 1,
        }
    ]
    tables["tournament_day_live_queue"] = [
        {
            "id": "queue-a",
            "run_id": "run-1",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "day_draw_id": "day-draw-a",
            "draw_id": "draw-a",
            "game_id": "game-a",
            "team_a_id": "team-a1",
            "team_b_id": "team-a2",
            "state": "WAITING",
            "priority": 1,
            "version": 1,
        }
    ]
    draw = tables["tournament_event_draws"][0]
    draw["hidden_from_primary_ops"] = True
    event = tables["tournament_event_options"][0]
    event["enabled"] = False
    event["scheduled_day_ids"] = ["day-2"]

    snapshot = build_admin_tournament_day_live_snapshot(
        FakeSupabase(tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )

    active = next(row for row in snapshot["draws"] if row["id"] == "draw-a")
    assignment_codes = {
        blocker["code"]
        for blocker in active["readiness"]["assignments"]["blockers"]
    }
    assert {"DRAW_UNAVAILABLE", "DRAW_UNSCHEDULED", "DRAW_SOURCE_CHANGED"} <= (
        assignment_codes
    )
    assert active["readiness"]["pause"]["ready"] is True
    assert all(row["game_id"] != "game-a" for row in snapshot["eligible_queue"])
    blocked = next(row for row in snapshot["blocked_games"] if row["game_id"] == "game-a")
    assert "New court assignments are stopped" in blocked["note"]

    tables["tournament_day_live_draws"][0]["state"] = "PAUSED"
    paused_snapshot = build_admin_tournament_day_live_snapshot(
        FakeSupabase(tables),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )
    paused = next(row for row in paused_snapshot["draws"] if row["id"] == "draw-a")
    assert paused["readiness"]["resume"]["ready"] is False
    assert any(
        blocker["code"] == "DRAW_UNSCHEDULED"
        for blocker in paused["readiness"]["resume"]["blockers"]
    )


def test_day_activation_and_close_require_supported_represented_draws() -> None:
    no_supported = snapshot_tables()
    for event in no_supported["tournament_event_options"]:
        event["enabled"] = False
    draft_snapshot = build_admin_tournament_day_live_snapshot(
        FakeSupabase(no_supported),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )
    assert any(
        blocker["code"] == "NO_SUPPORTED_DRAWS"
        for blocker in draft_snapshot["readiness"]["activate_day"]["blockers"]
    )

    active = snapshot_tables()
    active["tournament_day_live_runs"] = [
        {
            "id": "run-1",
            "club_id": "club-1",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "state": "ACTIVE",
            "version": 1,
            "queue_version": 1,
        }
    ]
    empty_close = build_admin_tournament_day_live_snapshot(
        FakeSupabase(active),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )
    empty_codes = {
        blocker["code"]
        for blocker in empty_close["readiness"]["close_day"]["blockers"]
    }
    assert {"NO_ACTIVATED_DRAWS", "SCHEDULED_DRAWS_NOT_ACTIVATED"} <= empty_codes

    active["tournament_day_live_draws"] = [
        {
            "id": "day-draw-a",
            "run_id": "run-1",
            "tournament_id": "tour-1",
            "registration_day_id": "day-1",
            "draw_id": "draw-a",
            "state": "ACTIVE",
            "source_draw_updated_at": "2026-08-17T08:00:00Z",
            "version": 1,
        }
    ]
    partial_close = build_admin_tournament_day_live_snapshot(
        FakeSupabase(active),
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )
    partial_codes = {
        blocker["code"]
        for blocker in partial_close["readiness"]["close_day"]["blockers"]
    }
    assert "NO_ACTIVATED_DRAWS" not in partial_codes
    assert "SCHEDULED_DRAWS_NOT_ACTIVATED" in partial_codes


def test_snapshot_excludes_only_the_current_guarded_intent_from_preflight() -> None:
    tables = snapshot_tables()
    tables["tournament_admin_operations"] = [
        {
            "operation_key": "current-operation",
            "club_id": "club-1",
            "surface": "tournament_live",
            "entity_type": "tournament_registration_day",
            "entity_id": "tour-1:day-1",
            "lock_scope": "tournament:tour-1:day:day-1",
            "action": "tournament_day_live_generate_playoffs",
            "status": "intent",
            "updated_at": "2026-08-17T10:00:00Z",
        }
    ]
    supabase = FakeSupabase(tables)

    visible = build_admin_tournament_day_live_snapshot(
        supabase,
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
    )
    internal = build_admin_tournament_day_live_snapshot(
        supabase,
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
        exclude_operation_key="current-operation",
    )

    assert [row["operation_key"] for row in visible["operations"]] == [
        "current-operation"
    ]
    assert internal["operations"] == []
    assert any(
        blocker["code"] == "OPERATION_UNSETTLED"
        for blocker in visible["draws"][0]["readiness"]["generate_playoffs"]["blockers"]
    )
    assert not any(
        blocker["code"] == "OPERATION_UNSETTLED"
        for blocker in internal["draws"][0]["readiness"]["generate_playoffs"]["blockers"]
    )


def test_activate_day_initializes_only_the_day_run_and_courts(monkeypatch) -> None:
    before = _preactivation_snapshot()
    after = _workspace_snapshot()
    supabase = FakeSupabase()
    harness = _install_command_harness(monkeypatch, supabase, (before, after))
    supabase.rpc_handlers["admin_activate_tournament_day_live_cas"] = lambda params: {
        "ok": True,
        "run_id": "run-1",
        "assignments": [],
    }
    request = _request("activate_day", before, payload={})

    result = _execute(supabase, request)

    assert [name for name, _params in supabase.rpc_calls] == [
        "admin_activate_tournament_day_live_cas"
    ]
    params = supabase.rpc_calls[0][1]
    assert params["p_club_id"] == "club-1"
    assert params["p_tournament_id"] == "tour-1"
    assert params["p_registration_day_id"] == "day-1"
    assert "p_draw_plan" not in params
    assert str(params["p_expected_run_version"]) == "0"
    assert params["p_activation_evidence"] == {
        "courts": [
            {"court_key": "court-1", "label": "Center 1", "position": 1},
            {"court_key": "court-2", "label": "Center 2", "position": 2},
        ]
    }
    assert params["p_operation_key"]
    assert params["p_request_fingerprint"]
    runner = harness["runner_calls"][0]
    assert runner["surface"] == "tournament_live"
    assert runner["lock_scope"] == "tournament:tour-1:day:day-1"
    assert runner["payload"]["activation_evidence"] == params[
        "p_activation_evidence"
    ]
    assert result["command"]["action"] == "activate_day"
    assert result["operation"]["status"] == "completed"
    assert result["snapshot"]["day_run"]["state"] == "ACTIVE"


@pytest.mark.parametrize(
    ("action", "before_state", "target_state", "rpc_action"),
    [
        ("activate_draw", "INACTIVE", "ACTIVE", "ACTIVATE"),
        ("pause_draw", "ACTIVE", "PAUSED", "PAUSE"),
        ("resume_draw", "PAUSED", "ACTIVE", "RESUME"),
    ],
)
def test_draw_lifecycle_transition_is_one_day_fenced_rpc(
    monkeypatch,
    action: str,
    before_state: str,
    target_state: str,
    rpc_action: str,
) -> None:
    before = _workspace_snapshot(draw_b_state=before_state)
    after = _workspace_snapshot(draw_b_state=target_state)
    supabase = FakeSupabase()
    _install_command_harness(monkeypatch, supabase, (before, after))
    supabase.rpc_handlers["admin_transition_tournament_day_draw_cas"] = lambda _params: {
        "ok": True
    }
    request = _request(
        action,
        before,
        payload={"draw_id": "draw-b"},
        draw_version="4",
    )

    result = _execute(supabase, request)

    assert len(supabase.rpc_calls) == 1
    name, params = supabase.rpc_calls[0]
    assert name == "admin_transition_tournament_day_draw_cas"
    assert params["p_draw_id"] == "draw-b"
    assert params["p_action"] == rpc_action
    assert str(params["p_expected_queue_version"]) == "11"
    if action == "activate_draw":
        assert params["p_expected_draw_updated_at"]
    else:
        assert str(params["p_expected_day_draw_version"]) == "4"
    assert result["snapshot"]["draws"][1]["activation_state"] == target_state


def test_auto_fill_uses_one_shared_queue_and_respects_cross_draw_player_claims(
    monkeypatch,
) -> None:
    before = _workspace_snapshot()
    before["eligible_queue"] = [
        {"game_id": "game-a", "draw_id": "draw-a", "position": 1, "state": "WAITING", "version": "1", "blockers": []},
        {"game_id": "game-b", "draw_id": "draw-b", "position": 2, "state": "WAITING", "version": "1", "blockers": []},
        {"game_id": "game-c", "draw_id": "draw-b", "position": 3, "state": "WAITING", "version": "1", "blockers": []},
    ]
    after = deepcopy(before)
    after["queue_version"] = "12"
    after["courts"][0]["current_assignment"] = {
        "id": "queue-a",
        "game_id": "game-a",
        "state": "ON_COURT",
        "version": "2",
    }
    after["courts"][0]["state"] = "ON_COURT"
    after["courts"][1]["current_assignment"] = {
        "id": "queue-c",
        "game_id": "game-c",
        "state": "ON_COURT",
        "version": "2",
    }
    after["courts"][1]["state"] = "ON_COURT"
    after["eligible_queue"] = []
    after["held_games"] = []
    after["blocked_games"] = [
        {
            "game_id": "game-b",
            "draw_id": "draw-b",
            "state": "BLOCKED",
            "reason": "PLAYER_BUSY",
            "version": "2",
            "blockers": [{"code": "PLAYER_ALREADY_CLAIMED", "message": "Player 1 is already assigned."}],
        }
    ]
    supabase = FakeSupabase()
    _install_command_harness(monkeypatch, supabase, (before, after))
    supabase.rpc_handlers["admin_fill_tournament_day_courts_cas"] = lambda _params: {
        "ok": True,
        "assigned_game_ids": ["game-a", "game-c"],
        "blocked_game_ids": ["game-b"],
    }

    result = _execute(supabase, _request("auto_fill_courts", before))

    assert [name for name, _params in supabase.rpc_calls] == [
        "admin_fill_tournament_day_courts_cas"
    ]
    params = supabase.rpc_calls[0][1]
    assert params["p_registration_day_id"] == "day-1"
    assert "p_draw_id" not in params
    assert str(params["p_expected_queue_version"]) == "11"
    assert {
        court["current_assignment"]["game_id"]
        for court in result["snapshot"]["courts"]
    } == {"game-a", "game-c"}
    assert result["snapshot"]["blocked_games"][0]["game_id"] == "game-b"
    assert result["snapshot"]["blocked_games"][0]["reason"] == "PLAYER_BUSY"


@pytest.mark.parametrize(
    ("action", "rpc_action", "selected_court_id", "selected_court_version"),
    [
        ("assign_next_court", "NEXT_OPEN", None, None),
        ("assign_game_to_court", "SELECTED", "court-row-2", "2"),
    ],
)
def test_operator_assigns_one_queued_game_to_next_or_selected_court(
    monkeypatch,
    action: str,
    rpc_action: str,
    selected_court_id: str | None,
    selected_court_version: str | None,
) -> None:
    before = _workspace_snapshot()
    before["eligible_queue"] = [
        {
            "game_id": "game-a",
            "draw_id": "draw-a",
            "position": 1,
            "state": "WAITING",
            "version": "5",
            "blockers": [],
        }
    ]
    before["games"] = [
        {
            "id": "game-a",
            "draw_id": "draw-a",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team-a1",
            "team_b_id": "team-a2",
            "version": "game-a-v1",
            "queue_entry_version": "5",
        }
    ]
    assigned_court_index = 1 if selected_court_id else 0
    after = deepcopy(before)
    after["queue_version"] = "12"
    after["eligible_queue"] = []
    after["courts"][assigned_court_index]["state"] = "ON_COURT"
    after["courts"][assigned_court_index]["current_assignment"] = {
        "id": "queue-a",
        "game_id": "game-a",
        "state": "ON_COURT",
        "version": "6",
    }
    supabase = FakeSupabase()
    _install_command_harness(monkeypatch, supabase, (before, after))
    supabase.rpc_handlers["admin_assign_tournament_day_game_cas"] = lambda _params: {
        "ok": True,
        "assignments": [{"game_id": "game-a"}],
    }
    payload = {"game_id": "game-a"}
    expected = {
        "game_version": "game-a-v1",
        "queue_entry_version": "5",
    }
    if selected_court_id:
        payload["court_id"] = selected_court_id
        expected["court_version"] = selected_court_version

    result = _execute(
        supabase,
        _request(action, before, payload=payload, **expected),
    )

    assert [name for name, _params in supabase.rpc_calls] == [
        "admin_assign_tournament_day_game_cas"
    ]
    params = supabase.rpc_calls[0][1]
    assert params["p_action"] == rpc_action
    assert params["p_game_id"] == "game-a"
    assert params["p_court_id"] == selected_court_id
    assert params["p_expected_queue_entry_version"] == 5
    assert params["p_expected_game_updated_at"] == "game-a-v1"
    assert (
        str(params["p_expected_court_version"])
        if params["p_expected_court_version"] is not None
        else None
    ) == selected_court_version
    assert result["snapshot"]["eligible_queue"] == []


@pytest.mark.parametrize(
    ("action", "rpc_action", "target_court_id", "target_court_version"),
    [
        ("requeue_game", "REQUEUE", None, None),
        ("move_game_to_court", "MOVE", "court-row-2", "2"),
    ],
)
def test_operator_requeues_or_moves_an_exact_on_court_assignment(
    monkeypatch,
    action: str,
    rpc_action: str,
    target_court_id: str | None,
    target_court_version: str | None,
) -> None:
    before = _workspace_snapshot()
    before["courts"][0].update(
        {
            "state": "ON_COURT",
            "current_assignment": {
                "id": "queue-a",
                "game_id": "game-a",
                "state": "ON_COURT",
                "version": "5",
            },
        }
    )
    before["games"] = [
        {
            "id": "game-a",
            "draw_id": "draw-a",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team-a1",
            "team_b_id": "team-a2",
            "version": "game-a-v1",
            "queue_entry_version": "5",
            "court_id": "court-row-1",
        }
    ]
    after = deepcopy(before)
    after["queue_version"] = "12"
    after["courts"][0].update({"state": "AVAILABLE", "current_assignment": None})
    if action == "requeue_game":
        after["eligible_queue"] = [
            {
                "game_id": "game-a",
                "draw_id": "draw-a",
                "position": 1,
                "state": "WAITING",
                "version": "6",
                "blockers": [],
            }
        ]
    else:
        after["courts"][1].update(
            {
                "state": "ON_COURT",
                "current_assignment": {
                    "id": "queue-a",
                    "game_id": "game-a",
                    "state": "ON_COURT",
                    "version": "6",
                },
            }
        )
    supabase = FakeSupabase()
    _install_command_harness(monkeypatch, supabase, (before, after))
    supabase.rpc_handlers["admin_reassign_tournament_day_game_cas"] = lambda _params: {
        "ok": True,
        "action": rpc_action,
    }
    payload = {"game_id": "game-a"}
    expected = {
        "game_version": "game-a-v1",
        "queue_entry_version": "5",
        "court_version": "2",
    }
    if target_court_id:
        payload["court_id"] = target_court_id
        expected["target_court_version"] = target_court_version

    result = _execute(
        supabase,
        _request(action, before, payload=payload, **expected),
    )

    assert [name for name, _params in supabase.rpc_calls] == [
        "admin_reassign_tournament_day_game_cas"
    ]
    params = supabase.rpc_calls[0][1]
    assert params["p_action"] == rpc_action
    assert params["p_game_id"] == "game-a"
    assert params["p_target_court_id"] == target_court_id
    assert params["p_expected_queue_entry_version"] == 5
    assert params["p_expected_source_court_version"] == 2
    assert (
        str(params["p_expected_target_court_version"])
        if params["p_expected_target_court_version"] is not None
        else None
    ) == target_court_version
    if action == "requeue_game":
        assert result["snapshot"]["eligible_queue"][0]["game_id"] == "game-a"
    else:
        assert result["snapshot"]["courts"][1]["current_assignment"]["game_id"] == "game-a"


def test_score_release_leaves_the_next_matchup_queued(monkeypatch) -> None:
    before = _workspace_snapshot()
    before["courts"][0]["current_assignment"] = {
        "id": "queue-a",
        "game_id": "game-a",
        "state": "ON_COURT",
        "version": "5",
    }
    before["courts"][0]["state"] = "ON_COURT"
    before["held_games"] = []
    before["eligible_queue"] = [
        {"game_id": "game-c", "draw_id": "draw-b", "position": 1, "state": "WAITING", "version": "3", "blockers": []}
    ]
    before["games"] = [
        {
            "id": "game-a",
            "draw_id": "draw-a",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team-a1",
            "team_b_id": "team-a2",
            "version": "game-a-v1",
        }
    ]
    after = deepcopy(before)
    after["queue_version"] = "12"
    after["held_games"] = []
    after["courts"][0]["current_assignment"] = None
    after["courts"][0]["state"] = "AVAILABLE"
    after["summary"]["completed_games"] = 1
    supabase = FakeSupabase()
    _install_command_harness(monkeypatch, supabase, (before, after))
    supabase.rpc_handlers["admin_score_release_tournament_day_game_cas"] = lambda _params: {
        "ok": True,
        "completed_game_id": "game-a",
        "released_court_id": "court-row-1",
        "assignments": [],
    }
    request = _request(
        "score_and_release",
        before,
        payload={"game_id": "game-a", "score_a": 11, "score_b": 7},
        draw_version="3",
        game_version="game-a-v1",
        court_version="2",
    )

    result = _execute(supabase, request)

    assert [name for name, _params in supabase.rpc_calls] == [
        "admin_score_release_tournament_day_game_cas"
    ]
    params = supabase.rpc_calls[0][1]
    assert params["p_game_id"] == "game-a"
    assert params["p_game_patch"]["score_a"] == 11
    assert params["p_game_patch"]["score_b"] == 7
    assert str(params["p_expected_game_updated_at"]) == "game-a-v1"
    assert str(params["p_expected_court_version"]) == "2"
    assert str(params["p_expected_queue_version"]) == "11"
    assert result["snapshot"]["courts"][0]["current_assignment"] is None
    assert result["snapshot"]["eligible_queue"][0]["game_id"] == "game-c"
    assert result["snapshot"]["summary"]["completed_games"] == 1


def test_unusual_score_requires_acknowledgement_before_atomic_score_rpc(monkeypatch) -> None:
    snapshot = _workspace_snapshot()
    snapshot["courts"][0].update(
        {
            "state": "ON_COURT",
            "current_assignment": {
                "id": "queue-a",
                "game_id": "game-a",
                "state": "ON_COURT",
                "version": "5",
            },
        }
    )
    snapshot["games"] = [
        {
            "id": "game-a",
            "draw_id": "draw-a",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team-a1",
            "team_b_id": "team-a2",
            "team_a": {"team_id": "team-a1"},
            "team_b": {"team_id": "team-a2"},
            "scoring": {"format": "GAME_TO_11", "target": 11, "win_by_two": True},
            "version": "game-a-v1",
        }
    ]
    supabase = FakeSupabase()
    _install_command_harness(monkeypatch, supabase, snapshot)
    request = _request(
        "score_and_release",
        snapshot,
        payload={
            "game_id": "game-a",
            "score_a": 76,
            "score_b": 11,
            "unusual_score_acknowledgement": False,
        },
        draw_version="3",
        game_version="game-a-v1",
        court_version="2",
    )
    with pytest.raises(ValueError, match="explicit acknowledgement"):
        _execute(supabase, request)
    assert supabase.rpc_calls == []


def test_explicit_missing_scoring_configuration_never_uses_legacy_default() -> None:
    with pytest.raises(ValueError, match="missing or unsupported"):
        day_live._score_review(
            {"id": "game-a", "scoring": {"format": None, "blocker": "missing"}},
            11,
            7,
            acknowledged=False,
        )

    assert day_live._score_review(
        {"id": "legacy-game"},
        11,
        7,
        acknowledged=False,
    )["scoring_format"] == "GAME_TO_11"


def test_non_played_result_uses_atomic_queue_cas_and_progression_evidence(monkeypatch) -> None:
    before = _workspace_snapshot()
    before["games"] = [
        {
            "id": "game-a",
            "draw_id": "draw-a",
            "state": "BLOCKED",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team-a1",
            "team_b_id": "team-a2",
            "team_a": {"team_id": "team-a1", "name": "Alpha", "participant_names": []},
            "team_b": {"team_id": "team-a2", "name": "Bravo", "participant_names": []},
            "scoring": {"format": "GAME_TO_11", "target": 11, "win_by_two": True},
            "version": "game-a-v1",
            "queue_entry_version": "6",
            "court_id": None,
        }
    ]
    after = deepcopy(before)
    after["games"][0].update(
        {
            "state": "COMPLETED",
            "score_a": 11,
            "score_b": 0,
            "winner_team_id": "team-a1",
            "result_type": "NO_SHOW",
            "result_note": "Bravo did not report by the published grace deadline.",
        }
    )
    after["summary"]["completed_games"] = 1
    supabase = FakeSupabase()
    state = _install_command_harness(monkeypatch, supabase, (before, after))
    supabase.rpc_handlers["admin_record_non_played_tournament_day_game_cas"] = lambda _params: {
        "ok": True,
        "non_played_result": True,
        "rating_publish_eligible": False,
    }
    request = _request(
        "record_non_played_result",
        before,
        payload={
            "game_id": "game-a",
            "result_type": "NO_SHOW",
            "winner_team_id": "team-a1",
            "result_note": "Bravo did not report by the published grace deadline.",
        },
        game_version="game-a-v1",
        queue_entry_version="6",
    )
    result = _execute(supabase, request)

    assert [name for name, _params in supabase.rpc_calls] == [
        "admin_record_non_played_tournament_day_game_cas"
    ]
    params = supabase.rpc_calls[0][1]
    assert params["p_expected_queue_entry_version"] == 6
    assert params["p_expected_court_version"] is None
    assert params["p_result_type"] == "NO_SHOW"
    assert params["p_game_patch"]["score_a"] == 11
    assert params["p_game_patch"]["score_b"] == 0
    assert params["p_dependency_updates"] == []
    guarded_payload = state["runner_calls"][0]["payload"]
    assert guarded_payload["score_evidence"]["outcome"]["rating_publish_eligible"] is False
    assert result["snapshot"]["games"][0]["result_type"] == "NO_SHOW"


def test_score_evidence_resolves_only_the_selected_draw_dependencies() -> None:
    snapshot = _workspace_snapshot()
    snapshot["draws"][0].update(
        {
            "source_updated_at": "draw-a-v1",
            "source_game_versions": [
                {"id": "draw-a-p1", "updated_at": "draw-a-p1-v1"},
                {"id": "draw-a-final", "updated_at": "draw-a-final-v1"},
            ],
        }
    )
    snapshot["draws"][1].update(
        {
            "source_updated_at": "draw-b-v1",
            "source_game_versions": [
                {"id": "draw-b-p1", "updated_at": "draw-b-p1-v1"},
                {"id": "draw-b-final", "updated_at": "draw-b-final-v1"},
            ],
        }
    )
    snapshot["games"] = [
        {
            "id": "draw-a-p1",
            "draw_id": "draw-a",
            "stage": "PLAYOFF",
            "playoff_game_code": "P1",
            "team_a_id": "team-a1",
            "team_b_id": "team-a2",
            "team_a_source": {"seed": 1},
            "team_b_source": {"seed": 2},
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
        },
        {
            "id": "draw-a-final",
            "draw_id": "draw-a",
            "stage": "PLAYOFF",
            "playoff_game_code": "P2",
            "team_a_id": None,
            "team_b_id": "team-a2",
            "team_a_source": {"winnerOf": "P1"},
            "team_b_source": {"seed": 2},
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
        },
        {
            "id": "draw-b-p1",
            "draw_id": "draw-b",
            "stage": "PLAYOFF",
            "playoff_game_code": "P1",
            "team_a_id": "team-b1",
            "team_b_id": "team-b2",
            "team_a_source": {"seed": 1},
            "team_b_source": {"seed": 2},
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
        },
        {
            "id": "draw-b-final",
            "draw_id": "draw-b",
            "stage": "PLAYOFF",
            "playoff_game_code": "P2",
            "team_a_id": None,
            "team_b_id": "team-b2",
            "team_a_source": {"winnerOf": "P1"},
            "team_b_source": {"seed": 2},
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
        },
    ]

    evidence = day_live._score_evidence(
        snapshot,
        {"game_id": "draw-a-p1", "score_a": 11, "score_b": 7},
    )

    assert [row["id"] for row in evidence["dependency_updates"]] == [
        "draw-a-final"
    ]
    assert evidence["dependency_updates"][0]["team_a_id"] == "team-a1"
    assert evidence["dependency_updates"][0]["expected_updated_at"] == (
        "draw-a-final-v1"
    )


def test_completed_round_robin_correction_uses_day_fenced_cas(monkeypatch) -> None:
    before = _workspace_snapshot()
    before["games"] = [
        {
            "id": "game-a",
            "draw_id": "draw-a",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team-a1",
            "team_b_id": "team-a2",
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-a1",
            "loser_team_id": "team-a2",
            "finalized_at": "2026-08-17T10:00:00Z",
            "version": "game-a-v1",
            "correction_readiness": {
                "ready": True,
                "blockers": [],
                "confirmation": "CORRECT COMPLETED SCORE",
            },
        }
    ]
    before["draws"][0]["source_updated_at"] = "draw-a-v1"
    before["draws"][0]["source_game_versions"] = [
        {"id": "game-a", "updated_at": "game-a-v1"}
    ]
    after = deepcopy(before)
    after["queue_version"] = "12"
    after["games"][0].update(
        {
            "score_a": 6,
            "score_b": 11,
            "winner_team_id": "team-a2",
            "loser_team_id": "team-a1",
            "version": "game-a-v2",
        }
    )
    supabase = FakeSupabase()
    _install_command_harness(monkeypatch, supabase, (before, after))
    supabase.rpc_handlers[
        "admin_correct_completed_tournament_day_game_cas"
    ] = lambda _params: {
        "ok": True,
        "corrected_game_id": "game-a",
        "corrected_completed_score": True,
    }
    request = _request(
        "correct_completed_score",
        before,
        payload={"game_id": "game-a", "score_a": 6, "score_b": 11},
        draw_version="3",
        game_version="game-a-v1",
    )

    result = _execute(supabase, request)

    assert [name for name, _params in supabase.rpc_calls] == [
        "admin_correct_completed_tournament_day_game_cas"
    ]
    params = supabase.rpc_calls[0][1]
    assert params["p_game_id"] == "game-a"
    assert params["p_expected_day_draw_version"] == 3
    assert params["p_expected_game_updated_at"] == "game-a-v1"
    assert params["p_expected_draw_updated_at"] == "draw-a-v1"
    assert params["p_dependency_updates"] == []
    assert params["p_game_patch"]["winner_team_id"] == "team-a2"
    assert "p_expected_court_version" not in params
    assert result["snapshot"]["games"][0]["score_b"] == 11


def test_completed_score_correction_surfaces_playoff_reset_blocker() -> None:
    snapshot = _workspace_snapshot()
    snapshot["games"] = [
        {
            "id": "game-a",
            "draw_id": "draw-a",
            "stage": "ROUND_ROBIN",
            "version": "game-a-v1",
            "correction_readiness": {
                "ready": False,
                "blockers": [
                    {
                        "code": "PLAYOFF_RESET_REQUIRED",
                        "message": "Reset playoffs first.",
                    }
                ],
            },
        }
    ]
    request = _request(
        "correct_completed_score",
        snapshot,
        payload={"game_id": "game-a", "score_a": 11, "score_b": 7},
        draw_version="3",
        game_version="game-a-v1",
    )

    with pytest.raises(ValueError, match="PLAYOFF_RESET_REQUIRED"):
        day_live._preflight(
            snapshot,
            request["action"],
            request["expected"],
            request["payload"],
        )


def test_generate_playoffs_uses_only_the_day_fenced_wrapper(monkeypatch) -> None:
    before = _workspace_snapshot()
    after = deepcopy(before)
    after["draws"][0]["total_games"] = 3
    after["draws"][0]["queued_games"] = 2
    after["eligible_queue"] = [
        {"game_id": "semi-1", "draw_id": "draw-a", "position": 1, "state": "WAITING", "version": "1", "blockers": []},
        {"game_id": "semi-2", "draw_id": "draw-a", "position": 2, "state": "WAITING", "version": "1", "blockers": []},
    ]
    supabase = FakeSupabase()
    _install_command_harness(monkeypatch, supabase, (before, after))
    monkeypatch.setattr(
        day_live,
        "_playoff_rows",
        lambda *_args, **_kwargs: [
            {
                "id": "semi-1",
                "draw_id": "draw-a",
                "stage": "PLAYOFF",
                "playoff_game_code": "SF1",
            }
        ],
    )
    supabase.rpc_handlers["admin_generate_tournament_day_playoffs_cas"] = lambda _params: {
        "ok": True,
        "inserted_game_ids": ["semi-1", "semi-2"],
    }
    request = _request(
        "generate_playoffs",
        before,
        payload={"draw_id": "draw-a", "advance_count": 4},
        draw_version="3",
    )

    result = _execute(supabase, request)

    assert [name for name, _params in supabase.rpc_calls] == [
        "admin_generate_tournament_day_playoffs_cas"
    ]
    params = supabase.rpc_calls[0][1]
    assert params["p_draw_id"] == "draw-a"
    assert params["p_advance_count"] == 4
    assert str(params["p_expected_queue_version"]) == "11"
    assert params["p_expected_team_versions"] == before["draws"][0]["team_versions"]
    assert params["p_expected_source_game_versions"] == before["draws"][0]["source_game_versions"]
    assert result["snapshot"]["draws"][0]["total_games"] == 3


def test_playoff_builder_preserves_team_numbers_and_exact_day_event_scope() -> None:
    team_rows = [
        {"id": "z-team", "team_number": 1, "seed": None},
        {"id": "a-team", "team_number": 2, "seed": None},
        {"id": "m-team", "team_number": 3, "seed": None},
        {"id": "b-team", "team_number": 4, "seed": None},
    ]
    games = []
    game_number = 0
    for left in range(len(team_rows)):
        for right in range(left + 1, len(team_rows)):
            game_number += 1
            team_a = team_rows[left]["id"]
            team_b = team_rows[right]["id"]
            games.append(
                {
                    "id": f"rr-{game_number}",
                    "draw_id": "draw-a",
                    "stage": "ROUND_ROBIN",
                    "team_a_id": team_a,
                    "team_b_id": team_b,
                    "score_a": 11,
                    "score_b": 5,
                    "winner_team_id": team_a,
                    "loser_team_id": team_b,
                    "finalized_at": "2026-08-17T10:00:00Z",
                }
            )

    rows = day_live._playoff_rows(
        {
            "day_scope": {"selected_day_id": "day-1"},
            "games": games,
        },
        {
            "id": "draw-a",
            "event_option_id": "event-a",
            # UUID/text sort order intentionally differs from team_number.
            "team_rows": team_rows,
        },
        "tour-1",
        advance_count=4,
    )

    assert len(rows) == 4
    assert all(row["registration_day_id"] == "day-1" for row in rows)
    assert all(row["event_option_id"] == "event-a" for row in rows)
    assert all(row["draw_id"] == "draw-a" for row in rows)
    semifinal = next(row for row in rows if row["playoff_game_code"] == "P1")
    assert semifinal["team_a_id"] == "z-team"
    assert semifinal["team_b_id"] == "b-team"


def test_close_podium_must_match_exact_final_and_bronze_results() -> None:
    playoff_games = [
        {
            "id": "final",
            "stage": "PLAYOFF",
            "playoff_round": "Final",
            "team_a_id": "team-1",
            "team_b_id": "team-2",
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "finalized_at": "2026-08-17T12:00:00Z",
        },
        {
            "id": "bronze",
            "stage": "PLAYOFF",
            "playoff_round": "Bronze",
            "team_a_id": "team-3",
            "team_b_id": "team-4",
            "score_a": 11,
            "score_b": 8,
            "winner_team_id": "team-3",
            "loser_team_id": "team-4",
            "finalized_at": "2026-08-17T12:00:00Z",
        },
    ]
    correct = [
        {"placement": 1, "team_id": "team-1"},
        {"placement": 2, "team_id": "team-2"},
        {"placement": 3, "team_id": "team-3"},
    ]
    wrong = [
        {"placement": 1, "team_id": "team-2"},
        {"placement": 2, "team_id": "team-1"},
        {"placement": 3, "team_id": "team-3"},
    ]

    assert day_live._podium_matches_playoff_results(playoff_games, correct) is True
    assert day_live._podium_matches_playoff_results(playoff_games, wrong) is False


def test_stale_review_or_wrong_confirmation_has_zero_rpc_or_table_write(monkeypatch) -> None:
    snapshot = _workspace_snapshot()
    supabase = FakeSupabase()
    harness = _install_command_harness(monkeypatch, supabase, snapshot)
    supabase.rpc_handlers["admin_fill_tournament_day_courts_cas"] = lambda _params: {
        "ok": True
    }
    stale = _request("auto_fill_courts", snapshot)
    stale["expected"]["state_fingerprint"] = "0" * 64

    with pytest.raises((StaleTournamentAdminStateError, ValueError), match="changed|stale"):
        _execute(supabase, stale)

    wrong_confirmation = _request("auto_fill_courts", snapshot)
    wrong_confirmation["confirmation_text"] = "fill"
    with pytest.raises(ValueError, match="AUTO FILL COURTS"):
        _execute(supabase, wrong_confirmation)

    wrong_version = _request("auto_fill_courts", snapshot)
    wrong_version["expected"]["day_run_version"] = "6"
    with pytest.raises((StaleTournamentAdminStateError, ValueError), match="version|changed|stale"):
        _execute(supabase, wrong_version)

    assert supabase.rpc_calls == []
    assert supabase.table_writes == []
    assert len(harness["runner_calls"]) <= 2


def test_exact_idempotent_replay_does_not_repeat_rpc_and_changed_payload_conflicts(
    monkeypatch,
) -> None:
    snapshot = _workspace_snapshot()
    supabase = FakeSupabase()
    state = _install_command_harness(monkeypatch, supabase, snapshot)
    supabase.rpc_handlers["admin_fill_tournament_day_courts_cas"] = lambda _params: {
        "ok": True,
        "assigned_game_ids": [],
    }
    seen: dict[str, tuple[str, dict]] = {}

    def idempotent_guarded(**kwargs):
        key = str(kwargs.get("idempotency_key") or "")
        fingerprint = json.dumps(kwargs.get("payload") or {}, sort_keys=True)
        existing = seen.get(key)
        if existing is not None:
            if existing[0] != fingerprint:
                raise ValueError("idempotency key was already used for a different request")
            return {**deepcopy(existing[1]), "idempotent_replay": True}
        if str(kwargs["current_state"]()) != str(kwargs["expected_state"]):
            raise StaleTournamentAdminStateError("Tournament day data changed")
        if kwargs.get("preflight") is not None:
            kwargs["preflight"]()
        result = {
            "ok": True,
            "operation_key": "operation-1",
            "request_fingerprint": fingerprint,
            "client_idempotency_key": key,
            "status": "completed",
            "idempotent_replay": False,
            **dict(kwargs["mutate"]() or {}),
        }
        seen[key] = (fingerprint, deepcopy(result))
        return result

    monkeypatch.setattr(day_live, "run_tournament_admin_guarded_operation", idempotent_guarded)
    key = str(uuid.uuid4())
    request = _request("auto_fill_courts", snapshot, key=key)

    first = _execute(supabase, request)
    replay = _execute(supabase, deepcopy(request))
    changed = _request(
        "pause_draw",
        snapshot,
        payload={"draw_id": "draw-a"},
        key=key,
        draw_version="3",
    )
    with pytest.raises(ValueError, match="already used for a different"):
        _execute(supabase, changed)

    assert [name for name, _params in supabase.rpc_calls] == [
        "admin_fill_tournament_day_courts_cas"
    ]
    assert first["operation"]["idempotent_replay"] is False
    assert replay["operation"]["idempotent_replay"] is True
    assert replay["operation"]["operation_key"] == first["operation"]["operation_key"]
    assert state["after_mutation"] is False


def test_scorekeeper_can_reconcile_correction_but_not_schedule_operation(
    monkeypatch,
) -> None:
    supabase = FakeSupabase()
    base_operation = {
        "operation_key": "operation-1",
        "surface": "tournament_live",
        "entity_type": "tournament_registration_day",
        "entity_id": "tour-1:day-1",
        "lock_scope": "tournament:tour-1:day:day-1",
        "action": "tournament_day_live_correct_completed_score",
    }
    monkeypatch.setattr(
        day_live, "require_tournament_admin_mutation_runtime", lambda *_args: None
    )
    monkeypatch.setattr(
        day_live,
        "get_tournament_admin_operation_record",
        lambda *_args, **_kwargs: dict(base_operation),
    )
    monkeypatch.setattr(
        day_live,
        "reconcile_tournament_admin_guarded_operation",
        lambda *_args, **_kwargs: {
            "operation_key": "operation-1",
            "recovery_disposition": "completed",
        },
    )
    monkeypatch.setattr(
        day_live,
        "build_admin_tournament_day_live_snapshot",
        lambda *_args, **_kwargs: _workspace_snapshot(),
    )

    result = day_live.reconcile_admin_tournament_day_live_operation(
        supabase,
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
        operation_key="operation-1",
        confirmation_text="RECONCILE DAY OPERATIONS",
        actor_email="scorekeeper@example.com",
        actor_role="scorekeeper",
    )
    assert result["operation"]["status"] == "completed"

    base_operation["action"] = "tournament_day_live_auto_fill_courts"
    with pytest.raises(PermissionError, match="insufficient permission"):
        day_live.reconcile_admin_tournament_day_live_operation(
            supabase,
            club_id="club-1",
            tournament_id="tour-1",
            registration_day_id="day-1",
            operation_key="operation-1",
            confirmation_text="RECONCILE DAY OPERATIONS",
            actor_email="scorekeeper@example.com",
            actor_role="scorekeeper",
        )
