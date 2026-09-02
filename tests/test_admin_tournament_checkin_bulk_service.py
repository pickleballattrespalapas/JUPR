from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from jupr_app.services.admin_tournament_checkin_service import (
    CHECK_IN_BULK_MAX_UPDATES,
    CHECK_IN_BULK_RPC,
    StaleTournamentCheckInError,
    TournamentCheckInIdempotencyConflictError,
    bulk_update_admin_tournament_checkins,
)
from services.api.admin_tournament_checkin_routes import (
    AdminTournamentBulkCheckInRow,
    _dump_patch_model,
)


class BulkRpc:
    def __init__(self, client, name: str, params: dict):
        self.client = client
        self.name = name
        self.params = params

    def execute(self):
        self.client.calls.append((self.name, self.params))
        if self.client.error:
            raise RuntimeError(self.client.error)
        rows = [
            {
                "registration_id": patch["registration_id"],
                "registration_day_id": self.params["p_registration_day_id"],
                "attendance_status": patch.get("attendance_status", "EXPECTED"),
                "waiver_verified": patch.get("waiver_verified", False),
                "notes": patch.get("notes", "preserved"),
                "updated_by": self.params["p_actor_email"],
                "updated_at": "2026-08-26T14:00:00Z",
            }
            for patch in self.params["p_updates"]
        ]
        return SimpleNamespace(
            data={
                "ok": True,
                "mode": "tournament_registration_check_in_bulk_update",
                "operation_key": self.params["p_operation_key"],
                "updated_count": len(rows),
                "check_ins": rows,
                "idempotent_replay": self.client.replay,
            }
        )


class BulkClient:
    def __init__(self, *, error: str | None = None, replay: bool = False):
        self.error = error
        self.replay = replay
        self.calls: list[tuple[str, dict]] = []

    def rpc(self, name: str, params: dict):
        return BulkRpc(self, name, params)


def invoke(client: BulkClient, updates: list[dict]) -> dict:
    return bulk_update_admin_tournament_checkins(
        client,
        club_id="club-1",
        tournament_id="tour-1",
        registration_day_id="day-1",
        operation_key="00000000-0000-4000-8000-000000000201",
        updates=updates,
        actor_email="admin@example.com",
        actor_role="club_owner",
    )


def test_bulk_check_in_makes_one_rpc_with_sorted_sparse_patches(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    client = BulkClient()

    result = invoke(
        client,
        [
            {
                "registration_id": "reg-z",
                "expected_updated_at": None,
                "notes": None,
            },
            {
                "registration_id": "reg-a",
                "expected_updated_at": "2026-08-26T13:00:00Z",
                "attendance_status": "checked_in",
                "waiver_verified": True,
            },
        ],
    )

    assert len(client.calls) == 1
    rpc_name, params = client.calls[0]
    assert rpc_name == CHECK_IN_BULK_RPC
    assert params["p_updates"] == [
        {
            "registration_id": "reg-a",
            "expected_updated_at": "2026-08-26T13:00:00Z",
            "attendance_status": "CHECKED_IN",
            "waiver_verified": True,
        },
        {
            "registration_id": "reg-z",
            "expected_updated_at": None,
            "notes": None,
        },
    ]
    assert params["p_actor_email"] == "admin@example.com"
    assert result["updated_count"] == 2
    assert [row["registration_id"] for row in result["check_ins"]] == [
        "reg-a",
        "reg-z",
    ]
    assert result["idempotent_replay"] is False


def test_bulk_check_in_returns_durable_replay_evidence(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")

    result = invoke(
        BulkClient(replay=True),
        [
            {
                "registration_id": "reg-a",
                "expected_updated_at": None,
                "attendance_status": "ABSENT",
            }
        ],
    )

    assert result["idempotent_replay"] is True
    assert "Replayed" in result["message"]


def test_bulk_route_model_preserves_sparse_patch_and_explicit_note_clear() -> None:
    row = AdminTournamentBulkCheckInRow(
        registration_id="reg-a",
        expected_updated_at=None,
        notes=None,
    )

    assert _dump_patch_model(row) == {
        "registration_id": "reg-a",
        "expected_updated_at": None,
        "notes": None,
    }
    with pytest.raises(ValidationError):
        AdminTournamentBulkCheckInRow(
            registration_id="reg-a",
            expected_updated_at=None,
            attendance_status="EXPECTED",
            approved_substitute_player_id=10,
        )
    with pytest.raises(ValidationError):
        AdminTournamentBulkCheckInRow(
            registration_id="reg-a",
            attendance_status="EXPECTED",
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ([], "Select at least one"),
        (
            [
                {
                    "registration_id": "reg-a",
                    "expected_updated_at": None,
                    "attendance_status": "EXPECTED",
                },
                {
                    "registration_id": "reg-a",
                    "expected_updated_at": None,
                    "waiver_verified": True,
                },
            ],
            "only once",
        ),
        (
            [{"registration_id": "reg-a", "expected_updated_at": None}],
            "needs an attendance",
        ),
        (
            [
                {
                    "registration_id": "reg-a",
                    "attendance_status": "EXPECTED",
                }
            ],
            "expected updated-at",
        ),
        (
            [
                {
                    "registration_id": "reg-a",
                    "expected_updated_at": None,
                    "approved_substitute_player_id": 10,
                }
            ],
            "does not support substitutions",
        ),
    ],
)
def test_bulk_check_in_rejects_noncanonical_batches_before_rpc(
    monkeypatch, updates: list[dict], message: str
) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    client = BulkClient()

    with pytest.raises(ValueError, match=message):
        invoke(client, updates)

    assert client.calls == []


def test_bulk_check_in_enforces_safe_batch_limit_before_rpc(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")
    client = BulkClient()
    updates = [
        {
            "registration_id": f"reg-{index:03d}",
            "expected_updated_at": None,
            "attendance_status": "EXPECTED",
        }
        for index in range(CHECK_IN_BULK_MAX_UPDATES + 1)
    ]

    with pytest.raises(ValueError, match="at most 100"):
        invoke(client, updates)

    assert client.calls == []


@pytest.mark.parametrize(
    ("error", "exception_type"),
    [
        ("JUPR_CHECK_IN_BULK_STALE", StaleTournamentCheckInError),
        (
            "JUPR_CHECK_IN_BULK_IDEMPOTENCY_CONFLICT",
            TournamentCheckInIdempotencyConflictError,
        ),
        ("JUPR_CHECK_IN_BULK_ROSTER", ValueError),
    ],
)
def test_bulk_check_in_maps_database_conflicts(
    monkeypatch, error: str, exception_type: type[Exception]
) -> None:
    monkeypatch.setenv("JUPR_ENV", "local")

    with pytest.raises(exception_type):
        invoke(
            BulkClient(error=error),
            [
                {
                    "registration_id": "reg-a",
                    "expected_updated_at": None,
                    "attendance_status": "EXPECTED",
                }
            ],
        )
