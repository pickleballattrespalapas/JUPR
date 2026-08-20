from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.services.admin_league_manager_roster_batch_service import (
    update_admin_league_manager_roster_batch,
)


class _Rpc:
    def __init__(self, payload: dict, error: Exception | None = None):
        self.payload = payload
        self.error = error

    def execute(self):
        if self.error is not None:
            raise self.error
        return SimpleNamespace(data=self.payload)


class _Supabase:
    def __init__(
        self,
        *,
        payload: dict | None = None,
        error: Exception | None = None,
    ):
        self.rpc_calls: list[tuple[str, dict]] = []
        self.payload = payload or {
            "ok": True,
            "committed": True,
            "updated_count": 1,
        }
        self.error = error

    def table(self, _name):
        raise AssertionError("roster lifecycle must be checked atomically by the guarded RPC")

    def rpc(self, name, params):
        self.rpc_calls.append((name, dict(params)))
        return _Rpc(dict(self.payload), self.error)


def _update(supabase: _Supabase) -> dict:
    return update_admin_league_manager_roster_batch(
        supabase,
        club_id="club",
        league_name="Open",
        action="activate",
        player_ids=[1],
        starting_rating=3.5,
        idempotency_key="roster:test-1",
        actor_email="owner@example.com",
        actor_role="club_owner",
        confirmation_text="SAVE LEAGUE ROSTER BATCH",
    )


def test_paused_new_roster_batch_is_rejected_by_the_authoritative_v3(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENV", "test")
    supabase = _Supabase(
        error=RuntimeError(
            "JUPR_LEAGUE_ROSTER_BATCH_READ_ONLY: only draft and active "
            "league rosters are mutable."
        )
    )

    with pytest.raises(ValueError, match="read-only unless the league is draft"):
        _update(supabase)

    assert supabase.rpc_calls[0][0] == "admin_apply_league_roster_batch_atomic_v3"


def test_paused_idempotent_receipt_can_return_through_v3(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "test")
    supabase = _Supabase(
        payload={
            "ok": True,
            "committed": True,
            "updated_count": 1,
            "idempotent": True,
        }
    )

    result = _update(supabase)

    assert result["idempotent"] is True
    assert supabase.rpc_calls[0][0] == "admin_apply_league_roster_batch_atomic_v3"


def test_active_roster_batch_still_returns_its_atomic_receipt(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "test")
    supabase = _Supabase()

    result = _update(supabase)

    assert result["committed"] is True
    assert supabase.rpc_calls[0][0] == "admin_apply_league_roster_batch_atomic_v3"
