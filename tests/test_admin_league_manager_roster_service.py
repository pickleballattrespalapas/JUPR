from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jupr_app.services.admin_league_manager_roster_service import (
    update_admin_league_manager_roster_membership,
)


class _ReadQuery:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.rows = [
            row for row in self.rows if str(row.get(key)) == str(value)
        ]
        return self

    def limit(self, value):
        self.rows = self.rows[: int(value)]
        return self

    def execute(self):
        return SimpleNamespace(data=[dict(row) for row in self.rows])


class _ReadOnlySupabase:
    def __init__(self):
        self.league_ratings: list[dict] = []

    def table(self, name):
        assert name == "league_ratings"
        return _ReadQuery([dict(row) for row in self.league_ratings])


def _update(supabase: _ReadOnlySupabase, **overrides):
    payload = {
        "club_id": "club",
        "league_name": "Open",
        "player_id": 7,
        "action": "activate",
        "starting_rating": 4.0,
        "actor_email": "owner@example.com",
        "actor_role": "club_owner",
        "confirmation_text": "SAVE ROSTER",
    }
    payload.update(overrides)
    return update_admin_league_manager_roster_membership(supabase, **payload)


def test_legacy_membership_is_a_one_player_atomic_batch_shim(monkeypatch):
    supabase = _ReadOnlySupabase()
    calls: list[dict] = []

    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_roster_service."
        "is_admin_league_manager_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_roster_service."
        "get_admin_league_manager_detail",
        lambda *_args, **_kwargs: {"league_roster_count": 1},
    )

    def _atomic_batch(client, **kwargs):
        assert client is supabase
        calls.append(dict(kwargs))
        exact_row = {
            "club_id": "club",
            "league_name": "Open",
            "player_id": 7,
            "rating": 1600,
            "starting_rating": 1600,
            "wins": 0,
            "losses": 0,
            "matches_played": 0,
            "is_active": True,
            "inactive_at": None,
        }
        supabase.league_ratings[:] = [dict(exact_row)]
        return {
            "ok": True,
            "committed": True,
            "updated_count": 1,
            "operation_id": "operation-1",
            "idempotent": False,
            "league_ratings": [dict(exact_row)],
        }

    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_roster_service."
        "update_admin_league_manager_roster_batch",
        _atomic_batch,
    )

    result = _update(supabase, idempotency_key="legacy:stable-retry")

    assert calls == [
        {
            "club_id": "club",
            "league_name": "Open",
            "action": "activate",
            "player_ids": [7],
            "starting_rating": 4.0,
            "idempotency_key": "legacy:stable-retry",
            "actor_email": "owner@example.com",
            "actor_role": "club_owner",
            "confirmation_text": "SAVE LEAGUE ROSTER BATCH",
            "source": "next_league_manager_roster_update",
        }
    ]
    assert result["mode"] == "league_manager_roster_membership_update"
    assert result["league_rating"]["rating"] == 1600
    assert result["operation_id"] == "operation-1"
    assert result["warnings"] == []


def test_legacy_membership_generates_one_shot_key_and_keeps_confirmation(
    monkeypatch,
):
    supabase = _ReadOnlySupabase()
    calls: list[dict] = []
    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_roster_service."
        "is_admin_league_manager_enabled",
        lambda: True,
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_roster_service."
        "get_admin_league_manager_detail",
        lambda *_args, **_kwargs: {},
    )

    def _atomic_batch(_client, **kwargs):
        calls.append(dict(kwargs))
        row = {
            "club_id": "club",
            "league_name": "Open",
            "player_id": 7,
            "rating": 1600,
        }
        return {
            "ok": True,
            "committed": True,
            "updated_count": 1,
            "league_ratings": [row],
        }

    monkeypatch.setattr(
        "jupr_app.services.admin_league_manager_roster_service."
        "update_admin_league_manager_roster_batch",
        _atomic_batch,
    )

    result = _update(supabase, confirmation_text=" save roster ")

    assert calls[0]["idempotency_key"].startswith("legacy-roster:")
    assert len(calls[0]["idempotency_key"]) <= 160
    assert result["ok"] is True

    with pytest.raises(ValueError, match="SAVE ROSTER"):
        _update(supabase, confirmation_text="SAVE")
    assert len(calls) == 1


def test_legacy_roster_service_contains_no_direct_mutation_or_compensation():
    source = Path(
        "jupr_app/services/admin_league_manager_roster_service.py"
    ).read_text(encoding="utf-8")

    assert "update_admin_league_manager_roster_batch(" in source
    assert "_rollback_roster_membership" not in source
    assert "write_admin_activity_log" not in source
    for mutation in ('.insert(', '.update(', '.delete('):
        assert mutation not in source
