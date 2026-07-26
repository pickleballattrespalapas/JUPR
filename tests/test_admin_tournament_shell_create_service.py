from __future__ import annotations

import pytest

from jupr_app.services.admin_tournament_shell_create_service import (
    create_admin_tournament_shell,
    get_tournament_shell_creation_state_fingerprint,
    reconcile_admin_tournament_shell_creation,
    tournament_shell_absent_state_fingerprint,
)
from tests.test_api_contract_admin_tournament_setup import FakeSupabase


TOURNAMENT_ID = "5f5d9096-f74e-40f0-83cb-1597485fc6b2"


def _create(supabase: FakeSupabase, **overrides):
    values = {
        "club_id": "club",
        "tournament_id": TOURNAMENT_ID,
        "name": "Winter Open",
        "start_date": "2026-12-05",
        "end_date": "2026-12-06",
        "actor_email": "owner@example.com",
        "actor_role": "club_owner",
        "confirmation_text": "CREATE TOURNAMENT",
    }
    values.update(overrides)
    return create_admin_tournament_shell(supabase, **values)


def test_shell_create_preflight_validates_without_writing(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    supabase = FakeSupabase()

    result = _create(supabase, dry_run=True)

    assert result["dry_run"] is True
    assert result["write_count"] == 0
    assert result["tournament"]["status"] == "DRAFT"
    assert len(supabase.storage["tournaments"]) == 1
    assert supabase.storage["admin_activity_log"] == []


def test_shell_create_inserts_one_draft_and_audits(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_REQUIRE_API_AUDIT_LOG", "1")
    supabase = FakeSupabase()
    absent = tournament_shell_absent_state_fingerprint(
        club_id="club",
        tournament_id=TOURNAMENT_ID,
    )

    result = _create(supabase)

    created = next(
        row for row in supabase.storage["tournaments"] if row["id"] == TOURNAMENT_ID
    )
    assert result["tournament"]["name"] == "Winter Open"
    assert created["status"] == "DRAFT"
    assert created["team_count"] == 4
    assert "December 2026" in created["event_tags"]["date_tags"]
    assert (
        get_tournament_shell_creation_state_fingerprint(
            supabase,
            club_id="club",
            tournament_id=TOURNAMENT_ID,
        )
        != absent
    )
    assert [
        row["action_type"] for row in supabase.storage["admin_activity_log"]
    ] == ["tournament_setup_shell_create"]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"confirmation_text": "CREATE"}, "CREATE TOURNAMENT"),
        ({"name": "   "}, "name is required"),
        ({"tournament_id": "not-a-uuid"}, "valid UUID"),
        (
            {"start_date": "2026-12-07", "end_date": "2026-12-06"},
            "cannot be before",
        ),
        ({"start_date": "12/05/2026"}, "YYYY-MM-DD"),
    ],
)
def test_shell_create_refuses_invalid_requests(
    monkeypatch,
    overrides,
    message,
) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    supabase = FakeSupabase()

    with pytest.raises(ValueError, match=message):
        _create(supabase, **overrides)

    assert len(supabase.storage["tournaments"]) == 1


def test_shell_create_exact_readback_reconciles_without_reinserting(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    supabase = FakeSupabase()
    _create(supabase)

    reconciled = reconcile_admin_tournament_shell_creation(
        supabase,
        club_id="club",
        tournament_id=TOURNAMENT_ID,
        name="Winter Open",
        start_date="2026-12-05",
        end_date="2026-12-06",
    )
    mismatch = reconcile_admin_tournament_shell_creation(
        supabase,
        club_id="club",
        tournament_id=TOURNAMENT_ID,
        name="Different tournament",
        start_date="2026-12-05",
        end_date="2026-12-06",
    )

    assert reconciled and reconciled["tournament"]["id"] == TOURNAMENT_ID
    assert mismatch is None
    assert len(supabase.storage["tournaments"]) == 2
