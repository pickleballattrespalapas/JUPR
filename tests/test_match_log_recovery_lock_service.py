from __future__ import annotations

import pytest

from jupr_app.services.match_log_recovery_lock_service import (
    MATCH_EDIT_KIND,
    MATCH_EXCLUSION_KIND,
    MatchLogRecoveryLocked,
    enforce_match_log_recovery_lock,
    list_open_match_log_recovery_locks,
)
from tests.test_admin_match_log_service import FakeSupabase


def _supabase(
    *,
    edit_operations: list[dict] | None = None,
    exclusion_operations: list[dict] | None = None,
) -> FakeSupabase:
    return FakeSupabase(
        {
            "match_edit_operations": list(edit_operations or []),
            "match_exclusion_operations": list(exclusion_operations or []),
        }
    )


def test_recovery_lock_is_club_scoped() -> None:
    supabase = _supabase(
        edit_operations=[
            {
                "id": "edit-other",
                "club_id": "other-club",
                "status": "recovery_required",
                "created_at": "2026-07-27T10:00:00Z",
            }
        ]
    )

    assert enforce_match_log_recovery_lock(supabase, club_id="club") is None


def test_open_edit_blocks_ordinary_write_and_allows_exact_recovery() -> None:
    supabase = _supabase(
        edit_operations=[
            {
                "id": "edit-1",
                "club_id": "club",
                "status": "pending_replay",
                "replay_job_id": "job-1",
                "created_at": "2026-07-27T10:00:00Z",
            }
        ]
    )

    with pytest.raises(MatchLogRecoveryLocked) as exc_info:
        enforce_match_log_recovery_lock(supabase, club_id="club")
    assert exc_info.value.lock.operation_kind == MATCH_EDIT_KIND
    assert exc_info.value.lock.operation_id == "edit-1"

    lock = enforce_match_log_recovery_lock(
        supabase,
        club_id="club",
        recovery_kind=MATCH_EDIT_KIND,
        recovery_operation_id="edit-1",
    )
    assert lock is not None
    assert lock.replay_job_id == "job-1"


def test_exclusion_lock_only_allows_its_exact_recovery() -> None:
    supabase = _supabase(
        exclusion_operations=[
            {
                "id": "exclude-1",
                "club_id": "club",
                "status": "pending_badge_reconcile",
                "replay_job_id": "job-1",
                "recovery_stage": "badge_reconcile",
                "created_at": "2026-07-27T10:00:00Z",
            }
        ]
    )

    lock = enforce_match_log_recovery_lock(
        supabase,
        club_id="club",
        recovery_kind=MATCH_EXCLUSION_KIND,
        recovery_operation_id="exclude-1",
    )
    assert lock is not None
    assert lock.recovery_stage == "badge_reconcile"

    with pytest.raises(MatchLogRecoveryLocked):
        enforce_match_log_recovery_lock(
            supabase,
            club_id="club",
            recovery_kind=MATCH_EDIT_KIND,
            recovery_operation_id="exclude-1",
        )


def test_multiple_open_ledgers_fail_closed_as_ambiguous() -> None:
    supabase = _supabase(
        edit_operations=[
            {
                "id": "edit-1",
                "club_id": "club",
                "status": "recovery_required",
                "created_at": "2026-07-27T10:00:00Z",
            }
        ],
        exclusion_operations=[
            {
                "id": "exclude-1",
                "club_id": "club",
                "status": "pending_badge_reconcile",
                "created_at": "2026-07-27T10:01:00Z",
            }
        ],
    )

    assert len(
        list_open_match_log_recovery_locks(supabase, club_id="club")
    ) == 2
    with pytest.raises(MatchLogRecoveryLocked) as exc_info:
        enforce_match_log_recovery_lock(
            supabase,
            club_id="club",
            recovery_kind=MATCH_EDIT_KIND,
            recovery_operation_id="edit-1",
        )
    assert exc_info.value.code == "MATCH_LOG_RECOVERY_LOCK_AMBIGUOUS"
