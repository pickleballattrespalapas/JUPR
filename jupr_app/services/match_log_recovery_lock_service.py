from __future__ import annotations

from dataclasses import dataclass
from typing import Any


MATCH_EDIT_KIND = "match_edit"
MATCH_EXCLUSION_KIND = "match_exclusion"

_EDIT_OPEN_STATUSES = ("pending_replay", "recovery_required")
_EXCLUSION_OPEN_STATUSES = (
    "pending_replay",
    "pending_badge_reconcile",
    "recovery_required",
)


@dataclass(frozen=True)
class MatchLogRecoveryLock:
    operation_kind: str
    operation_id: str
    operation_status: str
    replay_job_id: str | None = None
    recovery_stage: str | None = None
    created_at: str | None = None

    def as_detail(
        self,
        *,
        code: str = "MATCH_LOG_RECOVERY_LOCKED",
    ) -> dict[str, Any]:
        return {
            "code": code,
            "operation_kind": self.operation_kind,
            "operation_id": self.operation_id,
            "operation_status": self.operation_status,
            "replay_job_id": self.replay_job_id,
            "recovery_stage": self.recovery_stage,
            "message": (
                "Complete the exact Match Log recovery operation before "
                "starting another write or replay."
            ),
        }


class MatchLogRecoveryLocked(RuntimeError):
    def __init__(
        self,
        lock: MatchLogRecoveryLock,
        *,
        code: str = "MATCH_LOG_RECOVERY_LOCKED",
    ):
        self.lock = lock
        self.code = str(code)
        super().__init__(lock.as_detail(code=self.code)["message"])


class MatchLogRecoveryLockUnavailable(RuntimeError):
    pass


def _operation_rows(
    supabase: Any,
    *,
    table_name: str,
    club_id: str,
    columns: str,
    open_statuses: tuple[str, ...],
) -> list[dict[str, Any]]:
    try:
        response = (
            supabase.table(table_name)
            .select(columns)
            .eq("club_id", str(club_id))
            .order("created_at", desc=False)
            .execute()
        )
    except Exception as exc:  # fail closed if the guard cannot be evaluated
        raise MatchLogRecoveryLockUnavailable(
            "Match Log recovery state could not be verified. "
            "No write or replay was started."
        ) from exc
    rows = [
        dict(row)
        for row in list(getattr(response, "data", None) or [])
        if isinstance(row, dict)
        and str(row.get("status") or "") in set(open_statuses)
    ]
    return rows[:2]


def list_open_match_log_recovery_locks(
    supabase: Any,
    *,
    club_id: str,
) -> list[MatchLogRecoveryLock]:
    edit_rows = _operation_rows(
        supabase,
        table_name="match_edit_operations",
        club_id=str(club_id),
        columns="id,status,replay_job_id,created_at",
        open_statuses=_EDIT_OPEN_STATUSES,
    )
    exclusion_rows = _operation_rows(
        supabase,
        table_name="match_exclusion_operations",
        club_id=str(club_id),
        columns="id,status,replay_job_id,recovery_stage,created_at",
        open_statuses=_EXCLUSION_OPEN_STATUSES,
    )
    locks = [
        MatchLogRecoveryLock(
            operation_kind=MATCH_EDIT_KIND,
            operation_id=str(row.get("id") or ""),
            operation_status=str(row.get("status") or ""),
            replay_job_id=str(row.get("replay_job_id") or "") or None,
            created_at=str(row.get("created_at") or "") or None,
        )
        for row in edit_rows
        if str(row.get("id") or "")
    ]
    locks.extend(
        MatchLogRecoveryLock(
            operation_kind=MATCH_EXCLUSION_KIND,
            operation_id=str(row.get("id") or ""),
            operation_status=str(row.get("status") or ""),
            replay_job_id=str(row.get("replay_job_id") or "") or None,
            recovery_stage=str(row.get("recovery_stage") or "") or None,
            created_at=str(row.get("created_at") or "") or None,
        )
        for row in exclusion_rows
        if str(row.get("id") or "")
    )
    return sorted(
        locks,
        key=lambda lock: (
            str(lock.created_at or ""),
            lock.operation_kind,
            lock.operation_id,
        ),
    )


def enforce_match_log_recovery_lock(
    supabase: Any,
    *,
    club_id: str,
    recovery_kind: str | None = None,
    recovery_operation_id: str | None = None,
) -> MatchLogRecoveryLock | None:
    locks = list_open_match_log_recovery_locks(
        supabase,
        club_id=str(club_id),
    )
    if not locks:
        return None
    if len(locks) == 1:
        lock = locks[0]
        if (
            str(recovery_kind or "") == lock.operation_kind
            and str(recovery_operation_id or "") == lock.operation_id
        ):
            return lock
        raise MatchLogRecoveryLocked(lock)

    # More than one open ledger indicates an invariant breach. Fail closed.
    raise MatchLogRecoveryLocked(
        locks[0],
        code="MATCH_LOG_RECOVERY_LOCK_AMBIGUOUS",
    )
