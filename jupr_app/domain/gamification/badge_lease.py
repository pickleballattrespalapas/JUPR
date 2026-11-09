"""Observable queue ownership. Expired claims require review, never blind replay.

The heartbeat only maintains ownership; it cannot stop a Python computation.
Check ownership immediately before effects and fence final acknowledgement.
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger("jupr.badge.worker")
WORKER_ID = f"{os.getenv('FLY_MACHINE_ID', 'local')}:{os.getpid()}:{uuid.uuid4().hex[:12]}"
LEASE_SECONDS = 120
HEARTBEAT_SECONDS = 30
_worker_context: ContextVar[bool] = ContextVar("badge_worker_context", default=False)


class BadgeLeaseLost(RuntimeError):
    pass


def leases_enabled(club_id: str | None = None) -> bool:
    if os.getenv("JUPR_BADGE_QUEUE_LEASES", "0").strip() != "1":
        return False
    # A scoped pilot must not silently change ownership semantics for clubs
    # that retain legacy inline processing and fallback. Without a club,
    # retain the global flag for startup validation and fail-closed ack checks.
    if club_id is not None and os.getenv("JUPR_BADGE_WORKER_MODE", "embedded") == "external":
        scope = external_worker_club_id()
        if scope is not None:
            return str(club_id).strip() == scope
    return True


def external_worker_club_id() -> str | None:
    """Optional pilot scope; absent retains the existing global mode contract."""
    scope = os.getenv("JUPR_BADGE_WORKER_CLUB_ID", "").strip()
    if scope and scope != "tres_palapas":
        raise ValueError("Dedicated badge worker club scope is not authorized.")
    return scope or None


def processing_deferred(club_id: str | None = None) -> bool:
    if os.getenv("JUPR_BADGE_WORKER_MODE", "embedded") != "external" or _worker_context.get():
        return False
    scope = external_worker_club_id()
    return scope is None or str(club_id or "").strip() == scope


def deferred_badge_summary(enqueue_result: dict[str, Any], *, club_id: str,
                           match_ref: str | None = None) -> dict[str, Any] | None:
    """Keep API-side saves out of evaluation, including queue failure fallback.

    Failed queue writes cannot be made durable by pretending they succeeded.
    Return an explicit recovery error tied to the already-saved source, so
    callers suppress incomplete notifications and operators retry that source.
    """
    external = os.getenv("JUPR_BADGE_WORKER_MODE", "embedded") == "external"
    scope = external_worker_club_id() if external else None
    if external and scope is not None and str(club_id).strip() != scope:
        # Other clubs retain legacy processing, claims and fallback together;
        # they must not execute an unfenced fallback after losing a lease.
        return None
    if not external and not leases_enabled():
        return None
    if enqueue_result.get("queued"):
        result = {"mode": "queue", "processed": 0, "errored": 0, "deferred": True}
        if match_ref:
            result["notification_badge_refs"] = [str(match_ref)]
        return result
    logger.error("badge.job.enqueue_failed", extra={"club_id": str(club_id), "phase": "enqueue"})
    return {
        "mode": "queue_error", "processed": 0, "errored": 1,
        "recovery_required": True,
        "recovery_match_refs": [str(match_ref)] if match_ref else [],
        "error": "The games are saved, but award evaluation could not be queued. Resume the saved operation to retry; do not resubmit the games.",
    }


@contextmanager
def isolated_worker_context():
    token = _worker_context.set(True)
    try:
        yield
    finally:
        _worker_context.reset(token)


class BadgeJobLease:
    def __init__(self, db: Any, job: dict[str, Any]):
        self.db, self.job = db, job
        self.token = job.get("claim_token")
        self._stop = threading.Event()
        self._lost = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.monotonic()
        self._cpu = time.thread_time()

    def event(self, event: str, **fields: Any) -> None:
        logger.info(event, extra={
            "job_id": str(self.job.get("id", "")),
            "club_id": str(self.job.get("club_id", "")),
            "worker_id": WORKER_ID,
            "elapsed_ms": round((time.monotonic() - self._started) * 1000, 2),
            **fields,
        })

    def renew(self) -> None:
        if not self.token:
            return
        if self._lost.is_set():
            raise BadgeLeaseLost("Badge job ownership was lost; review before recovery.")
        try:
            ok = self.db.rpc("heartbeat_badge_eval_queue_job", {
                "p_job_id": str(self.job["id"]), "p_claim_token": str(self.token),
                "p_lease_seconds": LEASE_SECONDS,
            }).execute().data
            if ok is not True:
                raise BadgeLeaseLost("Badge job lease expired or ownership changed.")
        except Exception as exc:
            self._lost.set()
            self.event("badge.job.lease_lost", exception_type=type(exc).__name__)
            raise BadgeLeaseLost("Badge job heartbeat could not verify ownership.") from exc

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                self.renew()
                self.event("badge.job.heartbeat")
            except BadgeLeaseLost:
                return

    def __enter__(self):
        if leases_enabled(str(self.job.get("club_id") or "")) and not self.token:
            raise BadgeLeaseLost("Lease-enabled worker received an unfenced job.")
        self.renew()
        self.job["_assert_owned"] = self.renew
        self.event("badge.job.started")
        if self.token:
            self._thread = threading.Thread(target=self._heartbeat, name="badge-job-heartbeat", daemon=True)
            self._thread.start()
        return self

    def stop_heartbeat(self) -> None:
        self._stop.set()
        # Do not share a client with the next job while its heartbeat is in flight.
        if self._thread:
            self._thread.join()

    def __exit__(self, kind, value, traceback):
        self.stop_heartbeat()
        self.job.pop("_assert_owned", None)
        self.event("badge.job.failed" if kind else "badge.job.completed",
                   cpu_ms=round((time.thread_time() - self._cpu) * 1000, 2),
                   **({"exception_type": kind.__name__} if kind else {}))


def assert_job_owned(job: dict[str, Any]) -> None:
    guard = job.get("_assert_owned")
    if guard:
        guard()
