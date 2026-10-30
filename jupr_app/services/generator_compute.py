"""Bounded process isolation for pure schedule planning; persistence stays in callers.

Only dictionaries/lists/scalars cross the process boundary. Never submit a DB
client, request, token, callback, or a service function that can write data.
"""
from __future__ import annotations

import atexit
import importlib
import logging
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from functools import wraps
from typing import Any

logger = logging.getLogger("jupr.api.compute")
_ALLOWED = {
    "engine": ("jupr_app.domain.adaptive_play_engine", {
        "create_generator_preview", "start_generator_event", "advance_generator_event",
        "mutate_generator_roster", "select_generator_round", "skip_generator_round",
        "reopen_generator_round", "save_generator_round", "mark_generator_round_played",
    }),
    "hub": ("jupr_app.domain.round_robin_hub", {"apply_hub_command", "reconcile_hub_events"}),
}


class GeneratorComputeUnavailable(ValueError):
    """Pure planning failed before any session write; safe to retry the operation."""


def compute_unavailable_cause(exc: BaseException) -> GeneratorComputeUnavailable | None:
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, GeneratorComputeUnavailable):
            return exc
        exc = exc.__cause__  # type: ignore[assignment]
    return None


def _execute(group: str, name: str, args: tuple, kwargs: dict, submitted: float):
    module, names = _ALLOWED[group]
    if name not in names:
        raise ValueError("Unsupported schedule operation")
    started = time.monotonic()
    cpu = time.process_time()
    result = getattr(importlib.import_module(module), name)(*args, **kwargs)
    return result, {
        "queue_ms": round((started - submitted) * 1000, 2),
        "compute_ms": round((time.monotonic() - started) * 1000, 2),
        "cpu_ms": round((time.process_time() - cpu) * 1000, 2),
    }


class ScheduleWorker:
    """One planner and at most one queued planner per API process.

    Slots stay occupied until the worker finishes, even after caller timeout.
    A late result cannot save: only the waiting caller owns the subsequent CAS.
    """
    def __init__(self, *, capacity: int = 2, timeout_seconds: float = 60):
        self._slots = threading.BoundedSemaphore(capacity)
        self._lock = threading.Lock()
        self._pool: ProcessPoolExecutor | None = None
        self._closed = False
        self.timeout_seconds = timeout_seconds

    def _executor(self):
        with self._lock:
            if self._closed:
                raise GeneratorComputeUnavailable("Schedule planning is restarting. Please try again.")
            if self._pool is None:
                self._pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
            return self._pool

    def run(self, group: str, name: str, *args, **kwargs):
        start = time.monotonic()
        if not self._slots.acquire(blocking=False):
            logger.info("generator.compute.busy", extra={"operation": name})
            raise GeneratorComputeUnavailable("Another schedule is being prepared. Please try again shortly.")
        future = None
        pool = None
        try:
            pool = self._executor()
            future = pool.submit(_execute, group, name, args, kwargs, start)
            future.add_done_callback(lambda _: self._slots.release())
            result, timing = future.result(timeout=self.timeout_seconds)
            logger.info("generator.compute.complete", extra={"operation": name, **timing,
                "elapsed_ms": round((time.monotonic() - start) * 1000, 2)})
            return result
        except TimeoutError as exc:
            # Queued jobs may be cancelled; running pure jobs have no DB access.
            future.cancel()
            logger.warning("generator.compute.timeout", extra={"operation": name,
                "elapsed_ms": round((time.monotonic() - start) * 1000, 2)})
            raise GeneratorComputeUnavailable("Schedule planning took too long. No changes were saved. Please try again.") from exc
        except BrokenProcessPool as exc:
            # A second waiter from a broken pool must not destroy a newer
            # pool another request has already created.
            with self._lock:
                if self._pool is pool:
                    self._pool = None
            if pool is not None:
                pool.shutdown(wait=False, cancel_futures=True)
            logger.error("generator.compute.failed", extra={"operation": name, "exception_type": type(exc).__name__})
            raise GeneratorComputeUnavailable("Schedule planning restarted. No changes were saved. Please try again.") from exc
        finally:
            if future is None:
                self._slots.release()

    def close(self):
        with self._lock:
            self._closed = True
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


_worker: ScheduleWorker | None = None
_worker_lock = threading.Lock()


def shutdown_schedule_worker():
    global _worker
    with _worker_lock:
        worker, _worker = _worker, None
    if worker is not None:
        worker.close()


atexit.register(shutdown_schedule_worker)


def _enabled() -> bool:
    configured = os.getenv("JUPR_GENERATOR_PROCESS_WORKER", "").strip().lower()
    if configured:
        return configured in {"1", "true", "yes"}
    return os.getenv("JUPR_ENV", "").strip().lower() in {"staging", "production"}


def _adapter(group: str, name: str):
    module, _ = _ALLOWED[group]
    original = getattr(importlib.import_module(module), name)

    @wraps(original)
    def plan(*args, **kwargs):
        if not _enabled():
            return original(*args, **kwargs)
        global _worker
        with _worker_lock:
            if _worker is None:
                _worker = ScheduleWorker()
            worker = _worker
        return worker.run(group, name, *args, **kwargs)
    return plan


# Explicit allowlist: these functions have no database or request dependency.
for _group, (_, _names) in _ALLOWED.items():
    for _name in _names:
        globals()[_name] = _adapter(_group, _name)
