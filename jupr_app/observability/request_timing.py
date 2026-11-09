"""Bounded request measurements, without query text, values, or response data.

DB elapsed time is the client ``execute`` interval: transport, server wait and
decoding. It is not database-server CPU. Phase CPU uses the current thread's CPU
clock, not process CPU (which includes concurrent requests). Durations can
overlap when callers use parallel work; they must not be subtracted from request
elapsed time to manufacture a CPU/wait percentage. Uninstrumented phases and
thread-pool queueing remain unmeasured.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter, thread_time
from typing import Any


_PHASES = frozenset({"leaderboard_calculation", "leaderboard_history_normalize", "badge_calculation"})


@dataclass
class RequestTiming:
    request_id: str
    db_calls: int = 0
    db_errors: int = 0
    db_elapsed_ms: float = 0
    db_thread_cpu_ms: float = 0
    phases: dict[str, dict[str, float | int]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def add_db(self, elapsed_ms: float, cpu_ms: float, failed: bool) -> None:
        with self._lock:
            self.db_calls += 1
            self.db_errors += int(failed)
            self.db_elapsed_ms += elapsed_ms
            self.db_thread_cpu_ms += cpu_ms

    def add_phase(self, name: str, elapsed_ms: float, cpu_ms: float) -> None:
        if name not in _PHASES:
            return
        with self._lock:
            phase = self.phases.setdefault(name, {"calls": 0, "elapsed_ms": 0.0, "thread_cpu_ms": 0.0})
            phase["calls"] += 1
            phase["elapsed_ms"] += elapsed_ms
            phase["thread_cpu_ms"] += cpu_ms

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "db_calls": self.db_calls,
                "db_errors": self.db_errors,
                "db_elapsed_ms": round(self.db_elapsed_ms, 2),
                "db_thread_cpu_ms": round(self.db_thread_cpu_ms, 2),
                "phases": {
                    name: {key: round(value, 2) for key, value in phase.items()}
                    for name, phase in self.phases.items()
                },
            }


current_timing: ContextVar[RequestTiming | None] = ContextVar("jupr_request_timing", default=None)


@contextmanager
def measured_phase(name: str):
    """Time an allowlisted synchronous phase; also usable as a decorator.

The mutable, locked accumulator follows the request's copied context into
Starlette/AnyIO worker threads. Unknown labels are ignored, never logged.
"""
    timing = current_timing.get()
    if timing is None or name not in _PHASES:
        yield
        return
    started, cpu = perf_counter(), thread_time()
    try:
        yield
    finally:
        try:
            cpu_ms = (thread_time() - cpu) * 1000
            timing.add_phase(name, (perf_counter() - started) * 1000, cpu_ms)
        except Exception:
            # Telemetry must not replace a successful result or original error.
            pass


class _QueryTimingProxy:
    def __init__(self, query: Any):
        self._query = query

    def execute(self, *args, **kwargs):
        timing = current_timing.get()
        if timing is None:
            return self._query.execute(*args, **kwargs)
        started, cpu = perf_counter(), thread_time()
        failed = False
        try:
            return self._query.execute(*args, **kwargs)
        except BaseException:
            failed = True
            raise
        finally:
            try:
                cpu_ms = (thread_time() - cpu) * 1000
                timing.add_db((perf_counter() - started) * 1000, cpu_ms, failed)
            except Exception:
                pass

    def __getattr__(self, name: str):
        value = getattr(self._query, name)
        if callable(value):
            def chained(*args, **kwargs):
                return _wrap_query(value(*args, **kwargs))
            return chained
        # PostgREST's not_ filter is a property rather than a method.
        return _wrap_query(value)


def _wrap_query(value: Any):
    return _QueryTimingProxy(value) if callable(getattr(value, "execute", None)) else value


class _ClientTimingProxy:
    """Instance-local adapter; all non-query facilities remain untouched."""
    def __init__(self, client: Any):
        self._client = client

    def __getattr__(self, name: str):
        value = getattr(self._client, name)
        if name in {"table", "from_", "rpc"} and callable(value):
            def query(*args, **kwargs):
                # table()/from_() return a root builder with no execute method
                # until select/insert/etc. chooses an operation.
                return _QueryTimingProxy(value(*args, **kwargs))
            return query
        return value


def instrument_supabase_client(client: Any):
    """Instrument request-owned clients only; no global SDK monkey-patching."""
    if current_timing.get() is None or isinstance(client, _ClientTimingProxy):
        return client
    return _ClientTimingProxy(client)
