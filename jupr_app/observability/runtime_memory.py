"""Small process/machine samples for correlating memory pressure with handlers.

RSS and available RAM describe the whole process/machine, including concurrent
work. Their differences must never be reported as a request's own allocation.
Handlers are counted until response headers are ready, matching request timing;
streamed response bodies can outlive this interval. No payloads are inspected.
"""
from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Callable


def runtime_memory_snapshot(proc_root: Path = Path("/proc")) -> dict[str, float]:
    values: dict[str, float] = {}
    try:
        resident_pages = int((proc_root / "self/statm").read_text().split()[1])
        if resident_pages >= 0:
            values["process_rss_mb"] = round(resident_pages * os.sysconf("SC_PAGE_SIZE") / 1048576, 2)
    except (OSError, ValueError, IndexError, AttributeError):
        pass
    try:
        for line in (proc_root / "meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                available_kb = int(line.split()[1])
                if available_kb >= 0:
                    values["machine_available_mb"] = round(available_kb / 1024, 2)
                break
    except (OSError, ValueError, IndexError):
        pass
    return values


class RequestMemoryTracker:
    def __init__(self, sampler: Callable[[], dict[str, float]] = runtime_memory_snapshot):
        self._sampler = sampler
        self._in_flight = 0
        self._lock = Lock()

    def _sample(self, suffix: str) -> dict[str, float]:
        try:
            return {f"{key}_{suffix}": value for key, value in self._sampler().items()}
        except Exception:
            # Unavailable telemetry must not replace a response or its error.
            return {}

    def start(self) -> dict[str, int | float]:
        with self._lock:
            self._in_flight += 1
            count = self._in_flight
        return {"handlers_in_flight_start": count, **self._sample("start")}

    def finish(self, started: dict[str, int | float]) -> dict[str, int | float]:
        with self._lock:
            self._in_flight -= 1
            count = self._in_flight
        return {**started, "handlers_in_flight_end": count, **self._sample("end")}
