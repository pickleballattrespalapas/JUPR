"""Share overlapping public result reads without retaining completed results.

Slow public statistics must not fill FastAPI's entire synchronous worker pool.
Only explicitly decorated, unauthenticated read endpoints use these budgets;
staff requests, score writes, and publication/alias middleware remain separate.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from functools import wraps
from inspect import signature
from typing import Any, Callable

from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool


class PublicResultsReads:
    def __init__(self, *, max_active: int, max_waiters: int = 64):
        self.max_active = max_active
        self.max_waiters = max_waiters
        self._pending: dict[Any, asyncio.Task] = {}
        self._waiters = 0

    @staticmethod
    def _busy() -> HTTPException:
        return HTTPException(
            status_code=503,
            detail="Results are busy. Please try again in a few seconds.",
            headers={"Retry-After": "5", "Cache-Control": "no-store"},
        )

    def _finished(self, key: Any, task: asyncio.Task) -> None:
        if self._pending.get(key) is task:
            self._pending.pop(key, None)
        # A disconnected caller must not leave an unobserved exception behind.
        if not task.cancelled():
            task.exception()

    async def run(self, key: Any, loader: Callable[[], Any]) -> Any:
        task = self._pending.get(key)
        if task is not None and task.done():
            self._finished(key, task)
            task = None
        if self._waiters >= self.max_waiters:
            raise self._busy()
        if task is None:
            if len(self._pending) >= self.max_active:
                raise self._busy()
            # Admission happens before entering the shared thread pool. Waiting
            # for the same read is asynchronous and consumes no worker thread.
            task = asyncio.create_task(run_in_threadpool(loader))
            self._pending[key] = task
            task.add_done_callback(lambda finished: self._finished(key, finished))
        self._waiters += 1
        try:
            # Cancelling one HTTP request cannot cancel another visitor's read
            # or free a slot while its synchronous calculation still runs.
            result = await asyncio.shield(task)
            return deepcopy(result)
        finally:
            self._waiters -= 1


def coordinate_public_read(app, *, operation: str, lane: str = "statistics"):
    """Decorate a synchronous public GET, preserving FastAPI's typed signature."""
    if lane not in {"statistics", "tournaments"}:
        raise ValueError("Unknown public results lane")
    if not hasattr(app.state, "public_results_reads"):
        app.state.public_results_reads = {
            "statistics": PublicResultsReads(max_active=4),
            "tournaments": PublicResultsReads(max_active=2),
        }
    reads = app.state.public_results_reads[lane]

    def decorate(loader):
        typed_signature = signature(loader, eval_str=True)

        @wraps(loader)
        async def coordinated(*args, **kwargs):
            bound = typed_signature.bind(*args, **kwargs)
            bound.apply_defaults()
            key = (operation, tuple(bound.arguments.items()))
            return await reads.run(key, lambda: loader(*args, **kwargs))

        coordinated.__signature__ = typed_signature
        return coordinated

    return decorate
