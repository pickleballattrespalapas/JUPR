"""Bounded, process-local reuse for public statistics, never access decisions.

Visibility, publication and authorization must be checked afresh by callers.
Failures are not cached, and expired values are never served on failure.
"""
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from threading import Lock
from time import monotonic
from typing import Any, Callable


class PublicReadCache:
    def __init__(self, *, ttl: float = 30, capacity: int = 128, clock=monotonic):
        self.ttl, self.capacity, self.clock = ttl, capacity, clock
        self._values: OrderedDict[Any, tuple[float, Any]] = OrderedDict()
        self._pending: dict[Any, Future] = {}
        self._lock = Lock()

    def get(self, key: Any, loader: Callable[[], Any]) -> Any:
        with self._lock:
            now = self.clock()
            for expired in [k for k, (until, _) in self._values.items() if until <= now]:
                del self._values[expired]
            if key in self._values:
                self._values.move_to_end(key)
                return deepcopy(self._values[key][1])
            pending = self._pending.get(key)
            owner = pending is None
            if owner:
                # Bound in-flight bookkeeping too. Unique cold requests can
                # still proceed, but cannot grow the cache without limit.
                pending = Future() if len(self._pending) < self.capacity else None
                if pending is not None:
                    self._pending[key] = pending
        if not owner:
            return deepcopy(pending.result())
        try:
            value = loader()
            snapshot = deepcopy(value)
            with self._lock:
                self._values[key] = (self.clock() + self.ttl, snapshot)
                self._values.move_to_end(key)
                while len(self._values) > self.capacity:
                    self._values.popitem(last=False)
            if pending is not None:
                pending.set_result(snapshot)
            return value
        except BaseException as exc:
            if pending is not None:
                pending.set_exception(exc)
            raise
        finally:
            if pending is not None:
                with self._lock:
                    self._pending.pop(key, None)


def public_source_key(client: Any) -> Any:
    """API service clients are recreated per request; separate database origins.

    Non-network clients use their object identity, avoiding fixture collisions.
    No credentials are retained in cache keys.
    """
    origin = getattr(client, "supabase_url", None)
    return str(origin).rstrip("/") if origin else client


public_read_cache = PublicReadCache()
