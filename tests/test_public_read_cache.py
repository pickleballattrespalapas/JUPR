from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from jupr_app.data.public_read_cache import PublicReadCache


def test_public_cache_expires_bounds_entries_and_never_shares_mutable_results():
    now = [0]
    cache = PublicReadCache(ttl=30, capacity=2, clock=lambda: now[0])
    calls = []

    def load():
        calls.append(1)
        return {"value": [len(calls)]}

    cache.get(("db", "club-a"), load)["value"].append("mutation")
    assert cache.get(("db", "club-a"), load) == {"value": [1]}
    assert cache.get(("db", "club-b"), load) == {"value": [2]}
    assert cache.get(("other-db", "club-a"), load) == {"value": [3]}
    assert len(cache._values) == 2
    assert cache.get(("db", "club-a"), load) == {"value": [4]}
    now[0] = 30
    assert cache.get(("db", "club-a"), load) == {"value": [5]}


def test_concurrent_public_reads_share_one_load():
    cache = PublicReadCache()
    started, release = Event(), Event()
    calls = []

    def load():
        calls.append(1)
        started.set()
        assert release.wait(2)
        return {"value": [1]}

    with ThreadPoolExecutor(max_workers=8) as pool:
        first = pool.submit(cache.get, "same", load)
        assert started.wait(2)
        others = [pool.submit(cache.get, "same", load) for _ in range(7)]
        release.set()
        results = [first.result(), *(future.result() for future in others)]
    assert len(calls) == 1
    results[0]["value"].append(2)
    assert all(result == {"value": [1]} for result in results[1:])


def test_failure_does_not_replace_expired_data_or_poison_retry():
    now = [0]
    cache = PublicReadCache(ttl=30, clock=lambda: now[0])
    cache.get("same", lambda: "old")
    now[0] = 31

    def failed():
        raise RuntimeError("unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        cache.get("same", failed)
    assert not cache._pending
    assert cache.get("same", lambda: "fresh") == "fresh"
