"""Paged reads that reject incomplete or changing table membership."""
from collections.abc import Callable
from typing import Any

from jupr_app.data.paged_reads import DataReadUnavailable


def read_counted_rows(
    query_factory: Callable[[str | None], Any], *, key: str = "id",
    descending: bool = False, limit: int | None = None, page_size: int = 500,
) -> list[dict]:
    """The factory applies selection/tenant filters and accepts a count mode.

    Server caps may shorten a page, so advance by rows received. Exact counts,
    unique keys, and a final count check prevent publishing partial results.
    This is not a transaction snapshot; mutation callers still need their
    existing optimistic concurrency and canonical-data guards.
    """
    if page_size <= 0 or (limit is not None and limit <= 0):
        raise ValueError("page_size and explicit limit must be positive")
    rows, seen = [], set()
    total = None
    while total is None or len(rows) < min(total, limit if limit is not None else total):
        size = min(page_size, limit - len(rows)) if limit is not None else page_size
        response = (query_factory("exact" if total is None else None)
                    .order(key, desc=descending).range(len(rows), len(rows) + size - 1).execute())
        if total is None:
            if response.count is None or int(response.count) < 0:
                raise DataReadUnavailable("Complete history requires an exact row count.")
            total = int(response.count)
        page = response.data
        if not isinstance(page, list):
            raise DataReadUnavailable("History read returned no row payload.")
        if not page and len(rows) < min(total, limit if limit is not None else total):
            raise DataReadUnavailable("History ended before the exact row count was reached.")
        for row in page:
            identity = row.get(key)
            if identity is None or identity in seen:
                raise DataReadUnavailable("History contains missing or repeated row identities.")
            seen.add(identity)
        rows.extend(page)
        if len(rows) > min(total, limit if limit is not None else total):
            raise DataReadUnavailable("History changed or exceeded the requested range.")
    final = query_factory("exact").order(key, desc=descending).limit(1).execute()
    if final.count is None or int(final.count) != total:
        raise DataReadUnavailable("History membership changed during loading; retry the operation.")
    return rows
