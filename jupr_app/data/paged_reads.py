"""Complete, ordered reads for computations that must not use partial history."""
from collections.abc import Callable
from typing import Any


class DataReadUnavailable(RuntimeError):
    pass


def read_all_rows(query_factory: Callable[[], Any], *, order: str = "id", page_size: int = 500) -> list[dict]:
    """Build a fresh query per page; even a server-capped short page is not EOF.

    Callers must supply a stable unique order and all tenant filters. Errors
    propagate instead of turning missing history into zero awards or new awards.
    """
    rows: list[dict] = []
    while True:
        try:
            response = query_factory().order(order).range(len(rows), len(rows) + page_size - 1).execute()
            if response.data is None:
                raise ValueError("Read returned no data payload")
            page = [dict(row) for row in response.data]
        except Exception as exc:
            raise DataReadUnavailable("The complete award data could not be loaded.") from exc
        if not page:
            return rows
        rows.extend(page)
