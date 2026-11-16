"""Publish ratings only after rated play in the corresponding discipline."""
from __future__ import annotations

import math
from typing import Any, Mapping


def has_rated_history(row: Mapping[str, Any], *, singles: bool = False) -> bool:
    prefix = "singles_" if singles else ""
    try:
        count = row.get(f"{prefix}matches_played")
        if count is None:
            count = float(row.get(f"{prefix}wins") or 0) + float(row.get(f"{prefix}losses") or 0)
        return math.isfinite(float(count)) and float(count) > 0
    except (TypeError, ValueError, OverflowError):
        return False


def public_rating(row: Mapping[str, Any], field: str = "rating", *, singles: bool = False) -> float | None:
    """Mask starting estimates without changing the stored/admin rating."""
    if not has_rated_history(row, singles=singles):
        return None
    try:
        value = float(row[field])
        return value if math.isfinite(value) else None
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
