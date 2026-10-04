"""Player identity availability is separate from leaderboard activity."""

from typing import Any, Mapping


def is_merged_player(row: Mapping[str, Any]) -> bool:
    """Recognize the retained source identity used by current and legacy merges."""
    return any(
        "(merged into " in str(row.get(field) or "").casefold()
        for field in ("name", "display_name")
    )
