"""Immutable participation and scoring policy for one-off exhibition games."""
from __future__ import annotations

from typing import Any


def validate_supplemental_update(before: dict[str, Any], after: dict[str, Any]) -> None:
    """Enforce the same policy across hub and legacy generator write routes."""
    if not before.get("hubSupplemental"):
        return

    def lineups(event: dict[str, Any]) -> list[tuple]:
        return [(row.get("number"), match.get("id"),
                 tuple(match.get("sideA") or match.get("teamA") or []),
                 tuple(match.get("sideB") or match.get("teamB") or []))
                for row in event.get("rounds") or [] for match in row.get("matches") or []]

    if (after.get("hubSupplemental") != before["hubSupplemental"]
            or after.get("participants") != before.get("participants")
            or after.get("rosterByRound") != before.get("rosterByRound")
            or after.get("playoff")
            or after.get("ratingMode") != "unrated"
            or after.get("scoringMode") != "scored"
            or after.get("playFormat") != "doubles"
            or after.get("eventFormat", "standard") != "standard"
            or after.get("hubFinishAfterRound") != 1
            or {row.get("number") for row in after.get("rounds") or []} != {1}
            or lineups(after) != lineups(before)):
        raise ValueError("Extra games keep their original players and one-round exhibition format.")
