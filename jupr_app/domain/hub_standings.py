"""Read-only standings views for a hub with overlapping rounds.

Official generator standings retain their completed-round contract. This optional
view shows each recorded game immediately without settling rounds or publishing
results, including results earned by players who have since left.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from jupr_app.domain.adaptive_play_engine import generator_event_standings


def _matches(row: dict[str, Any]) -> list[dict[str, Any]]:
    return row.get("matches") or [match for court in row.get("courts") or [] for match in court.get("matches") or []]


def _recorded(match: dict[str, Any]) -> bool:
    return match.get("status") not in {"not_played", "skipped", "cancelled", "canceled"} and all(
        match.get(key) is not None for key in ("scoreA", "scoreB"))


def _resolved(match: dict[str, Any]) -> bool:
    return _recorded(match) or match.get("status") in {"not_played", "skipped", "cancelled", "canceled"}


def hub_standings_view(event: dict[str, Any]) -> dict[str, Any]:
    """Return additive admin presentation data; never mutate the event."""
    if not event or event.get("generatorKind") != "round_robin" or event.get("scoringMode") == "unscored" \
            or event.get("eventFormat") == "moneyball_expectations" or event.get("playoff"):
        return {}
    rounds = sorted(event.get("rounds") or [], key=lambda row: int(row["number"]))
    finish_after = event.get("hubFinishAfterRound")
    complete_through = 0
    for row in rounds:
        if finish_after is not None and int(row["number"]) > int(finish_after):
            break
        if row.get("status") not in {"saved", "played", "skipped"}:
            break
        complete_through = int(row["number"])
    assigned = set(event.get("hubAssignedMatchIds") or [])
    touched = [int(row["number"]) for row in rounds if row.get("hubStartedAt") or any(
        match.get("hubStartedAt") or str(match.get("id")) in assigned or _recorded(match)
        for match in _matches(row))]
    boundary = int(finish_after) if finish_after is not None else max(
        int(event.get("currentRoundNumber") or 1), max(touched, default=0))
    names = {str(person["id"]): str(person.get("name") or person["id"]) for person in event.get("participants") or []}
    pending = [{"match_id": str(match["id"]), "round_number": int(row["number"]),
        "side_a": [names.get(str(pid), str(pid)) for pid in match.get("sideA") or match.get("teamA") or []],
        "side_b": [names.get(str(pid), str(pid)) for pid in match.get("sideB") or match.get("teamB") or []]}
        for row in rounds if int(row["number"]) <= boundary and row.get("status") != "skipped"
        for match in _matches(row) if not _resolved(match)]
    partial_results = sum(_recorded(match) for row in rounds if row.get("status") not in {"saved", "skipped"}
                          for match in _matches(row))
    provisional = deepcopy(event)
    for row in provisional.get("rounds") or []:
        if row.get("status") != "skipped":
            row["status"] = "saved"
    return {
        "provisional_standings": generator_event_standings(provisional),
        "standings_view": {
            "completed_through_round": complete_through,
            "scored_matches_in_partial_rounds": partial_results,
            "pending_matches": pending,
            "pending_through_round": boundary,
            "finish_after_round": finish_after,
            "final": event.get("status") == "completed" and not pending,
        },
    }
