"""Optional seeded doubles playoffs for a scored round-robin session."""

from __future__ import annotations

import copy
from typing import Any


PLAYOFF_FORMATS = {"groups_of_four", "top_eight"}


def generator_playoff_options(event: dict[str, Any]) -> dict[str, Any]:
    from jupr_app.domain.adaptive_play_engine import (
        active_participant_ids, generator_event_standings, normalize_scoring_mode,
    )

    current = int(event.get("currentRoundNumber") or 1)
    current_row = next((r for r in event.get("rounds", []) if r.get("number") == current), {})
    reason = None
    if event.get("generatorKind") != "round_robin" or event.get("playFormat") == "singles":
        reason = "Playoffs are available for doubles round robins."
    elif normalize_scoring_mode(event.get("scoringMode")) != "scored":
        reason = "Playoffs require scored standings."
    elif event.get("playoff"):
        reason = "This session already has a playoff."
    elif event.get("status") not in {"active", "completed"}:
        reason = "Start the round robin before creating a playoff."
    elif current_row.get("status") not in {"saved", "skipped"}:
        reason = "Save or skip the current round before starting a playoff."
    elif any(r.get("number", 0) < current and r.get("status") not in {"saved", "skipped"} for r in event.get("rounds", [])):
        reason = "Save or skip every open round before starting a playoff."
    elif any(r.get("number", 0) > current and r.get("status") != "preview" for r in event.get("rounds", [])):
        reason = "Finish the recorded rounds before starting a playoff."

    active = set(active_participant_ids(event, current + 1))
    standings = generator_event_standings(event)
    seeds = [
        {"participantId": row["participantId"], "name": row["name"], "seed": index}
        for index, row in enumerate(
            (row for row in standings if row["participantId"] in active and int(row.get("matches") or 0) > 0), 1
        )
    ]
    ids = [row["participantId"] for row in seeds]

    def matchup(label: str, a: list[int], b: list[int]) -> dict[str, Any]:
        return {"label": label, "sideA": [ids[i] for i in a], "sideB": [ids[i] for i in b]}

    group_matches = [
        matchup(f"Places {i + 1}–{i + 4}", [i, i + 3], [i + 1, i + 2])
        for i in range(0, len(ids) - 3, 4)
    ]
    semifinal_matches = [
        matchup("Semifinal 1", [0, 7], [3, 4]),
        matchup("Semifinal 2", [1, 6], [2, 5]),
    ] if len(ids) >= 8 else []
    return {
        "seeds": seeds,
        "sourceRound": current,
        "formats": [
            {"format": "groups_of_four", "matches": group_matches,
             "reason": reason or (None if group_matches else "At least four available players need a completed game."),
             "sitOutParticipantIds": ids[len(group_matches) * 4:]},
            {"format": "top_eight", "matches": semifinal_matches,
             "reason": reason or (None if semifinal_matches else "At least eight available players need a completed game."),
             "sitOutParticipantIds": ids[8:]},
        ],
    }


def start_generator_playoff(event: dict[str, Any], *, playoff_format: str) -> dict[str, Any]:
    from jupr_app.domain.adaptive_play_engine import generator_event_standings, _now_iso

    if playoff_format not in PLAYOFF_FORMATS:
        raise ValueError("Choose games in groups of four or a top-eight bracket.")
    options = generator_playoff_options(event)
    option = next(row for row in options["formats"] if row["format"] == playoff_format)
    if option["reason"]:
        raise ValueError(option["reason"])
    next_event = copy.deepcopy(event)
    current = int(options["sourceRound"])
    # Preserve all played history; replace only unused future previews.
    next_event["rounds"] = [r for r in next_event["rounds"] if int(r["number"]) <= current]
    next_event["playoff"] = {
        "format": playoff_format, "sourceRound": current, "createdAt": _now_iso(),
        "seeds": copy.deepcopy(options["seeds"]),
        "seedStandings": generator_event_standings(event),
        "sitOutParticipantIds": list(option["sitOutParticipantIds"]),
    }
    court_count = max(1, int(event.get("courtCount") or event.get("doublesCourtCount") or len(option["matches"])))
    seeded_ids = [row["participantId"] for row in options["seeds"]]
    semifinal_ids = []

    def append_round(matches: list[dict[str, Any]], label: str) -> None:
        number = int(next_event["rounds"][-1]["number"]) + 1
        playing = {pid for match in matches for pid in match["sideA"] + match["sideB"]}
        next_event["rounds"].append({
            "number": number, "stage": "playoff", "label": label,
            "status": "active" if number == current + 1 else "preview",
            "matches": matches, "byeParticipantIds": [pid for pid in seeded_ids if pid not in playing],
            "warnings": [], "formatCounts": {"doubles": len(matches), "singles": 0},
        })

    for offset in range(0, len(option["matches"]), court_count):
        number = int(next_event["rounds"][-1]["number"]) + 1
        batch = []
        for court, template in enumerate(option["matches"][offset:offset + court_count], 1):
            match = {**copy.deepcopy(template), "id": f"playoff-r{number}-c{court}",
                     "round": number, "court": court, "playFormat": "doubles",
                     "scoreA": None, "scoreB": None, "status": "scheduled"}
            batch.append(match)
            semifinal_ids.append(match["id"])
        label = "Playoff semifinals" if playoff_format == "top_eight" else "Playoff games"
        append_round(batch, label)
    if playoff_format == "top_eight":
        number = int(next_event["rounds"][-1]["number"]) + 1
        append_round([{
            "id": f"playoff-r{number}-final", "round": number, "court": 1,
            "label": "Final", "playFormat": "doubles", "sideA": [], "sideB": [],
            "sourceMatchIds": semifinal_ids, "scoreA": None, "scoreB": None, "status": "scheduled",
        }], "Playoff final")
    next_event["currentRoundNumber"] = current + 1
    next_event["totalRounds"] = int(next_event["rounds"][-1]["number"])
    next_event["status"] = "active"
    next_event.pop("completedAt", None)
    return next_event


def populate_playoff_final(event: dict[str, Any], next_round: dict[str, Any]) -> None:
    """Resolve fixed semifinal teams only when both winning scores are saved."""
    matches = {m["id"]: (r, m) for r in event.get("rounds", []) for m in r.get("matches", [])}
    for final in next_round.get("matches", []):
        sources = final.get("sourceMatchIds")
        if not sources:
            continue
        winners = []
        for source in sources:
            row, match = matches[source]
            a, b = match.get("scoreA"), match.get("scoreB")
            if row.get("status") != "saved" or a is None or b is None or a == b:
                raise ValueError("Save both semifinal scores before starting the final.")
            winners.append(list(match["sideA"] if a > b else match["sideB"]))
        final["sideA"], final["sideB"] = winners
        playing = set(final["sideA"] + final["sideB"])
        next_round["byeParticipantIds"] = [
            s["participantId"] for s in event["playoff"]["seeds"] if s["participantId"] not in playing
        ]
