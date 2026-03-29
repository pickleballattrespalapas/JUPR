from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from typing import Any

from jupr_app.domain.schedule import (
    ORGANIZED_RR_DEFAULT_MODE,
    SUPPORTED_DOUBLES_PLAYER_COUNTS,
    get_match_schedule,
)
from jupr_app.domain.league_night_roster import suggest_court_sizes
from jupr_app.domain.tournament_match_payload import build_tournament_match_payload


SUPPORTED_RR_FORMATS = SUPPORTED_DOUBLES_PLAYER_COUNTS
SUPPORTED_TOURNAMENT_TEAM_COUNTS = [4, 5, 6, 7, 8]


def normalize_name(value: object) -> str:
    return " ".join(str(value or "").replace("\u00A0", " ").split()).strip()


def _uid(prefix: str, index: int) -> str:
    return f"{prefix}-{index}"


def _build_participants(names: list[str], resolved_ids: dict[str, int] | None = None) -> list[dict[str, Any]]:
    resolved_ids = dict(resolved_ids or {})
    participants: list[dict[str, Any]] = []
    for idx, raw_name in enumerate(names, start=1):
        name = normalize_name(raw_name) or f"Player {idx}"
        participant = {
            "id": _uid("p", idx),
            "seed": idx,
            "name": name,
        }
        if name in resolved_ids:
            participant["player_id"] = int(resolved_ids[name])
        participants.append(participant)
    return participants


def _participant_map(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(p["id"]): p for p in (event.get("participants") or [])}


def _extract_round_number(desc: str, fallback: int) -> int:
    match = re.search(r"Rnd\s*(\d+)", str(desc or ""), flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            pass
    return int(fallback)


def _group_schedule_matches(schedule: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for idx, match in enumerate(schedule, start=1):
        round_number = _extract_round_number(str(match.get("desc", "")), idx)
        grouped[round_number].append(
            {
                "id": f"{prefix}-r{round_number}-m{len(grouped[round_number]) + 1}",
                "desc": str(match.get("desc", "") or f"Rnd {round_number}"),
                "teamA": list(match.get("t1") or []),
                "teamB": list(match.get("t2") or []),
                "scoreA": None,
                "scoreB": None,
            }
        )
    rounds: list[dict[str, Any]] = []
    for round_number in sorted(grouped):
        rounds.append({
            "number": int(round_number),
            "matches": grouped[round_number],
        })
    return rounds


def _infer_bye(participant_ids: list[str], matches: list[dict[str, Any]]) -> str | None:
    active_ids: set[str] = set()
    for match in matches:
        active_ids.update([str(x) for x in (match.get("teamA") or [])])
        active_ids.update([str(x) for x in (match.get("teamB") or [])])
    missing = [pid for pid in participant_ids if str(pid) not in active_ids]
    return str(missing[0]) if len(missing) == 1 else None


def _build_league_round(round_number: int, court_groups: list[list[str]]) -> dict[str, Any]:
    courts: list[dict[str, Any]] = []
    for court_number, participant_ids in enumerate(court_groups, start=1):
        size = len(participant_ids)
        if size not in (4, 5):
            raise ValueError("League / Ladder currently supports 4-player and 5-player courts only.")
        schedule = get_match_schedule(f"{size}-Player", participant_ids)
        round_blocks = _group_schedule_matches(schedule, prefix=f"lg-r{round_number}-c{court_number}")
        mini_rounds = []
        for block in round_blocks:
            bye_pid = _infer_bye(participant_ids, block["matches"])
            mini_rounds.append(
                {
                    "number": int(block["number"]),
                    "byeParticipantId": bye_pid,
                    "matches": block["matches"],
                }
            )
        courts.append(
            {
                "courtNumber": int(court_number),
                "size": int(size),
                "participantIds": [str(x) for x in participant_ids],
                "miniRounds": mini_rounds,
            }
        )
    return {"number": int(round_number), "courts": courts}


def compute_standings(matches: list[dict[str, Any]], participants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for participant in participants:
        rows[str(participant["id"])] = {
            "participantId": str(participant["id"]),
            "name": str(participant.get("name", "")),
            "wins": 0,
            "losses": 0,
            "ties": 0,
            "matches": 0,
            "pointsFor": 0,
            "pointsAgainst": 0,
            "differential": 0,
        }
    for match in matches:
        if match.get("scoreA") is None or match.get("scoreB") is None:
            continue
        score_a = int(match.get("scoreA") or 0)
        score_b = int(match.get("scoreB") or 0)
        if score_a == score_b:
            result = "tie"
        else:
            result = "A" if score_a > score_b else "B"
        for pid in [str(x) for x in (match.get("teamA") or [])]:
            row = rows.get(pid)
            if row is None:
                continue
            row["matches"] += 1
            row["pointsFor"] += score_a
            row["pointsAgainst"] += score_b
            row["differential"] += score_a - score_b
            if result == "tie":
                row["ties"] += 1
            elif result == "A":
                row["wins"] += 1
            else:
                row["losses"] += 1
        for pid in [str(x) for x in (match.get("teamB") or [])]:
            row = rows.get(pid)
            if row is None:
                continue
            row["matches"] += 1
            row["pointsFor"] += score_b
            row["pointsAgainst"] += score_a
            row["differential"] += score_b - score_a
            if result == "tie":
                row["ties"] += 1
            elif result == "B":
                row["wins"] += 1
            else:
                row["losses"] += 1
    ordered = list(rows.values())
    ordered.sort(
        key=lambda row: (
            -int(row["wins"]),
            -int(row["differential"]),
            -int(row["pointsFor"]),
            int(row["losses"]),
            str(row["name"]).lower(),
        )
    )
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = int(rank)
    return ordered


def create_round_robin_event(
    *,
    name: str,
    participant_names: list[str],
    resolved_ids: dict[str, int] | None = None,
    official_context: dict[str, Any] | None = None,
    schedule_mode: str = ORGANIZED_RR_DEFAULT_MODE,
) -> dict[str, Any]:
    names = [normalize_name(x) for x in participant_names if normalize_name(x)]
    count = len(names)
    if count not in SUPPORTED_RR_FORMATS:
        raise ValueError(
            "Round Robin currently supports every JUPR doubles format from 4 to 20 participants in JUPR Live Beta."
        )
    participants = _build_participants(names, resolved_ids=resolved_ids)
    schedule = get_match_schedule(f"{count}-Player", [p["id"] for p in participants], schedule_mode=schedule_mode)
    rounds = _group_schedule_matches(schedule, prefix="rr")
    return {
        "schemaVersion": 1,
        "name": normalize_name(name) or "JUPR Live Round Robin",
        "type": "round_robin",
        "participants": participants,
        "scheduleMode": str(schedule_mode),
        "rounds": rounds,
        "substitutions": [],
        "official_context": dict(official_context or {}),
        "saved_rounds": [],
    }


def suggest_exact_league_court_sizes(participant_count: int) -> list[int]:
    suggestion = suggest_court_sizes(int(participant_count or 0))
    if not suggestion.get("ok"):
        return []
    if int(suggestion.get("bench", 0) or 0) != 0:
        return []
    return [int(x) for x in suggestion.get("sizes", [])]


def create_league_event(
    *,
    name: str,
    participant_names: list[str],
    total_rounds: int,
    resolved_ids: dict[str, int] | None = None,
    court_sizes: list[int] | None = None,
    official_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    names = [normalize_name(x) for x in participant_names if normalize_name(x)]
    count = len(names)
    if count < 4:
        raise ValueError("League / Ladder needs at least 4 participants.")
    sizes = [int(x) for x in (court_sizes or suggest_exact_league_court_sizes(count)) if int(x) > 0]
    if not sizes or sum(sizes) != count:
        raise ValueError("League / Ladder requires an exact 4-player / 5-player court fit.")
    if any(size not in (4, 5) for size in sizes):
        raise ValueError("League / Ladder currently supports 4-player and 5-player courts only.")
    participants = _build_participants(names, resolved_ids=resolved_ids)
    participant_ids = [str(p["id"]) for p in participants]
    cursor = 0
    court_groups: list[list[str]] = []
    for size in sizes:
        court_groups.append(participant_ids[cursor:cursor + size])
        cursor += size
    return {
        "schemaVersion": 1,
        "name": normalize_name(name) or "JUPR Live League",
        "type": "league",
        "participants": participants,
        "courtSizes": sizes,
        "totalRounds": int(total_rounds),
        "currentRoundNumber": 1,
        "rounds": [_build_league_round(1, court_groups)],
        "pendingAssignments": None,
        "substitutions": [],
        "official_context": dict(official_context or {}),
        "saved_rounds": [],
    }


def round_robin_matches(event: dict[str, Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for round_data in event.get("rounds") or []:
        matches.extend(round_data.get("matches") or [])
    return matches


def current_league_round(event: dict[str, Any]) -> dict[str, Any] | None:
    current = int(event.get("currentRoundNumber") or 1)
    for round_data in event.get("rounds") or []:
        if int(round_data.get("number") or 0) == current:
            return round_data
    return None


def league_round_matches(round_data: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not round_data:
        return []
    matches: list[dict[str, Any]] = []
    for court in round_data.get("courts") or []:
        for mini_round in court.get("miniRounds") or []:
            matches.extend(mini_round.get("matches") or [])
    return matches


def set_match_score(match: dict[str, Any], score_a: int | None, score_b: int | None) -> None:
    match["scoreA"] = None if score_a in (None, "") else int(score_a)
    match["scoreB"] = None if score_b in (None, "") else int(score_b)


def match_is_scored(match: dict[str, Any] | None) -> bool:
    if not match:
        return False
    return match.get("scoreA") is not None and match.get("scoreB") is not None


def find_match_by_id(event: dict[str, Any], match_id: str) -> dict[str, Any] | None:
    event_type = str(event.get("type") or "")
    if event_type == "round_robin":
        for match in round_robin_matches(event):
            if str(match.get("id")) == str(match_id):
                return match
        return None
    if event_type == "league":
        for round_data in event.get("rounds") or []:
            for match in league_round_matches(round_data):
                if str(match.get("id")) == str(match_id):
                    return match
        return None
    return None


def round_robin_current_round_number(event: dict[str, Any]) -> int:
    rounds = event.get("rounds") or []
    if not rounds:
        return 1
    for round_data in rounds:
        matches = round_data.get("matches") or []
        if any(not match_is_scored(match) for match in matches):
            return int(round_data.get("number") or 1)
    return int(rounds[-1].get("number") or 1)


def matches_for_round(event: dict[str, Any], round_number: int) -> list[dict[str, Any]]:
    event_type = str(event.get("type") or "")
    if event_type == "round_robin":
        for round_data in event.get("rounds") or []:
            if int(round_data.get("number") or 0) == int(round_number):
                return list(round_data.get("matches") or [])
        return []
    if event_type == "league":
        for round_data in event.get("rounds") or []:
            if int(round_data.get("number") or 0) == int(round_number):
                return league_round_matches(round_data)
        return []
    return []


def _substitution_matches(substitution: dict[str, Any]) -> set[str]:
    match_ids = {str(x) for x in (substitution.get("affected_match_ids") or []) if str(x)}
    if substitution.get("match_id"):
        match_ids.add(str(substitution.get("match_id")))
    return match_ids


def _saved_round_markers(event: dict[str, Any]) -> set[str]:
    return {str(x) for x in (event.get("saved_rounds") or [])}


def substitution_is_locked(event: dict[str, Any], substitution: dict[str, Any]) -> bool:
    for match_id in _substitution_matches(substitution):
        if match_is_scored(find_match_by_id(event, match_id)):
            return True
    return False


def substitution_is_active(event: dict[str, Any], substitution: dict[str, Any]) -> bool:
    scope = str(substitution.get("scope") or "")
    round_number = int(substitution.get("round_number") or 0)
    if scope == "game":
        match_id = str(substitution.get("match_id") or "")
        return bool(match_id) and not match_is_scored(find_match_by_id(event, match_id))
    if scope == "round":
        current_round = (
            int(event.get("currentRoundNumber") or 1)
            if str(event.get("type")) == "league"
            else round_robin_current_round_number(event)
        )
        if current_round != round_number:
            return False
        return any(
            not match_is_scored(find_match_by_id(event, match_id))
            for match_id in _substitution_matches(substitution)
        )
    return False


def clear_expired_substitutions(event: dict[str, Any]) -> None:
    saved_rounds = _saved_round_markers(event)
    cleared: list[dict[str, Any]] = []
    for substitution in event.get("substitutions") or []:
        round_marker = str(substitution.get("round_number") or "")
        if "rr" in saved_rounds and str(event.get("type")) == "round_robin":
            continue
        if round_marker and round_marker in saved_rounds:
            continue
        if not _substitution_matches(substitution):
            continue
        cleared.append(substitution)
    event["substitutions"] = cleared


def get_active_sub_for_match(
    event: dict[str, Any],
    match_id: str,
    original_participant_id: str | None = None,
    *,
    include_inactive: bool = False,
) -> dict[str, Any] | None:
    substitutions = list(event.get("substitutions") or [])
    substitutions.sort(key=lambda item: str(item.get("created_at") or ""))
    game_level: dict[str, Any] | None = None
    round_level: dict[str, Any] | None = None
    for substitution in substitutions:
        if str(match_id) not in _substitution_matches(substitution):
            continue
        if (
            original_participant_id is not None
            and str(substitution.get("original_participant_id")) != str(original_participant_id)
        ):
            continue
        if not include_inactive and not substitution_is_active(event, substitution):
            continue
        if str(substitution.get("scope")) == "game":
            game_level = substitution
        elif str(substitution.get("scope")) == "round":
            round_level = substitution
    return game_level or round_level


def resolve_active_player_name(
    event: dict[str, Any],
    match_id: str,
    original_participant_id: str,
) -> str:
    participant = _participant_map(event).get(str(original_participant_id), {})
    base_name = str(participant.get("name") or original_participant_id)
    substitution = get_active_sub_for_match(event, match_id, original_participant_id, include_inactive=True)
    if not substitution:
        return base_name
    return str(substitution.get("substitute_name") or base_name)


def resolve_display_name(
    event: dict[str, Any],
    match_id: str,
    original_participant_id: str,
) -> str:
    return resolve_active_player_name(event, match_id, original_participant_id)


def resolve_player_of_record(
    event: dict[str, Any],
    match_id: str,
    original_participant_id: str,
) -> int:
    substitution = get_active_sub_for_match(event, match_id, original_participant_id, include_inactive=True)
    if substitution and substitution.get("substitute_player_id") is not None:
        return int(substitution["substitute_player_id"])
    participant = _participant_map(event).get(str(original_participant_id))
    if participant is None or participant.get("player_id") is None:
        raise ValueError(f"Participant {original_participant_id} is not resolved to a JUPR player.")
    return int(participant["player_id"])


def apply_round_substitution(
    event: dict[str, Any],
    *,
    round_number: int,
    original_participant_id: str,
    substitute_player_id: int,
    substitute_name: str,
    created_by: str,
    created_at: str = "",
    note: str = "",
    substitution_id: str | None = None,
) -> dict[str, Any]:
    affected_match_ids = [
        str(match.get("id"))
        for match in matches_for_round(event, round_number)
        if not match_is_scored(match)
        and str(original_participant_id) in [str(x) for x in (match.get("teamA") or []) + (match.get("teamB") or [])]
    ]
    if not affected_match_ids:
        raise ValueError("No remaining unscored matches are affected by that round substitution.")
    substitution = {
        "id": substitution_id or f"sub-{len(event.get('substitutions') or []) + 1}",
        "scope": "round",
        "round_number": int(round_number),
        "match_id": None,
        "original_participant_id": str(original_participant_id),
        "original_slot": f"participant:{str(original_participant_id)}",
        "original_player_name": str(
            _participant_map(event).get(str(original_participant_id), {}).get("name") or original_participant_id
        ),
        "substitute_player_id": int(substitute_player_id),
        "substitute_name": normalize_name(substitute_name),
        "affected_match_ids": affected_match_ids,
        "created_by": normalize_name(created_by) or "admin",
        "created_at": str(created_at or ""),
        "note": str(note or "").strip(),
    }
    return substitution


def apply_single_game_substitution(
    event: dict[str, Any],
    *,
    round_number: int,
    match_id: str,
    original_participant_id: str,
    substitute_player_id: int,
    substitute_name: str,
    created_by: str,
    created_at: str = "",
    note: str = "",
    substitution_id: str | None = None,
) -> dict[str, Any]:
    match = find_match_by_id(event, match_id)
    if match is None:
        raise ValueError("Match not found.")
    if match_is_scored(match):
        raise ValueError("Scored matches cannot be changed retroactively.")
    participants = [str(x) for x in (match.get("teamA") or []) + (match.get("teamB") or [])]
    if str(original_participant_id) not in participants:
        raise ValueError("Selected player is not part of that match.")
    substitution = {
        "id": substitution_id or f"sub-{len(event.get('substitutions') or []) + 1}",
        "scope": "game",
        "round_number": int(round_number),
        "match_id": str(match_id),
        "original_participant_id": str(original_participant_id),
        "original_slot": f"participant:{str(original_participant_id)}",
        "original_player_name": str(
            _participant_map(event).get(str(original_participant_id), {}).get("name") or original_participant_id
        ),
        "substitute_player_id": int(substitute_player_id),
        "substitute_name": normalize_name(substitute_name),
        "affected_match_ids": [str(match_id)],
        "created_by": normalize_name(created_by) or "admin",
        "created_at": str(created_at or ""),
        "note": str(note or "").strip(),
    }
    return substitution


def update_round_robin_score(event: dict[str, Any], match_id: str, score_a: int | None, score_b: int | None) -> None:
    for match in round_robin_matches(event):
        if str(match.get("id")) == str(match_id):
            set_match_score(match, score_a, score_b)
            return


def update_league_score(event: dict[str, Any], match_id: str, score_a: int | None, score_b: int | None) -> None:
    round_data = current_league_round(event)
    for match in league_round_matches(round_data):
        if str(match.get("id")) == str(match_id):
            set_match_score(match, score_a, score_b)
            return


def round_robin_standings(event: dict[str, Any]) -> list[dict[str, Any]]:
    return compute_standings(round_robin_matches(event), event.get("participants") or [])


def league_round_summary(event: dict[str, Any], round_number: int | None = None) -> list[dict[str, Any]]:
    number = int(round_number or event.get("currentRoundNumber") or 1)
    participant_map = _participant_map(event)
    result: list[dict[str, Any]] = []
    for round_data in event.get("rounds") or []:
        if int(round_data.get("number") or 0) != number:
            continue
        for court in round_data.get("courts") or []:
            participant_ids = [str(x) for x in (court.get("participantIds") or [])]
            participants = [participant_map[pid] for pid in participant_ids if pid in participant_map]
            matches: list[dict[str, Any]] = []
            for mini_round in court.get("miniRounds") or []:
                matches.extend(mini_round.get("matches") or [])
            standings = compute_standings(matches, participants)
            for row in standings:
                row["currentCourt"] = int(court.get("courtNumber") or 0)
            result.append(
                {
                    "courtNumber": int(court.get("courtNumber") or 0),
                    "size": int(court.get("size") or len(participant_ids)),
                    "participantIds": participant_ids,
                    "standings": standings,
                }
            )
    return sorted(result, key=lambda item: int(item["courtNumber"]))


def league_aggregate_standings(event: dict[str, Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for round_data in event.get("rounds") or []:
        matches.extend(league_round_matches(round_data))
    return compute_standings(matches, event.get("participants") or [])


def is_league_round_complete(event: dict[str, Any], round_number: int | None = None) -> bool:
    round_data = current_league_round(event) if round_number is None else next(
        (r for r in (event.get("rounds") or []) if int(r.get("number") or 0) == int(round_number)),
        None,
    )
    matches = league_round_matches(round_data)
    return bool(matches) and all(m.get("scoreA") is not None and m.get("scoreB") is not None for m in matches)


def build_league_movement(event: dict[str, Any], round_number: int | None = None) -> dict[str, Any]:
    summary = league_round_summary(event, round_number=round_number)
    assignments: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    total_courts = len(event.get("courtSizes") or [])
    for court_info in summary:
        for row in court_info.get("standings") or []:
            pid = str(row["participantId"])
            assignments[pid] = int(court_info["courtNumber"])
            rows.append(
                {
                    "participantId": pid,
                    "name": row["name"],
                    "currentCourt": int(court_info["courtNumber"]),
                    "currentRank": int(row["rank"]),
                    "wins": int(row["wins"]),
                    "losses": int(row["losses"]),
                    "ties": int(row["ties"]),
                    "pointsFor": int(row["pointsFor"]),
                    "pointsAgainst": int(row["pointsAgainst"]),
                    "differential": int(row["differential"]),
                }
            )
    by_court: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_court[int(row["currentCourt"])].append(row)
    for court_num, court_rows in sorted(by_court.items()):
        court_rows.sort(key=lambda item: (item["currentRank"], item["name"].lower()))
        if court_rows and court_num > 1:
            assignments[str(court_rows[0]["participantId"])] = int(court_num) - 1
        if court_rows and court_num < total_courts:
            assignments[str(court_rows[-1]["participantId"])] = int(court_num) + 1
    for row in rows:
        row["proposedCourt"] = int(assignments[str(row["participantId"])])
    rows.sort(key=lambda item: (item["currentCourt"], item["currentRank"], item["name"].lower()))
    return {"assignments": assignments, "rows": rows}


def validate_assignments(event: dict[str, Any], assignments: dict[str, int]) -> dict[str, Any]:
    participants = event.get("participants") or []
    court_sizes = [int(x) for x in (event.get("courtSizes") or [])]
    errors: list[str] = []
    counts = [0 for _ in court_sizes]
    for participant in participants:
        pid = str(participant["id"])
        court_number = assignments.get(pid)
        if court_number is None or int(court_number) < 1 or int(court_number) > len(court_sizes):
            errors.append(f"{participant.get('name', pid)} is assigned to an invalid court.")
            continue
        counts[int(court_number) - 1] += 1
    for idx, expected in enumerate(court_sizes, start=1):
        actual = counts[idx - 1]
        if int(actual) != int(expected):
            errors.append(f"Court {idx} needs {expected} players, but has {actual}.")
    return {"ok": not errors, "counts": counts, "errors": errors}


def set_pending_assignment(event: dict[str, Any], participant_id: str, proposed_court: int) -> None:
    pending = dict(event.get("pendingAssignments") or {})
    pending[str(participant_id)] = int(proposed_court)
    event["pendingAssignments"] = pending


def start_next_league_round(event: dict[str, Any]) -> None:
    current_round = int(event.get("currentRoundNumber") or 1)
    total_rounds = int(event.get("totalRounds") or 1)
    if current_round >= total_rounds:
        raise ValueError("All configured rounds are already complete.")
    if not is_league_round_complete(event, current_round):
        raise ValueError("Complete all scores in the current round first.")
    base = build_league_movement(event, round_number=current_round)
    assignments = dict(base["assignments"])
    assignments.update(dict(event.get("pendingAssignments") or {}))
    validation = validate_assignments(event, assignments)
    if not validation["ok"]:
        raise ValueError(" ".join(validation["errors"]))
    movement_rows = copy.deepcopy(base["rows"])
    for row in movement_rows:
        row["proposedCourt"] = int(assignments[str(row["participantId"])])
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in movement_rows:
        grouped[int(row["proposedCourt"])].append(row)
    next_courts: list[list[str]] = []
    for court_num in range(1, len(event.get("courtSizes") or []) + 1):
        rows = grouped.get(court_num, [])
        rows.sort(key=lambda item: (item["currentCourt"], item["currentRank"], item["name"].lower()))
        next_courts.append([str(row["participantId"]) for row in rows])
    next_round_number = current_round + 1
    event.setdefault("rounds", []).append(_build_league_round(next_round_number, next_courts))
    event["currentRoundNumber"] = next_round_number
    event["pendingAssignments"] = None


def _seed_order(bracket_size: int) -> list[int]:
    order = [1, 2]
    while len(order) < int(bracket_size):
        max_seed = (len(order) * 2) + 1
        expanded: list[int] = []
        for seed in order:
            expanded.extend([seed, max_seed - seed])
        order = expanded
    return order[: int(bracket_size)]


def _next_power_of_two(value: int) -> int:
    result = 1
    while result < int(value):
        result *= 2
    return result


def _build_tournament_teams(team_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    teams: list[dict[str, Any]] = []
    for idx, entry in enumerate(team_entries, start=1):
        name = normalize_name(entry.get("name")) or f"Team {idx}"
        team = {
            "id": _uid("team", idx),
            "seed": idx,
            "name": name,
            "player1_name": normalize_name(entry.get("player1_name")),
            "player2_name": normalize_name(entry.get("player2_name")),
        }
        if entry.get("player1_id") is not None:
            team["player1_id"] = int(entry["player1_id"])
        if entry.get("player2_id") is not None:
            team["player2_id"] = int(entry["player2_id"])
        teams.append(team)
    return teams


def _find_tournament_match(event: dict[str, Any], round_number: int, slot: int) -> dict[str, Any] | None:
    for round_data in event.get("rounds") or []:
        if int(round_data.get("number") or 0) != int(round_number):
            continue
        for match in round_data.get("matches") or []:
            if int(match.get("slot") or 0) == int(slot):
                return match
    return None


def create_tournament_event(
    *,
    name: str,
    team_entries: list[dict[str, Any]],
    official_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if len(team_entries) not in SUPPORTED_TOURNAMENT_TEAM_COUNTS:
        raise ValueError("Tournament currently supports 4 to 8 fixed teams in JUPR Live Beta.")
    teams = _build_tournament_teams(team_entries)
    bracket_size = _next_power_of_two(len(teams))
    order = _seed_order(bracket_size)
    seeded = {int(team["seed"]): str(team["id"]) for team in teams}
    rounds: list[dict[str, Any]] = []
    matches_in_round = bracket_size // 2
    total_rounds = 0
    temp = bracket_size
    while temp > 1:
        total_rounds += 1
        temp //= 2
    for round_number in range(1, total_rounds + 1):
        matches: list[dict[str, Any]] = []
        for slot in range(1, matches_in_round + 1):
            if round_number == 1:
                idx = (slot - 1) * 2
                participant_a = seeded.get(order[idx])
                participant_b = seeded.get(order[idx + 1])
                matches.append(
                    {
                        "slot": int(slot),
                        "name": f"Round {round_number} Match {slot}",
                        "participantAId": participant_a,
                        "participantBId": participant_b,
                        "scoreA": None,
                        "scoreB": None,
                        "winnerId": None,
                        "sourceA": None,
                        "sourceB": None,
                    }
                )
            else:
                matches.append(
                    {
                        "slot": int(slot),
                        "name": f"Round {round_number} Match {slot}",
                        "participantAId": None,
                        "participantBId": None,
                        "scoreA": None,
                        "scoreB": None,
                        "winnerId": None,
                        "sourceA": {"roundNumber": round_number - 1, "slot": (slot * 2) - 1},
                        "sourceB": {"roundNumber": round_number - 1, "slot": slot * 2},
                    }
                )
        rounds.append({"number": int(round_number), "matches": matches})
        matches_in_round = max(1, matches_in_round // 2)
    event = {
        "schemaVersion": 1,
        "type": "tournament",
        "name": normalize_name(name) or "JUPR Live Tournament",
        "teams": teams,
        "bracketSize": int(bracket_size),
        "rounds": rounds,
        "official_context": dict(official_context or {}),
        "saved_match_ids": [],
    }
    resolve_tournament(event)
    return event


def resolve_tournament(event: dict[str, Any]) -> dict[str, Any]:
    for round_data in event.get("rounds") or []:
        for match in round_data.get("matches") or []:
            incoming_a = match.get("participantAId")
            incoming_b = match.get("participantBId")
            source_a = None
            source_b = None
            if match.get("sourceA"):
                source_a = _find_tournament_match(event, match["sourceA"]["roundNumber"], match["sourceA"]["slot"])
                incoming_a = source_a.get("winnerId") if source_a else None
            if match.get("sourceB"):
                source_b = _find_tournament_match(event, match["sourceB"]["roundNumber"], match["sourceB"]["slot"])
                incoming_b = source_b.get("winnerId") if source_b else None
            participants_changed = False
            if match.get("participantAId") != incoming_a:
                participants_changed = True
            if match.get("participantBId") != incoming_b:
                participants_changed = True
            match["participantAId"] = incoming_a
            match["participantBId"] = incoming_b
            if participants_changed:
                match["scoreA"] = None
                match["scoreB"] = None
                match["winnerId"] = None
            left = match.get("participantAId")
            right = match.get("participantBId")
            source_a_pending = bool(
                source_a
                and source_a.get("winnerId") is None
                and (source_a.get("participantAId") or source_a.get("participantBId"))
            )
            source_b_pending = bool(
                source_b
                and source_b.get("winnerId") is None
                and (source_b.get("participantAId") or source_b.get("participantBId"))
            )
            if left and not right:
                if source_b_pending:
                    match["winnerId"] = None
                    continue
                match["winnerId"] = left
                continue
            if right and not left:
                if source_a_pending:
                    match["winnerId"] = None
                    continue
                match["winnerId"] = right
                continue
            if not left or not right:
                match["winnerId"] = None
                continue
            if match.get("scoreA") is None or match.get("scoreB") is None:
                match["winnerId"] = None
                continue
            if int(match.get("scoreA") or 0) == int(match.get("scoreB") or 0):
                match["winnerId"] = None
                continue
            match["winnerId"] = left if int(match.get("scoreA") or 0) > int(match.get("scoreB") or 0) else right
    return event


def update_tournament_score(event: dict[str, Any], round_number: int, slot: int, score_a: int | None, score_b: int | None) -> None:
    match = _find_tournament_match(event, round_number, slot)
    if match is None:
        return
    set_match_score(match, score_a, score_b)
    resolve_tournament(event)


def tournament_champion(event: dict[str, Any]) -> str | None:
    rounds = event.get("rounds") or []
    if not rounds:
        return None
    final_round = rounds[-1]
    if not final_round.get("matches"):
        return None
    return final_round["matches"][0].get("winnerId")


def tournament_team_map(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(team["id"]): team for team in (event.get("teams") or [])}


def tournament_bracket_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    team_map = tournament_team_map(event)
    rows: list[dict[str, Any]] = []
    for round_data in event.get("rounds") or []:
        for match in round_data.get("matches") or []:
            team_a = team_map.get(str(match.get("participantAId")))
            team_b = team_map.get(str(match.get("participantBId")))
            winner = team_map.get(str(match.get("winnerId")))
            rows.append(
                {
                    "round_number": int(round_data.get("number") or 0),
                    "slot": int(match.get("slot") or 0),
                    "match_name": str(match.get("name") or f"Round {round_data.get('number')} Match {match.get('slot')}") ,
                    "team_a": team_a.get("name") if team_a else "TBD",
                    "team_b": team_b.get("name") if team_b else "TBD",
                    "score_a": match.get("scoreA"),
                    "score_b": match.get("scoreB"),
                    "winner": winner.get("name") if winner else "Pending",
                    "match_id": f"r{int(round_data.get('number') or 0)}-s{int(match.get('slot') or 0)}",
                }
            )
    return rows


def tournament_completed_match_payloads(event: dict[str, Any], *, unsaved_only: bool = False) -> list[dict[str, Any]]:
    team_map = tournament_team_map(event)
    official_context = dict(event.get("official_context") or {})
    tournament = {
        "id": official_context.get("tournament_id") or f"jupr-live-{normalize_name(event.get('name')).lower().replace(' ', '-')}",
        "name": event.get("name") or "Tournament",
    }
    saved_match_ids = set(event.get("saved_match_ids") or [])
    payloads: list[dict[str, Any]] = []
    for round_data in event.get("rounds") or []:
        for match in round_data.get("matches") or []:
            if match.get("scoreA") is None or match.get("scoreB") is None:
                continue
            if match.get("winnerId") is None:
                continue
            game_id = f"r{int(round_data.get('number') or 0)}-s{int(match.get('slot') or 0)}"
            if unsaved_only and game_id in saved_match_ids:
                continue
            payload = build_tournament_match_payload(
                tournament,
                {
                    "id": game_id,
                    "team_a_id": match.get("participantAId"),
                    "team_b_id": match.get("participantBId"),
                },
                team_map,
                score_a=int(match.get("scoreA") or 0),
                score_b=int(match.get("scoreB") or 0),
            )
            payloads.append(payload)
    return payloads


def mark_tournament_matches_saved(event: dict[str, Any], payloads: list[dict[str, Any]]) -> None:
    saved = set(event.get("saved_match_ids") or [])
    for payload in payloads:
        saved.add(str(payload.get("tournament_game_id")))
    event["saved_match_ids"] = sorted(saved)


def export_event_json(event: dict[str, Any]) -> str:
    return json.dumps(event, indent=2, sort_keys=False)


def match_payloads_from_rr(event: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for round_data in event.get("rounds") or []:
        for match in round_data.get("matches") or []:
            if match.get("scoreA") is None or match.get("scoreB") is None:
                continue
            team_a = [str(x) for x in (match.get("teamA") or [])]
            team_b = [str(x) for x in (match.get("teamB") or [])]
            payloads.append(
                {
                    "round_number": int(round_data.get("number") or 0),
                    "match_id": str(match.get("id")),
                    "t1_p1": team_a[0],
                    "t1_p2": team_a[1],
                    "t2_p1": team_b[0],
                    "t2_p2": team_b[1],
                    "s1": int(match.get("scoreA") or 0),
                    "s2": int(match.get("scoreB") or 0),
                }
            )
    return payloads


def match_payloads_from_current_league_round(event: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    round_data = current_league_round(event)
    if not round_data:
        return payloads
    for court in round_data.get("courts") or []:
        for mini_round in court.get("miniRounds") or []:
            for match in mini_round.get("matches") or []:
                if match.get("scoreA") is None or match.get("scoreB") is None:
                    continue
                team_a = [str(x) for x in (match.get("teamA") or [])]
                team_b = [str(x) for x in (match.get("teamB") or [])]
                payloads.append(
                    {
                        "round_number": int(round_data.get("number") or 0),
                        "court_number": int(court.get("courtNumber") or 0),
                        "mini_round_number": int(mini_round.get("number") or 0),
                        "match_id": str(match.get("id")),
                        "t1_p1": team_a[0],
                        "t1_p2": team_a[1],
                        "t2_p1": team_b[0],
                        "t2_p2": team_b[1],
                        "s1": int(match.get("scoreA") or 0),
                        "s2": int(match.get("scoreB") or 0),
                    }
                )
    return payloads


def resolve_payload_player_ids(
    event: dict[str, Any],
    payloads: list[dict[str, Any]],
    *,
    materialize_substitutions: bool = False,
) -> list[dict[str, Any]]:
    participant_map = _participant_map(event)
    resolved_payloads: list[dict[str, Any]] = []
    for payload in payloads:
        resolved = dict(payload)
        match_id = str(payload.get("match_id") or "")
        for key in ["t1_p1", "t1_p2", "t2_p1", "t2_p2"]:
            participant_id = str(resolved.get(key))
            if materialize_substitutions and match_id:
                resolved[key] = resolve_player_of_record(event, match_id, participant_id)
                continue
            participant = participant_map.get(participant_id)
            if participant is None or participant.get("player_id") is None:
                raise ValueError(f"Participant for {key} is not resolved to a JUPR player.")
            resolved[key] = int(participant["player_id"])
        resolved_payloads.append(resolved)
    return resolved_payloads


def standings_csv_rows(standings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in standings:
        rows.append(
            {
                "Rank": int(row.get("rank") or 0),
                "Player": str(row.get("name") or ""),
                "W": int(row.get("wins") or 0),
                "L": int(row.get("losses") or 0),
                "T": int(row.get("ties") or 0),
                "GP": int(row.get("matches") or 0),
                "PF": int(row.get("pointsFor") or 0),
                "PA": int(row.get("pointsAgainst") or 0),
                "Diff": int(row.get("differential") or 0),
            }
        )
    return rows
