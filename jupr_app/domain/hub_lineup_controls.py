"""Atomic operator edits to an existing hub plan.

These functions only edit the supplied in-memory event set. The hub command
service remains responsible for its version check and durable atomic commit.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def set_hub_match_lineup(hub: dict[str, Any], events: dict[str, dict[str, Any]],
                         payload: dict[str, Any], *, now: str) -> None:
    """Set two exact doubles teams, exchanging displaced waiting players safely."""
    from jupr_app.domain.round_robin_hub import (
        _edit_match_players, _find_match, _matches, _participants, _complete,
        _roster_change_unavailable_reason, _key, player_identity, _identity_data,
        _ensure_players, _claims, _actually_committed_match,
    )
    from jupr_app.domain.adaptive_play_engine import active_participant_ids, generator_match_play_format

    sides = [payload.get("side_a"), payload.get("side_b")]
    if any(not isinstance(side, list) or len(side) != 2 or
           any(not isinstance(pid, str) or not pid for pid in side) for side in sides):
        raise ValueError("Choose two players for each team.")
    desired = [*sides[0], *sides[1]]
    if len(set(desired)) != 4:
        raise ValueError("Choose four different players.")

    # A failed swap or validation must leave both the court state and plan intact.
    state, updated = deepcopy(hub), deepcopy(events)
    key = str(payload.get("session_key") or "")
    event, row, match = _find_match(state, updated, payload)
    reason = _roster_change_unavailable_reason(event)
    if reason:
        raise ValueError(reason)
    if generator_match_play_format(match, event.get("playFormat") or "doubles") != "doubles":
        raise ValueError("Setting both teams requires a doubles match.")
    if row.get("status") not in {"active", "preview"} or _complete(match, event) or any(
            match.get(field) is not None for field in ("scoreA", "scoreB")):
        raise ValueError("Recorded games keep their players.")
    assignment = next(((court, slot) for court in state.get("courts") or []
        for slot in ("current", "next") if court.get(slot) and _key(court[slot]) == _key(payload)), None)
    if payload.get("court_id"):
        if not assignment or assignment[0]["id"] != payload["court_id"] or assignment[1] != "current":
            raise ValueError("The court assignment changed. Reload before editing players.")
    elif assignment:
        raise ValueError("This game now has a court assignment. Open Edit players from its court.")

    number = int(row["number"])
    future_planning = (not assignment and number > int(event.get("currentRoundNumber") or 1)
        and not _actually_committed_match(event, match))
    by_id = {str(p["id"]): p for p in event.get("participants") or []}
    active = set(active_participant_ids(event, number))
    if any(pid not in active or pid not in by_id for pid in desired):
        raise ValueError("Choose players active in this round.")
    _ensure_players(state, updated, now)
    identities = [player_identity(by_id[pid], key, state) for pid in desired]
    if len(set(identities)) != 4:
        raise ValueError("The same person cannot appear twice in a game.")
    if any(state["players"][identity].get("paused") for identity in identities):
        raise ValueError("Resume these players before adding them to a game.")
    if set(identities) & _identity_data(state, updated)["pending"]:
        raise ValueError("Confirm matching player names in Players before editing this match.")
    claims = _claims(state, updated)
    def conflicting_claim(identity: str) -> bool:
        claim = claims.get(identity)
        if claim is None or claim == _key(payload):
            return False
        if not future_planning:
            return True
        if claim[0] != key:
            return False  # Other groups keep their claims; dispatch still waits.
        claimed_round = next((int(planned["number"]) for planned in event.get("rounds") or []
            if any(str(other["id"]) == claim[1] for other in _matches(planned))), number)
        return claimed_round >= number

    if any(conflicting_claim(identity) for identity in identities):
        raise ValueError("A selected player is already assigned or reserved on another court.")

    original = [str(p["id"]) for p in _participants(event, match)]
    outgoing = [pid for pid in original if pid not in desired]
    incoming = [pid for pid in desired if pid not in original]
    incoming_set = set(incoming)
    donor_matches = [other for other in _matches(row) if other is not match and
        incoming_set & {str(p["id"]) for p in _participants(event, other)}]
    if any(other.get("hubPlanLocked") for other in donor_matches):
        raise ValueError("A selected player has another locked lineup. Edit that lineup first.")
    originals = deepcopy([match, *donor_matches])
    audit_count = len(event.get("hubLineupEdits") or [])
    for target, replacement in zip(outgoing, incoming):
        # The small-swap command locks each affected game. All donor games were
        # unlocked before this atomic edit, so a second incoming player may
        # still come from the same donor; publish its final lock only at commit.
        for donor in donor_matches:
            donor.pop("hubPlanLocked", None)
        _edit_match_players(state, updated, {
            "session_key": key, "match_id": str(match["id"]), "round_number": number,
            **({"court_id": payload["court_id"]} if payload.get("court_id") else {}),
            "participant_id": target, "replacement_participant_id": replacement,
        }, now, _planning=future_planning)

    event, row, match = _find_match(state, updated, payload)
    match["sideA"], match["teamA"] = list(sides[0]), list(sides[0])
    match["sideB"], match["teamB"] = list(sides[1]), list(sides[1])
    affected_ids = {str(item["id"]) for item in originals}
    for other in _matches(row):
        if str(other["id"]) in affected_ids:
            people = [player_identity(p, key, state) for p in _participants(event, other)]
            if len(people) != len(set(people)):
                raise ValueError("The same person cannot appear twice in a game.")
            other["hubPlanLocked"] = True
    event["hubLineupEdits"] = list(event.get("hubLineupEdits") or [])[:audit_count] + [{
        "at": now, "round": number, "matchId": str(match["id"]), "action": "set_match_lineup",
        "sideA": list(sides[0]), "sideB": list(sides[1]), "originalMatches": originals,
    }]
    event.update(perfectRotation=False, adaptiveFromRound=min(int(event.get("adaptiveFromRound") or number), number))
    hub.clear()
    hub.update(state)
    events.clear()
    events.update(updated)


def remove_hub_players(event: dict[str, Any], participant_ids: list[str], *, now: str) -> dict[str, Any]:
    """Validate all departures, then repair the shared remaining plan once."""
    from jupr_app.domain.adaptive_play_engine import mutate_generator_roster, _participant_active_in_round
    from jupr_app.domain.round_robin_hub import (
        _roster_change_unavailable_reason, _can_remove_participant, _departure_round,
        repair_hub_roster_plan,
    )

    if (not isinstance(participant_ids, list) or not 1 <= len(participant_ids) <= 40 or
            any(not isinstance(pid, str) or not pid for pid in participant_ids) or
            len(set(participant_ids)) != len(participant_ids)):
        raise ValueError("Choose between one and 40 different players to remove.")
    reason = _roster_change_unavailable_reason(event)
    if reason:
        raise ValueError(reason)
    by_id = {str(p["id"]): p for p in event.get("participants") or []}
    for pid in participant_ids:
        if pid not in by_id:
            raise ValueError("Choose players in this round robin.")
        if not _can_remove_participant(event, by_id[pid]):
            raise ValueError("A selected player has already left this round robin.")
    departures = {pid: _departure_round(event, pid) for pid in participant_ids}
    result = deepcopy(event)
    for pid in participant_ids:
        result = mutate_generator_roster(result, action="remove", participant_id=pid, _hub_managed=True)
        participant = next(p for p in result["participants"] if str(p["id"]) == pid)
        participant["inactive_from_round"] = departures[pid]
        result["rosterRevisions"][-1].update(effectiveRound=departures[pid], at=now)
    by_id = {str(p["id"]): p for p in result["participants"]}
    changed = set(participant_ids)
    current = int(result.get("currentRoundNumber") or 1)
    for number, ids in list((result.get("rosterByRound") or {}).items()):
        if int(number) >= current:
            result["rosterByRound"][number] = [str(pid) for pid in ids if
                str(pid) not in changed or _participant_active_in_round(by_id[str(pid)], int(number))]
    result.setdefault("hubBatchDepartures", []).append({"at": now, "effectiveRounds": departures})
    return repair_hub_roster_plan(result)
