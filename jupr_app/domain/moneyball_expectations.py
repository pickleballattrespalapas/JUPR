"""Frozen Moneyball rotations and provisional/final standings."""
from __future__ import annotations

import copy
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any
from threading import Lock

import numpy as np

from jupr_app.domain import moneyball_probability as probability

FORMAT = "moneyball_expectations"
_PREPARATION_LOCK = Lock()


def is_moneyball(event: dict[str, Any]) -> bool:
    return event.get("eventFormat") == FORMAT


def validate_score(a: Any, b: Any, target_score: int = 11) -> None:
    if type(target_score) is not int or target_score not in {11, 15}:
        raise ValueError("Moneyball supports games to 11 or 15, win by two.")
    if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int)
            or not isinstance(b, int) or min(a, b) < 0 or abs(a-b) < 2
            or max(a, b) < target_score or (max(a, b) > target_score and abs(a-b) != 2)
            or abs(a-b) > target_score):
        raise ValueError(f"Moneyball games are to {target_score}, win by two. Enter a completed game score.")


def config_from_event(event: dict[str, Any], ratings: dict[str, float]) -> dict[str, Any]:
    slots = {p["id"]: p.get("moneyballSlot", p["id"]) for p in event["participants"]}
    config = {"name": event["name"], "players": [
        {"id": p["id"], "name": p["name"], "rating": ratings[p["id"]]}
        for p in event["participants"] if p["id"] in ratings
    ], "matches": [
        {"id": m["id"], "round": r["number"], "team1": [slots[pid] for pid in m["sideA"]], "team2": [slots[pid] for pid in m["sideB"]]}
        for r in event["rounds"] for m in r["matches"]
    ]}
    goal = (event.get("moneyball") or {}).get("targetScore", 11)
    if goal != 11:
        config["goal_points"] = goal
    return config


@lru_cache(maxsize=32)
def _prepare(config_json: str) -> dict[str, Any]:
    model = json.loads(Path(__file__).with_name("moneyball_model_v1.json").read_text())
    config = json.loads(config_json)
    model = probability.model_for_target(model, config.get("goal_points", 11))
    lock = probability.prepare(config, model,
                               calibration_sims=500000, audit_sims=1000000)
    if not lock["fairness"]["simulation_fairness_check_passed"]:
        raise ValueError("This Moneyball setup did not pass the balance check. Please use a standard round robin.")
    return lock


def freeze_moneyball(event: dict[str, Any], ratings: dict[str, float], *, target_score: int = 11) -> dict[str, Any]:
    event = copy.deepcopy(event)
    if type(target_score) is not int or target_score not in {11, 15}:
        raise ValueError("Moneyball supports games to 11 or 15, win by two.")
    event["moneyball"] = {"targetScore": target_score}
    config = config_from_event(event, ratings)
    probability.validate_config(config)
    # Bound simulation memory when several organizers preview concurrently.
    with _PREPARATION_LOCK:
        lock = copy.deepcopy(_prepare(json.dumps(config, sort_keys=True)))
    event["moneyballPrivate"] = lock
    event["standingsSort"] = FORMAT
    event["moneyball"] = {
        "label": "Moneyball vs Expectations", "version": lock["version"],
        "entryFeeMxn": 100, "prizePoolMxn": 100*len(config["players"]),
        "targetScore": target_score, "winBy": 2, "requiredRounds": len(event["rounds"]), "startingPlayers": len(config["players"]),
        "ratingsFrozen": True, "experimental": True,
        "tiePolicy": "Tied winners split the pool.",
    }
    if target_score == 15:
        event["moneyball"]["modelNote"] = (
            "Games to 15 use an experimental extension of the 11-point expectation model. "
            "It has not been validated against historical 15-point results."
        )
    if len(config["players"]) == 6:
        event["moneyball"]["rotationSummary"] = (
            "Balanced six-player rotation: 1 court, 6 rounds. Everyone plays 4 games with 4 different partners "
            "and has 2 byes, one in each three-round block, with no consecutive byes. "
            "Each opponent is faced once or twice. This is not a perfect round robin."
        )
    event["previewFingerprint"] = probability.digest({
        "schedule": event["previewFingerprint"], "setup": lock["lock_sha256"],
    })
    return event


def convert_existing_moneyball(event: dict[str, Any], ratings: dict[str, float], *,
                              player_ids: dict[str, int], target_score: int,
                              converted_at: str) -> dict[str, Any]:
    """Prepare an in-place conversion; persistence must compare the session version.

    Caller supplies verified club-scoped player IDs and starting ratings. This
    function never regenerates a round or changes scores, dates, or event identity.
    """
    if (is_moneyball(event) or event.get("moneyballPrivate") or event.get("playoff")
            or event.get("rosterRevisions") or event.get("publishedMatchIds")
            or event.get("generatorKind") != "round_robin" or event.get("playFormat") != "doubles"
            or event.get("scoringMode") != "scored" or event.get("status") not in {"active", "completed"}):
        raise ValueError("Convert only an unpublished, fixed doubles round robin with its original lineup.")
    participants = event.get("participants", [])
    ids = {p["id"] for p in participants}
    if (set(ratings) != ids or set(player_ids) != ids
            or any(type(v) is not int or v <= 0 for v in player_ids.values())
            or len(set(player_ids.values())) != len(ids)):
        raise ValueError("Every starting player needs a distinct club profile and a verified starting rating.")
    result = copy.deepcopy(event)
    for p in result["participants"]:
        if p.get("player_id") not in (None, player_ids[p["id"]]):
            raise ValueError("Conversion cannot replace an existing player profile.")
        p["player_id"] = player_ids[p["id"]]
    for row in result["rounds"]:
        if row.get("status") not in {"saved", "active", "preview"}:
            raise ValueError("Every scheduled game must remain playable; skipped rounds cannot be converted.")
        for match in row["matches"]:
            a, b = match.get("scoreA"), match.get("scoreB")
            if row["status"] == "saved" or a is not None or b is not None:
                validate_score(a, b, target_score)
    result["eventFormat"] = FORMAT
    result = freeze_moneyball(result, ratings, target_score=target_score)
    result["moneyball"]["convertedAt"] = converted_at
    result["moneyball"]["convertedAfterGames"] = sum(len(r["matches"]) for r in event["rounds"] if r["status"] == "saved")
    result["moneyball"]["conversionNote"] = (
        "Converted from the existing round robin. Matchups and saved scores were preserved. "
        "Starting ratings were supplied at conversion; played results were not used to set expectations."
    )
    standings(result)
    return result


def public_event(event: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(event)
    result.pop("moneyballPrivate", None)
    if is_moneyball(event) and event.get("moneyballPrivate"):
        result["moneyballExpectations"] = full_event_expectations(event)
    return result


def _verified_lock(event: dict[str, Any]) -> dict[str, Any]:
    lock = event.get("moneyballPrivate")
    if not lock:
        raise ValueError("The frozen Moneyball setup is missing. Start a new event.")
    verified = dict(lock)
    supplied = verified.pop("lock_sha256", None)
    if probability.digest(verified) != supplied or not lock["fairness"]["simulation_fairness_check_passed"]:
        raise ValueError("The frozen Moneyball setup changed. Start a new event.")
    ratings = {p["id"]: p["rating"] for p in lock["config"]["players"]}
    try:
        current = config_from_event(event, ratings)
    except (KeyError, TypeError) as exc:
        raise ValueError("Moneyball players and matchups must stay fixed after setup.") from exc
    # Title edits are permitted; player identities and schedule stay frozen.
    current["name"] = lock["config"]["name"]
    if probability.digest(current) != lock["config_sha256"]:
        raise ValueError("Moneyball players and matchups must stay fixed after setup.")
    if lock["model"]["goal_points"] != current.get("goal_points", 11):
        raise ValueError("Moneyball scoring target must match its frozen model.")
    return lock


def full_event_expectations(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Public starting-lineup totals from every frozen game, excluding byes."""
    lock = _verified_lock(event)
    support = np.array(lock["model"]["margin_support"], dtype=int)
    rows = {p["id"]: {
        "participantId": p["id"], "name": p["name"], "scheduledGames": 0,
        "expectedWins": 0., "expectedLosses": 0., "expectedDifferential": 0.,
    } for p in lock["config"]["players"]}
    for match, probabilities in zip(lock["config"]["matches"], lock["margin_probabilities"]):
        p = np.array(probabilities)
        for team, sign in (("team1", 1), ("team2", -1)):
            wins = float(p[support * sign > 0].sum())
            differential = float(p @ support) * sign
            for pid in match[team]:
                row = rows[pid]
                row["scheduledGames"] += 1
                row["expectedWins"] += wins
                row["expectedDifferential"] += differential
    for row in rows.values():
        row["expectedLosses"] = row["scheduledGames"] - row["expectedWins"]
    return list(rows.values())


def all_games_saved(event: dict[str, Any]) -> bool:
    required = len((event.get("moneyballPrivate") or {}).get("config", {}).get("matches", []))
    return bool(required) and sum(len(r.get("matches", [])) for r in event.get("rounds", [])) == required and all(r.get("status") == "saved" for r in event["rounds"])


def standings(event: dict[str, Any]) -> list[dict[str, Any]]:
    lock = _verified_lock(event)
    config, model = lock["config"], lock["model"]
    support = np.array(model["margin_support"], dtype=int)
    bound = int(np.max(np.abs(support)))
    probabilities = np.array(lock["margin_probabilities"])
    matches = {m["id"]: (r, m) for r in event["rounds"] for m in r["matches"]}
    final = all_games_saved(event)
    rows = []
    disqualified = set(event.get("moneyballDisqualified", []))
    for i, player in enumerate(config["players"]):
        pid = player["id"]
        row = {"participantId": pid, "name": player["name"], "matches": 0,
               "wins": 0, "losses": 0, "pointsFor": 0, "pointsAgainst": 0,
               "differential": 0, "expectedWins": 0., "expectedDifferential": 0.,
               "moneyballScore": None, "moneyballFinal": final, "standingsSort": FORMAT,
               "moneyballEligible": pid not in disqualified}
        pmf = np.array([1.])
        for j, frozen in enumerate(config["matches"]):
            r, m = matches[frozen["id"]]
            if r.get("status") != "saved" or pid not in m["sideA"]+m["sideB"]:
                continue
            a, b = m.get("scoreA"), m.get("scoreB")
            validate_score(a, b, model["goal_points"])
            sign = 1 if pid in frozen["team1"] else -1
            scored, conceded = (a, b) if sign == 1 else (b, a)
            row["matches"] += 1
            row["wins"] += int(scored > conceded)
            row["losses"] += int(scored < conceded)
            row["pointsFor"] += scored
            row["pointsAgainst"] += conceded
            row["differential"] += scored-conceded
            p = probabilities[j]
            row["expectedWins"] += float(p[support*sign > 0].sum())
            row["expectedDifferential"] += float(p @ support)*sign
            single = np.zeros(2*bound+1)
            single[support*sign+bound] = p
            pmf = np.convolve(pmf, single)
        if row["matches"] and row["moneyballEligible"]:
            position = row["differential"]+bound*row["matches"]
            u = float(np.clip(pmf[:position].sum()+.5*pmf[position], 1e-12, 1-1e-12))
            z = math.log(u)-math.log1p(-u)+lock["balance_logit_offsets"][i]
            row["moneyballScore"] = 100/(1+math.exp(-z))
        rows.append(row)
    for player in event["participants"]:
        if player["id"] not in disqualified or player["id"] in {p["id"] for p in config["players"]}:
            continue
        pid = player["id"]
        row = {"participantId": pid, "name": player["name"], "matches": 0, "wins": 0, "losses": 0,
               "pointsFor": 0, "pointsAgainst": 0, "differential": 0, "expectedWins": None,
               "expectedDifferential": None, "moneyballScore": None, "moneyballFinal": final,
               "standingsSort": FORMAT, "moneyballEligible": False}
        for r, m in matches.values():
            if r.get("status") != "saved" or pid not in m["sideA"]+m["sideB"]:
                continue
            a, b = m["scoreA"], m["scoreB"]
            own, other = (a,b) if pid in m["sideA"] else (b,a)
            row["matches"] += 1
            row["wins"] += int(own > other)
            row["losses"] += int(own < other)
            row["pointsFor"] += own
            row["pointsAgainst"] += other
            row["differential"] += own-other
        rows.append(row)
    rows.sort(key=lambda r: (not r["moneyballEligible"], -(r["moneyballScore"] if r["moneyballScore"] is not None else -1)))
    for index, row in enumerate(rows):
        if not row["moneyballEligible"]:
            row["rank"] = 0
            continue
        tied = index > 0 and (row["moneyballScore"] is None or
            abs(row["moneyballScore"]-rows[index-1]["moneyballScore"]) < 1e-10)
        row["rank"] = rows[index-1]["rank"] if tied else index+1
    return rows


def settlement(event: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(event.get("moneyball") or {})
    final = event.get("status") == "completed" and all_games_saved(event)
    winners = [r for r in rows if r["rank"] == 1 and r.get("moneyballEligible", True)] if final else []
    result.update(final=final, provisional=not final, completedGames=sum(r["matches"] for r in rows)//4,
                  totalGames=len(event["moneyballPrivate"]["config"]["matches"]), winners=[],
                  noEligibleWinner=final and not winners)
    if winners:
        cents, remainder = divmod(round(result["prizePoolMxn"]*100), len(winners))
        result["winners"] = [{"participantId": r["participantId"], "name": r["name"],
                              "payoutMxn": (cents+int(i < remainder))/100}
                             for i, r in enumerate(winners)]
    return result


def substitute_moneyball(event: dict[str, Any], *, participant_id: str | None,
                         name: str | None, player_id: int | None, scope: str) -> dict[str, Any]:
    """Replace a frozen lineup slot without regenerating or recalibrating games."""
    _verified_lock(event)
    if event.get("status") != "active":
        raise ValueError("Substitutions are allowed only during an active Moneyball event.")
    if scope not in {"round", "rest"}:
        raise ValueError("Choose this round or the rest of the session.")
    by_id = {p["id"]: p for p in event["participants"]}
    original = by_id.get(participant_id)
    clean_name = " ".join(str(name or "").split())[:160]
    if original is None or not clean_name:
        raise ValueError("Choose a current player and enter the substitute’s name.")
    if any(p["name"].casefold() == clean_name.casefold() or
           (player_id is not None and p.get("player_id") == int(player_id)) for p in event["participants"]):
        raise ValueError("The substitute must be a different player who is not already in this event.")
    current = int(event["currentRoundNumber"])
    candidates = [r for r in event["rounds"] if r.get("status") in {"active", "preview"}
                  and (scope == "rest" or r["number"] == current)]
    affected = [(r,m) for r in candidates for m in r["matches"]
                if participant_id in m["sideA"]+m["sideB"]]
    if not affected:
        raise ValueError("This player has no unplayed games in the selected substitution scope.")
    if any(m.get("scoreA") is not None or m.get("scoreB") is not None for _,m in affected):
        raise ValueError("Save or clear entered scores before substituting.")
    next_event = copy.deepcopy(event)
    counter = 1
    while f"p-sub-{counter}" in by_id:
        counter += 1
    new_id = f"p-sub-{counter}"
    slot = original.get("moneyballSlot", participant_id)
    new = {"id": new_id, "name": clean_name, "moneyballSlot": slot,
           "substitutes_for": participant_id, "roster_order": original.get("roster_order", 0)}
    if player_id is not None:
        new["player_id"] = int(player_id)
    next_event["participants"].append(new)
    match_ids = {m["id"] for _,m in affected}
    candidate_numbers = {r["number"] for r in candidates}
    for r in next_event["rounds"]:
        if r["number"] in candidate_numbers:
            r["byeParticipantIds"] = [new_id if pid == participant_id else pid for pid in r.get("byeParticipantIds", [])]
        for m in r["matches"]:
            if m["id"] in match_ids:
                for side in ("sideA", "sideB"):
                    m[side] = [new_id if pid == participant_id else pid for pid in m[side]]
    dq = set(next_event.get("moneyballDisqualified", []))
    dq.update([participant_id, slot, new_id])
    next_event["moneyballDisqualified"] = sorted(dq)
    next_event.setdefault("moneyballSubstitutions", []).append({"originalId": participant_id,
        "substituteId": new_id, "slotId": slot, "matchIds": sorted(match_ids), "scope": scope})
    _verified_lock(next_event)
    return next_event
