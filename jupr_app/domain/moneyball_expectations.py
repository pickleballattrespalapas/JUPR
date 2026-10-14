"""Frozen eight-player Moneyball format and provisional/final standings."""
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


def validate_score(a: Any, b: Any) -> None:
    if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, int)
            or not isinstance(b, int) or min(a, b) < 0 or abs(a-b) < 2
            or max(a, b) < 11 or (max(a, b) > 11 and abs(a-b) != 2)
            or abs(a-b) > 11):
        raise ValueError("Moneyball games are to 11, win by two. Enter a completed game score.")


def config_from_event(event: dict[str, Any], ratings: dict[str, float]) -> dict[str, Any]:
    return {"name": event["name"], "players": [
        {"id": p["id"], "name": p["name"], "rating": ratings[p["id"]]}
        for p in event["participants"]
    ], "matches": [
        {"id": m["id"], "round": r["number"], "team1": m["sideA"], "team2": m["sideB"]}
        for r in event["rounds"] for m in r["matches"]
    ]}


@lru_cache(maxsize=32)
def _prepare(config_json: str) -> dict[str, Any]:
    model = json.loads(Path(__file__).with_name("moneyball_model_v1.json").read_text())
    lock = probability.prepare(json.loads(config_json), model,
                               calibration_sims=500000, audit_sims=1000000)
    if not lock["fairness"]["simulation_fairness_check_passed"]:
        raise ValueError("This Moneyball setup did not pass the balance check. Please use a standard round robin.")
    return lock


def freeze_moneyball(event: dict[str, Any], ratings: dict[str, float]) -> dict[str, Any]:
    event = copy.deepcopy(event)
    config = config_from_event(event, ratings)
    probability.validate_config(config)
    # Bound simulation memory when several organizers preview concurrently.
    with _PREPARATION_LOCK:
        lock = copy.deepcopy(_prepare(json.dumps(config, sort_keys=True)))
    event["moneyballPrivate"] = lock
    event["standingsSort"] = FORMAT
    event["moneyball"] = {
        "label": "Moneyball vs Expectations", "version": lock["version"],
        "entryFeeMxn": 100, "prizePoolMxn": 800,
        "targetScore": 11, "winBy": 2, "requiredRounds": 7,
        "ratingsFrozen": True, "experimental": True,
        "tiePolicy": "Tied winners split the pool.",
    }
    event["previewFingerprint"] = probability.digest({
        "schedule": event["previewFingerprint"], "setup": lock["lock_sha256"],
    })
    return event


def public_event(event: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(event)
    result.pop("moneyballPrivate", None)
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
    return lock


def all_games_saved(event: dict[str, Any]) -> bool:
    return len(event.get("rounds", [])) == 7 and all(r.get("status") == "saved" for r in event["rounds"])


def standings(event: dict[str, Any]) -> list[dict[str, Any]]:
    lock = _verified_lock(event)
    config, model = lock["config"], lock["model"]
    support = np.array(model["margin_support"], dtype=int)
    probabilities = np.array(lock["margin_probabilities"])
    matches = {m["id"]: (r, m) for r in event["rounds"] for m in r["matches"]}
    final = all_games_saved(event)
    rows = []
    for i, player in enumerate(config["players"]):
        pid = player["id"]
        row = {"participantId": pid, "name": player["name"], "matches": 0,
               "wins": 0, "losses": 0, "pointsFor": 0, "pointsAgainst": 0,
               "differential": 0, "expectedWins": 0., "expectedDifferential": 0.,
               "moneyballScore": None, "moneyballFinal": final, "standingsSort": FORMAT}
        pmf = np.array([1.])
        for j, frozen in enumerate(config["matches"]):
            r, m = matches[frozen["id"]]
            if r.get("status") != "saved" or pid not in frozen["team1"]+frozen["team2"]:
                continue
            a, b = m.get("scoreA"), m.get("scoreB")
            validate_score(a, b)
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
            single = np.zeros(23)
            single[support*sign+11] = p
            pmf = np.convolve(pmf, single)
        if row["matches"]:
            position = row["differential"]+11*row["matches"]
            u = float(np.clip(pmf[:position].sum()+.5*pmf[position], 1e-12, 1-1e-12))
            z = math.log(u)-math.log1p(-u)+lock["balance_logit_offsets"][i]
            row["moneyballScore"] = 100/(1+math.exp(-z))
        rows.append(row)
    rows.sort(key=lambda r: -(r["moneyballScore"] if r["moneyballScore"] is not None else -1))
    for index, row in enumerate(rows):
        tied = index > 0 and (row["moneyballScore"] is None or
            abs(row["moneyballScore"]-rows[index-1]["moneyballScore"]) < 1e-10)
        row["rank"] = rows[index-1]["rank"] if tied else index+1
    return rows


def settlement(event: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = dict(event.get("moneyball") or {})
    final = event.get("status") == "completed" and all_games_saved(event)
    winners = [r for r in rows if r["rank"] == 1] if final else []
    result.update(final=final, provisional=not final, completedGames=sum(r["matches"] for r in rows)//4,
                  totalGames=14, winners=[])
    if winners:
        cents, remainder = divmod(80000, len(winners))
        result["winners"] = [{"participantId": r["participantId"], "name": r["name"],
                              "payoutMxn": (cents+int(i < remainder))/100}
                             for i, r in enumerate(winners)]
    return result
