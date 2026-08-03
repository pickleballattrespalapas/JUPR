"""Adaptive Round-Robin and Ladder scheduling engine.

The engine is intentionally deterministic: the same roster order, history, court
count, and round number produce the same schedule. Completed and skipped rounds
are immutable inputs when future rounds are regenerated.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


STANDINGS_SORTS = {"wins", "points", "differential"}
SCORING_MODES = {"scored", "unscored"}
PLAY_FORMATS = {"singles", "doubles", "doubles_singles"}


def normalize_play_format(value: Any) -> str:
    play_format = str(value or "doubles").strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "mixed": "doubles_singles",
        "mixed_doubles_singles": "doubles_singles",
        "doubles_singles_mix": "doubles_singles",
        "singles_doubles": "doubles_singles",
        "singles_doubles_mix": "doubles_singles",
    }
    play_format = aliases.get(play_format, play_format)
    return play_format if play_format in PLAY_FORMATS else "doubles"


def generator_match_play_format(match: dict[str, Any], default_format: str = "doubles") -> str:
    """Return the actual singles/doubles format for one scheduled match.

    New mixed-format sessions persist ``playFormat`` on every match. Side sizes
    keep legacy sessions and imported payloads safe when that field is absent.
    """
    explicit = normalize_play_format(match.get("playFormat"))
    if explicit in {"singles", "doubles"} and str(match.get("playFormat") or "").strip():
        return explicit
    side_a = [str(value) for value in match.get("sideA") or match.get("teamA") or []]
    side_b = [str(value) for value in match.get("sideB") or match.get("teamB") or []]
    if len(side_a) == len(side_b) == 1:
        return "singles"
    if len(side_a) == len(side_b) == 2:
        return "doubles"
    fallback = normalize_play_format(default_format)
    return fallback if fallback in {"singles", "doubles"} else "doubles"


def normalize_scoring_mode(value: Any) -> str:
    mode = str(value or "scored").strip().lower().replace("-", "_")
    aliases = {
        "score": "scored",
        "scores": "scored",
        "no_scores": "unscored",
        "round_played": "unscored",
        "played": "unscored",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in SCORING_MODES else "scored"


def normalize_standings_sort(value: Any) -> str:
    mode = str(value or "wins").strip().lower().replace("-", "_")
    aliases = {
        "total_wins": "wins",
        "total_points": "points",
        "point_differential": "differential",
        "diff": "differential",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in STANDINGS_SORTS else "wins"

def _clean_name(value: Any) -> str:
    return " ".join(str(value or "").replace("\u00a0", " ").split()).strip()[:160]

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _stable_seed(*parts: Any) -> int:
    text = "||".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)

def _pair_key(a: Any, b: Any) -> str:
    return "|".join(sorted((str(a), str(b))))

def _team_key(team: list[str]) -> str:
    return ",".join(sorted(str(x) for x in team))

def _exact_match_key(team_a: list[str], team_b: list[str]) -> str:
    return "|".join(sorted((_team_key(team_a), _team_key(team_b))))

def _participant_map(event: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): row for row in event.get("participants") or [] if row.get("id")}

def _participant_active_in_round(participant: dict[str, Any], round_number: int) -> bool:
    start = int(participant.get("active_from_round") or 1)
    end = participant.get("inactive_from_round")
    inactive_rounds = {int(x) for x in participant.get("inactive_rounds") or []}
    if round_number < start:
        return False
    if end not in (None, "") and round_number >= int(end):
        return False
    return int(round_number) not in inactive_rounds

def active_participant_ids(event: dict[str, Any], round_number: int) -> list[str]:
    rows = [
        row for row in event.get("participants") or []
        if _participant_active_in_round(row, round_number)
    ]
    rows.sort(key=lambda row: (int(row.get("roster_order") or 0), str(row.get("name") or "").lower(), str(row.get("id"))))
    return [str(row["id"]) for row in rows]

def _blank_history(event: dict[str, Any]) -> dict[str, Any]:
    ids = [str(row["id"]) for row in event.get("participants") or []]
    return {
        "games": {pid: 0 for pid in ids},
        "byes": {pid: 0 for pid in ids},
        "singles_games": {pid: 0 for pid in ids},
        "doubles_games": {pid: 0 for pid in ids},
        "partners": defaultdict(int),
        "opponents": defaultdict(int),
        "singles_opponents": defaultdict(int),
        "doubles_opponents": defaultdict(int),
        "exact_matches": defaultdict(int),
    }

def _round_matches(round_row: dict[str, Any]) -> list[dict[str, Any]]:
    if round_row.get("matches"):
        return [dict(row) for row in round_row.get("matches") or []]
    rows: list[dict[str, Any]] = []
    for court in round_row.get("courts") or []:
        rows.extend(dict(row) for row in court.get("matches") or [])
    return rows

def _apply_match_history(history: dict[str, Any], match: dict[str, Any], play_format: str) -> None:
    side_a = [str(x) for x in match.get("sideA") or match.get("teamA") or []]
    side_b = [str(x) for x in match.get("sideB") or match.get("teamB") or []]
    actual_format = generator_match_play_format(match, play_format)
    for pid in side_a + side_b:
        history["games"][pid] = int(history["games"].get(pid, 0)) + 1
        history[f"{actual_format}_games"][pid] = int(
            history[f"{actual_format}_games"].get(pid, 0)
        ) + 1
    if actual_format == "singles":
        if len(side_a) == 1 and len(side_b) == 1:
            key = _pair_key(side_a[0], side_b[0])
            history["opponents"][key] += 1
            history["singles_opponents"][key] += 1
            history["exact_matches"][key] += 1
        return
    if len(side_a) == 2:
        history["partners"][_pair_key(side_a[0], side_a[1])] += 1
    if len(side_b) == 2:
        history["partners"][_pair_key(side_b[0], side_b[1])] += 1
    for left in side_a:
        for right in side_b:
            key = _pair_key(left, right)
            history["opponents"][key] += 1
            history["doubles_opponents"][key] += 1
    if len(side_a) == 2 and len(side_b) == 2:
        history["exact_matches"][_exact_match_key(side_a, side_b)] += 1

def history_before_round(event: dict[str, Any], round_number: int, *, include_preview: bool = False) -> dict[str, Any]:
    history = _blank_history(event)
    play_format = normalize_play_format(event.get("playFormat"))
    for round_row in event.get("rounds") or []:
        number = int(round_row.get("number") or 0)
        if number >= round_number:
            continue
        status = str(round_row.get("status") or "")
        if status == "skipped":
            continue
        if status not in {"saved", "played", "active", "preview"}:
            continue
        if status in {"active", "preview"} and not include_preview:
            continue
        for pid in round_row.get("byeParticipantIds") or []:
            history["byes"][str(pid)] = int(history["byes"].get(str(pid), 0)) + 1
        for match in _round_matches(round_row):
            _apply_match_history(history, match, play_format)
    return history

def _selection_priority(pid: str, history: dict[str, Any], roster_pos: dict[str, int]) -> tuple[Any, ...]:
    return (
        int(history["games"].get(pid, 0)),
        -int(history["byes"].get(pid, 0)),
        int(roster_pos.get(pid, 9999)),
        str(pid),
    )

def _candidate_playing_sets(
    active_ids: list[str],
    slots: int,
    history: dict[str, Any],
    roster_pos: dict[str, int],
    *,
    seed: int,
    attempts: int = 18,
) -> list[tuple[list[str], list[str]]]:
    if slots >= len(active_ids):
        return [(list(active_ids), [])]
    ordered = sorted(active_ids, key=lambda pid: _selection_priority(pid, history, roster_pos))
    bye_count = len(active_ids) - slots
    pool = ordered[: min(len(ordered), slots + max(bye_count * 2, 3))]
    seen: set[tuple[str, ...]] = set()
    results: list[tuple[list[str], list[str]]] = []
    baseline_playing = ordered[:slots]
    baseline_key = tuple(sorted(baseline_playing))
    seen.add(baseline_key)
    results.append((baseline_playing, [pid for pid in ordered if pid not in set(baseline_playing)]))
    rng = random.Random(seed)
    for _ in range(attempts):
        sample_pool = pool[:]
        rng.shuffle(sample_pool)
        playing = sorted(sample_pool[:slots], key=lambda pid: roster_pos.get(pid, 9999))
        key = tuple(sorted(playing))
        if len(playing) < slots or key in seen:
            continue
        seen.add(key)
        playing_set=set(playing)
        byes=[pid for pid in ordered if pid not in playing_set]
        results.append((playing,byes))
    return results

def _find_matching_backtrack(
    players: tuple[str, ...],
    pair_cost,
    *,
    max_nodes: int = 200000,
) -> tuple[list[tuple[str, str]] | None, float]:
    best: list[tuple[str, str]] | None = None
    best_cost = float("inf")
    nodes = 0

    def recurse(remaining: tuple[str, ...], pairs: list[tuple[str, str]], cost: float) -> None:
        nonlocal best, best_cost, nodes
        nodes += 1
        if nodes > max_nodes or cost >= best_cost:
            return
        if not remaining:
            best = list(pairs)
            best_cost = cost
            return
        # Pick the most constrained player.
        chosen = min(
            remaining,
            key=lambda pid: (
                sum(1 for other in remaining if other != pid and pair_cost(pid, other) <= 0.001),
                pid,
            ),
        )
        rest = tuple(pid for pid in remaining if pid != chosen)
        candidates = sorted(rest, key=lambda other: (pair_cost(chosen, other), other))
        for other in candidates:
            next_remaining = tuple(pid for pid in rest if pid != other)
            recurse(next_remaining, [*pairs, (chosen, other)], cost + float(pair_cost(chosen, other)))

    recurse(tuple(players), [], 0.0)
    return best, best_cost

def _singles_round(
    *,
    active_ids: list[str],
    court_count: int,
    history: dict[str, Any],
    roster_pos: dict[str, int],
    round_number: int,
    court_offset: int = 0,
    seed_salt: Any = 0,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    match_count = min(len(active_ids) // 2, max(1, int(court_count or 0)) if court_count else len(active_ids) // 2)
    slots = match_count * 2
    if slots < 2:
        return [], list(active_ids), ["At least two active players are required."]
    candidate_sets = _candidate_playing_sets(
        active_ids,
        slots,
        history,
        roster_pos,
        seed=_stable_seed("singles-subsets", seed_salt, round_number, *active_ids),
    )
    best_matches: list[tuple[str, str]] | None = None
    best_byes: list[str] = []
    best_cost = float("inf")
    for playing, byes in candidate_sets:
        def cost(a: str, b: str) -> float:
            key = _pair_key(a, b)
            singles_repeats = int(history["singles_opponents"].get(key, 0))
            all_repeats = int(history["opponents"].get(key, 0))
            return (
                singles_repeats * 10000.0
                + all_repeats * 100.0
                + abs(roster_pos.get(a, 0) - roster_pos.get(b, 0)) * -0.001
            )
        matching, cost_value = _find_matching_backtrack(tuple(playing), cost, max_nodes=60000)
        if matching is None:
            continue
        fairness_penalty = sum(
            max(0, history["byes"].get(pid, 0) - min(history["byes"].get(x, 0) for x in active_ids))
            for pid in byes
        ) * 10
        total = cost_value + fairness_penalty
        if total < best_cost:
            best_cost = total
            best_matches = matching
            best_byes = byes
    if best_matches is None:
        # Safe deterministic fallback.
        playing = sorted(active_ids, key=lambda pid: _selection_priority(pid, history, roster_pos))[:slots]
        best_byes = [pid for pid in active_ids if pid not in set(playing)]
        best_matches = list(zip(playing[::2], playing[1::2]))
    warnings = []
    repeated = sum(
        1
        for a, b in best_matches
        if history["singles_opponents"].get(_pair_key(a, b), 0)
    )
    if repeated:
        warnings.append(f"{repeated} singles matchup repeat(s) were unavoidable.")
    matches = [
        {
            "id": f"r{round_number}-c{court_number + court_offset}",
            "round": int(round_number),
            "court": int(court_number + court_offset),
            "playFormat": "singles",
            "sideA": [a],
            "sideB": [b],
            "teamA": [a],
            "teamB": [b],
            "scoreA": None,
            "scoreB": None,
            "status": "scheduled",
        }
        for court_number, (a, b) in enumerate(best_matches, start=1)
    ]
    return matches, best_byes, warnings

def _doubles_round(
    *,
    active_ids: list[str],
    court_count: int,
    history: dict[str, Any],
    roster_pos: dict[str, int],
    round_number: int,
    court_offset: int = 0,
    seed_salt: Any = 0,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    match_count = min(len(active_ids) // 4, max(1, int(court_count or 0)) if court_count else len(active_ids) // 4)
    slots = match_count * 4
    if slots < 4:
        return [], list(active_ids), ["At least four active players are required."]
    candidate_sets = _candidate_playing_sets(
        active_ids,
        slots,
        history,
        roster_pos,
        seed=_stable_seed("doubles-subsets", seed_salt, round_number, *active_ids),
        attempts=16,
    )
    best = None
    best_byes=[]
    best_cost=float("inf")
    for set_index,(playing,byes) in enumerate(candidate_sets):
        attempts=max(48,len(playing)*8)
        for attempt in range(attempts):
            rng=random.Random(_stable_seed("doubles",seed_salt,round_number,set_index,attempt,*playing))
            remaining=playing[:]; rng.shuffle(remaining)
            teams=[]; partner_cost=0.0
            while remaining:
                a=remaining.pop(0)
                candidates=[]
                for idx,b in enumerate(remaining):
                    repeats=int(history["partners"].get(_pair_key(a,b),0))
                    value=repeats*20000.0 - abs(roster_pos.get(a,0)-roster_pos.get(b,0))*0.001 + rng.random()*0.0001
                    candidates.append((value,idx,b))
                value,idx,b=min(candidates)
                remaining.pop(idx)
                teams.append(tuple(sorted((a,b))))
                partner_cost+=value
            team_queue=teams[:]; rng.shuffle(team_queue)
            match_pairs=[]; opponent_cost=0.0
            while team_queue:
                ta=team_queue.pop(0)
                choices=[]
                for idx,tb in enumerate(team_queue):
                    exact=int(history["exact_matches"].get(_exact_match_key(list(ta),list(tb)),0))
                    doubles_opp=sum(int(history["doubles_opponents"].get(_pair_key(x,y),0)) for x in ta for y in tb)
                    all_opp=sum(int(history["opponents"].get(_pair_key(x,y),0)) for x in ta for y in tb)
                    value=exact*200000.0+doubles_opp*1000.0+all_opp*100.0+rng.random()*0.0001
                    choices.append((value,idx,tb))
                value,idx,tb=min(choices)
                team_queue.pop(idx)
                match_pairs.append((ta,tb)); opponent_cost+=value
            total=partner_cost+opponent_cost+sum(int(history["byes"].get(pid,0)) for pid in byes)*5
            if total<best_cost:
                best_cost=total;best=match_pairs;best_byes=byes
    if best is None:
        playing=sorted(active_ids,key=lambda pid:_selection_priority(pid,history,roster_pos))[:slots]
        best_byes=[pid for pid in active_ids if pid not in set(playing)]
        teams=[tuple(sorted(playing[i:i+2])) for i in range(0,len(playing),2)]
        best=list(zip(teams[::2],teams[1::2]))
    warnings=[]
    partner_repeats=sum(int(history["partners"].get(_pair_key(team[0],team[1]),0)>0) for match in best for team in match)
    exact_repeats=sum(int(history["exact_matches"].get(_exact_match_key(list(a),list(b)),0)>0) for a,b in best)
    if partner_repeats: warnings.append(f"{partner_repeats} repeated partner pairing(s) were unavoidable.")
    if exact_repeats: warnings.append(f"{exact_repeats} repeated exact matchup(s) were unavoidable.")
    matches=[{
        "id":f"r{round_number}-c{court + court_offset}",
        "round":round_number,"court":court + court_offset,"playFormat":"doubles",
        "sideA":list(ta),"sideB":list(tb),"teamA":list(ta),"teamB":list(tb),
        "scoreA":None,"scoreB":None,"status":"scheduled"
    } for court,(ta,tb) in enumerate(best,1)]
    return matches,best_byes,warnings


def _count_spread(
    history: dict[str, Any],
    key: str,
    active_ids: list[str],
    increment_ids: set[str],
) -> tuple[int, int, int]:
    values = [
        int(history[key].get(pid, 0)) + int(pid in increment_ids)
        for pid in active_ids
    ]
    if not values:
        return (0, 0, 0)
    return (max(values) - min(values), sum(value * value for value in values), max(values))


def _candidate_singles_groups(
    playing: list[str],
    singles_slots: int,
    history: dict[str, Any],
    roster_pos: dict[str, int],
    *,
    seed: int,
    attempts: int = 14,
) -> list[list[str]]:
    if singles_slots <= 0:
        return [[]]
    if singles_slots >= len(playing):
        return [sorted(playing, key=lambda pid: roster_pos.get(pid, 9999))]

    def role_priority(pid: str) -> tuple[Any, ...]:
        return (
            int(history["singles_games"].get(pid, 0)),
            -int(history["doubles_games"].get(pid, 0)),
            int(history["games"].get(pid, 0)),
            int(roster_pos.get(pid, 9999)),
            str(pid),
        )

    ordered = sorted(playing, key=role_priority)
    seen: set[tuple[str, ...]] = set()
    groups: list[list[str]] = []

    def add(values: list[str]) -> None:
        group = sorted(values[:singles_slots], key=lambda pid: roster_pos.get(pid, 9999))
        key = tuple(sorted(group))
        if len(group) == singles_slots and key not in seen:
            seen.add(key)
            groups.append(group)

    add(ordered)

    baseline = ordered[:singles_slots]
    outside = ordered[singles_slots:]
    # Single swaps provide relationship diversity without sacrificing the role
    # debt ordering that keeps singles and doubles counts balanced.
    for outgoing in reversed(baseline):
        for incoming in outside:
            add([pid for pid in baseline if pid != outgoing] + [incoming])
            if len(groups) >= attempts:
                return groups

    rng = random.Random(seed)
    maximum_groups = min(attempts, math.comb(len(playing), singles_slots))
    for _ in range(maximum_groups * 6):
        if len(groups) >= maximum_groups:
            break
        ranked = sorted(
            playing,
            key=lambda pid: (
                int(history["singles_games"].get(pid, 0)),
                -int(history["doubles_games"].get(pid, 0)),
                rng.random(),
                int(roster_pos.get(pid, 9999)),
            ),
        )
        add(ranked)
    return groups


def _greedy_singles_metrics(
    player_ids: list[str],
    history: dict[str, Any],
    roster_pos: dict[str, int],
) -> tuple[int, int, int]:
    remaining = list(player_ids)
    repeated_pairs = 0
    singles_repeat_sum = 0
    all_repeat_sum = 0
    while remaining:
        chosen = min(
            remaining,
            key=lambda pid: (
                sum(
                    1
                    for other in remaining
                    if other != pid
                    and int(history["singles_opponents"].get(_pair_key(pid, other), 0)) == 0
                ),
                int(roster_pos.get(pid, 9999)),
                str(pid),
            ),
        )
        remaining.remove(chosen)
        other = min(
            remaining,
            key=lambda pid: (
                int(history["singles_opponents"].get(_pair_key(chosen, pid), 0)),
                int(history["opponents"].get(_pair_key(chosen, pid), 0)),
                int(roster_pos.get(pid, 9999)),
                str(pid),
            ),
        )
        remaining.remove(other)
        key = _pair_key(chosen, other)
        singles_repeats = int(history["singles_opponents"].get(key, 0))
        repeated_pairs += int(singles_repeats > 0)
        singles_repeat_sum += singles_repeats
        all_repeat_sum += int(history["opponents"].get(key, 0))
    return repeated_pairs, singles_repeat_sum, all_repeat_sum


def _greedy_doubles_metrics(
    player_ids: list[str],
    history: dict[str, Any],
    roster_pos: dict[str, int],
) -> tuple[int, int, int, int, int, int]:
    remaining = list(player_ids)
    teams: list[tuple[str, str]] = []
    repeated_partners = 0
    partner_repeat_sum = 0
    while remaining:
        chosen = min(
            remaining,
            key=lambda pid: (
                sum(
                    1
                    for other in remaining
                    if other != pid
                    and int(history["partners"].get(_pair_key(pid, other), 0)) == 0
                ),
                int(roster_pos.get(pid, 9999)),
                str(pid),
            ),
        )
        remaining.remove(chosen)
        other = min(
            remaining,
            key=lambda pid: (
                int(history["partners"].get(_pair_key(chosen, pid), 0)),
                int(roster_pos.get(pid, 9999)),
                str(pid),
            ),
        )
        remaining.remove(other)
        key = _pair_key(chosen, other)
        repeats = int(history["partners"].get(key, 0))
        repeated_partners += int(repeats > 0)
        partner_repeat_sum += repeats
        teams.append(tuple(sorted((chosen, other))))

    repeated_exact = 0
    exact_repeat_sum = 0
    doubles_opponent_sum = 0
    all_opponent_sum = 0
    while teams:
        team_a = teams.pop(0)
        idx, team_b = min(
            enumerate(teams),
            key=lambda item: (
                int(history["exact_matches"].get(_exact_match_key(list(team_a), list(item[1])), 0)),
                sum(
                    int(history["doubles_opponents"].get(_pair_key(left, right), 0))
                    for left in team_a
                    for right in item[1]
                ),
                sum(
                    int(history["opponents"].get(_pair_key(left, right), 0))
                    for left in team_a
                    for right in item[1]
                ),
                item[1],
            ),
        )
        teams.pop(idx)
        exact = int(history["exact_matches"].get(_exact_match_key(list(team_a), list(team_b)), 0))
        repeated_exact += int(exact > 0)
        exact_repeat_sum += exact
        doubles_opponent_sum += sum(
            int(history["doubles_opponents"].get(_pair_key(left, right), 0))
            for left in team_a
            for right in team_b
        )
        all_opponent_sum += sum(
            int(history["opponents"].get(_pair_key(left, right), 0))
            for left in team_a
            for right in team_b
        )
    return (
        repeated_partners,
        partner_repeat_sum,
        repeated_exact,
        exact_repeat_sum,
        doubles_opponent_sum,
        all_opponent_sum,
    )


def _effective_mixed_court_counts(
    active_count: int,
    desired_doubles: int,
    desired_singles: int,
) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for doubles_count in range(max(0, desired_doubles) + 1):
        for singles_count in range(max(0, desired_singles) + 1):
            if doubles_count == 0 and singles_count == 0:
                continue
            slots = doubles_count * 4 + singles_count * 2
            if slots <= active_count:
                candidates.append((doubles_count, singles_count))
    if not candidates:
        return (0, 0)
    both = [value for value in candidates if value[0] > 0 and value[1] > 0]
    eligible = both if active_count >= 6 and both else candidates
    return max(
        eligible,
        key=lambda value: (
            value[0] * 4 + value[1] * 2,
            -abs(desired_doubles - value[0]) - abs(desired_singles - value[1]),
            int(value[0] > 0 and value[1] > 0),
            value[0] + value[1],
            value[0],
        ),
    )


def _mixed_round(
    *,
    active_ids: list[str],
    doubles_court_count: int,
    singles_court_count: int,
    history: dict[str, Any],
    roster_pos: dict[str, int],
    round_number: int,
    seed_salt: Any = 0,
) -> tuple[list[dict[str, Any]], list[str], list[str], dict[str, int]]:
    desired_doubles = max(0, int(doubles_court_count or 0))
    desired_singles = max(0, int(singles_court_count or 0))
    actual_doubles, actual_singles = _effective_mixed_court_counts(
        len(active_ids), desired_doubles, desired_singles
    )
    slots = actual_doubles * 4 + actual_singles * 2
    warnings: list[str] = []
    if (actual_doubles, actual_singles) != (desired_doubles, desired_singles):
        warnings.append(
            "The active roster supports "
            f"{actual_doubles} doubles and {actual_singles} singles court(s) this round; "
            f"the configured mix is {desired_doubles} doubles and {desired_singles} singles."
        )
    if actual_doubles == 0 or actual_singles == 0:
        warnings.append(
            "Both formats could not run this round because the active roster is too small for the configured mix."
        )
    if slots < 2:
        return [], list(active_ids), [*warnings, "At least two active players are required."], {
            "doubles": actual_doubles,
            "singles": actual_singles,
        }

    candidate_sets = _candidate_playing_sets(
        active_ids,
        slots,
        history,
        roster_pos,
        seed=_stable_seed("mixed-playing", seed_salt, round_number, *active_ids),
        attempts=10,
    )
    singles_slots = actual_singles * 2
    best: tuple[list[str], list[str], list[str]] | None = None
    best_score: tuple[Any, ...] | None = None
    for set_index, (playing, byes) in enumerate(candidate_sets):
        role_groups = _candidate_singles_groups(
            playing,
            singles_slots,
            history,
            roster_pos,
            seed=_stable_seed("mixed-roles", seed_salt, round_number, set_index, *playing),
        )
        for singles_ids in role_groups:
            singles_set = set(singles_ids)
            doubles_ids = [pid for pid in playing if pid not in singles_set]
            doubles_set = set(doubles_ids)
            playing_set = set(playing)
            bye_set = set(byes)
            game_spread = _count_spread(history, "games", active_ids, playing_set)
            bye_spread = _count_spread(history, "byes", active_ids, bye_set)
            singles_spread = _count_spread(history, "singles_games", active_ids, singles_set)
            doubles_spread = _count_spread(history, "doubles_games", active_ids, doubles_set)
            singles_metrics = _greedy_singles_metrics(singles_ids, history, roster_pos)
            doubles_metrics = _greedy_doubles_metrics(doubles_ids, history, roster_pos)
            score = (
                game_spread[0],
                bye_spread[0],
                singles_spread[0],
                doubles_spread[0],
                game_spread[1],
                bye_spread[1],
                singles_spread[1],
                doubles_spread[1],
                singles_metrics[0],
                doubles_metrics[0],
                doubles_metrics[2],
                singles_metrics[1],
                doubles_metrics[1],
                doubles_metrics[3],
                doubles_metrics[4],
                singles_metrics[2] + doubles_metrics[5],
                tuple(sorted(singles_ids)),
                tuple(sorted(byes)),
            )
            if best_score is None or score < best_score:
                best_score = score
                best = (singles_ids, doubles_ids, byes)

    if best is None:
        playing = sorted(active_ids, key=lambda pid: _selection_priority(pid, history, roster_pos))[:slots]
        singles_ids = playing[:singles_slots]
        doubles_ids = playing[singles_slots:]
        byes = [pid for pid in active_ids if pid not in set(playing)]
    else:
        singles_ids, doubles_ids, byes = best

    doubles_matches: list[dict[str, Any]] = []
    singles_matches: list[dict[str, Any]] = []
    if actual_doubles:
        doubles_matches, _doubles_byes, doubles_warnings = _doubles_round(
            active_ids=doubles_ids,
            court_count=actual_doubles,
            history=history,
            roster_pos=roster_pos,
            round_number=round_number,
            seed_salt=seed_salt,
        )
        warnings.extend(doubles_warnings)
    if actual_singles:
        singles_matches, _singles_byes, singles_warnings = _singles_round(
            active_ids=singles_ids,
            court_count=actual_singles,
            history=history,
            roster_pos=roster_pos,
            round_number=round_number,
            court_offset=actual_doubles,
            seed_salt=seed_salt,
        )
        warnings.extend(singles_warnings)
    matches = sorted(
        [*doubles_matches, *singles_matches],
        key=lambda match: (int(match.get("court") or 0), str(match.get("id") or "")),
    )
    return matches, sorted(byes, key=lambda pid: roster_pos.get(pid, 9999)), warnings, {
        "doubles": actual_doubles,
        "singles": actual_singles,
    }

def _update_history_for_generated_round(history: dict[str, Any], round_row: dict[str, Any], play_format: str) -> None:
    for pid in round_row.get("byeParticipantIds") or []:
        history["byes"][str(pid)] = int(history["byes"].get(str(pid), 0)) + 1
    for match in _round_matches(round_row):
        _apply_match_history(history, match, play_format)

def _generate_round_robin_round(
    event: dict[str, Any],
    round_number: int,
    history: dict[str, Any],
) -> dict[str, Any]:
    active_ids = active_participant_ids(event, round_number)
    roster_pos = {
        str(row["id"]): int(row.get("roster_order") or 0)
        for row in event.get("participants") or []
    }
    play_format = normalize_play_format(event.get("playFormat"))
    if play_format == "singles":
        matches, byes, warnings = _singles_round(
            active_ids=active_ids,
            court_count=int(event.get("courtCount") or 0),
            history=history,
            roster_pos=roster_pos,
            round_number=round_number,
        )
        format_counts = {"doubles": 0, "singles": len(matches)}
    elif play_format == "doubles_singles":
        matches, byes, warnings, format_counts = _mixed_round(
            active_ids=active_ids,
            doubles_court_count=int(event.get("doublesCourtCount") or 0),
            singles_court_count=int(event.get("singlesCourtCount") or 0),
            history=history,
            roster_pos=roster_pos,
            round_number=round_number,
            seed_salt=event.get("mixedScheduleSeed") or 0,
        )
    else:
        matches, byes, warnings = _doubles_round(
            active_ids=active_ids,
            court_count=int(event.get("courtCount") or 0),
            history=history,
            roster_pos=roster_pos,
            round_number=round_number,
        )
        format_counts = {"doubles": len(matches), "singles": 0}
    return {
        "number": int(round_number),
        "status": "preview",
        "matches": matches,
        "byeParticipantIds": [str(pid) for pid in byes],
        "warnings": warnings,
        "formatCounts": format_counts,
        "savedAt": None,
        "skippedAt": None,
    }

def _balanced_group_sizes(player_count: int, court_count: int, play_format: str) -> list[int]:
    minimum = 2 if play_format == "singles" else 4
    if player_count < minimum:
        return []

    if court_count > 0:
        group_count = min(int(court_count), max(1, player_count // minimum))
    elif play_format == "doubles":
        # Preserve the familiar 4-player / 5-player ladder court model whenever
        # the roster can fit it exactly. Prefer the most simultaneous courts.
        exact: list[tuple[int, int, int]] = []
        for fives in range(0, (player_count // 5) + 1):
            remainder = player_count - (fives * 5)
            if remainder >= 0 and remainder % 4 == 0:
                fours = remainder // 4
                if fours + fives:
                    exact.append((fours + fives, fours, fives))
        if exact:
            _groups, fours, fives = max(exact, key=lambda item: (item[0], item[1]))
            return [5] * fives + [4] * fours
        group_count = min(
            max(1, math.ceil(player_count / 5)),
            max(1, player_count // minimum),
        )
    else:
        # Singles ladder courts are kept at five players or fewer when the
        # roster permits it, while still supporting every roster size.
        group_count = min(
            max(1, math.ceil(player_count / 5)),
            max(1, player_count // minimum),
        )

    while group_count > 1 and player_count // group_count < minimum:
        group_count -= 1
    sizes = [player_count // group_count] * group_count
    for idx in range(player_count % group_count):
        sizes[idx] += 1
    return sorted(sizes, reverse=True)

def _circle_singles_matches(ids: list[str], *, round_number: int, court_number: int) -> list[dict[str, Any]]:
    if len(ids) < 2:
        return []
    players = list(ids)
    bye_token = "__BYE__"
    if len(players) % 2:
        players.append(bye_token)
    fixed = players[0]
    rotating = players[1:]
    matches: list[dict[str, Any]] = []
    slot = 1
    for mini in range(1, len(players)):
        current = [fixed, *rotating]
        pairs = []
        half = len(current) // 2
        for idx in range(half):
            a = current[idx]
            b = current[-(idx + 1)]
            if bye_token not in {a, b}:
                pairs.append((a, b))
        for a, b in pairs:
            matches.append({
                "id": f"r{round_number}-c{court_number}-m{slot}",
                "round": int(round_number),
                "court": int(court_number),
                "playFormat": "singles",
                "miniRound": int(mini),
                "sideA": [a],
                "sideB": [b],
                "teamA": [a],
                "teamB": [b],
                "scoreA": None,
                "scoreB": None,
                "status": "scheduled",
            })
            slot += 1
        rotating = [rotating[-1], *rotating[:-1]]
    return matches

def _ladder_group_matches(
    event: dict[str, Any],
    ids: list[str],
    *,
    round_number: int,
    court_number: int,
) -> list[dict[str, Any]]:
    if str(event.get("playFormat") or "doubles") == "singles":
        return _circle_singles_matches(ids, round_number=round_number, court_number=court_number)
    local_event = {
        "participants": [
            {
                "id": pid,
                "roster_order": idx,
                "active_from_round": 1,
                "inactive_from_round": None,
                "inactive_rounds": [],
            }
            for idx, pid in enumerate(ids)
        ],
        "playFormat": "doubles",
        "courtCount": 1,
    }
    history = _blank_history(local_event)
    matches: list[dict[str, Any]] = []
    mini_count = 3 if len(ids) == 4 else max(3, len(ids))
    for mini in range(1, mini_count + 1):
        generated, byes, _warnings = _doubles_round(
            active_ids=ids,
            court_count=1,
            history=history,
            roster_pos={pid: idx for idx, pid in enumerate(ids)},
            round_number=mini,
        )
        if not generated:
            break
        match = dict(generated[0])
        match["id"] = f"r{round_number}-c{court_number}-m{mini}"
        match["round"] = round_number
        match["court"] = court_number
        match["miniRound"] = mini
        matches.append(match)
        _update_history_for_generated_round(
            history,
            {"matches": [match], "byeParticipantIds": byes},
            "doubles",
        )
    return matches

def _create_ladder_round(event: dict[str, Any], round_number: int, ordered_ids: list[str]) -> dict[str, Any]:
    sizes = _balanced_group_sizes(
        len(ordered_ids),
        int(event.get("courtCount") or 0),
        str(event.get("playFormat") or "doubles"),
    )
    courts: list[dict[str, Any]] = []
    cursor = 0
    for court_number, size in enumerate(sizes, start=1):
        ids = ordered_ids[cursor:cursor + size]
        cursor += size
        matches = _ladder_group_matches(
            event,
            ids,
            round_number=round_number,
            court_number=court_number,
        )
        courts.append({
            "courtNumber": court_number,
            "participantIds": list(ids),
            "size": len(ids),
            "matches": matches,
        })
    return {
        "number": round_number,
        "status": "preview",
        "courts": courts,
        "matches": [match for court in courts for match in court["matches"]],
        "byeParticipantIds": [],
        "warnings": [],
        "savedAt": None,
        "skippedAt": None,
    }


def _mixed_schedule_score(event: dict[str, Any], history: dict[str, Any]) -> tuple[Any, ...]:
    participant_ids = [str(row.get("id")) for row in event.get("participants") or [] if row.get("id")]

    def spread(key: str) -> tuple[int, int]:
        values = [int(history[key].get(pid, 0)) for pid in participant_ids]
        return (
            (max(values) - min(values)) if values else 0,
            sum(value * value for value in values),
        )

    partner_excess = sum(max(0, int(value) - 1) for value in history["partners"].values())
    singles_excess = sum(max(0, int(value) - 1) for value in history["singles_opponents"].values())
    exact_excess = sum(max(0, int(value) - 1) for value in history["exact_matches"].values())
    doubles_opponent_excess = sum(
        max(0, int(value) - 1) for value in history["doubles_opponents"].values()
    )
    all_opponent_excess = sum(max(0, int(value) - 1) for value in history["opponents"].values())
    games = spread("games")
    byes = spread("byes")
    singles = spread("singles_games")
    doubles = spread("doubles_games")
    return (
        games[0],
        byes[0],
        singles[0],
        doubles[0],
        partner_excess + singles_excess,
        partner_excess,
        singles_excess,
        exact_excess,
        doubles_opponent_excess,
        all_opponent_excess,
        games[1],
        byes[1],
        singles[1],
        doubles[1],
    )


def _generate_round_robin_schedule(event: dict[str, Any]) -> list[dict[str, Any]]:
    play_format = normalize_play_format(event.get("playFormat"))
    if play_format != "doubles_singles":
        history = _blank_history(event)
        rounds: list[dict[str, Any]] = []
        for round_number in range(1, int(event.get("totalRounds") or 1) + 1):
            round_row = _generate_round_robin_round(event, round_number, history)
            rounds.append(round_row)
            _update_history_for_generated_round(history, round_row, play_format)
        return rounds

    # Mixed schedules have more degrees of freedom than a single-format round
    # robin. Evaluate several deterministic whole-schedule candidates so an
    # early locally-good role assignment does not force avoidable repeats late.
    best_rounds: list[dict[str, Any]] | None = None
    best_score: tuple[Any, ...] | None = None
    best_seed = 0
    participant_count = len(event.get("participants") or [])
    round_count = int(event.get("totalRounds") or 1)
    schedule_work = participant_count * round_count
    candidate_count = (
        12
        if schedule_work < 500
        else 6
        if schedule_work < 1000
        else 2
        if schedule_work < 1600
        else 1
    )
    for seed in range(candidate_count):
        candidate_event = copy.deepcopy(event)
        candidate_event["mixedScheduleSeed"] = seed
        history = _blank_history(candidate_event)
        rounds = []
        for round_number in range(1, int(candidate_event.get("totalRounds") or 1) + 1):
            round_row = _generate_round_robin_round(candidate_event, round_number, history)
            rounds.append(round_row)
            _update_history_for_generated_round(history, round_row, play_format)
        score = (*_mixed_schedule_score(candidate_event, history), seed)
        if best_score is None or score < best_score:
            best_score = score
            best_rounds = rounds
            best_seed = seed
            if score[:10] == (0, 0, 0, 0, 0, 0, 0, 0, 0, 0):
                break
    event["mixedScheduleSeed"] = best_seed
    return best_rounds or []


def create_generator_preview(
    *,
    generator_kind: str,
    play_format: str,
    title: str,
    participant_names: list[str],
    player_ids: list[int] | None = None,
    total_rounds: int = 3,
    court_count: int = 0,
    doubles_court_count: int = 0,
    singles_court_count: int = 0,
    standings_sort: str = "wins",
    scoring_mode: str = "scored",
) -> dict[str, Any]:
    kind = str(generator_kind or "").strip().lower().replace("-", "_")
    if kind not in {"round_robin", "ladder"}:
        raise ValueError("generator_kind must be round_robin or ladder")
    raw_format = str(play_format or "").strip().lower().replace("-", "_").replace("+", "_")
    format_aliases = {
        "mixed": "doubles_singles",
        "mixed_doubles_singles": "doubles_singles",
        "doubles_singles_mix": "doubles_singles",
        "singles_doubles": "doubles_singles",
        "singles_doubles_mix": "doubles_singles",
    }
    raw_format = format_aliases.get(raw_format, raw_format)
    if raw_format not in PLAY_FORMATS:
        raise ValueError("play_format must be singles, doubles, or doubles_singles")
    fmt = raw_format
    if fmt == "doubles_singles" and kind != "round_robin":
        raise ValueError("Doubles + Singles Mix is available only for Round-Robin Generator sessions.")
    scoring = normalize_scoring_mode(scoring_mode)
    if kind == "ladder" and scoring != "scored":
        raise ValueError("Ladder Generator requires scored rounds because later rounds depend on results.")
    names = [_clean_name(x) for x in participant_names if _clean_name(x)]
    seen = set()
    unique_names = []
    for name in names:
        key = name.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique_names.append(name)
    minimum = 2 if fmt == "singles" else 6 if fmt == "doubles_singles" else 4
    format_label = (
        "Doubles + Singles Mix"
        if fmt == "doubles_singles"
        else fmt.title()
    )
    if len(unique_names) < minimum:
        raise ValueError(f"{format_label} requires at least {minimum} players.")
    if len(unique_names) > 40:
        raise ValueError("Generators support at most 40 players.")
    ids = [int(x) for x in (player_ids or [])]
    if ids and len(ids) != len(unique_names):
        raise ValueError("player_ids and participant_names must have the same length.")
    participants = []
    for idx, name in enumerate(unique_names, start=1):
        row = {
            "id": f"p-{idx}",
            "name": name,
            "seed": idx,
            "roster_order": idx,
            "active_from_round": 1,
            "inactive_from_round": None,
            "inactive_rounds": [],
        }
        if ids:
            row["player_id"] = ids[idx-1]
        participants.append(row)
    mixed_doubles = max(0, min(int(doubles_court_count or 0), 20))
    mixed_singles = max(0, min(int(singles_court_count or 0), 20))
    if fmt == "doubles_singles":
        if mixed_doubles < 1 or mixed_singles < 1:
            raise ValueError("Choose at least one doubles court and one singles court.")
        if mixed_doubles + mixed_singles > 20:
            raise ValueError("The combined doubles and singles court count cannot exceed 20.")
        required_players = mixed_doubles * 4 + mixed_singles * 2
        if required_players > len(unique_names):
            raise ValueError(
                f"This court mix requires {required_players} players, but the roster has {len(unique_names)}."
            )
        resolved_court_count = mixed_doubles + mixed_singles
    else:
        mixed_doubles = 0
        mixed_singles = 0
        resolved_court_count = max(0, min(int(court_count or 0), 20))
    event = {
        "schemaVersion": 3,
        "sourceEventUid": f"generator-{uuid4().hex}",
        "name": _clean_name(title) or ("Round-Robin Generator" if kind == "round_robin" else "Ladder Generator"),
        "type": "round_robin" if kind == "round_robin" else "league",
        "generatorKind": kind,
        "playFormat": fmt,
        "scoringMode": scoring,
        "standingsSort": normalize_standings_sort(standings_sort),
        "status": "preview",
        "participants": participants,
        "totalRounds": max(1, min(int(total_rounds or 1), 50)),
        "courtCount": resolved_court_count,
        "doublesCourtCount": mixed_doubles,
        "singlesCourtCount": mixed_singles,
        "currentRoundNumber": 1,
        "rounds": [],
        "rosterRevisions": [],
        "publishedMatchIds": [],
        "createdAt": _now_iso(),
    }
    if kind == "round_robin":
        event["rounds"] = _generate_round_robin_schedule(event)
    else:
        ordered_ids = active_participant_ids(event, 1)
        event["rounds"] = [_create_ladder_round(event, 1, ordered_ids)]
    event["previewFingerprint"] = hashlib.sha256(
        json.dumps(
            {
                "kind": kind,
                "format": fmt,
                "names": unique_names,
                "ids": ids,
                "rounds": event["totalRounds"],
                "courts": event["courtCount"],
                "doubles_courts": event["doublesCourtCount"],
                "singles_courts": event["singlesCourtCount"],
                "standings_sort": event["standingsSort"],
                "scoring_mode": event["scoringMode"],
                "schedule": event["rounds"],
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return event

def start_generator_event(event: dict[str, Any]) -> dict[str, Any]:
    next_event = copy.deepcopy(event)
    if str(next_event.get("status")) != "preview":
        raise ValueError("Only a preview can be started.")
    if not next_event.get("rounds"):
        raise ValueError("The preview has no playable rounds.")
    next_event["status"] = "active"
    next_event["startedAt"] = _now_iso()
    next_event["currentRoundNumber"] = 1
    for row in next_event.get("rounds") or []:
        row["status"] = "active" if int(row.get("number") or 0) == 1 else "preview"
    return next_event

def _get_round(event: dict[str, Any], round_number: int) -> dict[str, Any]:
    for row in event.get("rounds") or []:
        if int(row.get("number") or 0) == int(round_number):
            return row
    raise ValueError(f"Round {round_number} was not found.")

def _round_has_any_scores(round_row: dict[str, Any]) -> bool:
    return any(
        match.get("scoreA") is not None or match.get("scoreB") is not None
        for match in _round_matches(round_row)
    )

def save_generator_round(
    event: dict[str, Any],
    *,
    round_number: int,
    scores: list[dict[str, Any]],
) -> dict[str, Any]:
    next_event = copy.deepcopy(event)
    if normalize_scoring_mode(next_event.get("scoringMode")) != "scored":
        raise ValueError("This unscored Round-Robin uses Round Played instead of score entry.")
    row = _get_round(next_event, round_number)
    if str(row.get("status")) not in {"active", "preview"}:
        raise ValueError("Only an active round can be scored.")
    by_id = {str(match.get("id")): match for match in _round_matches(row)}
    score_by_id = {str(score.get("match_id")): score for score in scores or []}
    for match_id, match in by_id.items():
        score = score_by_id.get(match_id)
        if not score:
            raise ValueError("Enter every score before saving the round, or skip the round.")
        a = score.get("score_a")
        b = score.get("score_b")
        if a in (None, "") or b in (None, ""):
            raise ValueError("Enter every score before saving the round, or skip the round.")
        a = int(a)
        b = int(b)
        if a < 0 or b < 0 or a > 99 or b > 99:
            raise ValueError("Scores must be between 0 and 99.")
        if a == b:
            raise ValueError("Matches cannot end in a tie.")
        match["scoreA"] = a
        match["scoreB"] = b
        match["status"] = "scored"
    # Because _round_matches returns copies for courts, update originals by id.
    originals = {}
    for match in row.get("matches") or []:
        originals[str(match.get("id"))] = match
    for court in row.get("courts") or []:
        for match in court.get("matches") or []:
            originals[str(match.get("id"))] = match
    for match_id, updated in by_id.items():
        originals[match_id].update(updated)
    row["status"] = "saved"
    row["savedAt"] = _now_iso()
    row["skippedAt"] = None
    return next_event


def mark_generator_round_played(event: dict[str, Any], *, round_number: int) -> dict[str, Any]:
    next_event = copy.deepcopy(event)
    if str(next_event.get("generatorKind") or "") != "round_robin":
        raise ValueError("Round Played is available only for Round-Robin Generator sessions.")
    if normalize_scoring_mode(next_event.get("scoringMode")) != "unscored":
        raise ValueError("Scored Round-Robins must save scores or skip the round.")
    row = _get_round(next_event, round_number)
    round_status = str(row.get("status"))
    if round_status == "played":
        return next_event
    if round_status not in {"active", "preview"}:
        raise ValueError("Only an active round can be marked played.")
    if _round_has_any_scores(row):
        raise ValueError("Clear entered scores before marking this round played.")
    row["status"] = "played"
    row["playedAt"] = _now_iso()
    row["savedAt"] = None
    row["skippedAt"] = None
    for match in row.get("matches") or []:
        match["status"] = "played"
    for court in row.get("courts") or []:
        for match in court.get("matches") or []:
            match["status"] = "played"
    return next_event


def skip_generator_round(event: dict[str, Any], *, round_number: int, reason: str = "") -> dict[str, Any]:
    next_event = copy.deepcopy(event)
    row = _get_round(next_event, round_number)
    if str(row.get("status")) not in {"active", "preview"}:
        raise ValueError("Only an active round can be skipped.")
    if _round_has_any_scores(row):
        raise ValueError("Clear entered scores before skipping this round.")
    row["status"] = "skipped"
    row["skippedAt"] = _now_iso()
    row["skipReason"] = _clean_name(reason)
    return next_event

def _round_standings(event: dict[str, Any], round_row: dict[str, Any], participant_ids: list[str]) -> list[dict[str, Any]]:
    participants = _participant_map(event)
    stats = {
        pid: {
            "participantId": pid,
            "name": str(participants.get(pid, {}).get("name") or pid),
            "wins": 0,
            "losses": 0,
            "pointsFor": 0,
            "pointsAgainst": 0,
            "differential": 0,
            "matches": 0,
        }
        for pid in participant_ids
    }
    if str(round_row.get("status")) == "skipped":
        return list(stats.values())
    for match in _round_matches(round_row):
        if match.get("scoreA") is None or match.get("scoreB") is None:
            continue
        a = int(match["scoreA"]); b = int(match["scoreB"])
        side_a = [str(x) for x in match.get("sideA") or match.get("teamA") or []]
        side_b = [str(x) for x in match.get("sideB") or match.get("teamB") or []]
        for pid in side_a:
            row = stats.get(pid)
            if not row: continue
            row["matches"] += 1; row["pointsFor"] += a; row["pointsAgainst"] += b; row["differential"] += a-b
            row["wins" if a>b else "losses"] += 1
        for pid in side_b:
            row = stats.get(pid)
            if not row: continue
            row["matches"] += 1; row["pointsFor"] += b; row["pointsAgainst"] += a; row["differential"] += b-a
            row["wins" if b>a else "losses"] += 1
    rows = list(stats.values())
    rows.sort(key=lambda row: (-row["wins"], -row["differential"], -row["pointsFor"], row["name"].lower()))
    for idx,row in enumerate(rows,1): row["rank"]=idx
    return rows


def generator_event_standings(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return complete saved-round standings using the event's selected primary sort.

    Every participant remains visible, including players who joined late, withdrew,
    substituted, or have not yet completed a game. Skipped and unsaved rounds do
    not affect the table.
    """
    if normalize_scoring_mode(event.get("scoringMode")) == "unscored":
        return []
    participants = _participant_map(event)
    stats: dict[str, dict[str, Any]] = {}
    for participant in event.get("participants") or []:
        pid = str(participant.get("id") or "")
        if not pid:
            continue
        stats[pid] = {
            "participantId": pid,
            "name": str(participant.get("name") or pid),
            "matches": 0,
            "wins": 0,
            "losses": 0,
            "pointsFor": 0,
            "pointsAgainst": 0,
            "differential": 0,
            "rosterOrder": int(participant.get("roster_order") or 9999),
        }

    for round_row in event.get("rounds") or []:
        if str(round_row.get("status") or "") != "saved":
            continue
        for match in _round_matches(round_row):
            if match.get("scoreA") is None or match.get("scoreB") is None:
                continue
            score_a = int(match.get("scoreA") or 0)
            score_b = int(match.get("scoreB") or 0)
            side_a = [str(value) for value in match.get("sideA") or match.get("teamA") or []]
            side_b = [str(value) for value in match.get("sideB") or match.get("teamB") or []]
            for pid, points_for, points_against in [
                *((pid, score_a, score_b) for pid in side_a),
                *((pid, score_b, score_a) for pid in side_b),
            ]:
                row = stats.get(pid)
                if row is None:
                    participant = participants.get(pid, {})
                    row = {
                        "participantId": pid,
                        "name": str(participant.get("name") or pid),
                        "matches": 0,
                        "wins": 0,
                        "losses": 0,
                        "pointsFor": 0,
                        "pointsAgainst": 0,
                        "differential": 0,
                        "rosterOrder": int(participant.get("roster_order") or 9999),
                    }
                    stats[pid] = row
                row["matches"] += 1
                row["pointsFor"] += points_for
                row["pointsAgainst"] += points_against
                row["differential"] += points_for - points_against
                row["wins" if points_for > points_against else "losses"] += 1

    mode = normalize_standings_sort(event.get("standingsSort"))

    def sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
        common_tail = (
            int(row.get("losses") or 0),
            int(row.get("rosterOrder") or 9999),
            str(row.get("name") or "").casefold(),
            str(row.get("participantId") or ""),
        )
        if mode == "points":
            return (
                -int(row.get("pointsFor") or 0),
                -int(row.get("wins") or 0),
                -int(row.get("differential") or 0),
                *common_tail,
            )
        if mode == "differential":
            return (
                -int(row.get("differential") or 0),
                -int(row.get("wins") or 0),
                -int(row.get("pointsFor") or 0),
                *common_tail,
            )
        return (
            -int(row.get("wins") or 0),
            -int(row.get("differential") or 0),
            -int(row.get("pointsFor") or 0),
            *common_tail,
        )

    rows = sorted(stats.values(), key=sort_key)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["standingsSort"] = mode
    return rows


def _ladder_next_order(event: dict[str, Any], round_row: dict[str, Any], next_round: int) -> list[str]:
    active_next = active_participant_ids(event, next_round)
    active_set = set(active_next)
    if str(round_row.get("status")) == "skipped":
        prior = [
            pid
            for court in round_row.get("courts") or []
            for pid in court.get("participantIds") or []
            if str(pid) in active_set
        ]
        return [*prior, *[pid for pid in active_next if pid not in set(prior)]]
    courts = sorted(round_row.get("courts") or [], key=lambda court: int(court.get("courtNumber") or 0))
    ranked_groups = []
    for court in courts:
        ids = [str(pid) for pid in court.get("participantIds") or [] if str(pid) in active_set]
        ranked_groups.append(_round_standings(event, round_row, ids))
    # One up / one down. Start with ranked court order.
    groups = [[str(row["participantId"]) for row in group] for group in ranked_groups]
    for idx in range(len(groups)-1):
        if not groups[idx] or not groups[idx+1]:
            continue
        down = groups[idx][-1]
        up = groups[idx+1][0]
        groups[idx][-1] = up
        groups[idx+1][0] = down
    flattened = [pid for group in groups for pid in group if pid in active_set]
    flattened.extend(pid for pid in active_next if pid not in set(flattened))
    return flattened

def advance_generator_event(event: dict[str, Any]) -> dict[str, Any]:
    next_event = copy.deepcopy(event)
    current = int(next_event.get("currentRoundNumber") or 1)
    row = _get_round(next_event, current)
    if str(row.get("status")) not in {"saved", "played", "skipped"}:
        raise ValueError("Save or skip the current round before continuing. Unscored Round-Robins may mark it played.")
    total = int(next_event.get("totalRounds") or 1)
    if current >= total:
        next_event["status"] = "completed"
        next_event["completedAt"] = _now_iso()
        return next_event
    next_number = current + 1
    kind = str(next_event.get("generatorKind") or "round_robin")
    if kind == "ladder":
        order = _ladder_next_order(next_event, row, next_number)
        next_round = _create_ladder_round(next_event, next_number, order)
        next_event.setdefault("rounds", []).append(next_round)
    else:
        next_round = _get_round(next_event, next_number)
    next_round["status"] = "active"
    next_event["currentRoundNumber"] = next_number
    return next_event

def _next_participant_id(event: dict[str, Any]) -> str:
    used = {str(row.get("id")) for row in event.get("participants") or []}
    counter = 1
    while f"p-new-{counter}" in used:
        counter += 1
    return f"p-new-{counter}"

def _effective_roster_round(event: dict[str, Any]) -> int:
    if str(event.get("status")) == "preview":
        return 1
    current = int(event.get("currentRoundNumber") or 1)
    row = _get_round(event, current)
    if str(row.get("status")) in {"saved", "played", "skipped"} or _round_has_any_scores(row):
        return current + 1
    return current

def _regenerate_round_robin_from(event: dict[str, Any], start_round: int) -> None:
    preserved = [
        copy.deepcopy(row)
        for row in event.get("rounds") or []
        if int(row.get("number") or 0) < int(start_round)
    ]
    history = _blank_history(event)
    for row in preserved:
        if str(row.get("status")) == "skipped":
            continue
        for pid in row.get("byeParticipantIds") or []:
            history["byes"][str(pid)] = int(history["byes"].get(str(pid), 0)) + 1
        for match in _round_matches(row):
            _apply_match_history(history, match, str(event.get("playFormat") or "doubles"))
    generated = []
    total = int(event.get("totalRounds") or 1)
    for number in range(int(start_round), total+1):
        row = _generate_round_robin_round(event, number, history)
        if str(event.get("status")) == "active" and number == int(event.get("currentRoundNumber") or 1):
            row["status"] = "active"
        generated.append(row)
        _update_history_for_generated_round(history, row, str(event.get("playFormat") or "doubles"))
    event["rounds"] = [*preserved, *generated]

def mutate_generator_roster(
    event: dict[str, Any],
    *,
    action: str,
    participant_id: str | None = None,
    name: str | None = None,
    player_id: int | None = None,
    substitute_scope: str = "rest",
    roster_order: list[str] | None = None,
) -> dict[str, Any]:
    next_event = copy.deepcopy(event)
    clean_action = str(action or "").strip().lower()
    effective = _effective_roster_round(next_event)
    participants = _participant_map(next_event)

    if clean_action == "reorder":
        order = [str(pid) for pid in roster_order or []]
        known = {str(row["id"]) for row in next_event.get("participants") or []}
        if set(order) != known:
            raise ValueError("Roster order must include every participant exactly once.")
        for idx, pid in enumerate(order, 1):
            participants[pid]["roster_order"] = idx
    elif clean_action == "add":
        clean_name = _clean_name(name)
        if not clean_name:
            raise ValueError("Player name is required.")
        new_id = _next_participant_id(next_event)
        row = {
            "id": new_id,
            "name": clean_name,
            "seed": len(next_event.get("participants") or []) + 1,
            "roster_order": len(next_event.get("participants") or []) + 1,
            "active_from_round": effective,
            "inactive_from_round": None,
            "inactive_rounds": [],
        }
        if player_id is not None:
            row["player_id"] = int(player_id)
        next_event.setdefault("participants", []).append(row)
    elif clean_action == "remove":
        pid = str(participant_id or "")
        if pid not in participants:
            raise ValueError("Player was not found.")
        participants[pid]["inactive_from_round"] = effective
    elif clean_action == "substitute":
        pid = str(participant_id or "")
        if pid not in participants:
            raise ValueError("Player was not found.")
        clean_name = _clean_name(name)
        if not clean_name:
            raise ValueError("Substitute name is required.")
        new_id = _next_participant_id(next_event)
        scope = str(substitute_scope or "rest").lower()
        if scope not in {"round", "rest"}:
            raise ValueError("Substitute scope must be round or rest.")
        if scope == "round":
            participants[pid].setdefault("inactive_rounds", []).append(effective)
            inactive_from = effective + 1
        else:
            participants[pid]["inactive_from_round"] = effective
            inactive_from = None
        row = {
            "id": new_id,
            "name": clean_name,
            "seed": len(next_event.get("participants") or []) + 1,
            "roster_order": int(participants[pid].get("roster_order") or len(participants)+1),
            "active_from_round": effective,
            "inactive_from_round": inactive_from,
            "inactive_rounds": [],
            "substitutes_for": pid,
        }
        if player_id is not None:
            row["player_id"] = int(player_id)
        next_event.setdefault("participants", []).append(row)
    else:
        raise ValueError("Unsupported roster action.")

    next_event.setdefault("rosterRevisions", []).append({
        "action": clean_action,
        "effectiveRound": effective,
        "participantId": participant_id,
        "name": _clean_name(name),
        "at": _now_iso(),
    })

    kind = str(next_event.get("generatorKind") or "round_robin")
    if kind == "round_robin":
        _regenerate_round_robin_from(next_event, effective)
    else:
        current = int(next_event.get("currentRoundNumber") or 1)
        if effective == current:
            # Replace the unstarted current ladder round.
            prior = [row for row in next_event.get("rounds") or [] if int(row.get("number") or 0) < current]
            ordered = active_participant_ids(next_event, current)
            current_row = _create_ladder_round(next_event, current, ordered)
            current_row["status"] = "preview" if str(next_event.get("status")) == "preview" else "active"
            next_event["rounds"] = [*prior, current_row]
    return next_event

def schedule_export_rows(event: dict[str, Any]) -> list[dict[str, Any]]:
    participants = _participant_map(event)
    rows = []
    for round_row in event.get("rounds") or []:
        byes = ", ".join(str(participants.get(str(pid), {}).get("name") or pid) for pid in round_row.get("byeParticipantIds") or [])
        for match in _round_matches(round_row):
            side_a = " / ".join(str(participants.get(str(pid), {}).get("name") or pid) for pid in match.get("sideA") or match.get("teamA") or [])
            side_b = " / ".join(str(participants.get(str(pid), {}).get("name") or pid) for pid in match.get("sideB") or match.get("teamB") or [])
            rows.append({
                "round": int(round_row.get("number") or 0),
                "court": int(match.get("court") or 0),
                "play_format": generator_match_play_format(
                    match,
                    str(event.get("playFormat") or "doubles"),
                ),
                "mini_round": int(match.get("miniRound") or 0) or None,
                "side_a": side_a,
                "score_a": "",
                "score_b": "",
                "side_b": side_b,
                "byes": byes,
                "status": str(round_row.get("status") or "preview"),
            })
        if not _round_matches(round_row):
            rows.append({
                "round": int(round_row.get("number") or 0),
                "court": None,
                "play_format": None,
                "mini_round": None,
                "side_a": "",
                "score_a": "",
                "score_b": "",
                "side_b": "",
                "byes": byes,
                "status": str(round_row.get("status") or "preview"),
            })
    return rows
