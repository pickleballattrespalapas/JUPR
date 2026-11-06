"""Conservative starting estimates from completed generator games.

The established JUPR score model predicts score share from the difference in
average team ratings. Its inverse gives a score-implied gap of
``log10((score_a + 1) / (score_b + 1))`` in JUPR units. The extra point prevents
shutouts from implying infinite ability.

All new players are fitted together, keeping supplied existing-player ratings
fixed. A ridge prior equivalent to four doubles appearances pulls sparse
evidence toward the mean known rating. This estimates a starting level only;
it does not publish games, change existing players, or count rated history.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


ALGORITHM = "generator-score-gap-ridge-v1"
DEFAULT_JUPR = 3.5
RIDGE_WEIGHT = 1.0


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def estimate_generator_player_ratings(
    matches: list[dict[str, Any]],
    known_ratings: dict[str, float],
    target_ids: list[str],
    *,
    match_format: str = "doubles",
) -> dict[str, dict[str, Any]]:
    """Estimate unknown participant IDs in one format, with no persistence.

    ``known_ratings`` is in JUPR units and must contain only verified existing
    profiles in this roster. The caller must keep newly created profiles among
    ``target_ids`` during recomputation, rather than using earlier estimates as
    anchors. It must pass only saved, completed games, including all unknown
    participants as targets. Games with unresolved IDs, invalid scores, mixed
    formats, duplicate players or non-play status are excluded defensively.

    A component with no connection to a known rating cannot establish absolute
    ability, so its players retain the group mean (or 3.5 if nobody is known).
    Singles and doubles must be invoked separately with their own anchors.
    """
    if match_format not in {"doubles", "singles"}:
        raise ValueError("Choose doubles or singles for a rating estimate.")
    targets = sorted({str(pid) for pid in target_ids})
    if not targets:
        return {}
    target_set = set(targets)
    known: dict[str, float] = {}
    for pid, value in known_ratings.items():
        rating = _finite(value)
        if str(pid) not in target_set and rating is not None and 1 <= rating <= 7:
            known[str(pid)] = rating
    prior = math.fsum(known.values()) / len(known) if known else DEFAULT_JUPR
    size = 2 if match_format == "doubles" else 1
    allowed = target_set | set(known)
    games: list[tuple[list[str], list[str], float]] = []
    neighbors: dict[str, set[str]] = {pid: set() for pid in allowed}
    seen_match_ids: set[tuple[str, str]] = set()
    for match in matches:
        if match.get("status") not in (None, "scored"):
            continue
        explicit_format = match.get("matchFormat") or match.get("match_format")
        if explicit_format and explicit_format != match_format:
            continue
        side_a = match.get("sideA") or match.get("teamA") or []
        side_b = match.get("sideB") or match.get("teamB") or []
        if not isinstance(side_a, list) or not isinstance(side_b, list):
            continue
        if len(side_a) != size or len(side_b) != size:
            continue
        a, b = [str(pid) for pid in side_a], [str(pid) for pid in side_b]
        ids = a + b
        if len(set(ids)) != len(ids) or not set(ids) <= allowed:
            continue
        score_a, score_b = _finite(match.get("scoreA")), _finite(match.get("scoreB"))
        if (score_a is None or score_b is None or min(score_a, score_b) < 0
                or not score_a.is_integer() or not score_b.is_integer()
                or score_a == score_b):
            continue
        match_id = match.get("id")
        if match_id is not None:
            # Some schedules number matches anew in each round. Their identity
            # is the same (round, match) pair used by score review/publication.
            key = (str(match.get("round_number", match.get("roundNumber", ""))), str(match_id))
            if key in seen_match_ids:
                continue
            seen_match_ids.add(key)
        games.append((a, b, math.log10((score_a + 1) / (score_b + 1))))
        for pid in ids:
            neighbors[pid].update(ids)

    # Propagate through other unknown players, without treating them as fixed
    # known ratings. Disconnected groups keep an explicitly fallback estimate.
    connected = set(known)
    frontier = list(known)
    while frontier:
        for neighbor in neighbors[frontier.pop()]:
            if neighbor not in connected:
                connected.add(neighbor)
                frontier.append(neighbor)

    index = {pid: i for i, pid in enumerate(targets)}
    matrix = np.eye(len(targets), dtype=float) * RIDGE_WEIGHT
    values = np.full(len(targets), prior * RIDGE_WEIGHT, dtype=float)
    counts = {pid: 0 for pid in targets}
    anchored_counts = {pid: 0 for pid in targets}
    for a, b, score_gap in games:
        ids = a + b
        if not any(pid in target_set for pid in ids) or not set(ids) <= connected:
            continue
        coefficients = np.zeros(len(targets), dtype=float)
        known_gap = 0.0
        for side, sign in ((a, 1.0), (b, -1.0)):
            for pid in side:
                if pid in known:
                    known_gap += sign * known[pid] / size
                else:
                    coefficients[index[pid]] = sign / size
                    counts[pid] += 1
                    if any(other in known for other in ids):
                        anchored_counts[pid] += 1
        matrix += np.outer(coefficients, coefficients)
        values += coefficients * (score_gap - known_gap)

    estimates = np.linalg.solve(matrix, values)
    return {
        pid: {
            "rating_jupr": round(float(np.clip(estimates[index[pid]], 1.0, 7.0)), 3),
            "prior_jupr": round(prior, 3),
            "method": "score_estimate" if counts[pid] else "group_average" if known else "default_estimate",
            "games_used": counts[pid],
            "anchored_games": anchored_counts[pid],
            "known_players": len(known),
            "provisional": True,
            "algorithm": ALGORITHM,
        }
        for pid in targets
    }
