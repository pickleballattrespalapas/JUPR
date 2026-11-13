"""Bounded, deterministic candidate search for adaptive doubles rotations."""
from __future__ import annotations

import itertools
import random
from functools import lru_cache
from typing import Any

Team = tuple[str, str]
Games = tuple[tuple[Team, Team], ...]
Cost = tuple[int, ...]
Option = tuple[Cost, Games, tuple[str, ...]]


def pair(a: str, b: str) -> str:
    return '|'.join(sorted((a, b)))


def _pairings(items):
    if not items:
        yield ()
        return
    for i in range(1, len(items)):
        for rest in _pairings(items[1:i] + items[i + 1:]):
            yield ((items[0], items[i]),) + rest


@lru_cache(maxsize=2)
def _layouts(size):
    return tuple(tuple((teams[a], teams[b]) for a, b in courts)
                 for teams in _pairings(tuple(range(size)))
                 for courts in _pairings(tuple(range(size // 2))))


def _run(rounds: list[int], number: int) -> int:
    """Include both sides of a gap when games were recorded out of order."""
    seen = set(rounds)
    length = 1
    for step in (-1, 1):
        n = number + step
        while n in seen:
            length += 1
            n += step
    return length


def rank_options(
    candidate_sets: list[tuple[list[str], list[str]]], history: dict[str, Any],
    number: int, seed: int, *, limit: int = 4,
) -> list[Option]:
    """Enumerate 3/315 small layouts; sample 64 larger ones per fair subset.

    Pair features are computed once. The caller caps fair subsets and expands
    only four candidates for one subsequent round, never the full session.
    """
    ids = sorted({pid for playing, _ in candidate_sets for pid in playing})
    partners, opponents = {}, {}
    for a, b in itertools.combinations(ids, 2):
        key = pair(a, b)
        partners[key] = (int(history['partners'].get(key, 0)),
                         _run(history.get('partner_rounds', {}).get(key, []), number))
        opponents[key] = (int(history['doubles_opponents'].get(key, 0)),
                          _run(history.get('opponent_rounds', {}).get(key, []), number))

    def opponent_cost(ta, tb):
        values = [opponents[pair(a, b)] for a in ta for b in tb]
        return (max(v[0] for v in values), max(v[1] for v in values),
                sum((v[1] - 1) ** 2 for v in values), sum(v[0] ** 2 for v in values))

    def score(games):
        ps = [partners[pair(*team)] for game in games for team in game]
        os = [opponents[pair(a, b)] for ta, tb in games for a in ta for b in tb]
        # Partner coverage first; then repeat concentration and spacing.
        return (sum(v[0] > 0 for v in ps), sum(v[0] for v in ps),
                max(v[0] for v in ps), max(v[1] for v in ps), sum((v[1]-1)**2 for v in ps),
                max(v[0] for v in os), max(v[1] for v in os), sum((v[1]-1)**2 for v in os),
                sum(v[0] ** 2 for v in os), sum(v[0] for v in os),
                sum(history['exact_matches'].get('|'.join(sorted((','.join(sorted(a)), ','.join(sorted(b))))), 0) for a, b in games),
                sum(history.get('groups', {}).get(','.join(sorted(a+b)), 0) for a, b in games))

    best = []
    seen = set()
    for index, (playing, byes) in enumerate(candidate_sets):
        playing = sorted(playing)
        if len(playing) <= 8:
            candidates = (tuple((tuple(playing[i] for i in a), tuple(playing[i] for i in b))
                                for a, b in layout) for layout in _layouts(len(playing)))
        else:
            def sampled():
                rng = random.Random(f'{seed}:{number}:{index}')
                for _ in range(64):
                    remaining = playing[:]
                    rng.shuffle(remaining)
                    teams = []
                    while remaining:
                        a = remaining.pop()
                        b = min(remaining, key=lambda b: (*partners[pair(a,b)], rng.random()))
                        remaining.remove(b)
                        teams.append(tuple(sorted((a,b))))
                    rng.shuffle(teams)
                    games = []
                    while teams:
                        a = teams.pop()
                        b = min(teams, key=lambda b: (*opponent_cost(a,b), rng.random()))
                        teams.remove(b)
                        games.append((a,b))
                    yield tuple(games)
            candidates = sampled()
        for games in candidates:
            key = tuple(sorted(tuple(sorted((tuple(sorted(a)), tuple(sorted(b))))) for a,b in games))
            if key in seen:
                continue
            seen.add(key)
            cost = score(games)
            entry = (cost, key, tuple(byes))
            if len(best) < limit or entry < best[-1]:
                best.append(entry)
                best.sort()
                del best[limit:]
    return best


def window_cost(first: Cost, second: Cost) -> Cost:
    """Preserve current coverage and worst repeats before looking ahead.

    Speculative future coverage must not introduce an avoidable current streak.
    """
    return (first[:4] + first[5:7] + (first[0] + second[0], first[1] + second[1])
            + tuple(max(first[i], second[i]) if i in (2,3,5,6) else first[i]+second[i]
                    for i in range(2, len(first))) + first)
