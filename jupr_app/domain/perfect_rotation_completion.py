"""Find and verify an exact doubles completion around immutable games.

Try the known rotations with compatible player relabelings first. Small rosters
also use a bounded exact-cover search. A failed search is never called perfect.
No runtime solver dependency is required.
"""
from collections import Counter
from itertools import combinations, product
from time import monotonic

from jupr_app.domain.perfect_round_robin import perfect_doubles_schedule


def game_key(game):
    return tuple(sorted(tuple(sorted(side)) for side in game))


def game_pairs(game):
    a, b = game
    return [tuple(sorted(a)), tuple(sorted(b))], [tuple(sorted(pair)) for pair in product(a, b)]


def is_perfect_rotation(ids, games):
    partners, opponents = Counter(), Counter()
    for game in games:
        a, b = game
        if len(a) != 2 or len(b) != 2 or len(set(a + b)) != 4 or not set(a + b) <= set(ids):
            return False
        p, o = game_pairs(game)
        partners.update(p)
        opponents.update(o)
    pairs = {tuple(sorted(pair)) for pair in combinations(ids, 2)}
    return partners == Counter({pair: 1 for pair in pairs}) and opponents == Counter({pair: 2 for pair in pairs})


def find_perfect_completion(ids, fixed_games):
    """Return a verified full game set containing every fixed game, or None."""
    template = perfect_doubles_schedule(ids)
    if not template:
        return None
    canonical = [game for row in template for game in row]
    partners, opponents = Counter(), Counter()
    for a, b in fixed_games:
        if len(a) != 2 or len(b) != 2 or len(set(a + b)) != 4 or not set(a + b) <= set(ids):
            return None
        p, o = game_pairs((a, b))
        partners.update(p)
        opponents.update(o)
    if any(count > 1 for count in partners.values()) or any(count > 2 for count in opponents.values()):
        return None
    required = {game_key(game) for game in fixed_games}

    def remap(mapping):
        unused = [pid for pid in ids if pid not in mapping.values()]
        mapping = dict(mapping)
        for pid in ids:
            if pid not in mapping:
                mapping[pid] = unused.pop(0)
        result = [([mapping[p] for p in a], [mapping[p] for p in b]) for a, b in canonical]
        return result if required <= {game_key(game) for game in result} and is_perfect_rotation(ids, result) else None

    # A single opening round can always anchor a known rotation, including
    # games appended after late arrivals. Keep both teams of every game.
    playing = [pid for game in fixed_games for side in game for pid in side]
    if len(playing) == len(set(playing)) and len(fixed_games) <= len(template[0]):
        mapping = {}
        for (a, b), (c, d) in zip(template[0], fixed_games):
            mapping.update(zip(a + b, c + d))
        return remap(mapping)

    if required <= {game_key(game) for game in canonical}:
        return canonical

    deadline = monotonic() + 0.5
    pair_index = {pair: i for i, game in enumerate(canonical) for pair in game_pairs(game)[0]}

    def embed(remaining, actual_to_template, used_games):
        if monotonic() >= deadline:
            return None
        if not remaining:
            return remap({value: key for key, value in actual_to_template.items()})
        chosen = max(remaining, key=lambda game: sum(p in actual_to_template for side in game for p in side))
        a, b = chosen
        candidates = range(len(canonical))
        for side in (a, b):
            if all(p in actual_to_template for p in side):
                candidates = [pair_index[tuple(sorted(actual_to_template[p] for p in side))]]
                break
        for index in candidates:
            if index in used_games:
                continue
            c, d = canonical[index]
            for left, right in ((c, d), (d, c)):
                for x, y in product((left, left[::-1]), (right, right[::-1])):
                    extension = dict(zip(a + b, x + y))
                    if any(p in actual_to_template and actual_to_template[p] != value for p, value in extension.items()):
                        continue
                    merged = {**actual_to_template, **extension}
                    if len(set(merged.values())) != len(merged):
                        continue
                    result = embed([game for game in remaining if game is not chosen], merged, used_games | {index})
                    if result:
                        return result
        return None

    result = embed(list(fixed_games), {}, set())
    if result:
        return result
    if len(ids) > 13:
        return None

    # Exact completion of the residual partner graph: every partner edge is
    # used once, while no opponent edge may exceed two. Full coverage then
    # forces every opponent edge to equal two as well.
    deadline = monotonic() + 0.75
    pairs = {tuple(sorted(pair)) for pair in combinations(ids, 2)}
    missing = pairs - set(partners)
    candidates = []
    by_pair = {pair: [] for pair in missing}
    for a, b, c, d in combinations(ids, 4):
        for game in (([a, b], [c, d]), ([a, c], [b, d]), ([a, d], [b, c])):
            p, o = game_pairs(game)
            if set(p) <= missing and all(opponents[pair] < 2 for pair in o):
                index = len(candidates)
                candidates.append((game, p, o))
                for pair in p:
                    by_pair[pair].append(index)

    def cover(unused, chosen):
        if not unused:
            result = [*fixed_games, *(candidates[index][0] for index in chosen)]
            return result if is_perfect_rotation(ids, result) else None
        if monotonic() >= deadline:
            return None
        options = None
        for pair in sorted(unused):
            valid = [i for i in by_pair[pair] if set(candidates[i][1]) <= unused and all(opponents[p] < 2 for p in candidates[i][2])]
            if not valid:
                return None
            if options is None or len(valid) < len(options):
                options = valid
        for index in options or []:
            _game, p, o = candidates[index]
            opponents.update(o)
            result = cover(unused - set(p), [*chosen, index])
            opponents.subtract(o)
            if result:
                return result
        return None

    return cover(missing, [])
