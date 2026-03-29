"""Reference generator for missing JUPR doubles round-robin formats.

Purpose
-------
JUPR currently exposes doubles round-robin schedules for:
    4, 5, 6, 8, 9, 12, and 14 players

This script generates the missing counts from 4..20:
    7, 10, 11, 13, 15, 16, 17, 18, 19, 20

Design goals
------------
- Keep existing hand-authored JUPR schedules untouched.
- Add deterministic, balanced schedules for the missing counts.
- Never repeat a partner pair inside a generated format.
- Keep byes as balanced as mathematically possible.
- Minimize repeated opponent pairings greedily when pairing teams into matches.

Output shape
------------
Each generated format is emitted as round blocks and as a flat match list.
The flat match list mirrors the shape used by JUPR's current schedule code:

    {
      "desc": "Rnd 1 • Ct 1",
      "t1": [1, 4],
      "t2": [2, 3]
    }

The numbers are 1-based player positions. JUPR can map those positions to
actual player IDs or names in its existing get_match_schedule flow.
"""
from __future__ import annotations

import collections
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

EXISTING_COUNTS = (4, 5, 6, 8, 9, 12, 14)
GENERATED_COUNTS = tuple(n for n in range(4, 21) if n not in EXISTING_COUNTS)

Pair = Tuple[int, int]
Match = Tuple[Pair, Pair]


def rr_partner_rounds(count: int) -> List[dict]:
    """Build round-robin partner rounds with the circle method.

    For counts divisible by 4 or congruent to 1 mod 4, every partner pair can be
    used directly. For counts congruent to 2 or 3 mod 4, one partner pair per
    round must be omitted so the remaining pairs can be grouped into doubles matches.
    """
    players: List[int | None] = list(range(1, count + 1))
    if count % 2 == 1:
        players.append(None)

    total = len(players)
    arr = players[:]
    rounds: List[dict] = []

    for round_number in range(1, total):
        pairs: List[Pair] = []
        ghost_bye = None
        for i in range(total // 2):
            a = arr[i]
            b = arr[total - 1 - i]
            if a is None or b is None:
                ghost_bye = b if a is None else a
            else:
                pairs.append(tuple(sorted((int(a), int(b)))))
        rounds.append({"round": round_number, "ghost_bye": ghost_bye, "pairs": pairs})

        fixed = arr[0]
        rest = arr[1:]
        rest = [rest[-1]] + rest[:-1]
        arr = [fixed] + rest

    return rounds


def choose_omitted_pairs(count: int, rounds: List[dict]) -> Dict[int, Pair]:
    """Choose one omitted partner pair in rounds where pair count is odd.

    Rules:
    - If the round already has an even number of partner pairs, nothing is omitted.
    - For counts congruent to 3 mod 4, each player should be omitted exactly twice
      in addition to their one ghost-bye round.
    - For counts congruent to 2 mod 4, all players should be omitted twice except
      two players who are omitted once.
    """
    pair_count = len(rounds[0]["pairs"])
    if pair_count % 2 == 0:
        return {}

    if count % 4 == 3:
        target_options = [{player: 2 for player in range(1, count + 1)}]
    elif count % 4 == 2:
        target_options = []
        players = list(range(1, count + 1))
        for a in range(1, count):
            for b in range(a + 1, count + 1):
                target = {player: 2 for player in players}
                target[a] = 1
                target[b] = 1
                target_options.append(target)
    else:
        raise ValueError(f"Unexpected player count {count} for omitted-pair solving")

    round_pairs = [round_info["pairs"][:] for round_info in rounds]
    round_total = len(rounds)

    def solve_target(target: Dict[int, int]) -> Dict[int, Pair] | None:
        @lru_cache(None)
        def dfs(done_mask: int, remaining_tuple: Tuple[int, ...]):
            remaining = [0] + list(remaining_tuple)
            if done_mask == (1 << round_total) - 1:
                return () if all(value == 0 for value in remaining[1:]) else None

            undecided = [r for r in range(round_total) if not ((done_mask >> r) & 1)]
            if sum(remaining[1:]) != 2 * len(undecided):
                return None

            for player in range(1, count + 1):
                if remaining[player] < 0 or remaining[player] > len(undecided):
                    return None

            best_round = None
            best_options = None

            for r in undecided:
                options = []
                for pair in round_pairs[r]:
                    a, b = pair
                    if remaining[a] > 0 and remaining[b] > 0:
                        options.append(pair)
                if not options:
                    return None
                if best_options is None or len(options) < len(best_options):
                    best_round = r
                    best_options = options
                    if len(best_options) == 1:
                        break

            assert best_round is not None
            assert best_options is not None

            def option_score(pair: Pair) -> Tuple[int, int, int]:
                a, b = pair
                return (
                    remaining[a] + remaining[b],
                    max(remaining[a], remaining[b]),
                    min(remaining[a], remaining[b]),
                )

            for pair in sorted(best_options, key=option_score, reverse=True):
                a, b = pair
                next_remaining = remaining[:]
                next_remaining[a] -= 1
                next_remaining[b] -= 1
                solved = dfs(done_mask | (1 << best_round), tuple(next_remaining[1:]))
                if solved is not None:
                    return ((best_round, pair),) + solved

            return None

        solved = dfs(0, tuple(target[player] for player in range(1, count + 1)))
        return dict(solved) if solved is not None else None

    for target in target_options:
        solved = solve_target(target)
        if solved is not None:
            return solved

    raise RuntimeError(f"Unable to choose omitted pairs for {count} players")


_PAIRING_CACHE: Dict[Tuple[Pair, ...], List[Tuple[Tuple[Pair, Pair], ...]]] = {}


def all_pairings(items: Tuple[Pair, ...]) -> List[Tuple[Tuple[Pair, Pair], ...]]:
    """Return all ways to pair the given partner-pairs into matches."""
    if items in _PAIRING_CACHE:
        return _PAIRING_CACHE[items]

    if not items:
        return [()]

    first = items[0]
    rest = items[1:]
    result: List[Tuple[Tuple[Pair, Pair], ...]] = []
    for index, other in enumerate(rest):
        remaining = rest[:index] + rest[index + 1 :]
        for pairing in all_pairings(remaining):
            result.append(((first, other),) + pairing)

    _PAIRING_CACHE[items] = result
    return result


def best_match_grouping(active_pairs: List[Pair], opponent_counts: collections.Counter) -> Tuple[Tuple[Tuple[Pair, Pair], ...], int]:
    """Choose the within-round match grouping that adds the least opponent-repeat cost."""
    best = None
    best_cost = None

    for grouping in all_pairings(tuple(active_pairs)):
        cost = 0
        for left, right in grouping:
            for x in left:
                for y in right:
                    key = (min(x, y), max(x, y))
                    current = opponent_counts[key]
                    cost += (current + 1) ** 2 - current ** 2

        if best_cost is None or cost < best_cost:
            best_cost = cost
            best = grouping

    assert best is not None
    assert best_cost is not None
    return best, best_cost


def build_template(count: int) -> dict:
    rounds = rr_partner_rounds(count)
    omitted = choose_omitted_pairs(count, rounds)
    opponent_counts: collections.Counter = collections.Counter()

    round_blocks = []
    flat_matches = []

    for round_info in rounds:
        round_number = int(round_info["round"])
        active_pairs = [pair for pair in round_info["pairs"] if omitted.get(round_number - 1) != pair]
        grouping, _ = best_match_grouping(active_pairs, opponent_counts)

        bye = []
        if round_info["ghost_bye"] is not None:
            bye.append(int(round_info["ghost_bye"]))
        if (round_number - 1) in omitted:
            bye.extend(list(omitted[round_number - 1]))
        bye = sorted(bye)

        match_rows = []
        for court_number, (team1, team2) in enumerate(grouping, start=1):
            t1 = list(team1)
            t2 = list(team2)
            row = {
                "court": court_number,
                "desc": f"Rnd {round_number} • Ct {court_number}",
                "t1": t1,
                "t2": t2,
            }
            flat_matches.append(row)
            match_rows.append(row)

            for x in t1:
                for y in t2:
                    opponent_counts[(min(x, y), max(x, y))] += 1

        round_blocks.append(
            {
                "round": round_number,
                "bye": bye,
                "matches": match_rows,
            }
        )

    return {
        "playerCount": count,
        "roundCount": len(round_blocks),
        "matchCount": len(flat_matches),
        "courtCount": len(round_blocks[0]["matches"]) if round_blocks else 0,
        "rounds": round_blocks,
        "flatMatches": flat_matches,
        "summary": summarize_template(count, round_blocks),
    }


def summarize_template(count: int, round_blocks: List[dict]) -> dict:
    partner_counts = collections.Counter()
    opponent_counts = collections.Counter()
    play_counts = collections.Counter()
    bye_counts = collections.Counter()

    for round_info in round_blocks:
        active = set()
        for match in round_info["matches"]:
            t1 = tuple(match["t1"])
            t2 = tuple(match["t2"])
            partner_counts[tuple(sorted(t1))] += 1
            partner_counts[tuple(sorted(t2))] += 1
            active.update(t1)
            active.update(t2)
            for x in t1:
                for y in t2:
                    opponent_counts[(min(x, y), max(x, y))] += 1

        for player in active:
            play_counts[player] += 1
        for player in round_info["bye"]:
            bye_counts[player] += 1

    return {
        "rounds": len(round_blocks),
        "matches": sum(len(round_info["matches"]) for round_info in round_blocks),
        "courtsPerRound": len(round_blocks[0]["matches"]) if round_blocks else 0,
        "playCountRange": [
            min(play_counts.values()) if play_counts else 0,
            max(play_counts.values()) if play_counts else 0,
        ],
        "byeCountRange": [
            min(bye_counts.values()) if bye_counts else 0,
            max(bye_counts.values()) if bye_counts else 0,
        ],
        "maxPartnerRepeat": max(partner_counts.values()) if partner_counts else 0,
        "maxOpponentRepeat": max(opponent_counts.values()) if opponent_counts else 0,
        "repeatedOpponentPairs": sum(1 for value in opponent_counts.values() if value > 1),
    }


def build_missing_templates() -> dict:
    return {
        f"{count}-Player": build_template(count)
        for count in GENERATED_COUNTS
    }


def write_outputs(out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "existingCountsInCurrentJUPR": list(EXISTING_COUNTS),
            "generatedCounts": list(GENERATED_COUNTS),
            "notes": [
                "All generated formats are switch-partner doubles schedules.",
                "Partner pairs do not repeat within a generated format.",
                "Byes are balanced as evenly as mathematically possible for each count.",
            ],
        },
        "formats": build_missing_templates(),
    }

    json_path = out_dir / "jupr_missing_round_robin_templates.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    summary_lines = [
        "# Missing JUPR doubles round robin templates",
        "",
        f"Existing counts in current JUPR: {', '.join(map(str, EXISTING_COUNTS))}",
        f"Generated counts: {', '.join(map(str, GENERATED_COUNTS))}",
        "",
        "| Format | Rounds | Matches | Courts/Round | Play Count Range | Bye Count Range | Max Partner Repeat | Max Opponent Repeat |",
        "|---|---:|---:|---:|---|---|---:|---:|",
    ]

    for label, spec in payload["formats"].items():
        summary = spec["summary"]
        summary_lines.append(
            f"| {label} | {summary['rounds']} | {summary['matches']} | {summary['courtsPerRound']} | "
            f"{summary['playCountRange'][0]}-{summary['playCountRange'][1]} | "
            f"{summary['byeCountRange'][0]}-{summary['byeCountRange'][1]} | "
            f"{summary['maxPartnerRepeat']} | {summary['maxOpponentRepeat']} |"
        )

    (out_dir / "jupr_missing_round_robin_summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_outputs(Path(__file__).resolve().parent)
