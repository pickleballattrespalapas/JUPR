import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from jupr_app.domain.generated_doubles_templates import GENERATED_DOUBLES_TEMPLATES


Match = dict[str, Any]

HAND_AUTHORED_DOUBLES_PLAYER_COUNTS = [4, 5, 6, 8, 9, 12, 14]
GENERATED_DOUBLES_PLAYER_COUNTS = sorted(
    int(format_name.split("-", 1)[0]) for format_name in GENERATED_DOUBLES_TEMPLATES
)
SUPPORTED_DOUBLES_PLAYER_COUNTS = sorted(HAND_AUTHORED_DOUBLES_PLAYER_COUNTS + GENERATED_DOUBLES_PLAYER_COUNTS)
SUPPORTED_DOUBLES_FORMAT_TYPES = [f"{count}-Player" for count in SUPPORTED_DOUBLES_PLAYER_COUNTS]
EXPECTED_DOUBLES_GAMES_BY_FORMAT = {
    "4-Player": 3,
    "5-Player": 5,
    "6-Player": 9,
    "8-Player": 14,
    "9-Player": 18,
    "12-Player": 33,
    "14-Player": 39,
    **{format_name: int(template["matchCount"]) for format_name, template in GENERATED_DOUBLES_TEMPLATES.items()},
}

ORGANIZED_RR_MAX_ROUNDS = 8
ORGANIZED_RR_DEFAULT_MODE = "organized"
SCHEDULE_MODE_FULL = "full"
SCHEDULE_MODE_ORGANIZED = "organized"
SCHEDULE_MODE_NAIVE_FIRST_EIGHT = "naive_first_eight"

SCHEDULE_QUALITY_WEIGHTS = {
    "unique_exposure_score": 10_000,
    "unique_partner_score": 1_200,
    "adjacent_interaction_penalty": -550,
    "partner_to_opponent_flip_penalty": -1_600,
    "repeated_opponent_penalty": -120,
    "bye_balance_penalty": -60,
}


@dataclass(frozen=True)
class RoundBlock:
    source_round_number: int
    matches: tuple[tuple[tuple[int, int], tuple[int, int]], ...]
    original_descs: tuple[str, ...]


@dataclass(frozen=True)
class SearchState:
    round_indexes: tuple[int, ...]
    objective: tuple[int, ...]
    tie_break: tuple[int, ...]


def _map_template_match_rows(players: list[Any], match_rows: list[dict[str, Any]]) -> list[Match]:
    mapped: list[Match] = []
    for row in match_rows:
        t1 = [players[int(position) - 1] for position in row["t1"]]
        t2 = [players[int(position) - 1] for position in row["t2"]]
        mapped.append({"t1": t1, "t2": t2, "desc": str(row["desc"])})
    return mapped


def _validate_14p_rounds(rounds: list[dict[str, Any]]) -> bool:
    total_matches = 0
    for round_data in rounds:
        matches = round_data["matches"]
        if len(matches) != 3:
            return False

        used_players: set[int] = set()
        for t1_p1, t1_p2, t2_p1, t2_p2 in matches:
            players_in_match = [t1_p1, t1_p2, t2_p1, t2_p2]
            if any(not 1 <= player_num <= 14 for player_num in players_in_match):
                return False
            if len(set(players_in_match)) != 4:
                return False
            if any(player_num in used_players for player_num in players_in_match):
                return False
            used_players.update(players_in_match)

        if len(used_players) != 12:
            return False

        byes = round_data.get("byes", [])
        if len(byes) != 2 or len(set(byes)) != 2:
            return False
        if any(not 1 <= bye <= 14 for bye in byes):
            return False
        if any(bye in used_players for bye in byes):
            return False

        total_matches += len(matches)

    return total_matches == 39


def _extract_round_number(desc: str, fallback: int) -> int:
    match = re.search(r"Rnd\s*(\d+)", str(desc or ""), flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            pass
    return int(fallback)


def _replace_round_number(desc: str, new_round_number: int, court_index: int) -> str:
    value = str(desc or "").strip()
    if re.search(r"Rnd\s*\d+", value, flags=re.IGNORECASE):
        return re.sub(r"Rnd\s*\d+", f"Rnd {new_round_number}", value, count=1, flags=re.IGNORECASE)
    if value:
        return f"Rnd {new_round_number} ({value})"
    return f"Rnd {new_round_number} (Ct {court_index})"


def _group_matches_into_round_blocks(schedule: list[Match], players: list[Any]) -> list[RoundBlock]:
    grouped_matches: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)
    grouped_descs: dict[int, list[str]] = defaultdict(list)
    for idx, match in enumerate(schedule, start=1):
        round_number = _extract_round_number(str(match.get("desc", "")), idx)
        t1 = tuple(players.index(player) + 1 for player in (match.get("t1") or []))
        t2 = tuple(players.index(player) + 1 for player in (match.get("t2") or []))
        grouped_matches[round_number].append((t1, t2))
        grouped_descs[round_number].append(str(match.get("desc") or ""))
    return [
        RoundBlock(
            source_round_number=int(round_number),
            matches=tuple(grouped_matches[round_number]),
            original_descs=tuple(grouped_descs[round_number]),
        )
        for round_number in sorted(grouped_matches)
    ]


def _round_player_roles(round_block: RoundBlock) -> tuple[dict[int, set[int]], dict[int, set[int]], set[int]]:
    partners: dict[int, set[int]] = defaultdict(set)
    opponents: dict[int, set[int]] = defaultdict(set)
    active_players: set[int] = set()
    for t1, t2 in round_block.matches:
        a, b = t1
        c, d = t2
        active_players.update([a, b, c, d])
        partners[a].add(b)
        partners[b].add(a)
        partners[c].add(d)
        partners[d].add(c)
        for left in t1:
            opponents[left].update(t2)
        for right in t2:
            opponents[right].update(t1)
    return partners, opponents, active_players


def calculate_schedule_metrics(schedule: list[Match], players: list[Any]) -> dict[str, Any]:
    player_indexes = list(range(1, len(players) + 1))
    round_blocks = _group_matches_into_round_blocks(schedule, players)
    exposure_sets: dict[int, set[int]] = {player: set() for player in player_indexes}
    partner_sets: dict[int, set[int]] = {player: set() for player in player_indexes}
    bye_counts: Counter[int] = Counter({player: 0 for player in player_indexes})
    opponent_pair_counts: Counter[tuple[int, int]] = Counter()
    adjacent_interaction_count = 0
    partner_to_opponent_flip_count = 0

    previous_interactions: dict[int, set[int]] | None = None
    previous_partners: dict[int, set[int]] | None = None
    previous_opponents: dict[int, set[int]] | None = None

    for round_block in round_blocks:
        partners, opponents, active_players = _round_player_roles(round_block)
        interactions = {player: set(partners.get(player, set())) | set(opponents.get(player, set())) for player in player_indexes}

        for player in player_indexes:
            exposure_sets[player].update(interactions[player])
            partner_sets[player].update(partners.get(player, set()))
            if player not in active_players:
                bye_counts[player] += 1

        for t1, t2 in round_block.matches:
            for left in t1:
                for right in t2:
                    opponent_pair_counts[tuple(sorted((left, right)))] += 1

        if previous_interactions is not None and previous_partners is not None and previous_opponents is not None:
            for player in player_indexes:
                repeated_people = previous_interactions[player] & interactions[player]
                adjacent_interaction_count += len(repeated_people)
                partner_to_opponent_flip_count += len(previous_partners[player] & opponents.get(player, set()))
                partner_to_opponent_flip_count += len(previous_opponents[player] & partners.get(player, set()))

        previous_interactions = interactions
        previous_partners = {player: set(partners.get(player, set())) for player in player_indexes}
        previous_opponents = {player: set(opponents.get(player, set())) for player in player_indexes}

    exposure_counts = {players[player - 1]: len(exposure_sets[player]) for player in player_indexes}
    partner_counts = {players[player - 1]: len(partner_sets[player]) for player in player_indexes}
    bye_count_map = {players[player - 1]: int(bye_counts[player]) for player in player_indexes}
    repeated_opponent_count = sum(max(0, count - 1) for count in opponent_pair_counts.values())
    bye_range = (max(bye_counts.values()) - min(bye_counts.values())) if bye_counts else 0
    bye_balance_penalty = sum(abs(bye_counts[player] - (sum(bye_counts.values()) / len(player_indexes))) for player in player_indexes)

    unique_exposure_score = sum(exposure_counts.values())
    unique_partner_score = sum(partner_counts.values())

    weighted_score = (
        unique_exposure_score * SCHEDULE_QUALITY_WEIGHTS["unique_exposure_score"]
        + unique_partner_score * SCHEDULE_QUALITY_WEIGHTS["unique_partner_score"]
        + adjacent_interaction_count * SCHEDULE_QUALITY_WEIGHTS["adjacent_interaction_penalty"]
        + partner_to_opponent_flip_count * SCHEDULE_QUALITY_WEIGHTS["partner_to_opponent_flip_penalty"]
        + repeated_opponent_count * SCHEDULE_QUALITY_WEIGHTS["repeated_opponent_penalty"]
        + int(bye_balance_penalty) * SCHEDULE_QUALITY_WEIGHTS["bye_balance_penalty"]
    )

    return {
        "rounds_used": len(round_blocks),
        "matches_used": sum(len(round_block.matches) for round_block in round_blocks),
        "player_exposure_counts": exposure_counts,
        "player_partner_counts": partner_counts,
        "bye_counts": bye_count_map,
        "unique_exposure_score": unique_exposure_score,
        "unique_partner_score": unique_partner_score,
        "adjacent_interaction_penalty": adjacent_interaction_count,
        "partner_to_opponent_flip_penalty": partner_to_opponent_flip_count,
        "repeated_opponent_penalty": repeated_opponent_count,
        "bye_balance_penalty": int(bye_balance_penalty),
        "exposure_range": (min(exposure_counts.values()), max(exposure_counts.values())) if exposure_counts else (0, 0),
        "partner_range": (min(partner_counts.values()), max(partner_counts.values())) if partner_counts else (0, 0),
        "bye_range": bye_range,
        "weighted_score": weighted_score,
    }


def _objective_tuple(metrics: dict[str, Any]) -> tuple[int, ...]:
    return (
        int(metrics["unique_exposure_score"]),
        int(metrics["unique_partner_score"]),
        -int(metrics["adjacent_interaction_penalty"]),
        -int(metrics["partner_to_opponent_flip_penalty"]),
        -int(metrics["repeated_opponent_penalty"]),
        -int(metrics["bye_balance_penalty"]),
        -int(metrics["bye_range"]),
        int(metrics["weighted_score"]),
    )


def _build_schedule_from_round_blocks(selected_rounds: list[RoundBlock], players: list[Any], renumber_rounds: bool) -> list[Match]:
    schedule: list[Match] = []
    for new_round_number, round_block in enumerate(selected_rounds, start=1):
        round_number = new_round_number if renumber_rounds else round_block.source_round_number
        for court_index, ((t1, t2), desc) in enumerate(zip(round_block.matches, round_block.original_descs), start=1):
            schedule.append(
                {
                    "t1": [players[t1[0] - 1], players[t1[1] - 1]],
                    "t2": [players[t2[0] - 1], players[t2[1] - 1]],
                    "desc": _replace_round_number(desc, round_number, court_index) if renumber_rounds else desc,
                }
            )
    return schedule


def _score_round_indexes(
    round_indexes: tuple[int, ...],
    round_blocks: list[RoundBlock],
    players: list[Any],
) -> tuple[tuple[int, ...], dict[str, Any]]:
    schedule = _build_schedule_from_round_blocks([round_blocks[index] for index in round_indexes], players, renumber_rounds=False)
    metrics = calculate_schedule_metrics(schedule, players)
    return _objective_tuple(metrics), metrics


def _beam_search_round_sequence(round_blocks: list[RoundBlock], players: list[Any], target_rounds: int) -> tuple[int, ...]:
    beam_width = 48
    states = [SearchState(round_indexes=tuple(), objective=(0, 0, 0, 0, 0, 0, 0, 0), tie_break=tuple())]
    for _depth in range(target_rounds):
        next_states: dict[tuple[int, ...], SearchState] = {}
        for state in states:
            used = set(state.round_indexes)
            for round_index in range(len(round_blocks)):
                if round_index in used:
                    continue
                candidate = state.round_indexes + (round_index,)
                objective, _ = _score_round_indexes(candidate, round_blocks, players)
                tie_break = tuple(round_blocks[index].source_round_number for index in candidate)
                existing = next_states.get(candidate)
                if existing is None or (objective, tuple(-value for value in tie_break)) > (
                    existing.objective,
                    tuple(-value for value in existing.tie_break),
                ):
                    next_states[candidate] = SearchState(round_indexes=candidate, objective=objective, tie_break=tie_break)
        states = sorted(
            next_states.values(),
            key=lambda state: (state.objective, tuple(-value for value in state.tie_break)),
            reverse=True,
        )[:beam_width]
    best = max(states, key=lambda state: (state.objective, tuple(-value for value in state.tie_break)))
    return best.round_indexes


def _improve_round_sequence(initial_indexes: tuple[int, ...], round_blocks: list[RoundBlock], players: list[Any]) -> tuple[int, ...]:
    current = initial_indexes
    current_objective, _ = _score_round_indexes(current, round_blocks, players)
    target_length = len(current)

    improved = True
    while improved:
        improved = False
        used = set(current)
        unused = [index for index in range(len(round_blocks)) if index not in used]
        best_neighbor = current
        best_objective = current_objective

        current_list = list(current)
        for left in range(target_length):
            for right in range(left + 1, target_length):
                candidate_list = current_list.copy()
                candidate_list[left], candidate_list[right] = candidate_list[right], candidate_list[left]
                candidate = tuple(candidate_list)
                candidate_objective, _ = _score_round_indexes(candidate, round_blocks, players)
                if candidate_objective > best_objective:
                    best_neighbor = candidate
                    best_objective = candidate_objective

        for source in range(target_length):
            for destination in range(target_length):
                if source == destination:
                    continue
                candidate_list = current_list.copy()
                moved = candidate_list.pop(source)
                candidate_list.insert(destination, moved)
                candidate = tuple(candidate_list)
                candidate_objective, _ = _score_round_indexes(candidate, round_blocks, players)
                if candidate_objective > best_objective:
                    best_neighbor = candidate
                    best_objective = candidate_objective

        for position in range(target_length):
            for replacement in unused:
                for destination in range(target_length):
                    candidate_list = current_list.copy()
                    candidate_list[position] = replacement
                    moved = candidate_list.pop(position)
                    candidate_list.insert(destination, moved)
                    if len(set(candidate_list)) != target_length:
                        continue
                    candidate = tuple(candidate_list)
                    candidate_objective, _ = _score_round_indexes(candidate, round_blocks, players)
                    if candidate_objective > best_objective:
                        best_neighbor = candidate
                        best_objective = candidate_objective

        if best_objective > current_objective:
            current = best_neighbor
            current_objective = best_objective
            improved = True

    return current


def _organized_schedule_from_full_schedule(format_type: str, players: list[Any], full_schedule: list[Match]) -> list[Match]:
    round_blocks = _group_matches_into_round_blocks(full_schedule, players)
    if len(round_blocks) <= ORGANIZED_RR_MAX_ROUNDS:
        selected_indexes = _improve_round_sequence(tuple(range(len(round_blocks))), round_blocks, players)
    else:
        seed_indexes = _beam_search_round_sequence(round_blocks, players, ORGANIZED_RR_MAX_ROUNDS)
        selected_indexes = _improve_round_sequence(seed_indexes, round_blocks, players)
    selected_rounds = [round_blocks[index] for index in selected_indexes]
    optimized_schedule = _build_schedule_from_round_blocks(selected_rounds, players, renumber_rounds=True)
    naive_schedule = _build_schedule_from_round_blocks(
        round_blocks[:ORGANIZED_RR_MAX_ROUNDS],
        players,
        renumber_rounds=True,
    )
    optimized_metrics = calculate_schedule_metrics(optimized_schedule, players)
    naive_metrics = calculate_schedule_metrics(naive_schedule, players)
    return optimized_schedule if _objective_tuple(optimized_metrics) >= _objective_tuple(naive_metrics) else naive_schedule


def _get_full_match_schedule(format_type: str, players: list[Any]) -> list[Match]:
    p = list(players or [])

    try:
        needed = int(str(format_type).split("-", 1)[0])
    except Exception:
        return []

    if len(p) < needed:
        return []

    if format_type == "4-Player":
        return [
            {"t1": [p[1], p[0]], "t2": [p[2], p[3]], "desc": "Rnd 1"},
            {"t1": [p[3], p[1]], "t2": [p[0], p[2]], "desc": "Rnd 2"},
            {"t1": [p[3], p[0]], "t2": [p[1], p[2]], "desc": "Rnd 3"},
        ]

    if format_type == "5-Player":
        return [
            {"t1": [p[0], p[1]], "t2": [p[2], p[3]], "desc": "Rnd 1"},
            {"t1": [p[1], p[3]], "t2": [p[2], p[4]], "desc": "Rnd 2"},
            {"t1": [p[0], p[4]], "t2": [p[1], p[2]], "desc": "Rnd 3"},
            {"t1": [p[0], p[2]], "t2": [p[3], p[4]], "desc": "Rnd 4"},
            {"t1": [p[0], p[3]], "t2": [p[1], p[4]], "desc": "Rnd 5"},
        ]

    if format_type == "6-Player":
        return [
            {"t1": [p[0], p[5]], "t2": [p[1], p[3]], "desc": "Rnd 1"},
            {"t1": [p[3], p[4]], "t2": [p[0], p[2]], "desc": "Rnd 2"},
            {"t1": [p[2], p[4]], "t2": [p[1], p[5]], "desc": "Rnd 3"},
            {"t1": [p[2], p[5]], "t2": [p[0], p[1]], "desc": "Rnd 4"},
            {"t1": [p[0], p[4]], "t2": [p[3], p[5]], "desc": "Rnd 5"},
            {"t1": [p[0], p[3]], "t2": [p[1], p[2]], "desc": "Rnd 6"},
            {"t1": [p[3], p[4]], "t2": [p[1], p[5]], "desc": "Rnd 7"},
            {"t1": [p[2], p[3]], "t2": [p[4], p[5]], "desc": "Rnd 8"},
            {"t1": [p[1], p[4]], "t2": [p[0], p[2]], "desc": "Rnd 9"},
        ]

    if format_type == "8-Player":
        return [
            {"t1": [p[0], p[5]], "t2": [p[1], p[4]], "desc": "Rnd 1 (Ct 1)"},
            {"t1": [p[2], p[7]], "t2": [p[3], p[6]], "desc": "Rnd 1 (Ct 2)"},
            {"t1": [p[1], p[2]], "t2": [p[4], p[7]], "desc": "Rnd 2 (Ct 1)"},
            {"t1": [p[0], p[3]], "t2": [p[5], p[6]], "desc": "Rnd 2 (Ct 2)"},
            {"t1": [p[0], p[7]], "t2": [p[2], p[5]], "desc": "Rnd 3 (Ct 1)"},
            {"t1": [p[1], p[6]], "t2": [p[3], p[4]], "desc": "Rnd 3 (Ct 2)"},
            {"t1": [p[0], p[1]], "t2": [p[2], p[3]], "desc": "Rnd 4 (Ct 1)"},
            {"t1": [p[4], p[5]], "t2": [p[6], p[7]], "desc": "Rnd 4 (Ct 2)"},
            {"t1": [p[0], p[6]], "t2": [p[1], p[7]], "desc": "Rnd 5 (Ct 1)"},
            {"t1": [p[2], p[4]], "t2": [p[3], p[5]], "desc": "Rnd 5 (Ct 2)"},
            {"t1": [p[1], p[5]], "t2": [p[2], p[6]], "desc": "Rnd 6 (Ct 1)"},
            {"t1": [p[0], p[4]], "t2": [p[3], p[7]], "desc": "Rnd 6 (Ct 2)"},
            {"t1": [p[1], p[3]], "t2": [p[5], p[7]], "desc": "Rnd 7 (Ct 1)"},
            {"t1": [p[0], p[2]], "t2": [p[4], p[6]], "desc": "Rnd 7 (Ct 2)"},
        ]

    if format_type == "9-Player":
        return [
            {"t1": [p[1], p[2]], "t2": [p[3], p[6]], "desc": "Rnd 1 (Ct 1)"},
            {"t1": [p[4], p[8]], "t2": [p[5], p[7]], "desc": "Rnd 1 (Ct 2)"},
            {"t1": [p[2], p[0]], "t2": [p[4], p[7]], "desc": "Rnd 2 (Ct 1)"},
            {"t1": [p[5], p[6]], "t2": [p[3], p[8]], "desc": "Rnd 2 (Ct 2)"},
            {"t1": [p[0], p[1]], "t2": [p[5], p[8]], "desc": "Rnd 3 (Ct 1)"},
            {"t1": [p[3], p[7]], "t2": [p[4], p[6]], "desc": "Rnd 3 (Ct 2)"},
            {"t1": [p[4], p[5]], "t2": [p[6], p[0]], "desc": "Rnd 4 (Ct 1)"},
            {"t1": [p[7], p[2]], "t2": [p[8], p[1]], "desc": "Rnd 4 (Ct 2)"},
            {"t1": [p[5], p[3]], "t2": [p[7], p[1]], "desc": "Rnd 5 (Ct 1)"},
            {"t1": [p[8], p[0]], "t2": [p[6], p[2]], "desc": "Rnd 5 (Ct 2)"},
            {"t1": [p[3], p[4]], "t2": [p[8], p[2]], "desc": "Rnd 6 (Ct 1)"},
            {"t1": [p[6], p[1]], "t2": [p[7], p[0]], "desc": "Rnd 6 (Ct 2)"},
            {"t1": [p[7], p[8]], "t2": [p[0], p[3]], "desc": "Rnd 7 (Ct 1)"},
            {"t1": [p[1], p[5]], "t2": [p[2], p[4]], "desc": "Rnd 7 (Ct 2)"},
            {"t1": [p[8], p[6]], "t2": [p[1], p[4]], "desc": "Rnd 8 (Ct 1)"},
            {"t1": [p[2], p[3]], "t2": [p[0], p[5]], "desc": "Rnd 8 (Ct 2)"},
            {"t1": [p[6], p[7]], "t2": [p[2], p[5]], "desc": "Rnd 9 (Ct 1)"},
            {"t1": [p[0], p[4]], "t2": [p[1], p[3]], "desc": "Rnd 9 (Ct 2)"},
        ]

    if format_type == "12-Player":
        return [
            {"t1": [p[2], p[5]], "t2": [p[3], p[10]], "desc": "Rnd 1 (Ct 1)"},
            {"t1": [p[4], p[6]], "t2": [p[8], p[9]], "desc": "Rnd 1 (Ct 2)"},
            {"t1": [p[11], p[0]], "t2": [p[1], p[7]], "desc": "Rnd 1 (Ct 3)"},
            {"t1": [p[5], p[8]], "t2": [p[6], p[2]], "desc": "Rnd 2 (Ct 1)"},
            {"t1": [p[7], p[9]], "t2": [p[0], p[1]], "desc": "Rnd 2 (Ct 2)"},
            {"t1": [p[11], p[3]], "t2": [p[4], p[10]], "desc": "Rnd 2 (Ct 3)"},
            {"t1": [p[10], p[1]], "t2": [p[3], p[4]], "desc": "Rnd 3 (Ct 1)"},
            {"t1": [p[11], p[6]], "t2": [p[7], p[2]], "desc": "Rnd 3 (Ct 2)"},
            {"t1": [p[8], p[0]], "t2": [p[9], p[5]], "desc": "Rnd 3 (Ct 3)"},
            {"t1": [p[11], p[9]], "t2": [p[10], p[5]], "desc": "Rnd 4 (Ct 1)"},
            {"t1": [p[0], p[3]], "t2": [p[1], p[8]], "desc": "Rnd 4 (Ct 2)"},
            {"t1": [p[2], p[4]], "t2": [p[6], p[7]], "desc": "Rnd 4 (Ct 3)"},
            {"t1": [p[3], p[6]], "t2": [p[4], p[0]], "desc": "Rnd 5 (Ct 1)"},
            {"t1": [p[5], p[7]], "t2": [p[9], p[10]], "desc": "Rnd 5 (Ct 2)"},
            {"t1": [p[11], p[1]], "t2": [p[2], p[8]], "desc": "Rnd 5 (Ct 3)"},
            {"t1": [p[8], p[10]], "t2": [p[1], p[2]], "desc": "Rnd 6 (Ct 1)"},
            {"t1": [p[11], p[4]], "t2": [p[5], p[0]], "desc": "Rnd 6 (Ct 2)"},
            {"t1": [p[6], p[9]], "t2": [p[7], p[3]], "desc": "Rnd 6 (Ct 3)"},
            {"t1": [p[11], p[7]], "t2": [p[8], p[3]], "desc": "Rnd 7 (Ct 1)"},
            {"t1": [p[9], p[1]], "t2": [p[10], p[6]], "desc": "Rnd 7 (Ct 2)"},
            {"t1": [p[0], p[2]], "t2": [p[4], p[5]], "desc": "Rnd 7 (Ct 3)"},
            {"t1": [p[1], p[4]], "t2": [p[2], p[9]], "desc": "Rnd 8 (Ct 1)"},
            {"t1": [p[3], p[5]], "t2": [p[7], p[8]], "desc": "Rnd 8 (Ct 2)"},
            {"t1": [p[11], p[10]], "t2": [p[0], p[6]], "desc": "Rnd 8 (Ct 3)"},
            {"t1": [p[6], p[8]], "t2": [p[10], p[0]], "desc": "Rnd 9 (Ct 1)"},
            {"t1": [p[4], p[7]], "t2": [p[5], p[1]], "desc": "Rnd 9 (Ct 2)"},
            {"t1": [p[11], p[2]], "t2": [p[3], p[9]], "desc": "Rnd 9 (Ct 3)"},
            {"t1": [p[11], p[5]], "t2": [p[6], p[1]], "desc": "Rnd 10 (Ct 1)"},
            {"t1": [p[9], p[0]], "t2": [p[2], p[3]], "desc": "Rnd 10 (Ct 2)"},
            {"t1": [p[7], p[10]], "t2": [p[8], p[4]], "desc": "Rnd 10 (Ct 3)"},
            {"t1": [p[10], p[2]], "t2": [p[0], p[7]], "desc": "Rnd 11 (Ct 1)"},
            {"t1": [p[11], p[8]], "t2": [p[9], p[4]], "desc": "Rnd 11 (Ct 2)"},
            {"t1": [p[1], p[3]], "t2": [p[5], p[6]], "desc": "Rnd 11 (Ct 3)"},
        ]

    if format_type == "14-Player":
        rounds = [
            {"matches": [(6, 10, 9, 1), (4, 3, 12, 2), (13, 7, 5, 11)], "byes": [8, 14]},
            {"matches": [(13, 14, 8, 12), (11, 4, 3, 1), (5, 9, 7, 2)], "byes": [6, 10]},
            {"matches": [(5, 14, 13, 10), (7, 11, 4, 12), (2, 3, 9, 6)], "byes": [1, 8]},
            {"matches": [(4, 8, 9, 11), (13, 1, 2, 10), (6, 12, 3, 5)], "byes": [14, 7]},
            {"matches": [(14, 12, 2, 5), (6, 7, 8, 9), (10, 1, 11, 13)], "byes": [3, 4]},
            {"matches": [(13, 3, 1, 7), (11, 10, 2, 14), (8, 5, 6, 4)], "byes": [12, 9]},
            {"matches": [(10, 12, 6, 8), (1, 4, 5, 7), (9, 13, 14, 3)], "byes": [2, 11]},
            {"matches": [(4, 5, 14, 12), (2, 13, 6, 11), (7, 9, 8, 1)], "byes": [10, 3]},
            {"matches": [(10, 14, 7, 4), (11, 8, 13, 5), (1, 2, 12, 3)], "byes": [9, 6]},
            {"matches": [(7, 8, 10, 3), (5, 6, 12, 11), (14, 1, 9, 2)], "byes": [4, 13]},
            {"matches": [(12, 1, 11, 14), (4, 2, 10, 8), (3, 7, 13, 6)], "byes": [9, 5]},
            {"matches": [(3, 9, 8, 13), (1, 11, 14, 6), (4, 10, 5, 12)], "byes": [7, 2]},
            {"matches": [(14, 4, 1, 6), (10, 5, 3, 8), (11, 2, 7, 9)], "byes": [12, 13]},
        ]

        if not _validate_14p_rounds(rounds):
            return []

        matches: list[Match] = []
        for round_index, round_data in enumerate(rounds, start=1):
            for court_index, (t1_p1, t1_p2, t2_p1, t2_p2) in enumerate(round_data["matches"], start=1):
                matches.append(
                    {
                        "t1": [p[t1_p1 - 1], p[t1_p2 - 1]],
                        "t2": [p[t2_p1 - 1], p[t2_p2 - 1]],
                        "desc": f"Rnd {round_index} (Ct {court_index})",
                    }
                )
        return matches

    generated_template = GENERATED_DOUBLES_TEMPLATES.get(format_type)
    if generated_template is not None:
        return _map_template_match_rows(p, generated_template["flatMatches"])

    return []


def get_match_schedule(
    format_type: str,
    players: list[Any],
    custom_text: str | None = None,
    schedule_mode: str = SCHEDULE_MODE_FULL,
) -> list[Match]:
    """
    Returns a list of matches, each match is:
      {"t1": [pA, pB], "t2": [pC, pD], "desc": "Rnd 1"}

    `players` may be player_ids (ints) or any identifiers; we preserve them unchanged.

    If custom_text is provided, it expects lines containing 4 numbers, e.g.:
      "1 2 3 4" (1-based indices into players)
    """
    p = list(players or [])

    if custom_text and len(custom_text.strip()) > 5:
        matches: list[Match] = []
        lines = custom_text.strip().splitlines()
        r_num = 1

        for line in lines:
            nums = [int(x) for x in re.findall(r"\d+", line)]
            if len(nums) < 4:
                continue

            idx = [n - 1 for n in nums[:4]]
            if all(0 <= i < len(p) for i in idx):
                matches.append(
                    {"t1": [p[idx[0]], p[idx[1]]], "t2": [p[idx[2]], p[idx[3]]], "desc": f"Game {r_num}"}
                )
                r_num += 1

        if matches:
            return matches

    if schedule_mode == SCHEDULE_MODE_NAIVE_FIRST_EIGHT:
        full_schedule = _get_full_match_schedule(format_type, p)
        round_blocks = _group_matches_into_round_blocks(full_schedule, p)
        return _build_schedule_from_round_blocks(round_blocks[:ORGANIZED_RR_MAX_ROUNDS], p, renumber_rounds=True)

    if schedule_mode == SCHEDULE_MODE_ORGANIZED:
        full_schedule = _get_full_match_schedule(format_type, p)
        return _organized_schedule_from_full_schedule(format_type, p, full_schedule)

    return _get_full_match_schedule(format_type, p)
