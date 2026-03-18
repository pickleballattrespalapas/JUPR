import re
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


def get_match_schedule(format_type: str, players: list[Any], custom_text: str | None = None) -> list[Match]:
    """
    Returns a list of matches, each match is:
      {"t1": [pA, pB], "t2": [pC, pD], "desc": "Rnd 1"}

    `players` may be player_ids (ints) or any identifiers; we preserve them unchanged.

    If custom_text is provided, it expects lines containing 4 numbers, e.g.:
      "1 2 3 4" (1-based indices into players)
    """
    p = list(players or [])

    # ---- Custom schedule override ----
    if custom_text and len(custom_text.strip()) > 5:
        matches: list[Match] = []
        lines = custom_text.strip().splitlines()
        r_num = 1

        for line in lines:
            nums = [int(x) for x in re.findall(r"\d+", line)]
            if len(nums) < 4:
                continue

            idx = [n - 1 for n in nums[:4]]  # convert 1-based -> 0-based
            if all(0 <= i < len(p) for i in idx):
                matches.append(
                    {"t1": [p[idx[0]], p[idx[1]]], "t2": [p[idx[2]], p[idx[3]]], "desc": f"Game {r_num}"}
                )
                r_num += 1

        if matches:
            return matches

    # ---- Standard templates ----
    try:
        needed = int(str(format_type).split("-", 1)[0])
    except Exception:
        return []

    if len(p) < needed:
        return []

    if format_type == "4-Player":
        # Expected games: 3
        return [
            {"t1": [p[1], p[0]], "t2": [p[2], p[3]], "desc": "Rnd 1"},
            {"t1": [p[3], p[1]], "t2": [p[0], p[2]], "desc": "Rnd 2"},
            {"t1": [p[3], p[0]], "t2": [p[1], p[2]], "desc": "Rnd 3"},
        ]

    if format_type == "5-Player":
        # Expected games: 5
        return [
            {"t1": [p[0], p[1]], "t2": [p[2], p[3]], "desc": "Rnd 1"},
            {"t1": [p[1], p[3]], "t2": [p[2], p[4]], "desc": "Rnd 2"},
            {"t1": [p[0], p[4]], "t2": [p[1], p[2]], "desc": "Rnd 3"},
            {"t1": [p[0], p[2]], "t2": [p[3], p[4]], "desc": "Rnd 4"},
            {"t1": [p[0], p[3]], "t2": [p[1], p[4]], "desc": "Rnd 5"},
        ]

    if format_type == "6-Player":
        # Expected games: 9
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
        # Expected games: 14
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
        # Expected games: 18
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
        # Expected games: 33
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
        # Expected games: 39
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
