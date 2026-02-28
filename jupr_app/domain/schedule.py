import re
from typing import Any


Match = dict[str, Any]


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

    return []
