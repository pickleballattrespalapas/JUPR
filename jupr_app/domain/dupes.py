import math


def to_int_or_neg1(x) -> int:
    """Convert to int if possible; return -1 for None/NaN/blank/bad values."""
    try:
        if x is None:
            return -1
        if isinstance(x, float) and math.isnan(x):
            return -1
        s = str(x).strip()
        if s == "" or s.lower() in ("nan", "none", "null"):
            return -1
        return int(float(s))  # handles "12.0"
    except Exception:
        return -1


def canonical_dup_key(row: dict, club_id: str) -> str:
    """
    Canonical key that matches duplicates even if:
      - players inside a team are swapped
      - team1/team2 are swapped (scores swapped too)

    `row` can be a dict (recommended) or any mapping with .get().
    """
    a1 = to_int_or_neg1(row.get("t1_p1"))
    a2 = to_int_or_neg1(row.get("t1_p2"))
    b1 = to_int_or_neg1(row.get("t2_p1"))
    b2 = to_int_or_neg1(row.get("t2_p2"))

    teamA = sorted([a1, a2])
    teamB = sorted([b1, b2])

    s1 = to_int_or_neg1(row.get("score_t1"))
    s2 = to_int_or_neg1(row.get("score_t2"))

    # Normalize ordering across teams; if swapped, swap scores too
    if tuple(teamB) < tuple(teamA):
        teamA, teamB = teamB, teamA
        s1, s2 = s2, s1

    league = str(row.get("league", "") or "").strip()
    week = str(row.get("week_tag", "") or "").strip()
    mtype = str(row.get("match_type", "") or "").strip()

    return f"{club_id}|{league}|{week}|{mtype}|{teamA[0]}-{teamA[1]}|{teamB[0]}-{teamB[1]}|{s1}-{s2}"
