"""Exact doubles rotations: every partner once and every opponent twice."""
from jupr_app.domain.schedule import get_match_schedule
from jupr_app.domain.perfect_doubles_starters import CYCLIC_STARTERS


def perfect_doubles_schedule(ids: list[str]) -> list[list[tuple[list[str], list[str]]]] | None:
    count = len(ids)
    courts = count // 4
    if count in {4, 5, 8, 9, 12}:
        rows = get_match_schedule(f"{count}-Player", ids, schedule_mode="full")
        return [[(list(row["t1"]), list(row["t2"])) for row in rows[start:start + courts]]
                for start in range(0, len(rows), courts)]
    starter = CYCLIC_STARTERS.get(count)
    if starter is None:
        return None
    modulus = count if count % 4 == 1 else count - 1
    return [[([ids[-1] if p == -1 else ids[(p + shift) % modulus] for p in a],
               [ids[-1] if p == -1 else ids[(p + shift) % modulus] for p in b])
              for a, b in starter] for shift in range(modulus)]
