"""Fixed Moneyball rotations, including the balanced six-player exception."""
from jupr_app.domain.perfect_round_robin import perfect_doubles_schedule


def moneyball_doubles_schedule(ids: list[str]) -> list[list[tuple[list[str], list[str]]]] | None:
    if len(ids) != 6:
        return perfect_doubles_schedule(ids)
    # One court: four games, four distinct partners and two byes per player.
    # Each opponent is faced once or twice; this is not a perfect rotation.
    rounds = [
        ((0, 1), (2, 5)),
        ((0, 5), (3, 4)),
        ((1, 3), (2, 4)),
        ((0, 4), (1, 5)),
        ((0, 3), (1, 2)),
        ((2, 3), (4, 5)),
    ]
    return [[([ids[p] for p in a], [ids[p] for p in b])] for a, b in rounds]
