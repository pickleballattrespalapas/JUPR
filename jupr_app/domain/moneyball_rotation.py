"""Fixed Moneyball rotations, including the balanced six-player exception."""
from jupr_app.domain.perfect_round_robin import perfect_doubles_schedule


def moneyball_doubles_schedule(ids: list[str], total_rounds: int = 6) -> list[list[tuple[list[str], list[str]]]] | None:
    if len(ids) != 6:
        return perfect_doubles_schedule(ids)
    if total_rounds not in {6, 9}:
        raise ValueError("Six-player Moneyball must use the full fixed rotation of 6 or 9 rounds.")
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
    if total_rounds == 9:
        # Six games each: all five partners, one repeated partner, three byes.
        # Everyone faces each opponent twice or three times and sits once per
        # three-round block, with no consecutive byes or repeated bye partners.
        rounds.extend([
            ((0, 1), (3, 4)),
            ((1, 4), (2, 5)),
            ((0, 2), (3, 5)),
        ])
    return [[([ids[p] for p in a], [ids[p] for p in b])] for a, b in rounds]
