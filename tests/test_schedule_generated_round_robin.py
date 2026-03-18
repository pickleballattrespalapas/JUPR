from __future__ import annotations

import re
from collections import Counter, defaultdict

import pytest

from jupr_app.domain.schedule import EXPECTED_DOUBLES_GAMES_BY_FORMAT, get_match_schedule


NEW_DOUBLE_COUNTS = [7, 10, 11, 13, 15, 16, 17, 18, 19, 20]


def _round_number(desc: str) -> int:
    match = re.search(r"Rnd\s*(\d+)", str(desc or ""), flags=re.IGNORECASE)
    assert match is not None, f"Missing round number in description: {desc!r}"
    return int(match.group(1))


@pytest.mark.parametrize("count", NEW_DOUBLE_COUNTS)
def test_generated_round_robin_formats_are_balanced_and_unique(count: int):
    players = list(range(1, count + 1))
    schedule = get_match_schedule(f"{count}-Player", players)

    assert schedule
    assert len(schedule) == EXPECTED_DOUBLES_GAMES_BY_FORMAT[f"{count}-Player"]

    seen_partner_pairs: set[tuple[int, int]] = set()
    play_counts = Counter({player: 0 for player in players})
    bye_counts = Counter({player: 0 for player in players})
    rounds: dict[int, list[dict]] = defaultdict(list)

    for match in schedule:
        round_number = _round_number(match["desc"])
        rounds[round_number].append(match)

        t1 = list(match["t1"])
        t2 = list(match["t2"])
        assert len(t1) == 2
        assert len(t2) == 2

        participants = t1 + t2
        assert len(set(participants)) == 4

        for player in participants:
            play_counts[player] += 1

        for team in (t1, t2):
            partner_pair = tuple(sorted(team))
            assert partner_pair not in seen_partner_pairs
            seen_partner_pairs.add(partner_pair)

    for round_matches in rounds.values():
        round_players: list[int] = []
        for match in round_matches:
            round_players.extend(match["t1"])
            round_players.extend(match["t2"])

        assert len(round_players) == len(set(round_players))

        active_players = set(round_players)
        for player in players:
            if player not in active_players:
                bye_counts[player] += 1

    assert max(play_counts.values()) - min(play_counts.values()) <= 1
    assert max(bye_counts.values()) - min(bye_counts.values()) <= 1
