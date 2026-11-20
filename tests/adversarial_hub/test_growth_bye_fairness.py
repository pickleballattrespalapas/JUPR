"""Arrival fairness at the Hub boundary, not just the standalone generator."""
from copy import deepcopy

import pytest

from tests.adversarial_hub.growth_audit import GrowthAudit
from tests.test_admin_round_robin_hub_service import TransactionalHubDB, session, create, command, raw_session


@pytest.mark.parametrize("preview_rounds", [1, 3])
def test_two_growing_groups_do_not_repeat_a_new_arrivals_bye(preview_rounds):
    audit = GrowthAudit(preview_rounds=preview_rounds)
    audit.fill()
    initial = deepcopy(audit.matches())
    for key in ("A", "B"):
        while len(audit.events[key]["participants"]) < 9:
            audit.arrive(key)
    for key, event in audit.events.items():
        first, second = event["rounds"][:2]
        assert not any(m.get("hubStartedAt") for m in second["matches"]), "Second-round lineups are still free to plan fairly"
        repeats = set(first["byeParticipantIds"]) & set(second["byeParticipantIds"])
        assert not repeats, f"Group {key} assigns avoidable consecutive byes to {sorted(repeats)}"
    for key, match in initial.items():
        if match.get("hubStartedAt"):
            assert audit.matches()[key]["sideA"] == match["sideA"]
            assert audit.matches()[key]["sideB"] == match["sideB"]


def test_service_commits_fair_next_round_for_two_groups_growing_from_four_and_six():
    db = TransactionalHubDB()
    session(db, key="rr-a", count=4, offset=1000)
    session(db, key="rr-b", count=6, offset=2000)
    detail = create(db, courts=10)
    for key in ("rr-a", "rr-b"):
        detail = command(db, detail, "add_group", {"session_key": key})
    detail = command(db, detail, "auto_fill_courts")
    for key, initial, offset in [("rr-a",4,1000),("rr-b",6,2000)]:
        for number in range(initial+1,10):
            detail = command(db, detail, "add_player", {"session_key":key,"name":f"Player {offset+number}","player_id":offset+number})
    for key in ("rr-a", "rr-b"):
        event = raw_session(db,key)["state"]["page_state"]["event"]
        first, second = event["rounds"][:2]
        repeats = set(first["byeParticipantIds"]) & set(second["byeParticipantIds"])
        assert not repeats, f"Persisted {key} gives consecutive byes to {sorted(repeats)}"
