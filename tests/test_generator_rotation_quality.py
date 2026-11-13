"""Player-facing rotation regressions after attendance changes."""
from collections import Counter
from copy import deepcopy

from jupr_app.domain import adaptive_play_engine as engine


# Anonymous 8 -> 9 -> 8 session: player 9 joins at round 2; player 8
# leaves at round 5. The original seventh round extended one opponent
# pairing to three consecutive rounds despite a legal better-spaced option.
PREFIX = (
    (((1, 6), (2, 5)), ((3, 8), (4, 7))),
    (((6, 9), (3, 7)), ((4, 5), (2, 8))),
    (((9, 1), (4, 8)), ((2, 7), (3, 5))),
    (((3, 4), (5, 9)), ((7, 6), (8, 1))),
    (((2, 4), (1, 7)), ((3, 9), (5, 6))),
    (((4, 9), (5, 7)), ((3, 6), (1, 2))),
)
ORIGINAL_SEVENTH = (((1, 5), (7, 9)), ((2, 3), (4, 6)))


def pair(a, b):
    return tuple(sorted((a, b)))


def round_row(number, games=(), byes=(), status="preview"):
    return {
        "number": number, "status": status,
        "matches": [
            {
                "id": f"r{number}-c{court}", "round": number,
                "court": court, "playFormat": "doubles",
                "sideA": [f"p-{pid}" for pid in left],
                "sideB": [f"p-{pid}" for pid in right],
                "teamA": [f"p-{pid}" for pid in left],
                "teamB": [f"p-{pid}" for pid in right],
                "scoreA": 11 if status == "saved" else None,
                "scoreB": 7 if status == "saved" else None,
                "status": "scored" if status == "saved" else "scheduled",
            }
            for court, (left, right) in enumerate(games, 1)
        ],
        "byeParticipantIds": [f"p-{pid}" for pid in byes],
        "warnings": [], "formatCounts": {"doubles": len(games), "singles": 0},
        "savedAt": None, "skippedAt": None,
    }


def attendance_fixture():
    event = engine.create_generator_preview(
        generator_kind="round_robin", play_format="doubles",
        title="Changing attendance", participant_names=[f"Player {i}" for i in range(1, 10)],
        total_rounds=9, court_count=2,
    )
    event.update(status="active", currentRoundNumber=6, perfectRotation=False,
                 adaptiveFromRound=1, scoringMode="scored")
    event["participants"][7]["inactive_from_round"] = 5
    event["participants"][8]["active_from_round"] = 2
    event["rosterRevisions"] = [
        {"action": "add", "effectiveRound": 2},
        {"action": "remove", "effectiveRound": 5, "participantIds": ["p-8"]},
    ]
    event["rosterByRound"] = {
        str(n): [f"p-{i}" for i in range(1, 10) if not (i == 9 and n == 1) and not (i == 8 and n >= 5)]
        for n in range(1, 10)
    }
    byes = {2: (1,), 3: (6,), 4: (2,)}
    event["rounds"] = [round_row(n, games, byes.get(n, ()), "saved") for n, games in enumerate(PREFIX, 1)]
    event["rounds"] += [round_row(n) for n in (7, 8, 9)]
    return event


def row(event, number):
    return next(item for item in event["rounds"] if item["number"] == number)


def partners(round_value):
    return {pair(*match[side]) for match in round_value["matches"] for side in ("sideA", "sideB")}


def opponents(round_value):
    return {pair(a, b) for match in round_value["matches"] for a in match["sideA"] for b in match["sideB"]}


def assert_full_round(event, number):
    current = row(event, number)
    playing = [pid for match in current["matches"] for pid in match["sideA"] + match["sideB"]]
    assert len(playing) == len(set(playing)) == 8
    assert set(playing) == set(engine.active_participant_ids(event, number))
    assert current["byeParticipantIds"] == []


def test_attendance_change_retains_history_and_breaks_avoidable_third_opponent_meeting():
    event = attendance_fixture()
    before = deepcopy(event)
    result = engine.advance_generator_event(event)
    assert event == before
    assert result["rounds"][:6] == before["rounds"][:6]
    assert_full_round(result, 7)
    prior_partners = set().union(*(partners(r) for r in before["rounds"][:6]))
    assert partners(row(result, 7)).isdisjoint(prior_partners)
    # This checks all players, so the fix cannot simply transfer the streak
    # from the reported pair to a different pair on the other court.
    assert not (opponents(row(result, 5)) & opponents(row(result, 6)) & opponents(row(result, 7)))
    assert pair("p-5", "p-9") not in opponents(row(result, 7))
    assert row(result, 8)["matches"]
    assert not row(result, 9)["matches"]
    assert engine.advance_generator_event(event)["rounds"][6]["matches"] == row(result, 7)["matches"]


def test_unavoidable_partner_repeats_are_spaced_without_increasing_their_number():
    event = attendance_fixture()
    event["rounds"][6] = round_row(7, ORIGINAL_SEVENTH, status="saved")
    event["currentRoundNumber"] = 7
    result = engine.advance_generator_event(event)
    assert result["rounds"][:7] == event["rounds"][:7]
    assert_full_round(result, 8)
    prior = Counter(p for r in event["rounds"][:7] for p in partners(r))
    chosen = partners(row(result, 8))
    # Exhausting the available fresh partners requires two repeats here;
    # a feasible arrangement spaces both repeats away from round 7.
    assert sum(prior[p] > 0 for p in chosen) == 2
    assert chosen.isdisjoint(partners(row(event, 7)))


def test_roster_edit_preserves_played_active_and_partially_scored_future_rounds():
    event = attendance_fixture()
    event["rounds"][6] = round_row(7, ORIGINAL_SEVENTH, status="active")
    event["currentRoundNumber"] = 7
    event["rounds"][8] = round_row(9, ORIGINAL_SEVENTH)
    row(event, 9)["matches"][0]["scoreA"] = 3
    before = deepcopy(event)
    result = engine.mutate_generator_roster(event, action="remove", participant_id="p-3")
    assert event == before
    assert result["rounds"][:7] == before["rounds"][:7]
    assert row(result, 9) == row(before, 9)
    assert "p-3" not in {
        pid for m in row(result, 8)["matches"] for pid in m["sideA"] + m["sideB"]
    }


def test_lookahead_keeps_fresh_partners_available_for_both_upcoming_rounds():
    games = (
        (((1, 7), (5, 6)), ((2, 3), (4, 8))),
        (((1, 2), (5, 7)), ((3, 4), (6, 8))),
        (((1, 8), (6, 7)), ((2, 5), (3, 4))),
        (((1, 5), (3, 4)), ((2, 6), (7, 8))),
    )
    event = engine.create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Two upcoming rounds",
        participant_names=[f"Player {i}" for i in range(1, 9)], total_rounds=7, court_count=2,
    )
    event.update(status="active", currentRoundNumber=4, perfectRotation=False, adaptiveFromRound=1)
    event["rounds"] = [round_row(n, pairs, status="saved") for n, pairs in enumerate(games, 1)]
    event["rounds"] += [round_row(n) for n in (5, 6, 7)]
    result = engine.advance_generator_event(event)
    # A locally preferred fifth round strands one repeated partner pair in
    # round 6. Another equally fresh fifth round allows both rounds to stay fresh.
    prior = set().union(*(partners(r) for r in event["rounds"][:4]))
    upcoming = Counter(p for r in result["rounds"][4:6] for p in partners(r))
    assert len(upcoming) == 8
    assert max(upcoming.values()) == 1
    assert set(upcoming).isdisjoint(prior)
    assert_full_round(result, 5)
    assert_full_round(result, 6)


def test_pair_recency_uses_round_numbers_and_excludes_skipped_or_unplayed_games():
    games = (((1, 2), (3, 4)), ((5, 6), (7, 8)))
    event = {
        "participants": [{"id": f"p-{i}"} for i in range(1, 9)],
        "playFormat": "doubles",
        # Imported or previously selected games need not arrive in round order.
        "rounds": [round_row(n, games, status="saved") for n in (5, 2, 4, 3, 1)],
    }
    row(event, 3)["status"] = "skipped"
    row(event, 4)["matches"][0]["status"] = "not_played"
    history = engine.history_before_round(event, 6)
    assert history["partner_rounds"]["p-1|p-2"] == [1, 2, 5]
    assert history["opponent_rounds"]["p-1|p-3"] == [1, 2, 5]
    assert history["partners"]["p-1|p-2"] == 3
    assert history["doubles_opponents"]["p-1|p-3"] == 3
    assert history["partner_rounds"]["p-5|p-6"] == [1, 2, 4, 5]
    assert history["opponent_rounds"]["p-5|p-7"] == [1, 2, 4, 5]


def test_next_round_roster_change_invalidates_prepared_rotation(monkeypatch):
    event = attendance_fixture()
    # Prepare round 7 while round 6 remains the current saved round.
    engine._regenerate_round_robin_from(event, 7, focus_round=7)
    old_key = row(event, 7)["planningKey"]
    generated = []
    original = engine._generate_round_robin_round

    def record(value, number, history):
        generated.append(number)
        return original(value, number, history)

    monkeypatch.setattr(engine, "_generate_round_robin_round", record)
    engine._regenerate_round_robin_from(event, 7, focus_round=7)
    assert generated == []
    # Round 7's own roster stays identical; its next-round options change.
    event["rosterByRound"]["8"].remove("p-9")
    engine._regenerate_round_robin_from(event, 7, focus_round=7)
    assert 7 in generated
    assert row(event, 7)["planningKey"] != old_key
    assert_full_round(event, 7)
    assert "p-9" not in {
        pid for m in row(event, 8)["matches"] for pid in m["sideA"] + m["sideB"]
    }


def test_two_round_search_has_a_fixed_candidate_expansion_budget(monkeypatch):
    calls = []
    original = engine.rank_options

    def record(candidate_sets, history, number, seed, **kwargs):
        calls.append((number, len(candidate_sets)))
        return original(candidate_sets, history, number, seed, **kwargs)

    monkeypatch.setattr(engine, "rank_options", record)
    ids = [f"p-{i}" for i in range(1, 41)]
    history = engine._blank_history({"participants": [{"id": pid} for pid in ids]})
    matches, byes, _ = engine._quality_doubles_round(
        active_ids=ids, court_count=9, history=history,
        roster_pos={pid: n for n, pid in enumerate(ids)}, round_number=10,
        next_active_ids=ids[:-1],
    )
    assert 1 < len(calls) <= 5
    assert calls[0][0] == 10 and calls[0][1] <= 17
    assert all(number == 11 and subsets <= 5 for number, subsets in calls[1:])
    assert len(matches) == 9 and len(byes) == 4
    playing = [pid for m in matches for pid in m["sideA"] + m["sideB"]]
    assert len(set(playing)) == len(playing) == 36
    assert set(playing) | set(byes) == set(ids)


def test_lookahead_does_not_trade_an_avoidable_opponent_streak_for_future_partners():
    games = (
        (((0, 6), (1, 7)), ((2, 5), (3, 4))),
        (((0, 4), (2, 3)), ((1, 7), (5, 6))),
        (((0, 6), (1, 3)), ((2, 7), (4, 5))),
        (((0, 2), (1, 6)), ((3, 5), (4, 7))),
        (((0, 5), (1, 4)), ((2, 6), (3, 7))),
        (((0, 3), (2, 7)), ((1, 4), (5, 6))),
    )
    event = engine.create_generator_preview(
        generator_kind="round_robin", play_format="doubles", title="Avoid concentrated repeats",
        participant_names=[f"Player {i}" for i in range(8)], total_rounds=9, court_count=2,
    )
    for i, participant in enumerate(event["participants"]):
        participant["id"] = f"p-{i}"
    event.update(status="active", currentRoundNumber=6, perfectRotation=False, adaptiveFromRound=1)
    event["rounds"] = [round_row(n, pairs, status="saved") for n, pairs in enumerate(games, 1)]
    event["rounds"] += [round_row(n) for n in (7, 8, 9)]
    result = engine.advance_generator_event(event)
    prior_partners = set().union(*(partners(r) for r in event["rounds"][:6]))
    assert partners(row(result, 7)).isdisjoint(prior_partners)
    assert not (opponents(row(result, 5)) & opponents(row(result, 6)) & opponents(row(result, 7)))
    assert_full_round(result, 7)
