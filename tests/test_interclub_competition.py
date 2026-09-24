"""Southern BCS rules exercised through real generated score documents."""
from collections import Counter
from copy import deepcopy

import pytest

from jupr_app.domain.interclub_competition import (
    club_cup,
    generate_championship,
    generate_round_robin,
    league_standings,
    matchup_points,
    prepare_reschedule,
    qualifying_clubs,
    rating_games,
    singles_court,
    singles_rotation,
    summarize_document,
    validate_document,
)


PLAYED_AT = "2027-01-16T17:00:00Z"


def entry(club, division="3.5"):
    return {"club_id": club, "division": division, "roster": [
        {"entry_id": f"{club}-{division}-w1", "gender": "female"},
        {"entry_id": f"{club}-{division}-w2", "gender": "female"},
        {"entry_id": f"{club}-{division}-m1", "gender": "male"},
        {"entry_id": f"{club}-{division}-m2", "gender": "male"},
    ]}


def regular(clubs=("a", "b"), division="3.5", meet="meet", format="gender"):
    return generate_round_robin(meet, [entry(c, division) for c in clubs], format=format, played_at=PLAYED_AT)


def complete_game(game, winner="a", loser_score=7):
    game.update(status="completed", a=11 if winner == "a" else loser_score,
                b=11 if winner == "b" else loser_score, winner=None)


def complete_pair(pairing, winner="a", scores=None):
    for index, game in enumerate(pairing["games"]):
        complete_game(game, scores[index] if scores else winner)


def complete_doc(doc, winner="a"):
    for encounter in doc["encounters"]:
        for pairing in encounter["pairings"]:
            complete_pair(pairing, winner)
    return doc


def final(division="3.5", meet="final", a="a", b="b", winners=("a", "a", "a", "b"), phase="final"):
    doc = generate_championship(meet, division, entry(a, division), entry(b, division), phase=phase, played_at=PLAYED_AT)
    for pairing, winner in zip(doc["encounters"][0]["pairings"], winners):
        complete_pair(pairing, winner)
    return doc


@pytest.mark.parametrize("count,games_per_player,pairings,rotations", [(2, 3, 2, 1), (3, 6, 6, 3), (4, 9, 12, 3)])
@pytest.mark.parametrize("format", ["gender", "mixed"])
def test_smaller_round_robins_play_every_opponent_once(count, games_per_player, pairings, rotations, format):
    doc = regular(tuple("abcd"[:count]), format=format)
    all_pairings = [p for e in doc["encounters"] for p in e["pairings"]]
    assert len(all_pairings) == pairings
    appearances = Counter(player for p in all_pairings for side in ("a", "b") for player in p[f"players_{side}"] for _ in p["games"])
    assert set(appearances.values()) == {games_per_player}
    assert len(appearances) == count * 4
    assert len({e["rotation"] for e in doc["encounters"]}) == rotations
    assert len({frozenset((e["club_a"], e["club_b"])) for e in doc["encounters"]}) == count * (count - 1) // 2


def test_two_levels_four_clubs_need_eight_courts_and_72_games():
    entries = [entry(club, division) for division in ("3.5", "4.0") for club in "abcd"]
    doc = generate_round_robin("meet", entries, courts=8)
    assert sum(len(p["games"]) for e in doc["encounters"] for p in e["pairings"]) == 72
    for rotation in (1, 2, 3):
        assert sorted(p["court"] for e in doc["encounters"] if e["rotation"] == rotation for p in e["pairings"]) == list(range(1, 9))
    with pytest.raises(ValueError, match="needs 8 courts"):
        generate_round_robin("meet", entries, courts=7)


@pytest.mark.parametrize("format", ["gender", "mixed"])
def test_staggered_draw_fits_eight_courts_without_losing_games_or_double_booking(format):
    # The reported case: three four-club divisions plus one three-club division.
    entries = [entry(club, division) for division, clubs in
               [("3.0", "abcd"), ("3.5", "abcd"), ("4.0", "abcd"), ("4.5", "abc")] for club in clubs]
    simultaneous = generate_round_robin("meet", entries, courts=14, format=format)
    doc = generate_round_robin("meet", entries, courts=8, format=format, schedule_mode="staggered")
    assert doc == generate_round_robin("meet", list(reversed(entries)), courts=8, format=format, schedule_mode="staggered")
    assert doc["schedule_mode"] == "staggered"
    assert {e["rotation"] for e in doc["encounters"]} == set(range(1, 7))
    assert {g["id"] for e in doc["encounters"] for p in e["pairings"] for g in p["games"]} == {
        g["id"] for e in simultaneous["encounters"] for p in e["pairings"] for g in p["games"]}
    for rotation in range(1, 7):
        pairings = [p for e in doc["encounters"] if e["rotation"] == rotation for p in e["pairings"]]
        courts = [p["court"] for p in pairings if p["court"]]
        assert len(courts) == len(set(courts)) <= 8
        assert min(courts) >= 1 and max(courts) <= 8
        players = [player for p in pairings for side in ("a", "b") for player in p[f"players_{side}"]]
        assert len(players) == len(set(players))
    complete_doc(doc)
    complete_doc(simultaneous)
    # Actual timestamps do not come from a wave estimate.
    for value in (doc, simultaneous):
        for e in value["encounters"]:
            for p in e["pairings"]:
                for g in p["games"]:
                    g["played_at"] = PLAYED_AT
    assert league_standings([doc]) == league_standings([simultaneous])
    assert len(rating_games(doc)) == 126


def test_staggered_odd_court_capacity_and_partial_forfeits():
    entries = [entry(club, division) for division in ("3.0", "3.5") for club in "abcd"]
    doc = generate_round_robin("odd", entries, courts=3, schedule_mode="staggered")
    assert len({e["rotation"] for e in doc["encounters"]}) == 12
    assert all(p["court"] in (1, 2) for e in doc["encounters"] for p in e["pairings"])
    partial = [entry(club) for club in "abcd"]
    for row in partial:
        row["roster"] = row["roster"][:2]
    doc = generate_round_robin("partial", partial, courts=1, schedule_mode="staggered")
    assert len({e["rotation"] for e in doc["encounters"]}) == 6
    assert all(e["pairings"][0]["court"] == 1 and e["pairings"][1]["court"] is None for e in doc["encounters"])
    assert all(g["status"] == "double_forfeit" for e in doc["encounters"] for g in e["pairings"][1]["games"])
    with pytest.raises(ValueError, match="at least 2"):
        generate_round_robin("full", entries, courts=1, schedule_mode="staggered")


def test_legacy_schedule_defaults_and_invalid_staggered_capacity():
    doc = regular()
    doc.pop("schedule_mode")
    assert validate_document(doc)["schedule_mode"] == "simultaneous"
    for courts in (None, 0, -1, 101, True):
        with pytest.raises(ValueError):
            generate_round_robin("meet", [entry("a"), entry("b")], courts=courts, schedule_mode="staggered")


@pytest.mark.parametrize("clubs", [("a",), ("a", "b", "c", "d", "e"), ("a", "a")])
def test_bad_round_robin_entries_rejected(clubs):
    with pytest.raises(ValueError):
        regular(clubs)


def test_generation_requires_two_men_and_two_women():
    bad = entry("a")
    bad["roster"][0]["gender"] = "male"
    with pytest.raises(ValueError, match="two women and two men"):
        generate_round_robin("meet", [bad, entry("b")])


def test_generated_ids_stable_despite_entry_order():
    one = regular(tuple("abcd"))
    two = regular(tuple("dcba"))
    assert one == two


@pytest.mark.parametrize("winners,expected", [(("a", "a"), (3, 0)), (("b", "b"), (0, 3)),
                                              (("a", "b"), (1, 1)), (("a", "draw"), (3, 1)),
                                              (("b", "draw"), (1, 3)), (("draw", "draw"), (1, 1))])
def test_all_standings_point_combinations(winners, expected):
    assert matchup_points(winners) == expected
    assert matchup_points(reversed(winners)) == expected


def test_regular_pairing_plays_all_three_even_after_two_wins():
    doc = regular()
    for p in doc["encounters"][0]["pairings"]:
        complete_game(p["games"][0])
        complete_game(p["games"][1])
    validate_document(doc)  # The incomplete pencil-and-paper transcription is a valid draft.
    with pytest.raises(ValueError, match="whole meet"):
        validate_document(doc, official=True)
    assert not summarize_document(doc)["complete"]


def test_incomplete_completed_score_can_be_saved_but_not_official():
    doc = regular()
    doc["encounters"][0]["pairings"][0]["games"][0].update(status="completed", a=11)
    assert not summarize_document(doc)["complete"]
    with pytest.raises(ValueError, match="both scores"):
        validate_document(doc, official=True)


@pytest.mark.parametrize("score", [(11, 0), (11, 9), (12, 10), (102, 100), (1002, 1000)])
def test_win_by_two_has_no_score_cap(score):
    doc = complete_doc(regular())
    doc["encounters"][0]["pairings"][0]["games"][0].update(a=score[0], b=score[1])
    validate_document(doc, official=True)


@pytest.mark.parametrize("score", [(10, 8), (11, 10), (14, 8), (12, 9), (13, 13)])
def test_invalid_final_score_rejected(score):
    doc = complete_doc(regular())
    doc["encounters"][0]["pairings"][0]["games"][0].update(a=score[0], b=score[1])
    with pytest.raises(ValueError, match="win by two"):
        validate_document(doc, official=True)


def test_injury_retirement_uses_actual_score_even_if_winning_side_concedes():
    doc = complete_doc(regular())
    game = doc["encounters"][0]["pairings"][0]["games"][0]
    game.update(status="retired", a=8, b=2, winner="b")
    results = summarize_document(validate_document(doc, official=True))["encounters"][0]
    assert results["games_a"] == 5 and results["games_b"] == 1
    assert results["point_differential"] == 26  # 5 x +4 and stopped game +6.
    assert len(rating_games(doc)) == 5
    assert game["id"] not in {g["id"] for g in rating_games(doc)}


def test_game_already_won_cannot_be_relabelled_as_injury_retired():
    doc = complete_doc(regular())
    game = doc["encounters"][0]["pairings"][0]["games"][0]
    game.update(status="retired", a=11, b=9, winner="b")
    with pytest.raises(ValueError, match="already finished"):
        validate_document(doc, official=True)


def test_missing_pairing_forfeits_only_its_three_games_without_fake_scores():
    doc = complete_doc(regular())
    pairing = doc["encounters"][0]["pairings"][0]
    pairing["players_a"] = []  # An absent pairing has no invented players.
    for game in pairing["games"]:
        game.update(status="forfeit", a=None, b=None, winner="b")
    result = summarize_document(validate_document(doc, official=True))["encounters"][0]
    assert (result["points_a"], result["points_b"]) == (1, 1)
    assert result["games_a"] == 3 and result["games_b"] == 3
    assert result["point_differential"] == 12
    assert len(rating_games(doc)) == 3


def test_no_games_no_reschedule_is_standings_draw_with_no_appearance():
    doc = regular()
    doc["weather"] = "finalized_partial"
    for pairing in doc["encounters"][0]["pairings"]:
        for game in pairing["games"]:
            game["status"] = "unplayed"
    rows = league_standings([doc])["divisions"]["3.5"]
    assert {r["points"] for r in rows} == {1}
    assert {r["games_won"] for r in rows} == {0}
    assert {r["meets_played"] for r in rows} == {0}
    assert qualifying_clubs([doc])["3.5"]["status"] == "insufficient_entries"
    assert rating_games(doc) == []


def test_one_game_wins_pairing_and_one_each_draws_when_weather_final():
    doc = regular()
    doc["weather"] = "finalized_partial"
    one, two = doc["encounters"][0]["pairings"]
    complete_game(one["games"][0], "a")
    complete_game(two["games"][0], "a")
    complete_game(two["games"][1], "b")
    for pairing in (one, two):
        for game in pairing["games"]:
            if game["status"] == "pending":
                game["status"] = "unplayed"
    result = summarize_document(validate_document(doc, official=True))["encounters"][0]
    assert [p["winner"] for p in result["pairings"]] == ["a", "draw"]
    assert (result["points_a"], result["points_b"]) == (3, 1)
    assert len(rating_games(doc)) == 3


def test_weather_delay_keeps_partial_score_but_cannot_be_official():
    doc = regular()
    doc["weather"] = "delay"
    doc["encounters"][0]["pairings"][0]["games"][0].update(a=5, b=3)
    saved = validate_document(doc)
    assert saved["encounters"][0]["pairings"][0]["games"][0]["a"] == 5
    with pytest.raises(ValueError):
        validate_document(doc, official=True)


def test_reschedule_replaces_every_game_of_unfinished_pairing_only():
    doc = regular()
    finished, unfinished = doc["encounters"][0]["pairings"]
    finished["eligibility_deadline"] = "2027-01-15T17:00:00Z"
    unfinished["eligibility_deadline"] = "2027-01-15T17:00:00Z"
    complete_pair(finished)
    complete_game(unfinished["games"][0], "b")
    complete_game(unfinished["games"][1], "a")
    before = deepcopy(doc)
    new = prepare_reschedule(doc, played_at="2027-01-23T17:00:00Z", eligibility_deadline="2027-01-22T17:00:00Z")
    assert doc == before
    first, second = new["encounters"][0]["pairings"]
    assert first["eligibility_deadline"] == finished["eligibility_deadline"]
    assert all(g["status"] == "completed" for g in first["games"])
    assert second["eligibility_deadline"] == "2027-01-22T17:00:00Z"
    assert all(g["status"] == "pending" and g["a"] is None and g["b"] is None for g in second["games"])
    assert all(g["played_at"] == "2027-01-23T17:00:00Z" for g in second["games"])


def test_replayed_mixed_pairing_can_use_new_eligible_lineup_without_injury():
    doc = regular(format="mixed")
    finished, unfinished = doc["encounters"][0]["pairings"]
    for pair in (finished, unfinished):
        pair["eligibility_deadline"] = "2027-01-15T17:00:00Z"
    complete_pair(finished)
    replay = prepare_reschedule(doc, played_at="2027-01-23T17:00:00Z", eligibility_deadline="2027-01-22T17:00:00Z")
    old, new = replay["encounters"][0]["pairings"]
    # The same players may now play the replayed pairing on the new date.
    new["players_a"] = old["players_a"][:]
    new["players_b"] = old["players_b"][:]
    complete_pair(new)
    validate_document(replay, official=True)
    assert len(rating_games(replay)) == 6


def test_timezone_spelling_cannot_fake_a_new_roster_cutoff():
    doc = regular(format="mixed")
    first, second = doc["encounters"][0]["pairings"]
    first["eligibility_deadline"] = "2027-01-15T17:00:00Z"
    second["eligibility_deadline"] = "2027-01-15T10:00:00-07:00"
    second["players_a"] = first["players_a"][:]
    with pytest.raises(ValueError, match="four different"):
        validate_document(doc)


def test_replayed_actual_game_cannot_precede_new_roster_deadline():
    doc = complete_doc(regular())
    doc["encounters"][0]["pairings"][0]["eligibility_deadline"] = "2027-01-22T17:00:00Z"
    with pytest.raises(ValueError, match="precede"):
        validate_document(doc, official=True)


def test_actual_injury_substitute_is_rated_for_the_games_they_played_only():
    doc = complete_doc(regular())
    pairing = doc["encounters"][0]["pairings"][0]
    replacement = [pairing["players_a"][0], "a-approved-substitute"]
    for game in pairing["games"][1:]:
        game["players_a"] = replacement
    pairing["games"][1]["injury_reason"] = "Ankle injury between games; eligible reserve replaces injured player."
    rates = rating_games(doc)
    by_id = {g["id"]: g for g in rates}
    assert by_id[pairing["games"][0]["id"]]["players_a"] == pairing["players_a"]
    assert by_id[pairing["games"][1]["id"]]["players_a"] == replacement
    assert by_id[pairing["games"][2]["id"]]["players_a"] == replacement


def test_healthy_lineup_changes_rejected():
    doc = complete_doc(regular())
    doc["encounters"][0]["pairings"][0]["games"][1]["players_a"] = ["a-reserve1", "a-reserve2"]
    with pytest.raises(ValueError, match="injury reason"):
        validate_document(doc, official=True)


def test_same_player_cannot_play_both_pairings():
    doc = regular()
    first, second = doc["encounters"][0]["pairings"]
    second["players_a"][0] = first["players_a"][0]
    with pytest.raises(ValueError, match="four different"):
        validate_document(doc)


def test_rating_order_uses_absolute_timestamps_not_offset_text():
    doc = complete_doc(regular())
    games = doc["encounters"][0]["pairings"][0]["games"]
    games[0]["played_at"] = "2027-01-16T10:00:00-07:00"  # 17:00 UTC
    games[1]["played_at"] = "2027-01-16T12:00:00Z"  # actually first
    assert rating_games(doc)[0]["id"] == games[1]["id"]


def test_rating_order_at_equal_actual_timestamps_follows_games_then_rotations():
    doc = complete_doc(regular(tuple("abcd")))
    games = rating_games(doc)
    schedule = {g["id"]: (e["rotation"], index) for e in doc["encounters"] for p in e["pairings"] for index, g in enumerate(p["games"])}
    assert [schedule[g["id"]] for g in games] == sorted(schedule.values())
    assert [g["sequence"] for g in games] == list(range(len(games)))


def test_games_have_exact_actual_player_ids_for_each_club():
    games = rating_games(complete_doc(regular()))
    assert len(games) == 6
    assert all(g["club_a"] == "a" and g["club_b"] == "b" for g in games)
    assert all(len(g["players_a"]) == len(g["players_b"]) == 2 for g in games)


def test_whole_completed_meet_has_three_points_per_opponent_not_per_pairing():
    doc = regular(tuple("abcd"))
    for encounter in doc["encounters"]:
        winner = "a" if encounter["club_a"] == "a" else "b" if encounter["club_b"] == "a" else "a"
        for pairing in encounter["pairings"]:
            complete_pair(pairing, winner)
    rows = league_standings([doc])["divisions"]["3.5"]
    a = next(r for r in rows if r["club_id"] == "a")
    assert a["points"] == 9 and a["pairings_won"] == 6 and a["games_won"] == 18
    assert a["meets_played"] == 1


def tied_three_clubs():
    doc = regular(tuple("abc"))
    for e in doc["encounters"]:
        winning_club = {frozenset(("a", "b")): "a", frozenset(("b", "c")): "b", frozenset(("a", "c")): "c"}[frozenset((e["club_a"], e["club_b"]))]
        for p in e["pairings"]:
            complete_pair(p, "a" if e["club_a"] == winning_club else "b")
    return doc


def test_unresolved_multiway_tie_cannot_advance_alphabetical_top_two():
    doc = tied_three_clubs()
    standings = league_standings([doc])
    assert {r["position"] for r in standings["divisions"]["3.5"]} == {1}
    assert standings["qualification"]["3.5"]["qualifiers"] == []
    assert set(standings["qualification"]["3.5"]["playoff_required"]) == {"a", "b", "c"}


def test_first_second_statistical_tie_needs_no_qualification_playoff():
    doc = regular()
    complete_pair(doc["encounters"][0]["pairings"][0], "a")
    complete_pair(doc["encounters"][0]["pairings"][1], "b")
    qualifying = qualifying_clubs([doc])["3.5"]
    assert qualifying["status"] == "ready" and set(qualifying["qualifiers"]) == {"a", "b"}


def test_forfeited_entire_meet_does_not_satisfy_club_appearance_minimum():
    doc = regular()
    for p in doc["encounters"][0]["pairings"]:
        for game in p["games"]:
            game.update(status="forfeit", winner="a")
    qual = qualifying_clubs([doc])["3.5"]
    assert qual["eligible"] == []
    assert qual["status"] == "insufficient_entries"


def test_retired_game_counts_actual_regular_season_appearance():
    doc = regular()
    doc["weather"] = "finalized_partial"
    for p in doc["encounters"][0]["pairings"]:
        p["games"][0].update(status="retired", a=1, b=0, winner="b")
        for game in p["games"][1:]:
            game["status"] = "unplayed"
    assert set(qualifying_clubs([doc])["3.5"]["eligible"]) == {"a", "b"}
    assert not rating_games(doc)


def test_mlp_is_four_single_games_and_not_two_three_game_pairings():
    doc = final()
    assert [p["kind"] for p in doc["encounters"][0]["pairings"]] == ["women", "men", "mixed_a", "mixed_b"]
    assert [len(p["games"]) for p in doc["encounters"][0]["pairings"]] == [1, 1, 1, 1]
    assert summarize_document(validate_document(doc, official=True))["encounters"][0]["winner"] == "a"


def test_mlp_two_two_requires_singles_and_singles_never_rated():
    doc = final(winners=("a", "b", "a", "b"))
    with pytest.raises(ValueError, match="rotating-singles"):
        validate_document(doc, official=True)
    e = doc["encounters"][0]
    e["tiebreak"] = {"status": "completed", "a": 25, "b": 23,
                     "order_a": [p["entry_id"] for p in entry("a")["roster"]],
                     "order_b": [p["entry_id"] for p in entry("b")["roster"]]}
    assert summarize_document(validate_document(doc, official=True))["encounters"][0]["winner"] == "a"
    assert len(rating_games(doc)) == 4


@pytest.mark.parametrize("division,court", [("3.0", "skinny"), ("3.5", "skinny"), ("4.0", "full"), ("4.5/Open", "full"), ("Open", "full")])
def test_singles_court_by_skill_level(division, court):
    assert singles_court(division) == court


def test_both_teams_rotate_after_every_four_rallies_and_cycle():
    assert [singles_rotation(rallies) for rallies in (0, 3, 4, 7, 8, 11, 12, 15, 16, 20, 40)] == [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2]


def test_same_mixed_pair_cannot_play_both_mlp_mixed_games():
    doc = final()
    e = doc["encounters"][0]
    e["pairings"][3]["players_a"] = e["pairings"][2]["players_a"]
    with pytest.raises(ValueError, match="same four"):
        validate_document(doc, official=True)


def test_no_singles_tiebreak_when_doubles_already_decided_matchup():
    doc = final()
    doc["encounters"][0]["tiebreak"] = {"status": "completed", "a": 21, "b": 18,
        "order_a": [p["entry_id"] for p in entry("a")["roster"]],
        "order_b": [p["entry_id"] for p in entry("b")["roster"]]}
    with pytest.raises(ValueError, match="2–2"):
        validate_document(doc, official=True)


def test_final_and_qualification_playoff_do_not_change_regular_standings():
    regular_doc = complete_doc(regular())
    playoff = final(meet="qualifier", phase="qualifier")
    championship = final()
    assert league_standings([regular_doc]) == league_standings([regular_doc, playoff, championship])
    before, after = club_cup([regular_doc, championship]), club_cup([regular_doc, championship, playoff])
    assert before == after


def test_club_cup_adds_every_category_and_six_three_final_bonuses():
    docs = [complete_doc(regular(division=d, meet=f"regular-{d}")) for d in ("3.0", "3.5", "4.0", "4.5")]
    docs.extend(final(division=d, meet=f"final-{d}") for d in ("3.0", "3.5", "4.0", "4.5"))
    cup = club_cup(docs)
    a, b = cup["standings"]
    assert (a["club_id"], a["regular_points"], a["championship_points"], a["points"]) == ("a", 12, 24, 36)
    assert (b["club_id"], b["regular_points"], b["championship_points"], b["points"]) == ("b", 0, 12, 12)
    assert cup["status"] == "complete" and cup["champions"] == ["a"]


def test_cup_joint_champions_after_all_tiebreaks():
    docs = []
    for division in ("3.5", "4.0"):
        doc = regular(division=division, meet=f"regular-{division}")
        complete_pair(doc["encounters"][0]["pairings"][0], "a")
        complete_pair(doc["encounters"][0]["pairings"][1], "b")
        docs.append(doc)
    docs.append(final("3.5", meet="final-35"))
    docs.append(final("4.0", meet="final-40", winners=("b", "b", "b", "a")))
    cup = club_cup(docs)
    assert cup["status"] == "complete"
    assert set(cup["champions"]) == {"a", "b"}
    assert {r["points"] for r in cup["standings"]} == {11}


def test_draft_or_duplicate_result_revisions_cannot_enter_official_aggregates():
    doc = complete_doc(regular())
    with pytest.raises(ValueError, match="latest approved revision"):
        league_standings([doc, deepcopy(doc)])
    with pytest.raises(ValueError, match="whole meet"):
        league_standings([regular()])


def test_same_division_cannot_receive_championship_bonus_twice():
    docs = [complete_doc(regular()), final(), final(meet="another-final")]
    with pytest.raises(ValueError, match="one official championship final"):
        club_cup(docs)


def test_club_catalog_objects_do_not_create_phantom_clubs_or_leak_fields():
    doc = complete_doc(regular())
    clubs = [{"id": "a", "name": "Club A", "private_email": "private@example.invalid"},
             {"id": "b", "name": "Club B"}, {"id": "c", "name": "Not entered"}]
    standings = league_standings([doc], clubs)["divisions"]["3.5"]
    cup = club_cup([doc], clubs)["standings"]
    for rows in (standings, cup):
        assert {r["club_id"] for r in rows} == {"a", "b", "c"}
        assert next(r for r in rows if r["club_id"] == "a")["name"] == "Club A"
        assert all("private_email" not in row for row in rows)


def four_clubs_tied_for_second():
    doc = regular(tuple("abcd"))
    for encounter in doc["encounters"]:
        clubs = {encounter["club_a"], encounter["club_b"]}
        if clubs == {"b", "c"}:
            complete_pair(encounter["pairings"][0], "a")
            complete_pair(encounter["pairings"][1], "b")
            continue
        winning_club = "a" if "a" in clubs else next(c for c in clubs if c != "d")
        for pairing in encounter["pairings"]:
            complete_pair(pairing, "a" if encounter["club_a"] == winning_club else "b")
    return doc


def test_second_third_residual_tie_requires_and_accepts_full_mlp_playoff():
    doc = four_clubs_tied_for_second()
    qualifying = qualifying_clubs([doc])["3.5"]
    assert qualifying["qualifiers"] == ["a"]
    assert qualifying["status"] == "playoff_required"
    assert set(qualifying["playoff_required"]) == {"b", "c"}
    playoff = final(meet="qualifier", a="b", b="c", phase="qualifier", winners=("b", "b", "b", "a"))
    decided = qualifying_clubs([doc, playoff])["3.5"]
    assert decided["status"] == "ready"
    assert set(decided["qualifiers"]) == {"a", "c"}
    assert not decided["playoff_required"]


def test_lower_placing_tie_does_not_require_a_qualification_playoff():
    doc = regular(tuple("abcd"))
    for encounter in doc["encounters"]:
        clubs = {encounter["club_a"], encounter["club_b"]}
        if clubs == {"c", "d"}:
            complete_pair(encounter["pairings"][0], "a")
            complete_pair(encounter["pairings"][1], "b")
            continue
        winning_club = "a" if "a" in clubs else "b"
        for pairing in encounter["pairings"]:
            complete_pair(pairing, "a" if encounter["club_a"] == winning_club else "b")
    qualifying = qualifying_clubs([doc])["3.5"]
    assert qualifying["status"] == "ready"
    assert set(qualifying["qualifiers"]) == {"a", "b"}


def test_injury_replacements_do_not_revert_to_original_players_next_game():
    doc = complete_doc(regular())
    pair = doc["encounters"][0]["pairings"][0]
    pair["games"][1].update(players_a=[pair["players_a"][0], "sub"], injury_reason="Injury replacement")
    pair["games"][2].update(injury_reason="Try to restore original player")
    with pytest.raises(ValueError, match="cannot return"):
        validate_document(doc, official=True)


def playoff(a, b, winner, meet=None, winners=None):
    if winners is None:
        winners = ("a", "a", "a", "b") if winner == a else ("b", "b", "b", "a")
    return final(meet=meet or f"qualifier-{a}-{b}", a=a, b=b, winners=winners, phase="qualifier")


def test_three_way_playoff_round_robin_resolves_top_two_only_when_complete():
    regular_doc = tied_three_clubs()
    docs = [regular_doc, playoff("a", "b", "a"), playoff("a", "c", "a")]
    incomplete = qualifying_clubs(docs)["3.5"]
    assert incomplete["qualifiers"] == [] and set(incomplete["playoff_required"]) == set("abc")
    docs.append(playoff("b", "c", "b"))
    complete = qualifying_clubs(docs)["3.5"]
    assert complete["qualifiers"] == ["a", "b"]
    assert complete["status"] == "ready"


def test_three_way_playoff_for_second_place_resolves_one_available_slot():
    regular_doc = regular(tuple("abcd"))
    cycle = {frozenset(("b", "c")): "b", frozenset(("c", "d")): "c", frozenset(("b", "d")): "d"}
    for encounter in regular_doc["encounters"]:
        clubs = {encounter["club_a"], encounter["club_b"]}
        winner = "a" if "a" in clubs else cycle[frozenset(clubs)]
        for pairing in encounter["pairings"]:
            complete_pair(pairing, "a" if encounter["club_a"] == winner else "b")
    undecided = qualifying_clubs([regular_doc])["3.5"]
    assert undecided["qualifiers"] == ["a"]
    assert set(undecided["playoff_required"]) == set("bcd")
    docs = [regular_doc, playoff("b", "c", "c"), playoff("b", "d", "d"), playoff("c", "d", "d")]
    assert qualifying_clubs(docs)["3.5"]["qualifiers"] == ["a", "d"]


def test_multiway_playoff_cycle_stays_unresolved_without_statistical_advancement():
    docs = [tied_three_clubs(), playoff("a", "b", "a"), playoff("b", "c", "b"), playoff("a", "c", "c")]
    qualifying = qualifying_clubs(docs)["3.5"]
    assert qualifying["qualifiers"] == []
    assert set(qualifying["playoff_required"]) == set("abc")
    assert qualifying["status"] == "playoff_required"


def test_multiway_playoff_uses_games_won_after_matchup_wins():
    docs = [tied_three_clubs(), playoff("a", "b", "a", winners=("a", "a", "a", "a")),
            playoff("b", "c", "b"), playoff("a", "c", "c")]
    assert qualifying_clubs(docs)["3.5"]["qualifiers"] == ["a", "c"]


def test_next_mlp_playoff_resolves_only_remaining_cutoff_tie():
    mixed_win = playoff("a", "c", "c", winners=("a", "b", "a", "b"))
    mixed_win["encounters"][0]["tiebreak"] = {"status": "completed", "a": 18, "b": 21,
        "order_a": [p["entry_id"] for p in entry("a")["roster"]],
        "order_b": [p["entry_id"] for p in entry("c")["roster"]]}
    docs = [tied_three_clubs(), playoff("a", "b", "a", winners=("a", "a", "a", "a")),
            playoff("b", "c", "b"), mixed_win]
    first_round = qualifying_clubs(docs)["3.5"]
    assert first_round["qualifiers"] == ["a"]
    assert first_round["playoff_required"] == ["b", "c"]
    rematch = playoff("b", "c", "c", meet="last-qualifier")
    for p in rematch["encounters"][0]["pairings"]:
        p["games"][0]["played_at"] = "2027-01-17T17:00:00Z"
    final_round = qualifying_clubs([*docs, rematch])["3.5"]
    assert final_round["qualifiers"] == ["a", "c"]
    assert final_round["status"] == "ready"


def test_repeated_playoffs_with_ambiguous_identical_times_do_not_use_uuid_order():
    doc = four_clubs_tied_for_second()
    first = playoff("b", "c", "b", meet="first-playoff")
    second = playoff("b", "c", "c", meet="second-playoff")
    qualification = qualifying_clubs([doc, first, second])["3.5"]
    assert qualification["qualifiers"] == ["a"]
    assert qualification["status"] == "playoff_required"


def partial(club, genders=("female", "female")):
    roster = entry(club)["roster"]
    selected = []
    for gender in genders:
        player = next(player for player in roster if player["gender"] == gender and player not in selected)
        selected.append(player)
    return {**entry(club), "roster": selected}


@pytest.mark.parametrize("format,genders,present_kind,missing_kind", [
    ("gender", ("female", "female"), "women", "men"),
    ("gender", ("male", "male"), "men", "women"),
    ("mixed", ("female", "male"), "mixed_a", "mixed_b"),
])
def test_partial_team_has_real_playing_pair_and_truthful_three_game_forfeit(format, genders, present_kind, missing_kind):
    doc = generate_round_robin("partial", [partial("a", genders), entry("b")], format=format, courts=1, played_at=PLAYED_AT)
    pairing = {p["kind"]: p for p in doc["encounters"][0]["pairings"]}
    missing = pairing[missing_kind]
    assert missing["players_a"] == [] and len(missing["players_b"]) == 2
    assert missing["court"] is None
    assert all(g["status"] == "forfeit" and g["winner"] == "b" and g["a"] is g["b"] is None for g in missing["games"])
    assert pairing[present_kind]["court"] == 1
    complete_pair(pairing[present_kind], "a")
    result = summarize_document(validate_document(doc, official=True))["encounters"][0]
    assert (result["points_a"], result["points_b"]) == (1, 1)
    assert (result["games_a"], result["games_b"]) == (3, 3)
    assert len(rating_games(doc)) == 3
    assert len(result["played_a"]) == len(result["played_b"]) == 2


def test_both_missing_same_pairing_have_double_losses_and_no_weather_draw_point():
    doc = generate_round_robin("partial", [partial("a"), partial("b")], courts=1, played_at=PLAYED_AT)
    women, men = doc["encounters"][0]["pairings"]
    assert men["players_a"] == men["players_b"] == []
    assert all(g["status"] == "double_forfeit" and g["winner"] is None for g in men["games"])
    complete_pair(women, "a")
    result = summarize_document(validate_document(doc, official=True))["encounters"][0]
    assert (result["points_a"], result["points_b"]) == (3, 0)
    assert (result["games_a"], result["games_b"]) == (3, 0)
    assert (result["losses_a"], result["losses_b"]) == (3, 6)
    rows = {r["club_id"]: r for r in league_standings([doc])["divisions"]["3.5"]}
    assert rows["a"]["games_lost"] == 3 and rows["b"]["games_lost"] == 6
    assert result["point_differential"] == 12
    assert len(rating_games(doc)) == 3


def test_opposite_missing_pairings_split_without_fake_played_games():
    doc = generate_round_robin("partial", [partial("a"), partial("b", ("male", "male"))], courts=1)
    result = summarize_document(validate_document(doc, official=True))["encounters"][0]
    assert (result["points_a"], result["points_b"]) == (1, 1)
    assert (result["games_a"], result["games_b"]) == (3, 3)
    assert result["point_differential"] == 0
    assert result["played_a"] == result["played_b"] == []
    assert rating_games(doc) == []
    assert all(p["court"] is None for p in doc["encounters"][0]["pairings"])


@pytest.mark.parametrize("format,genders", [("gender", ("female", "male")), ("mixed", ("female", "female"))])
def test_partial_pair_must_match_the_selected_meet_format(format, genders):
    with pytest.raises(ValueError, match="partial team"):
        generate_round_robin("partial", [partial("a", genders), entry("b")], format=format)


def test_partial_roster_cannot_create_championship_or_qualifier_team():
    for phase in ("final", "qualifier"):
        with pytest.raises(ValueError, match="4 different"):
            generate_championship("final", "3.5", partial("a"), entry("b"), phase=phase)


def test_double_forfeit_status_cannot_have_fake_scores_or_a_winner():
    doc = generate_round_robin("partial", [partial("a"), partial("b")], courts=1)
    game = doc["encounters"][0]["pairings"][1]["games"][0]
    game["winner"] = "a"
    with pytest.raises(ValueError, match="cannot have a winner"):
        validate_document(doc)
    game.update(winner=None, a=0, b=0)
    with pytest.raises(ValueError, match="no numeric score"):
        validate_document(doc)


def test_two_double_forfeited_pairings_award_no_standings_points():
    assert matchup_points(["double_forfeit", "double_forfeit"]) == (0, 0)
