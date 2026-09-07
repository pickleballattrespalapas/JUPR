from copy import deepcopy

from jupr_app.domain.gamification.program_badges import evaluate_program_badges
from jupr_app.domain.live_beta_engine import round_robin_standings, round_robin_result_fingerprint


def snapshot():
    return {"club_id": "club", "revision": 1, "as_of": "2026-09-07T12:00:00Z",
            "players": [{"id": i, "name": f"Player {i}", "club_id": "club"} for i in range(1, 9)]}


def match(i, **patch):
    return {"id": i, "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4,
            "score_t1": 11, "score_t2": 4, "date": f"2026-01-{i % 28 + 1:02}T12:00:00Z", "league": "L", **patch}


def ids(result, pid=1):
    return {a["badge_id"] for a in result["awards"] if a["player_id"] == pid}


def test_partnership_boundaries_and_both_members():
    s = snapshot()
    for count, expected in ((9, set()), (10, {"matches_together_10", "wins_together_10"}),
                            (25, {"matches_together_10", "wins_together_10", "matches_together_25", "wins_together_25"})):
        s["matches"] = [match(i) for i in range(count)]
        result = evaluate_program_badges(s)
        assert ids(result) == expected == ids(result, 2)
        assert ids(result, 3) == {b for b in expected if b.startswith("matches")}
    assert len(result["awards"]) == 12
    assert all(a["value_json"]["partner_id"] != a["player_id"] for a in result["awards"])


def test_pair_order_identity_deletions_other_club_and_future():
    s = snapshot()
    s["matches"] = [match(i, t1_p1=2 if i % 2 else 1, t1_p2=1 if i % 2 else 2) for i in range(10)]
    assert "wins_together_10" in ids(evaluate_program_badges(s))
    s["matches"][0]["deleted_at"] = "2026-02-01"
    assert not ids(evaluate_program_badges(s))
    s["matches"][0]["deleted_at"] = None
    s["players"][1]["club_id"] = "other"
    assert not ids(evaluate_program_badges(s))
    s["players"][1]["club_id"] = "club"
    s["matches"][0]["date"] = "2030-01-01"
    assert not ids(evaluate_program_badges(s))


def test_league_minimum_and_finalization():
    s = snapshot()
    s["leagues_metadata"] = [{"id": i, "league_name": f"L{i}", "status": "ended", "min_games": 99} for i in range(5)]
    s["finalizations"] = [{"source_type": "league", "source_id": str(i), "completed_at": "2026-02-01", "min_games": 2} for i in range(5)]
    s["matches"] = [match(i * 2 + j, league=f"L{i}") for i in range(5) for j in range(2)]
    r = evaluate_program_badges(s)
    assert "leagues_completed_5" in ids(r)
    assert next(a for a in r["awards"] if a["badge_id"] == "leagues_completed_5")["earned_at"].startswith("2026-02-01")
    s["finalizations"][0]["min_games"] = 3
    assert "leagues_completed_5" not in ids(evaluate_program_badges(s))
    s["finalizations"][0]["min_games"] = None
    assert evaluate_program_badges(s)["review"]


def rr(uid="rr", rounds=None):
    event = {"sourceEventUid": uid, "name": "Round robin", "type": "round_robin",
             "participants": [{"id": str(i), "name": f"Player {i}", "player_id": i} for i in range(1, 9)],
             "rounds": [{"matches": rounds or [{"id": "m1", "teamA": ["1", "2"], "teamB": ["3", "4"], "scoreA": 11, "scoreB": 3}]}]}
    return {"id": uid, "source": "admin_web", "status": "completed", "completed_at": "2026-01-01", "state": {"event": event}}


def test_numerical_tie_never_awards_alphabetical_first():
    s = snapshot()
    s["live_sessions"] = [rr()]
    result = evaluate_program_badges(s)
    assert "round_robin_wins_1" not in ids(result)
    assert len(result["pending_ties"]) == 1
    tie = result["pending_ties"][0]
    assert {p["player_id"] for p in tie["leaders"]} == {1, 2}
    s["decisions"] = [{**tie, "winner_player_id": 2}]
    assert "round_robin_wins_1" in ids(evaluate_program_badges(s), 2)
    s["live_sessions"][0]["state"]["event"]["rounds"][0]["matches"][0]["scoreB"] = 4
    result = evaluate_program_badges(s)
    assert "round_robin_wins_1" not in ids(result, 2)
    assert result["pending_ties"]


def test_admin_choice_display_expires_on_correction():
    event = rr()["state"]["event"]
    event["round_robin_winner"] = {"participant_id": "2", "display_fingerprint": round_robin_result_fingerprint(event)}
    ranks = round_robin_standings(event)
    assert ranks[0]["participantId"] == "2"
    assert ranks[0]["winnerDecidedByAdmin"]
    event["rounds"][0]["matches"][0]["scoreA"] = 12
    assert sum(r["winnerNeedsAdmin"] for r in round_robin_standings(event)) == 2


def test_rr_uses_wins_differential_and_points_only():
    from unittest.mock import patch
    rows = [{"participantId": "1", "name": "Z", "wins": 3, "differential": 10, "pointsFor": 40, "losses": 3},
            {"participantId": "2", "name": "A", "wins": 3, "differential": 10, "pointsFor": 40, "losses": 0},
            {"participantId": "3", "name": "B", "wins": 3, "differential": 10, "pointsFor": 39, "losses": 0}]
    with patch("jupr_app.domain.live_beta_engine.compute_standings", return_value=rows):
        standings = round_robin_standings({})
    assert [r["rank"] for r in standings] == [1, 1, 3]


def test_five_partners_one_completed_session_and_not_separate_events():
    s = snapshot()
    matches = [{"id": str(i), "teamA": ["1", str(i)], "teamB": ["7", "8"], "scoreA": 11, "scoreB": 3} for i in range(2, 7)]
    s["live_sessions"] = [rr(rounds=matches)]
    result = evaluate_program_badges(s)
    assert {"five_winning_partners", "round_robin_wins_1"} <= ids(result)
    s["live_sessions"] = [rr(str(i), [m]) for i, m in enumerate(matches)]
    assert "five_winning_partners" not in ids(evaluate_program_badges(s))
    s["live_sessions"] = [rr(rounds=matches)]
    s["live_sessions"][0]["status"] = "active"
    assert not evaluate_program_badges(s)["awards"]


def test_round_robin_completion_milestones_no_first_and_wins_50():
    s = snapshot()
    matches = [{"id": "a", "teamA": ["1", "2"], "teamB": ["3", "4"], "scoreA": 11, "scoreB": 0},
               {"id": "b", "teamA": ["1", "3"], "teamB": ["2", "4"], "scoreA": 11, "scoreB": 1}]
    s["live_sessions"] = [rr(str(i), matches) for i in range(50)]
    r = evaluate_program_badges(s)
    assert {b for b in ids(r) if b.startswith("round_robin")} == {f"round_robin_wins_{n}" for n in (1, 5, 10, 25, 50)} | {f"round_robins_completed_{n}" for n in (5, 10, 25)}
    assert not any("completed_1" == a["badge_id"] for a in r["awards"])
    assert evaluate_program_badges(deepcopy(s)) == r


def tournament_data(s, count=1):
    s["tournaments"], s["tournament_games"], s["tournament_teams"], s["tournament_event_draws"], s["tournament_podium"], s["finalizations"] = [], [], [], [], [], []
    for tid in range(count):
        s["tournaments"].append({"id": str(tid), "name": "Tournament", "status": "COMPLETED"})
        s["finalizations"].append({"source_type": "tournament", "source_id": str(tid), "completed_at": "2026-03-01"})
        for event in range(3):
            k = f"{tid}-{event}"
            s["tournament_event_draws"].append({"id": k, "event_option_id": str(event), "name": f"Event {event}"})
            s["tournament_teams"].extend([{"id": k+'a', "tournament_id": str(tid), "player1_id": 1, "player2_id": 2, "event_option_id": str(event)},
                                           {"id": k+'b', "tournament_id": str(tid), "player1_id": 3, "player2_id": 4, "event_option_id": str(event)}])
            s["tournament_games"].append({"id": k, "tournament_id": str(tid), "draw_id": k, "team_a_id": k+'a', "team_b_id": k+'b', "winner_team_id": k+'a',
                                           "score_a": 2, "score_b": 1, "finalized_at": "2026-02-01", "result_type": "PLAYED"})
            s["tournament_podium"].append({"id": k, "tournament_id": str(tid), "draw_id": k, "team_id": k+'a', "placement": event+1})


def test_triple_crown_three_distinct_events_and_any_medal():
    s = snapshot()
    tournament_data(s)
    assert ids(evaluate_program_badges(s)) == {"triple_crown"}
    s["tournament_podium"].pop()
    s["tournament_podium"].append(deepcopy(s["tournament_podium"][0]))
    assert not ids(evaluate_program_badges(s))
    tournament_data(s, 5)
    result = evaluate_program_badges(s)
    assert "tournaments_completed_5" in ids(result)
    assert len([a for a in result["awards"] if a["badge_id"] == "triple_crown" and a["player_id"] == 1]) == 5


def test_best_of_three_rating_children_are_one_match_not_three():
    s = snapshot()
    tournament_data(s, 3)  # nine matches, below the first partnership threshold
    s["matches"] = []
    parents = list(s["tournament_games"])
    for g in parents:
        for n in (1, 2, 3):
            gid = g["id"]+f'-child{n}'
            s["tournament_games"].append({**g, "id": gid, "stage": "SERIES_GAME", "series_parent_game_id": g["id"]})
            s["matches"].append(match(len(s["matches"]), tournament_game_id=gid))
    assert "matches_together_10" not in ids(evaluate_program_badges(s))
    s["matches"].append(match(100))
    assert "matches_together_10" in ids(evaluate_program_badges(s))
    s["tournament_games"][0]["result_type"] = "WALKOVER"
    assert "matches_together_10" not in ids(evaluate_program_badges(s))


def test_missing_historical_completion_date_is_held():
    s = snapshot()
    tournament_data(s, 5)
    s["finalizations"] = []
    r = evaluate_program_badges(s)
    assert not any(a["badge_id"] in {"triple_crown", "tournaments_completed_5"} for a in r["awards"])
    assert len(r["review"]) == 5


def test_substitute_receives_played_results_instead_of_slot_owner():
    s = snapshot()
    session = rr()
    event = session['state']['event']
    event['substitutions'] = [{'id': 'sub1', 'scope': 'game', 'match_id': 'm1', 'original_participant_id': '1',
                              'substitute_player_id': 5, 'substitute_name': 'Player 5', 'created_at': '2026-01-01'}]
    s['live_sessions'] = [session]
    result = evaluate_program_badges(s)
    assert {p['player_id'] for p in result['pending_ties'][0]['leaders']} == {2, 5}
    standings = round_robin_standings(event)
    assert next(r for r in standings if r['participantId'] == '1')['matches'] == 0


def test_saved_social_session_counts_once_and_waits_for_complete_persisted_scores():
    s = snapshot()
    sessions = [rr(str(i)) for i in range(5)]
    s['live_sessions'] = sessions
    s['live_events'] = [{'id': r['id'], 'source_event_uid': r['id'], 'status': 'saved', 'event_date': '2026-01-01',
                         'raw_event_json': r['state']['event']} for r in sessions]
    s['live_event_participants'] = [{'event_id': r['id'], 'participant_key': str(p), 'linked_player_id': p} for r in sessions for p in range(1, 9)]
    s['live_event_matches'] = [{'event_id': r['id'], 'match_key': 'm1', 'score_t1': 11, 'score_t2': 3} for r in sessions]
    result = evaluate_program_badges(s)
    assert 'round_robins_completed_5' in ids(result)
    assert 'round_robins_completed_10' not in ids(result)
    s['live_event_matches'].pop()
    assert 'round_robins_completed_5' not in ids(evaluate_program_badges(s))
    s['live_events'][0]['status'] = 'pending'
    assert not any(a['badge_id'].startswith('round_robins_completed') for a in evaluate_program_badges(s)['awards'])


def test_unlinked_numerical_winner_does_not_give_linked_runner_up_credit():
    s = snapshot()
    session = rr(rounds=[{'id':'a','teamA':['1','2'],'teamB':['3','4'],'scoreA':11,'scoreB':1},
                         {'id':'b','teamA':['1','3'],'teamB':['2','4'],'scoreA':11,'scoreB':3}])
    session['state']['event']['participants'][0].pop('player_id')
    s['live_sessions'] = [session]
    r = evaluate_program_badges(s)
    assert not any(a['badge_id'].startswith('round_robin_wins') for a in r['awards'])
    assert 'not linked' in r['review'][0]['reason']


def test_public_quick_session_cannot_self_award_badges():
    s = snapshot()
    s['live_sessions'] = [{**rr(str(i)), 'source':'public_web'} for i in range(25)]
    assert not evaluate_program_badges(s)['awards']


def test_singles_and_mixed_generator_round_robin_wins():
    from jupr_app.domain.adaptive_play_engine import generator_event_standings
    s = snapshot()
    session = rr(rounds=[{'id':'a','teamA':['1'],'teamB':['2'],'scoreA':11,'scoreB':3}])
    session['source'] = 'play_generator'
    event = session['state']['event']
    event.update(status='completed', generatorKind='round_robin', playFormat='singles', standingsSort='points')
    event['rounds'][0]['status'] = 'saved'
    event['rounds'].append({'status':'skipped','matches':[{'id':'skip','teamA':['2'],'teamB':['1'],'scoreA':None,'scoreB':None}]})
    s['live_sessions'] = [session]
    r = evaluate_program_badges(s)
    assert 'round_robin_wins_1' in ids(r)
    assert not any('together' in a['badge_id'] for a in r['awards'])
    rows = generator_event_standings(event)
    assert rows[0]['participantId'] == '1' and rows[0]['roundRobinVictoryRanking']
