from copy import deepcopy

import pytest

from jupr_app.domain.tournaments.pools import build_draw_round_robin_games
from jupr_app.domain.tournaments import build_playoff_games, compute_podium_from_playoffs, finalize_game, resolve_playoff_dependencies
from jupr_app.services import admin_tournament_day_live_service as live
from jupr_app.services.admin_tournament_podium_review_service import _validate_reviewable_podium
from tests.test_admin_tournament_day_live_service import FakeSupabase, snapshot_tables


def pool_tables(sizes=(3, 3), advances=(2, 2)):
    tables = snapshot_tables()
    tables['tournament_event_options'] = [tables['tournament_event_options'][0]]
    tables['tournament_event_options'][0]['scoring_default'] = 'GAME_TO_11'
    tables['tournament_event_draws'] = [tables['tournament_event_draws'][0]]
    teams = [dict(id=f'team-{n}', tournament_id='tour-1', draw_id='draw-a', registration_day_id='day-1', event_option_id='event-a', team_number=n, player1_id=2*n-1, player2_id=2*n, competition_status='ACTIVE', updated_at=f'team-{n}-v1') for n in range(1, sum(sizes)+1)]
    tables['tournament_teams'] = teams
    tables['players'] = [dict(id=n, club_id='club-1', name=f'Player {n}', active=True) for n in range(1,2*len(teams)+1)]
    tables['tournament_registrations'] = [dict(id=f'registration-{n}', tournament_id='tour-1', player_id=n, status='CONFIRMED', payment_status='paid') for n in range(1,2*len(teams)+1)]
    pools, start = [], 0
    for i, size in enumerate(sizes):
        pools.append(dict(id=chr(65+i), name=f'Pool {chr(65+i)}', team_ids=[t['id'] for t in teams[start:start+size]], advance_count=advances[i]))
        start += size
    config = dict(version=1, pools=pools)
    tables['tournament_event_draws'][0]['pool_config'] = config
    games = []
    for n, game in enumerate(build_draw_round_robin_games(tournament_id='tour-1', teams=teams, pool_config=config), 1):
        a, b = game['team_a_id'], game['team_b_id']
        a_wins = int(a.split('-')[-1]) < int(b.split('-')[-1])
        games.append(dict(game, id=f'rr-{n}', draw_id='draw-a', registration_day_id='day-1', event_option_id='event-a', score_a=11 if a_wins else 5, score_b=5 if a_wins else 11, winner_team_id=a if a_wins else b, loser_team_id=b if a_wins else a, finalized_at='2026-08-17T09:00:00Z', updated_at=f'rr-{n}-v1'))
    tables['tournament_games'] = games
    tables['tournament_day_live_runs'] = [dict(id='run-1', club_id='club-1', tournament_id='tour-1', registration_day_id='day-1', state='ACTIVE', version=1, queue_version=len(games))]
    tables['tournament_day_live_draws'] = [dict(id='day-draw-a', run_id='run-1', tournament_id='tour-1', registration_day_id='day-1', draw_id='draw-a', state='ACTIVE', source_draw_updated_at=tables['tournament_event_draws'][0]['updated_at'], version=1)]
    tables['tournament_day_live_queue'] = [dict(id=f'q-{n}', run_id='run-1', draw_id='draw-a', game_id=game['id'], team_a_id=game['team_a_id'], team_b_id=game['team_b_id'], state='COMPLETED', priority=n, version=1, released_at='2026-08-17T09:01:00Z', completed_at='2026-08-17T09:01:00Z') for n,game in enumerate(games,1)]
    return tables


def snapshot(tables):
    return live.build_admin_tournament_day_live_snapshot(FakeSupabase(tables), club_id='club-1', tournament_id='tour-1', registration_day_id='day-1')


def configuration(draw, ids=None):
    review = draw['playoff_review']
    return dict(template_code=review['default_template_code'], seed_team_ids=ids if ids is not None else review['default_seed_team_ids'], round_scoring=review['default_round_scoring'])


@pytest.mark.parametrize('sizes,total', [((3,3),6), ((4,4),12), ((4,3),9)])
def test_pool_snapshot_keeps_standings_separate_and_recommends_crossover(sizes,total):
    result = snapshot(pool_tables(sizes))
    draw = result['draws'][0]
    second = sizes[0]+1
    expected = ['team-1',f'team-{second}','team-2',f'team-{second+1}']
    assert draw['round_robin_complete'] is True
    assert draw['readiness']['generate_playoffs']['ready'] is True
    assert draw['readiness']['generate_playoffs']['allowed_advance_counts'] == list(range(2,sum(sizes)+1))
    assert draw['playoff_review']['default_seed_team_ids'] == expected
    assert draw['playoff_review']['recommended_seed_team_ids'] == expected
    assert [len(group['standings']) for group in draw['pool_standings']] == list(sizes)
    assert all(game['pool_id'] in {'A','B'} for game in result['games'])
    assert len(result['games']) == total
    games = live._playoff_rows(result, draw, 'tour-1', 4, configuration(draw))
    semifinals = [g for g in games if g['playoff_round'] == 'SF']
    assert {(g['team_a_id'],g['team_b_id']) for g in semifinals} == {('team-1',f'team-{second+1}'), (f'team-{second}','team-2')}


def test_pool_qualifier_override_requires_reason_and_stays_exact_draw():
    result = snapshot(pool_tables())
    draw = result['draws'][0]
    edited = configuration(draw,['team-1','team-4','team-3','team-5'])
    with pytest.raises(ValueError,match='Explain why'):
        live._validate_playoff_configuration(draw,4,edited)
    edited['override_reason'] = 'Director approved alternate after player withdrawal.'
    reviewed = live._validate_playoff_configuration(draw,4,edited)
    assert reviewed['seed_team_ids'] == edited['seed_team_ids']
    envelope = live._operation_payload_base('generate_playoffs',{},dict(draw_id='draw-a',advance_count=4,playoff_configuration=reviewed))
    assert envelope['payload'] == dict(draw_id='draw-a',advance_count=4)
    assert envelope['playoff_configuration']['override_reason'] == edited['override_reason']
    edited['seed_team_ids'][2] = 'foreign-team'
    with pytest.raises(ValueError,match='exact draw'):
        live._validate_playoff_configuration(draw,4,edited)
    with pytest.raises(ValueError,match='confirm'):
        live._playoff_rows(result,draw,'tour-1',4)


@pytest.mark.parametrize('corruption',['cross_pool','missing_pair','duplicate_pair','unassigned'])
def test_pool_readiness_fails_closed_for_invalid_schedule(corruption):
    tables = pool_tables()
    if corruption == 'cross_pool': tables['tournament_games'][0]['team_b_id']='team-4'
    elif corruption == 'missing_pair': tables['tournament_games'].pop()
    elif corruption == 'duplicate_pair': tables['tournament_games'].append(dict(tables['tournament_games'][0],id='duplicate'))
    else: tables['tournament_event_draws'][0]['pool_config']['pools'][0]['team_ids'].pop()
    draw = snapshot(tables)['draws'][0]
    assert draw['round_robin_complete'] is False
    assert not draw['readiness']['generate_playoffs']['ready']
    assert 'POOL_SETUP_INVALID' in {b['code'] for b in draw['readiness']['activate']['blockers']}


def test_pool_advancement_change_invalidates_both_review_fingerprints():
    tables = pool_tables()
    before = snapshot(tables)
    tables['tournament_event_draws'][0]['pool_config']['pools'][0]['advance_count']=1
    after = snapshot(tables)
    assert before['state_fingerprint'] != after['state_fingerprint']
    assert before['draws'][0]['playoff_review_fingerprint'] != after['draws'][0]['playoff_review_fingerprint']
    assert after['draws'][0]['playoff_review']['configured_advance_count'] == 3


def test_unfinished_pool_result_prevents_advancement():
    tables = pool_tables()
    tables['tournament_games'][0].update(score_a=None,score_b=None,winner_team_id=None,loser_team_id=None,finalized_at=None)
    draw = snapshot(tables)['draws'][0]
    assert not draw['round_robin_complete']
    assert 'ROUND_ROBIN_INCOMPLETE' in {b['code'] for b in draw['readiness']['generate_playoffs']['blockers']}


@pytest.mark.parametrize('advances', [(1,1),(1,2),(3,3)])
def test_pool_playoff_count_comes_from_config_and_rounds_follow_template(advances):
    result = snapshot(pool_tables(advances=advances))
    draw = result['draws'][0]
    count=sum(advances)
    assert draw['readiness']['generate_playoffs']['allowed_advance_counts'] == [2,3,4,5,6]
    conf=configuration(draw)
    games=live._playoff_rows(result,draw,'tour-1',count,conf)
    assert set(conf['round_scoring']) == {g['playoff_round'].upper() for g in games}
    assert len(games) == count-1+(1 if count>=4 else 0)


@pytest.mark.parametrize('count', [2,3,4])
def test_small_pool_playoff_podium_review_matches_real_medal_results(count):
    teams=[dict(id=f'team-{i}',team_number=i) for i in range(1,7)]
    standings=[dict(team_id=t['id'],seed=i) for i,t in enumerate(teams,1)]
    playoffs=build_playoff_games(tournament_id='tour-1',advance_count=count,standings=standings)
    for index, game in enumerate(playoffs):
        game['id'] = f'playoff-{index}'
    for game in playoffs:
        # Domain helper returns patches for winner-source materialization.
        for patch in resolve_playoff_dependencies(playoffs):
            next(g for g in playoffs if g.get('id')==patch.get('id')).update(patch)
        game.update(score_a=11,score_b=5)
        game.update(finalize_game(game))
    podium=compute_podium_from_playoffs(playoffs)
    assert live._podium_matches_playoff_results(playoffs,podium)
    draw=dict(pool_config=dict(version=1,pools=[dict(id='A',name='Pool A',team_ids=['team-1','team-2','team-3'],advance_count=1),dict(id='B',name='Pool B',team_ids=['team-4','team-5','team-6'],advance_count=count-1)]))
    rr_games = build_draw_round_robin_games(tournament_id='tour-1',teams=teams,pool_config=draw['pool_config'])
    for game in rr_games:
        game.update(score_a=11,score_b=5)
        game.update(finalize_game(game))
    _validate_reviewable_podium(draw=draw,teams=teams,games=rr_games+playoffs,podium=podium)
    with pytest.raises(ValueError):
        _validate_reviewable_podium(draw=draw,teams=teams,games=rr_games+playoffs,podium=podium[:-1])


def test_pool_podium_review_refuses_aggregate_rr_podium_without_playoffs():
    tables = pool_tables()
    with pytest.raises(ValueError, match="combined playoffs"):
        _validate_reviewable_podium(draw=tables['tournament_event_draws'][0],teams=tables['tournament_teams'],games=tables['tournament_games'],podium=[dict(placement=i,team_id=f'team-{i}') for i in range(1,4)])


def test_retired_pool_qualifier_does_not_silently_fill_from_other_pool():
    tables = pool_tables(advances=(3,1))
    tables['tournament_teams'][2]['competition_status'] = 'RETIRED'
    tables['tournament_teams'][2]['retirement_max_score'] = 11
    draw = snapshot(tables)['draws'][0]
    assert len(draw['playoff_review']['default_seed_team_ids']) == 3
    assert len(draw['playoff_review']['recommended_seed_team_ids']) == 3
    replacement = configuration(draw,draw['playoff_review']['default_seed_team_ids']+['team-5'])
    with pytest.raises(ValueError,match='Explain why'):
        live._validate_playoff_configuration(draw,4,replacement)
    replacement['override_reason']='Pool A withdrawal; director selects Pool B alternate.'
    assert live._validate_playoff_configuration(draw,4,replacement)['seed_team_ids'][-1]=='team-5'


def test_retirement_allows_smaller_reviewed_playoff_with_required_reason():
    tables = pool_tables((2,2),(2,2))
    tables['tournament_teams'][3]['competition_status'] = 'RETIRED'
    tables['tournament_teams'][3]['retirement_max_score'] = 11
    result = snapshot(tables)
    draw=result['draws'][0]
    assert draw['readiness']['generate_playoffs']['ready']
    assert draw['readiness']['generate_playoffs']['allowed_advance_counts']==[2,3]
    assert draw['playoff_review']['configured_advance_count']==4
    assert draw['playoff_review']['default_template_code']=='SINGLE_ELIMINATION_3'
    conf=configuration(draw)
    with pytest.raises(ValueError,match='Explain why'):
        live._validate_playoff_configuration(draw,3,conf)
    conf['override_reason']='Three active teams remain after withdrawal; all advance.'
    assert len(live._playoff_rows(result,draw,'tour-1',3,conf))==2


@pytest.mark.parametrize("scheduled_games", [0, 2])
def test_saved_pool_roster_stays_printable_before_complete_schedule(scheduled_games):
    tables = pool_tables()
    tables["tournament_games"] = tables["tournament_games"][:scheduled_games]
    tables["tournament_day_live_queue"] = []
    tables["tournament_day_live_draws"] = []
    tables["tournament_day_live_runs"] = []

    result = snapshot(tables)
    draw = result["draws"][0]
    groups = draw["round_robin_summary"]["pools"]
    assert [(group["pool_id"], group["pool_name"]) for group in groups] == [
        ("A", "Pool A"), ("B", "Pool B")
    ]
    assert [sorted(row["team_id"] for row in group["standings"]) for group in groups] == [
        ["team-1", "team-2", "team-3"], ["team-4", "team-5", "team-6"]
    ]
    named_standings = {
        row["team_id"]: row["team_name"]
        for row in draw["round_robin_summary"]["standings"]
    }
    assert named_standings == {
        f"team-{number}": f"Player {number * 2 - 1} / Player {number * 2}"
        for number in range(1, 7)
    }
    assert draw["pool_standings"] == groups
    assert len(result["games"]) == scheduled_games
    assert draw["round_robin_complete"] is False
    assert draw["readiness"]["activate"]["ready"] is False
    assert draw["readiness"]["generate_playoffs"]["ready"] is False
    for action in ("activate", "generate_playoffs"):
        assert "POOL_SETUP_INVALID" in {
            blocker["code"] for blocker in draw["readiness"][action]["blockers"]
        }
    assert draw["playoff_review"]["recommended_seed_team_ids"] == []
    assert draw["playoff_review"]["default_seed_team_ids"] == []
    assert not any(
        row["recommended_qualifier"]
        for row in draw["round_robin_summary"]["standings"]
    )
