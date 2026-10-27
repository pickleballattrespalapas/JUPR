"""Integration checks for explicit unplayed results using fake stores only."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from jupr_app.domain import adaptive_play_engine as engine
from jupr_app.domain.generator_playoffs import start_generator_playoff
from jupr_app.domain.gamification.program_badges import evaluate_program_badges
from jupr_app.services.admin_play_generator_service import _saved_matches, _publish_payloads
from jupr_app.services.public_live_service import public_live_session_detail
from services.api.admin_play_generator_routes import GeneratorScoresRequest
from services.api.public_play_generator_routes import PublicGeneratorScoresRequest
from test_generator_round_controls_service import GeneratorSession
from test_program_badges import snapshot, rr, ids


def scores_for(event, number, a=11, b=7):
    row = next(r for r in event['rounds'] if r['number'] == number)
    return [{'match_id': m['id'], 'score_a': a, 'score_b': b} for m in engine._round_matches(row)]


@pytest.mark.parametrize('audience', ['admin', 'public'])
def test_all_blank_round_persists_resolves_advances_and_never_publishes(audience):
    game = GeneratorSession(audience, player_count=8)
    game.call('save', round_number=1, scores=scores_for(game.session['event'], 1, None, None))
    saved = game.reload()
    assert saved['event']['rounds'][0]['status'] == 'saved'
    assert all(m['status'] == 'not_played' and m['scoreA'] is None and m['scoreB'] is None
               for m in saved['event']['rounds'][0]['matches'])
    assert sum(row['matches'] for row in saved['standings']) == 0
    assert _saved_matches(saved['event']) == []
    game.call('advance')
    assert game.reload()['current_round_number'] == 2


@pytest.mark.parametrize('audience', ['admin', 'public'])
def test_half_blank_validation_then_same_action_key_corrected_input_saves(audience):
    game = GeneratorSession(audience)
    model = PublicGeneratorScoresRequest if audience == 'public' else GeneratorScoresRequest
    data = {'expected_version': game.session['version'], 'idempotency_key': 'corrected-score-input-0001',
            'scores': scores_for(game.session['event'], 1, 11, None)}
    if audience == 'public':
        data['edit_token'] = game.edit_token
    before = deepcopy(game.db.db)
    with pytest.raises(ValidationError):
        model.model_validate(data)
    assert game.db.db == before
    data['scores'][0]['score_b'] = 10
    parsed = model.model_validate(data)
    kwargs = {'idempotency_key': data['idempotency_key']} if audience == 'public' else {}
    game.call('save', round_number=1, scores=[s.model_dump() for s in parsed.scores], **kwargs)
    assert game.reload()['event']['rounds'][0]['matches'][0]['scoreB'] == 10


@pytest.mark.parametrize('audience', ['admin', 'public'])
def test_completed_session_can_clear_an_old_saved_round_without_reopening_later_games(audience):
    game = GeneratorSession(audience)
    game.score(1)
    game.call('advance')
    game.score(2)
    game.call('complete')
    before = deepcopy(game.reload())
    game.call('skip', round_number=1, reason='This round was recorded in error')
    after = game.reload()
    assert after['status'] == after['event']['status'] == 'completed'
    assert after['event']['rounds'][0]['status'] == 'skipped'
    assert after['event']['rounds'][1:] == before['event']['rounds'][1:]
    assert {number for number, _ in _saved_matches(after['event'])} == {2}


@pytest.mark.parametrize('audience', ['admin', 'public'])
@pytest.mark.parametrize('lock', ['pending', 'approved', 'rejected', 'published_ids', 'published_at'])
def test_clear_saved_round_is_locked_after_submission_or_publication(audience, lock):
    game = GeneratorSession(audience)
    game.score(1)
    if lock.startswith('published_'):
        game.stored['state']['official_publish'] = (
            {'published_match_ids': ['r1-c1']} if lock == 'published_ids'
            else {'published_at': '2026-10-23T00:00:00Z'})
    else:
        game.stored['state']['generator_submission'] = {'status': lock}
    before = deepcopy(game.stored)
    with pytest.raises(ValueError, match='[Pp]ublished|[Ss]ubmitted|locked'):
        game.call('skip', round_number=1, reason='Recorded in error')
    assert game.stored == before


def test_multiround_placements_finish_once_unused_rounds_are_explicitly_resolved():
    event = engine.start_generator_event(engine.create_generator_preview(
        generator_kind='round_robin', play_format='doubles', title='Four placement games',
        participant_names=[f'Player {i}' for i in range(16)], total_rounds=1, court_count=4))
    event = engine.save_generator_round(event, round_number=1, scores=scores_for(event, 1))
    event['courtCount'] = 1
    event = start_generator_playoff(event, playoff_format='groups_of_four')
    assert len([r for r in event['rounds'] if r.get('stage') == 'playoff']) == 4
    event = engine.save_generator_round(event, round_number=2, scores=scores_for(event, 2, 11, 10))
    with pytest.raises(ValueError, match='every playoff game'):
        engine.complete_generator_event(event)
    for number in (3, 4, 5):
        event = engine.advance_generator_event(event)
        event = engine.save_generator_round(event, round_number=number, scores=scores_for(event, number, None, None))
    completed = engine.complete_generator_event(event)
    assert completed['status'] == 'completed'
    assert len(_saved_matches(completed)) == 5
    assert all(r['status'] == 'saved' for r in completed['rounds'])


@pytest.mark.parametrize('stale_scores', [False, True])
def test_unplayed_match_does_not_block_or_change_valid_round_robin_badges(stale_scores):
    played = [
        {'id': 'a', 'teamA': ['1', '2'], 'teamB': ['3', '4'], 'scoreA': 11, 'scoreB': 0},
        {'id': 'b', 'teamA': ['1', '3'], 'teamB': ['2', '4'], 'scoreA': 11, 'scoreB': 1},
    ]
    data = snapshot()
    data['live_sessions'] = [rr(rounds=deepcopy(played))]
    before = evaluate_program_badges(data)
    assert 'round_robin_wins_1' in ids(before)
    data['live_sessions'][0]['state']['event']['rounds'][0]['matches'].append({
        'id': 'unplayed', 'teamA': ['2', '4'], 'teamB': ['1', '3'], 'status': 'not_played',
        'scoreA': 99 if stale_scores else None, 'scoreB': 0 if stale_scores else None})
    after = evaluate_program_badges(data)
    assert ids(after) == ids(before)
    assert not after['review']


def test_not_played_status_prevents_publication_even_with_stale_numeric_fields():
    event = engine.start_generator_event(engine.create_generator_preview(
        generator_kind='round_robin', play_format='doubles', title='No fictitious rating',
        participant_names=[f'Player {i}' for i in range(8)], player_ids=list(range(1, 9)), total_rounds=1, court_count=2))
    event = engine.save_generator_round(event, round_number=1, scores=scores_for(event, 1))
    skipped_id = event['rounds'][0]['matches'][0]['id']
    event['rounds'][0]['matches'][0]['status'] = 'not_played'
    published = _publish_payloads(event, session_key='fake', match_date='2026-10-23', operation_key='a' * 64, published_ids=set())
    assert len(published) == 1
    assert skipped_id not in {m['live_match_id'] for m in published}


def test_finished_generator_badges_ignore_unused_future_schedule():
    data = snapshot()
    session = rr(rounds=[
        {'id': 'a', 'teamA': ['1', '2'], 'teamB': ['3', '4'], 'scoreA': 11, 'scoreB': 0},
        {'id': 'b', 'teamA': ['1', '3'], 'teamB': ['2', '4'], 'scoreA': 11, 'scoreB': 1}])
    event = session['state']['event']
    event['generatorKind'] = 'round_robin'
    event['rounds'][0]['status'] = 'saved'
    data['live_sessions'] = [session]
    before = evaluate_program_badges(data)
    event['rounds'].append({'number': 2, 'status': 'preview', 'matches': [
        {'id': 'future', 'teamA': ['2', '4'], 'teamB': ['1', '3'], 'scoreA': None, 'scoreB': None}]})
    after = evaluate_program_badges(data)
    assert ids(after) == ids(before) == {'round_robin_wins_1'}
    assert not after['review']


def test_generic_public_live_detail_keeps_generator_teams_results_and_sort():
    event = {'type': 'round_robin', 'generatorKind': 'round_robin', 'standingsSort': 'points',
             'status': 'completed', 'participants': [
                 {'id': str(i), 'name': f'P{i}', 'roster_order': i + 1} for i in range(4)],
             'rounds': [{'number': 1, 'status': 'saved', 'matches': [
                 {'id': 'a', 'sideA': ['0'], 'sideB': ['1'], 'scoreA': 1, 'scoreB': 0, 'status': 'scored'},
                 {'id': 'b', 'sideA': ['0'], 'sideB': ['2'], 'scoreA': 1, 'scoreB': 0, 'status': 'scored'},
                 {'id': 'c', 'sideA': ['1'], 'sideB': ['2'], 'scoreA': 20, 'scoreB': 19, 'status': 'scored'},
                 {'id': 'unplayed', 'sideA': ['2'], 'sideB': ['3'], 'scoreA': None, 'scoreB': None, 'status': 'not_played'}]}]}
    row = {'session_key': 'fake', 'status': 'completed', 'state': {'mode': 'play_generator', 'page_state': {'event': event}}}
    detail = public_live_session_detail(row)
    assert detail['standings'] == engine.generator_event_standings(event)
    assert detail['rounds'][0]['matches'][0]['team_a'] == ['P0']
    assert detail['rounds'][0]['matches'][0]['team_b'] == ['P1']
    assert detail['rounds'][0]['matches'][-1]['status'] == 'not_played'


def test_real_generator_badges_use_side_assignments_and_ignore_not_played_and_future_games():
    event = engine.start_generator_event(engine.create_generator_preview(
        generator_kind='round_robin', play_format='singles', title='Actual generator schema',
        participant_names=[f'Player {i}' for i in range(1, 5)], player_ids=list(range(1, 5)), total_rounds=3, court_count=2))
    scores = scores_for(event, 1)
    scores[-1].update(score_a=None, score_b=None)
    event = engine.save_generator_round(event, round_number=1, scores=scores)
    event = engine.complete_generator_event(event)
    data = snapshot()
    session = rr()
    session['state']['event'] = event
    data['live_sessions'] = [session]
    result = evaluate_program_badges(data)
    winner = event['rounds'][0]['matches'][0]['sideA'][0]
    player_id = next(p['player_id'] for p in event['participants'] if p['id'] == winner)
    assert 'round_robin_wins_1' in ids(result, player_id)
    assert not result['review']


@pytest.mark.parametrize('kind,mode', [('round_robin', 'scored'), ('round_robin', 'unscored'), ('ladder', 'scored')])
def test_generic_public_summary_preserves_generator_round_and_outcomes(kind, mode):
    event = engine.start_generator_event(engine.create_generator_preview(
        generator_kind=kind, play_format='doubles', scoring_mode=mode, title='Public generator',
        participant_names=[f'Player {i}' for i in range(8)], total_rounds=3, court_count=2))
    if mode == 'scored':
        event = engine.save_generator_round(event, round_number=1, scores=scores_for(event, 1, None, None))
    else:
        event['rounds'][0]['matches'][0]['status'] = 'not_played'
        event = engine.mark_generator_round_played(event, round_number=1)
    event = engine.advance_generator_event(event)
    before = deepcopy(event)
    detail = public_live_session_detail({'session_key': 'fake', 'status': 'active', 'state': {'page_state': {'event': event}}})
    assert detail['current_round'] == 2
    assert detail['generator_kind'] == kind
    assert detail['rounds'][0]['status'] == ('saved' if mode == 'scored' else 'played')
    assert detail['rounds'][0]['matches'][0]['status'] == 'not_played'
    assert detail['rounds'][0]['matches'][0]['team_a'] and detail['rounds'][0]['matches'][0]['team_b']
    assert detail['rounds'][0]['matches'][0]['is_scored'] is False
    assert detail['standings'] == engine.generator_event_standings(event)
    assert event == before


def test_legacy_public_scores_cannot_bypass_published_generator_results_lock():
    from jupr_app.services.public_live_write_service import update_public_live_scores
    game = GeneratorSession('public')
    game.score(1)
    game.stored['state']['official_publish'] = {'published_match_ids': ['r1-c1']}
    before = deepcopy(game.stored)
    with pytest.raises(ValueError, match='[Gg]enerator|[Pp]ublished|locked'):
        update_public_live_scores(game.db, club_id='club', session_key=game.session['session_key'],
                                  edit_token=game.edit_token, expected_version=game.session['version'],
                                  idempotency_key='legacy-generator-bypass-0001', requester_hash='a' * 64,
                                  scores=[{'match_id': 'r1-c1', 'score_a': 0, 'score_b': 11}])
    assert game.stored == before
