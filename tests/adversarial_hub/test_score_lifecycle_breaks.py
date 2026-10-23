"""Offline score and closeout regressions from the RR Hub adversarial audit."""
from copy import deepcopy

import pytest
from pydantic import ValidationError

from jupr_app.domain import adaptive_play_engine as engine
from jupr_app.domain.generator_playoffs import generator_playoff_options, start_generator_playoff
from jupr_app.domain.round_robin_hub import apply_hub_command, save_hub_match
from jupr_app.services.admin_play_generator_service import _publish_payloads, _saved_matches
from services.api.admin_play_generator_routes import GeneratorScorePayload, GeneratorScoresRequest
from services.api.public_play_generator_routes import PublicGeneratorScorePayload, PublicGeneratorScoresRequest

NOW = '2026-10-23T00:00:00+00:00'
POINTS_TRACE = [(0, 11), (11, 10), (0, 11), (11, 10), (0, 11), (10, 11), (11, 0), (0, 11)]
DIFFERENTIAL_TRACE = [(10, 11), (11, 10), (10, 11), (0, 11)]


def event(count=8, rounds=4, sort='wins'):
    return engine.start_generator_event(engine.create_generator_preview(
        generator_kind='round_robin', play_format='doubles', title='Adversarial score lifecycle',
        participant_names=[f'P{i}' for i in range(count)], player_ids=list(range(100, 100 + count)),
        total_rounds=rounds, court_count=max(1, count // 4), standings_sort=sort))


def payload(value, number=1, a=11, b=7):
    return [{'match_id': m['id'], 'score_a': a, 'score_b': b}
            for m in value['rounds'][number - 1]['matches']]


def record_trace(sort, trace):
    value = event(rounds=len(trace) // 2, sort=sort)
    for number in range(1, len(trace) // 2 + 1):
        scores = payload(value, number)
        for index, score in enumerate(scores):
            score['score_a'], score['score_b'] = trace[(number - 1) * 2 + index]
        value = engine.save_generator_round(value, round_number=number, scores=scores)
        if number < len(trace) // 2:
            value = engine.advance_generator_event(value)
    return value


def attach(value):
    state = {'id': 'audit-hub', 'courts': [
        {'id': 'c0', 'label': 'Court 1', 'open': True, 'current': None, 'next': None}],
        'groups': [], 'players': {}}
    return apply_hub_command(state, {'a': value}, action='add_group', payload={'session_key': 'a'}, now=NOW)


def ranks(value):
    return [r['participantId'] for r in engine.generator_event_standings(value)]


@pytest.mark.parametrize('sort,trace', [('points', POINTS_TRACE), ('differential', DIFFERENTIAL_TRACE)])
@pytest.mark.parametrize('surface', ['generator', 'hub'])
def test_finish_must_preserve_selected_standings_order(sort, trace, surface):
    before = record_trace(sort, trace)
    if surface == 'generator':
        completed = engine.complete_generator_event(before)
    else:
        state, events = attach(before)
        _, events = apply_hub_command(state, events, action='finish_group', payload={'session_key': 'a'}, now=NOW)
        completed = events['a']
    assert completed['status'] == 'completed'
    assert completed['standingsSort'] == before['standingsSort']
    assert ranks(completed) == ranks(before)


@pytest.mark.parametrize('sort,trace', [('points', POINTS_TRACE), ('differential', DIFFERENTIAL_TRACE)])
def test_playoff_seeds_must_not_depend_on_finish_button_order(sort, trace):
    before = record_trace(sort, trace)
    active = start_generator_playoff(before, playoff_format='groups_of_four')['playoff']['seeds']
    completed = start_generator_playoff(engine.complete_generator_event(before),
                                        playoff_format='groups_of_four')['playoff']['seeds']
    assert active == completed


@pytest.mark.parametrize('played_count', [0, 1, 3])
def test_blank_games_must_not_prevent_saving_real_results(played_count):
    value = event(count=16)
    scores = payload(value)
    for score in scores[played_count:]:
        score.update(score_a=None, score_b=None)
    saved = engine.save_generator_round(value, round_number=1, scores=scores)
    row = saved['rounds'][0]
    assert row['status'] == 'saved'
    assert sum(m['status'] == 'scored' for m in row['matches']) == played_count
    assert all(m['status'] == 'not_played' and m['scoreA'] is None and m['scoreB'] is None
               for m in row['matches'][played_count:])
    assert sum(r['matches'] for r in engine.generator_event_standings(saved)) == played_count * 4


def test_hub_blank_score_pair_can_resolve_unplayed_game():
    value = event(count=4)
    saved = save_hub_match(value, match_id=value['rounds'][0]['matches'][0]['id'],
                           score_a=None, score_b=None, now=NOW)
    assert sum(r['matches'] for r in engine.generator_event_standings(saved)) == 0


def test_only_played_playoff_game_can_save():
    value = event(count=16)
    value = engine.save_generator_round(value, round_number=1, scores=payload(value))
    value = start_generator_playoff(value, playoff_format='groups_of_four')
    number = value['currentRoundNumber']
    scores = payload(value, number)
    for score in scores[1:]:
        score.update(score_a=None, score_b=None)
    saved = engine.save_generator_round(value, round_number=number, scores=scores)
    assert saved['rounds'][number - 1]['status'] == 'saved'
    assert len(_saved_matches(saved)) == 5


def test_saved_round_can_be_corrected_to_unplayed_without_rewriting_other_rounds():
    value = engine.save_generator_round(event(), round_number=1, scores=payload(event()))
    value = engine.advance_generator_event(value)
    later = deepcopy(value['rounds'][1:])
    changed = engine.skip_generator_round(value, round_number=1, reason='Games were not played')
    assert changed['rounds'][0]['status'] == 'skipped'
    assert changed['rounds'][1:] == later


@pytest.mark.parametrize('model', [GeneratorScoresRequest, PublicGeneratorScoresRequest])
def test_duplicate_score_ids_are_rejected(model):
    value = event()
    scores = payload(value)
    scores.append({**scores[0], 'score_a': 0, 'score_b': 11})
    extra = {'edit_token': 'token'} if model is PublicGeneratorScoresRequest else {}
    with pytest.raises(ValidationError, match='[Dd]uplicate|[Uu]nique'):
        model(expected_version=1 if extra else '2026-10-23T00:00:00Z',
              idempotency_key='audit-duplicate-0001', scores=scores, **extra)
    with pytest.raises(ValueError, match='[Dd]uplicate|[Uu]nique'):
        engine.save_generator_round(value, round_number=1, scores=scores)


def test_unknown_extra_match_id_is_rejected():
    value = event()
    scores = payload(value) + [{'match_id': 'different-session-match', 'score_a': 99, 'score_b': 0}]
    with pytest.raises(ValueError, match='[Uu]nknown|[Uu]nexpected|[Mm]atch'):
        engine.save_generator_round(value, round_number=1, scores=scores)


@pytest.mark.parametrize('model', [GeneratorScorePayload, PublicGeneratorScorePayload])
def test_api_rejects_boolean_scores(model):
    with pytest.raises(ValidationError):
        model(match_id='r1-c1', score_a=True, score_b=False)


@pytest.mark.parametrize('a,b', [(None, 7), (11, None), (-1, 7), (11, -1), (100, 7), (11, 100), (11, 11)])
def test_invalid_round_scores_reject_without_mutating_input(a, b):
    value = event()
    before = deepcopy(value)
    scores = payload(value)
    scores[-1].update(score_a=a, score_b=b)
    with pytest.raises(ValueError):
        engine.save_generator_round(value, round_number=1, scores=scores)
    assert value == before


@pytest.mark.parametrize('a,b', [(None, 7), (11, None), (-1, 7), (100, 7), (11, 11),
                                (True, False), (11.0, 9.0), (11.9, 9.1), ('11', '9')])
def test_hub_rejects_invalid_or_non_integer_scores_without_mutation(a, b):
    value = event()
    before = deepcopy(value)
    with pytest.raises(ValueError):
        save_hub_match(value, match_id=value['rounds'][0]['matches'][0]['id'],
                       score_a=a, score_b=b, now=NOW)
    assert value == before


@pytest.mark.parametrize('model', [GeneratorScorePayload, PublicGeneratorScorePayload])
@pytest.mark.parametrize('a,b', [(11.9, 9.1), (-1, 7), (100, 7), ('bad', 'score')])
def test_api_rejects_fractional_out_of_range_and_non_numeric_scores(model, a, b):
    with pytest.raises(ValidationError):
        model(match_id='r1-c1', score_a=a, score_b=b)


@pytest.mark.parametrize('a,b', [(11, 10), (1, 0), (99, 0), (0, 99), (0, 11)])
def test_valid_standard_rr_scores_have_same_result_in_hub_and_generator(a, b):
    value = event(count=4)
    generated = engine.save_generator_round(value, round_number=1, scores=payload(value, a=a, b=b))
    hub = save_hub_match(value, match_id=value['rounds'][0]['matches'][0]['id'], score_a=a, score_b=b, now=NOW)
    assert generated['rounds'][0]['matches'][0]['scoreA'] == hub['rounds'][0]['matches'][0]['scoreA'] == a
    assert generated['rounds'][0]['matches'][0]['scoreB'] == hub['rounds'][0]['matches'][0]['scoreB'] == b


def test_unknown_only_payload_cannot_score_another_match():
    value = event()
    before = deepcopy(value)
    with pytest.raises(ValueError):
        engine.save_generator_round(value, round_number=1, scores=[{'match_id': 'unknown', 'score_a': 11, 'score_b': 0}])
    assert value == before


def test_repeat_save_does_not_count_same_games_twice():
    value = event()
    first = engine.save_generator_round(value, round_number=1, scores=payload(value))
    second = engine.save_generator_round(first, round_number=1, scores=payload(first))
    assert engine.generator_event_standings(first) == engine.generator_event_standings(second)
    assert len(_saved_matches(first)) == len(_saved_matches(second)) == 2


def test_repeated_hub_score_is_rejected_without_overwriting_first_result():
    value = event(count=8)
    mid = value['rounds'][0]['matches'][0]['id']
    first = save_hub_match(value, match_id=mid, score_a=11, score_b=4, now=NOW)
    before = deepcopy(first)
    with pytest.raises(ValueError):
        save_hub_match(first, match_id=mid, score_a=4, score_b=11, now=NOW)
    assert first == before


def test_partial_hub_results_are_retained_but_wait_for_whole_round_standings():
    value = event(count=8)
    first = save_hub_match(value, match_id=value['rounds'][0]['matches'][0]['id'], score_a=11, score_b=4, now=NOW)
    assert first['rounds'][0]['matches'][0]['scoreA'] == 11
    assert first['rounds'][0]['status'] == 'active'
    assert sum(r['matches'] for r in engine.generator_event_standings(first)) == 0
    assert _saved_matches(first) == []
    second = save_hub_match(first, match_id=value['rounds'][0]['matches'][1]['id'], score_a=11, score_b=9, now=NOW)
    assert sum(r['matches'] for r in engine.generator_event_standings(second)) == 8


def test_finish_rejects_partially_scored_round():
    value = event()
    value['rounds'][0]['matches'][0].update(scoreA=11, scoreB=7, status='scored')
    before = deepcopy(value)
    with pytest.raises(ValueError):
        engine.complete_generator_event(value)
    assert value == before


def test_saved_round_correction_preserves_later_matchups():
    value = event()
    value = engine.save_generator_round(value, round_number=1, scores=payload(value))
    value = engine.advance_generator_event(value)
    later = deepcopy(value['rounds'][1:])
    changed = engine.save_generator_round(value, round_number=1, scores=payload(value, a=7, b=11))
    assert changed['rounds'][1:] == later
    assert changed['currentRoundNumber'] == 2


def test_playoff_locks_old_rr_scores_and_roster():
    value = engine.save_generator_round(event(), round_number=1, scores=payload(event()))
    value = start_generator_playoff(value, playoff_format='groups_of_four')
    before = deepcopy(value)
    with pytest.raises(ValueError):
        engine.save_generator_round(value, round_number=1, scores=payload(value))
    with pytest.raises(ValueError):
        engine.mutate_generator_roster(value, action='add', name='Arrived during playoff')
    assert value == before


def test_departed_and_zero_game_arrival_excluded_from_playoff_seeds():
    value = engine.save_generator_round(event(), round_number=1, scores=payload(event()))
    departed = engine.generator_event_standings(value)[0]['participantId']
    value = engine.mutate_generator_roster(value, action='remove', participant_id=departed)
    value = engine.mutate_generator_roster(value, action='add', name='New arrival')
    seeds = generator_playoff_options(value)['seeds']
    assert departed not in {r['participantId'] for r in seeds}
    assert 'New arrival' not in {r['name'] for r in seeds}


def test_playoff_score_preserves_regular_standings_and_is_publishable_once():
    value = engine.save_generator_round(event(), round_number=1, scores=payload(event()))
    regular = engine.generator_event_standings(value)
    value = start_generator_playoff(value, playoff_format='groups_of_four')
    number = value['currentRoundNumber']
    value = engine.save_generator_round(value, round_number=number, scores=payload(value, number))
    assert engine.generator_event_standings(value) == regular
    published = _publish_payloads(value, session_key='audit', match_date='2026-10-22', operation_key='a' * 64, published_ids=set())
    assert len(published) == 4
    assert _publish_payloads(value, session_key='audit', match_date='2026-10-22', operation_key='a' * 64,
                             published_ids={p['live_match_id'] for p in published}) == []


def test_skipped_round_never_appears_in_standings_or_publish_payloads():
    value = engine.skip_generator_round(event(), round_number=1)
    value = engine.advance_generator_event(value)
    value = engine.save_generator_round(value, round_number=2, scores=payload(value, 2))
    assert {number for number, _ in _saved_matches(value)} == {2}
    assert sum(r['matches'] for r in engine.generator_event_standings(value)) == 8


def test_finish_with_unplayed_current_round_skips_only_that_round():
    value = event()
    value = engine.save_generator_round(value, round_number=1, scores=payload(value))
    value = engine.advance_generator_event(value)
    before = deepcopy(value)
    complete = engine.complete_generator_event(value)
    assert complete['rounds'][0] == before['rounds'][0]
    assert complete['rounds'][1]['status'] == 'skipped'
    assert complete['rounds'][2:] == before['rounds'][2:]
    assert engine.complete_generator_event(complete) == complete


@pytest.mark.parametrize('a,b', [(True, False), (11.0, 7.0), (11.9, 7.2), ('11', '7')])
def test_domain_scores_reject_coercion_without_mutation(a, b):
    value = event()
    before = deepcopy(value)
    with pytest.raises(ValueError, match='whole numbers'):
        engine.save_generator_round(value, round_number=1, scores=payload(value, a=a, b=b))
    assert value == before


def test_domain_rejects_duplicate_score_ids_without_api_validation():
    value = event()
    scores = payload(value)
    with pytest.raises(ValueError, match='Duplicate'):
        engine.save_generator_round(value, round_number=1, scores=scores + [scores[0]])


@pytest.mark.parametrize('played_count', [0, 1, 3])
def test_not_played_matches_do_not_count_in_history_or_publication_and_allow_next_round(played_count):
    value = event(count=16)
    scores = payload(value)
    for score in scores[played_count:]:
        score.update(score_a=None, score_b=None)
    saved = engine.save_generator_round(value, round_number=1, scores=scores)
    assert sum(engine.history_before_round(saved, 2)['games'].values()) == 4 * played_count
    assert len(_saved_matches(saved)) == played_count
    published = _publish_payloads(saved, session_key='audit', match_date='2026-10-22', operation_key='a' * 64, published_ids=set())
    assert len(published) == played_count
    assert all(row['score_t1'] == 11 and row['score_t2'] == 7 for row in published)
    advanced = engine.advance_generator_event(saved)
    assert advanced['currentRoundNumber'] == 2
    assert advanced['rounds'][0] == saved['rounds'][0]


def test_all_not_played_round_does_not_assign_fictional_byes():
    value = event(count=9)
    saved = engine.save_generator_round(value, round_number=1, scores=payload(value, a=None, b=None))
    history = engine.history_before_round(saved, 2)
    assert not any(history['games'].values())
    assert not any(history['byes'].values())


@pytest.mark.parametrize('completed', [False, True])
def test_clear_saved_round_removes_only_its_results_and_requires_reason(completed):
    value = event()
    value = engine.save_generator_round(value, round_number=1, scores=payload(value))
    value = engine.advance_generator_event(value)
    value = engine.save_generator_round(value, round_number=2, scores=payload(value, 2))
    if completed:
        value = engine.complete_generator_event(value)
    before = deepcopy(value)
    with pytest.raises(ValueError, match='reason'):
        engine.skip_generator_round(value, round_number=1)
    assert value == before
    cleared = engine.skip_generator_round(value, round_number=1, reason='Round did not happen')
    row = cleared['rounds'][0]
    assert row['status'] == 'skipped' and row['savedAt'] is None
    assert row['skipReason'] == 'Round did not happen' and row['skippedAt']
    assert all(m['status'] == 'not_played' and m['scoreA'] is None and m['scoreB'] is None for m in row['matches'])
    assert cleared['rounds'][1:] == before['rounds'][1:]
    assert cleared['currentRoundNumber'] == 2 and cleared['status'] == before['status']
    assert sum(r['matches'] for r in engine.generator_event_standings(cleared)) == 8
    assert {number for number, _ in _saved_matches(cleared)} == {2}
    assert value == before
    reopened = engine.reopen_generator_round(cleared, round_number=1)
    assert all(m['status'] == 'scheduled' for m in reopened['rounds'][0]['matches'])
    rescored = engine.save_generator_round(reopened, round_number=1, scores=payload(reopened))
    assert sum(r['matches'] for r in engine.generator_event_standings(rescored)) == 16


def test_saved_round_cannot_be_cleared_after_playoff_seeds_freeze():
    value = engine.save_generator_round(event(), round_number=1, scores=payload(event()))
    value = start_generator_playoff(value, playoff_format='top_four')
    before = deepcopy(value)
    with pytest.raises(ValueError, match='fixed once the playoff'):
        engine.skip_generator_round(value, round_number=1, reason='Incorrect scores')
    assert value == before


@pytest.mark.parametrize('count', [4, 8, 16])
def test_top_four_creates_one_championship_and_completes_with_one_real_result(count):
    value = event(count=count)
    value = engine.save_generator_round(value, round_number=1, scores=payload(value))
    standings = engine.generator_event_standings(value)
    playoff = start_generator_playoff(value, playoff_format='top_four')
    rounds = [r for r in playoff['rounds'] if r.get('stage') == 'playoff']
    assert len(rounds) == 1 and len(rounds[0]['matches']) == 1
    ids = [row['participantId'] for row in playoff['playoff']['seeds']]
    match = rounds[0]['matches'][0]
    assert match['sideA'] == [ids[0], ids[3]] and match['sideB'] == [ids[1], ids[2]]
    assert playoff['playoff']['sitOutParticipantIds'] == ids[4:]
    assert rounds[0]['label'] == 'Top four championship'
    number = playoff['currentRoundNumber']
    scored = engine.save_generator_round(playoff, round_number=number, scores=payload(playoff, number, 11, 10))
    completed = engine.complete_generator_event(scored)
    assert completed['status'] == 'completed'
    assert engine.generator_event_standings(completed) == standings
    assert len(_saved_matches(completed)) == count // 4 + 1


def test_top_eight_requires_semifinal_winners_without_discarding_other_scores():
    value = engine.save_generator_round(event(), round_number=1, scores=payload(event()))
    value = start_generator_playoff(value, playoff_format='top_eight')
    before = deepcopy(value)
    number = value['currentRoundNumber']
    scores = payload(value, number)
    scores[-1].update(score_a=None, score_b=None)
    with pytest.raises(ValueError, match='semifinal'):
        engine.save_generator_round(value, round_number=number, scores=scores)
    assert value == before


def test_unscored_round_played_preserves_not_played_matches():
    value = engine.start_generator_event(engine.create_generator_preview(
        generator_kind='round_robin', play_format='doubles', scoring_mode='unscored',
        title='Mixed terminal outcomes', participant_names=[f'P{i}' for i in range(8)], total_rounds=2, court_count=2))
    value['rounds'][0]['matches'][0]['status'] = 'not_played'
    played = engine.mark_generator_round_played(value, round_number=1)
    assert [m['status'] for m in played['rounds'][0]['matches']] == ['not_played', 'played']
    assert sum(engine.history_before_round(played, 2)['games'].values()) == 4


def test_explicit_not_played_ignores_stale_numbers_in_standings_and_badge_results():
    from jupr_app.domain.live_beta_engine import round_robin_result_matches
    value = engine.save_generator_round(event(), round_number=1, scores=payload(event()))
    value['rounds'][0]['matches'][0]['status'] = 'not_played'
    assert sum(r['matches'] for r in engine.generator_event_standings(value)) == 4
    matches = round_robin_result_matches(value)
    assert value['rounds'][0]['matches'][0]['id'] not in {m['id'] for m in matches}
    assert sum(m.get('scoreA') is not None for m in matches) == 1
