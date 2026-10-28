"""Stable full-group plans, independently of court speed and arrival cohorts."""
from collections import Counter
from copy import deepcopy

import pytest

from jupr_app.domain.adaptive_play_engine import active_participant_ids, mutate_generator_roster
from jupr_app.domain.round_robin_hub import (
    _prepare_rolling_rounds, _settle_hub_rounds, apply_hub_command, build_hub_snapshot,
)
from test_round_robin_hub import START, command, event, score, setup


def people(match):
    return list(match['sideA']) + list(match['sideB'])


def row(value, number):
    return next(r for r in value['rounds'] if r['number'] == number)


def legacy_arrival_event(count):
    """Redacted incident shape: groups of four repeated across twenty previews."""
    value = event(count=4)
    value['participants'] = [dict(id=f'p-{i}', name=f'Guest {i}', seed=i, roster_order=i,
        active_from_round=1, inactive_from_round=None, inactive_rounds=[]) for i in range(1, count + 1)]
    ids = [p['id'] for p in value['participants']]
    value['rounds'] = []
    for number in range(1, 21):
        matches = []
        for offset in range(0, count // 4 * 4, 4):
            a, b = ids[offset:offset + 2], ids[offset + 2:offset + 4]
            matches.append(dict(id=f'legacy-r{number}-m{offset // 4 + 1}', round=number,
                court=offset // 4 + 1, playFormat='doubles', sideA=a, sideB=b,
                teamA=a[:], teamB=b[:], scoreA=None, scoreB=None, status='scheduled'))
        value['rounds'].append(dict(number=number, status='active' if number == 1 else 'preview',
            matches=matches, byeParticipantIds=ids[count // 4 * 4:], warnings=[],
            formatCounts={'doubles': len(matches), 'singles': 0}))
    value.update(hubRollingEnabled=True, hubPreparedThroughRound=20, totalRounds=20,
        perfectRotation=False, adaptiveFromRound=1,
        rosterByRound={str(n): ids[:] for n in range(1, 21)})
    return value


def assert_round_membership(value, number):
    planned = row(value, number)
    playing = [pid for m in planned['matches'] for pid in people(m)]
    byes = planned['byeParticipantIds']
    assert len(playing) == len(set(playing))
    assert not set(playing) & set(byes)
    assert set(playing + byes) == set(active_participant_ids(value, number))


@pytest.mark.parametrize('count', [15, 29])
def test_incident_arrival_cohorts_are_mixed_once_then_reused(count):
    value = legacy_arrival_event(count)
    original = deepcopy(value)
    state, events = setup({'a': value}, courts=9)
    result = events['a']
    assert value == original
    assert [r['number'] for r in result['rounds']] == [1, 2]
    assert row(result, 1)['matches'] == row(original, 1)['matches']
    old_partners = {tuple(sorted(team)) for m in row(original, 1)['matches'] for team in (m['sideA'], m['sideB'])}
    old_groups = {tuple(sorted(people(m))) for m in row(original, 1)['matches']}
    future = deepcopy(row(result, 2))
    assert all(tuple(sorted(team)) not in old_partners for m in future['matches'] for team in (m['sideA'], m['sideB']))
    assert all(tuple(sorted(people(m))) not in old_groups for m in future['matches'])
    assert not set(row(result, 1)['byeParticipantIds']) & set(future['byeParticipantIds'])
    assert_round_membership(result, 2)
    state, events = command(state, events, 'auto_fill_courts')
    saved_state, saved_events = deepcopy(state), deepcopy(events)
    for court in ('c0', 'c1'):
        _, updated = score(deepcopy(saved_state), deepcopy(saved_events), court)
        assert row(updated['a'], 2) == future
        assert max(r['number'] for r in updated['a']['rounds']) == 2


@pytest.mark.parametrize('count', [15, 29])
def test_real_incremental_arrivals_get_one_complete_next_plan(count):
    state, events = setup({'a': event(count=4)}, courts=9)
    state, events = command(state, events, 'auto_fill_courts')
    first = deepcopy(row(events['a'], 1)['matches'][0])
    for index in range(4, count):
        state, events = command(state, events, 'add_player', session_key='a', name=f'Arrival {index}')
        assert len(events['a']['rounds']) == 2
        assert first == row(events['a'], 1)['matches'][0]
        assert_round_membership(events['a'], 2)
    current_groups = {tuple(sorted(people(m))) for m in row(events['a'], 1)['matches']}
    assert all(tuple(sorted(people(m))) not in current_groups for m in row(events['a'], 2)['matches'])
    assert not set(row(events['a'], 1)['byeParticipantIds']) & set(row(events['a'], 2)['byeParticipantIds'])


@pytest.mark.parametrize('count', [15, 29])
def test_eight_planned_rotations_balance_byes_without_consecutive_byes(count):
    value = _prepare_rolling_rounds(legacy_arrival_event(count))
    byes = Counter()
    previous = set()
    for number in range(1, 9):
        assert_round_membership(value, number)
        current = row(value, number)
        assert not previous & set(current['byeParticipantIds'])
        previous = set(current['byeParticipantIds'])
        byes.update(previous)
        for match in current['matches']:
            match.update(scoreA=11, scoreB=5, status='scored')
        value = _prepare_rolling_rounds(_settle_hub_rounds(value))
        assert value['currentRoundNumber'] == number + 1
        assert max(r['number'] for r in value['rounds']) == number + 2
    counts = [byes[p['id']] for p in value['participants']]
    assert max(counts) - min(counts) <= 1


def test_manual_locks_and_actual_future_commitments_survive_legacy_upgrade():
    value = legacy_arrival_event(16)
    locked = row(value, 2)['matches'][0]
    locked['hubPlanLocked'] = True
    announced = row(value, 7)['matches'][0]
    announced['hubPlanLocked'] = True
    started = row(value, 8)['matches'][1]
    started['hubStartedAt'] = START
    originals = deepcopy([locked, announced, started])
    result = _prepare_rolling_rounds(value)
    assert [r['number'] for r in result['rounds']] == [1, 2, 7, 8]
    assert [row(result, 2)['matches'][0], row(result, 7)['matches'][0], row(result, 8)['matches'][0]] == originals
    assert result['hubPreparedThroughRound'] == 8


def test_departure_repairs_affected_manual_lock_with_fresh_ids_but_keeps_other_lock():
    state, events = setup({'a': legacy_arrival_event(16)})
    a, b = row(events['a'], 2)['matches'][:2]
    a['hubPlanLocked'] = b['hubPlanLocked'] = True
    original_b = deepcopy(b)
    departed = people(a)[0]
    before_id = a['id']
    state, events = command(state, events, 'remove_players', session_key='a', participant_ids=[departed])
    value = events['a']
    assert next(p for p in value['participants'] if p['id'] == departed)['inactive_from_round'] == 1
    assert original_b in row(value, 2)['matches']
    assert before_id not in {m['id'] for r in value['rounds'] for m in r['matches']}
    assert all(departed not in people(m) for r in value['rounds'] for m in r['matches'])
    assert value['hubPlanAdjustments'][-1]['unlockedMatchIds'] == [before_id]
    assert_round_membership(value, 1)
    assert_round_membership(value, 2)


def test_legacy_selected_next_match_does_not_disappear_during_first_write():
    value = legacy_arrival_event(8)
    # Use an untouched current round with all four selected next-round players on bye.
    value['rounds'][0]['matches'] = value['rounds'][0]['matches'][1:]
    value['rounds'][0]['byeParticipantIds'] = people(row(value, 2)['matches'][0])
    target = deepcopy(row(value, 2)['matches'][0])
    state = {'groups': [{'session_key': 'a'}], 'players': {}, 'courts': [
        {'id': 'c0', 'label': 'Court 1', 'open': True, 'current': None, 'next': None, 'dedicated_session_key': None}]}
    state, events = command(state, {'a': value}, 'assign_game', session_key='a', match_id=target['id'], court_id='c0')
    assert state['courts'][0]['current']['match_id'] == target['id']
    assert people(row(events['a'], 2)['matches'][0]) == people(target)


def test_choose_next_round_as_final_and_reject_cutting_off_an_announced_court():
    state, events = setup({'a': event(count=8)}, courts=2)
    details = build_hub_snapshot(state, events, now=START)['groups'][0]
    assert [o['round_number'] for o in details['finish_round_options']] == [0, 1, 2]
    state, events = command(state, events, 'finish_group', session_key='a', finish_after_round=2)
    assert events['a']['hubFinishAfterRound'] == 2
    for number in (1, 2):
        for match in deepcopy(row(events['a'], number)['matches']):
            state, events = command(state, events, 'assign_game', session_key='a', match_id=match['id'], court_id='c0')
            state, events = score(state, events, 'c0')
    assert events['a']['status'] == 'completed'
    assert max(r['number'] for r in events['a']['rounds']) == 2
    state, events = setup({'a': legacy_arrival_event(12)}, courts=2)
    state, events = command(state, events, 'assign_game', session_key='a', match_id=row(events['a'], 1)['matches'][0]['id'], court_id='c0')
    with pytest.raises(ValueError, match='prepared final round'):
        command(state, events, 'finish_group', session_key='a', finish_after_round=0)
    with pytest.raises(ValueError, match='prepared final round'):
        command(state, events, 'finish_group', session_key='a', finish_after_round=3)


@pytest.mark.parametrize('count', [15, 29])
def test_next_planned_game_can_dispatch_when_its_four_are_free_before_round_finishes(count):
    state, events = setup({'a': legacy_arrival_event(count)}, courts=9)
    state, events = command(state, events, 'auto_fill_courts')
    value = events['a']
    current_matches = row(value, 1)['matches']
    dependencies = lambda match: [m for m in current_matches if set(people(m)) & set(people(match))]
    next_match = min(row(value, 2)['matches'], key=lambda m: len(dependencies(m)))
    next_id = next_match['id']
    plan = [(m['id'], people(m)) for m in row(value, 2)['matches']]
    needed = dependencies(next_match)
    assert 1 < len(needed) < len(current_matches)
    for match in needed:
        court = next(c for c in state['courts'] if c.get('current') and c['current']['match_id'] == match['id'])
        state, events = score(state, events, court['id'])
    assert events['a']['currentRoundNumber'] == 1
    snapshot = build_hub_snapshot(state, events, now=START)
    assert next_id in {m['match_id'] for m in snapshot['eligible_queue']}
    court = next(c for c in state['courts'] if not c.get('current') and not c.get('next'))
    state, events = command(state, events, 'assign_game', session_key='a', match_id=next_id, court_id=court['id'])
    state, events = score(state, events, court['id'])
    assert events['a']['currentRoundNumber'] == 1
    assert [(m['id'], people(m)) for m in row(events['a'], 2)['matches']] == plan
    snapshot = build_hub_snapshot(state, events, now=START)
    full_plan = snapshot['groups'][0]['next_round_matches']
    assert len(full_plan) == count // 4
    assert next(m for m in full_plan if m['match_id'] == next_id)['result_status'] == 'played'


def test_legacy_seat_arrivals_uses_bounded_plan_and_preserves_locked_next_game(monkeypatch):
    from jupr_app.domain import adaptive_play_engine as engine
    from jupr_app.domain.round_robin_hub import reconcile_hub_events

    value = event(count=7)
    state, events = setup({'a': value})
    value = events['a']
    arrival = {'id': 'late', 'name': 'Late guest', 'seed': 8, 'roster_order': 8,
        'active_from_round': 2, 'inactive_from_round': None, 'inactive_rounds': []}
    value['participants'].append(arrival)
    value['rosterByRound']['2'].append('late')
    locked = row(value, 2)['matches'][0]
    locked['hubPlanLocked'] = True
    protected = deepcopy(locked)
    before = deepcopy(events)
    def full_rebuild(*args, **kwargs):
        pytest.fail('Legacy hub seating must use the bounded planner')
    monkeypatch.setattr(engine, '_regenerate_round_robin_from', full_rebuild)
    seated = engine.mutate_generator_roster(value, action='seat_arrivals',
        participant_ids=[*row(value, 1)['byeParticipantIds'], 'late'], court_number=2)
    state, updated = reconcile_hub_events(state, before, {'a': seated}, now=START)
    assert protected in row(updated['a'], 2)['matches']
    assert len(updated['a']['rounds']) == 2
    assert row(updated['a'], 1)['matches'][-1]['hubPlanLocked']
    assert_round_membership(updated['a'], 1)
    assert_round_membership(updated['a'], 2)


def test_distant_locked_game_does_not_preserve_unlocked_cohorts_when_round_enters_window():
    value = legacy_arrival_event(16)
    locked = row(value, 7)['matches'][0]
    locked['hubPlanLocked'] = True
    promise = deepcopy(locked)
    removed_ids = {m['id'] for m in row(value, 7)['matches'][1:]}
    value = _prepare_rolling_rounds(value)
    assert row(value, 7)['matches'] == [promise]
    assert row(value, 7)['hubPlanPartial']
    for number in range(1, 6):
        for match in row(value, number)['matches']:
            match.update(scoreA=11, scoreB=5, status='scored')
        value = _prepare_rolling_rounds(_settle_hub_rounds(value))
    assert value['currentRoundNumber'] == 6
    prepared = row(value, 7)
    assert not prepared.get('hubPlanPartial')
    assert promise in prepared['matches']
    assert len(prepared['matches']) == 4
    assert not removed_ids & {m['id'] for m in prepared['matches']}
    assert_round_membership(value, 7)


def test_partial_current_round_is_completed_without_changing_its_locked_game():
    value = legacy_arrival_event(12)
    current = row(value, 1)
    locked = current['matches'][0]
    locked['hubPlanLocked'] = True
    current['matches'] = [locked]
    current['hubPlanPartial'] = True
    value['hubPlanVersion'] = 1
    promise = deepcopy(locked)
    value = _prepare_rolling_rounds(value)
    assert promise in row(value, 1)['matches']
    assert not row(value, 1).get('hubPlanPartial')
    assert len(row(value, 1)['matches']) == 3
    assert_round_membership(value, 1)


def test_finishing_still_completes_partial_final_plan_but_never_plans_past_boundary():
    value = legacy_arrival_event(12)
    promised = row(value, 4)['matches'][0]
    promised['hubStartedAt'] = START
    promise = deepcopy(promised)
    value = _prepare_rolling_rounds(value)
    assert row(value, 4)['hubPlanPartial']
    value['hubFinishAfterRound'] = 4
    for number in range(1, 4):
        for match in row(value, number)['matches']:
            match.update(scoreA=11, scoreB=5, status='scored')
        value = _prepare_rolling_rounds(_settle_hub_rounds(value))
        assert max(r['number'] for r in value['rounds']) == 4
    assert value['currentRoundNumber'] == 4
    assert not row(value, 4).get('hubPlanPartial')
    assert promise in row(value, 4)['matches']
    assert len(row(value, 4)['matches']) == 3
    assert_round_membership(value, 4)
