from copy import deepcopy
import pytest
from jupr_app.domain.gamification.reconciliation import build_reconciliation_plan, simulate_reconciliation


def snapshot():
    return {
        'players': [{'id': 1, 'club_id': 'club', 'rating': 1400, 'matches_played': 1, 'wins': 1, 'losses': 0}],
        'league_ratings': [], 'leagues_metadata': [],
        'badges': [{'badge_id': 'participant', 'is_active': True, 'state': 'live'}],
        'matches': [{'id': 1, 'club_id': 'club', 'league': 'Open', 'date': '2026-01-01', 't1_p1': 1, 't2_p1': 2, 'score_t1': 11, 'score_t2': 3}],
        'player_badges': [
            {'id': 'old', 'club_id': 'club', 'player_id': 1, 'badge_id': 'participant', 'context_type': 'overall', 'context_id': None, 'earned_at': '2026-01-01'},
            {'id': 'new', 'club_id': 'club', 'player_id': 1, 'badge_id': 'participant', 'context_type': 'overall', 'context_id': 'overall', 'earned_at': '2026-01-02'},
            {'id': 'loss', 'club_id': 'club', 'player_id': 2, 'badge_id': 'hall_of_fame_night', 'match_id': 1, 'earned_at': '2026-01-01'},
            {'id': 'peak', 'club_id': 'club', 'player_id': 1, 'badge_id': 'level_up', 'context_type': 'league', 'context_id': 'milestone:5.0', 'earned_at': '2026-01-01'},
            {'id': 'trophy', 'club_id': 'club', 'player_id': 1, 'badge_id': 'tournament_champion', 'context_type': 'tournament', 'context_id': 't', 'earned_at': '2026-01-01'},
        ]}


def plan(data):
    return build_reconciliation_plan(data, club_id='club', as_of='2026-02-01T00:00:00Z')


def test_repair_preserves_old_dates_trophies_peaks_and_reaches_fixed_point():
    data = snapshot(); original = deepcopy(data)
    preview = plan(data)
    assert {r['row']['id'] for r in preview['revocations']} == {'new', 'loss'}
    assert {r['row']['id'] for r in preview['review']} == {'peak'}
    assert data == original
    result = simulate_reconciliation(data, preview)
    by_id = {r['id']: r for r in result['player_badges']}
    for id_ in ('old', 'trophy', 'peak'):
        assert by_id[id_] == next(r for r in data['player_badges'] if r['id'] == id_)
    assert by_id['new']['revoked_at'] and by_id['loss']['revoked_at']
    assert not plan(result)['revocations'] and not plan(result)['additions']


def test_changed_snapshot_blocks_plan_and_other_club_is_rejected():
    data = snapshot(); preview = plan(data)
    data['player_badges'][0]['earned_at'] = '2026-01-03'
    with pytest.raises(ValueError, match='Snapshot changed'):
        simulate_reconciliation(data, preview)
    data['players'][0]['club_id'] = 'another'
    with pytest.raises(ValueError, match='Mixed club'):
        plan(data)


def test_revoked_lifetime_award_is_never_reinstated():
    data = snapshot()
    data['player_badges'] = [dict(data['player_badges'][0], revoked_at='2026-01-02')]
    assert not [a for a in plan(data)['additions'] if a['badge_id'] == 'participant']
