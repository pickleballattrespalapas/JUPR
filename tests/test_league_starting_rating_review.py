from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from jupr_app.services import admin_league_live_service as live
from jupr_app.services.starting_rating_review_service import StartingRatingReviewRequired
from services.api.main import app
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_league_live import (
    COURTS, ROSTER, _create, _install_env, league_live_tables,
)


def test_unreviewed_estimate_blocks_initial_courts_and_create_before_writes(monkeypatch):
    tables = league_live_tables()
    tables['players'][0]['__starting_review_confirmed__'] = False
    db = FakeSupabase(tables)
    _install_env(monkeypatch, db)
    client = TestClient(app)
    suggestion = client.post('/admin/clubs/club/league-manager/live/roster-suggestion',
        headers={'Authorization': 'Bearer local'}, json={'roster': ROSTER})
    assert suggestion.status_code == 409
    assert suggestion.json()['detail']['code'] == 'STARTING_RATING_REVIEW_REQUIRED'
    assert [row['player_id'] for row in suggestion.json()['detail']['players']] == [1]
    created = _create(client)
    assert created.status_code == 409
    assert not tables['league_live_sessions']
    assert not tables.get('admin_guarded_operations')


def test_actual_rated_history_allows_seed_even_without_estimate_confirmation(monkeypatch):
    tables = league_live_tables()
    tables['players'][0].update({'__starting_review_confirmed__': False, '__starting_review_matches__': 8})
    db = FakeSupabase(tables)
    _install_env(monkeypatch, db)
    result = _create(TestClient(app))
    assert result.status_code == 200, result.text
    assert result.json()['session']['roster_json'][0]['rating'] == 1400


def test_saved_session_exemption_never_covers_new_players(monkeypatch):
    tables = league_live_tables()
    for row in tables['players']:
        row['__starting_review_confirmed__'] = False
    tables['players'].append({'id': 5, 'club_id': 'club', 'name': 'New arrival', 'rating': 1200, '__starting_review_confirmed__': False})
    tables['league_live_sessions'].append({'id': 'saved', 'club_id': 'club', 'roster_json': deepcopy(ROSTER)})
    db = FakeSupabase(tables)
    _install_env(monkeypatch, db)
    kept = live.suggest_admin_league_live_roster(db, club_id='club', roster=ROSTER, session_id='saved')
    assert len(kept['roster']) == 4
    with pytest.raises(StartingRatingReviewRequired) as error:
        live.suggest_admin_league_live_roster(db, club_id='club', roster=[*ROSTER, {'player_id': 5, 'player_name': 'New arrival', 'rating': 1200}], session_id='saved')
    assert [row['player_id'] for row in error.value.detail['players']] == [5]


def test_prepared_first_courts_cannot_commit_after_reviewed_seed_changes(monkeypatch):
    tables = league_live_tables()
    db = FakeSupabase(tables)
    _install_env(monkeypatch, db)
    prepared = live.prepare_admin_league_live_session_create(db, club_id='club', session_id='new-session',
        league_name='Sunday Ladder', week_tag='Week 1', total_rounds=3, roster=ROSTER, courts=COURTS, actor_email='admin@example.com')
    tables['players'][0]['rating'] = 1500
    with pytest.raises(live.LeagueLiveStartingRatingChangedError, match='Rebuild the courts'):
        live.create_admin_league_live_session(db, club_id='club', league_name='Sunday Ladder', week_tag='Week 1',
            total_rounds=3, roster=ROSTER, courts=COURTS, actor_email='admin@example.com', actor_role='club_owner',
            confirmation_text='CREATE LIVE SESSION', prepared_plan=prepared)
    assert not tables['league_live_sessions']


@pytest.mark.parametrize('change', ['attendance', 'late_arrival'])
def test_new_arrival_review_is_required_before_next_round_plan(monkeypatch, change):
    tables = league_live_tables()
    db = FakeSupabase(tables)
    _install_env(monkeypatch, db)
    created = _create(TestClient(app)).json()['session']
    for row in tables['players']:
        row['__starting_review_confirmed__'] = False
    incoming = {'player_id': 5, 'player_name': 'New arrival', 'rating': 1200}
    tables['players'].append({'id': 5, 'club_id': 'club', 'name': 'New arrival', 'rating': 1200, '__starting_review_confirmed__': False})
    roster_change = ({'action': 'attendance', 'players': [incoming], 'active_player_ids': [1, 2, 3, 4, 5]}
        if change == 'attendance' else {'action': 'late_arrival', 'player': incoming})
    with pytest.raises(StartingRatingReviewRequired) as error:
        live.build_admin_league_live_round_plan(db, club_id='club', session_id=created['id'],
            round_number=1, expected_updated_at=created['updated_at'], matches=[], roster_change=roster_change)
    assert [row['player_id'] for row in error.value.detail['players']] == [5]
    assert len(tables['league_live_sessions'][0]['roster_json']) == 4


def test_snapshot_cannot_add_unreviewed_player_but_keeps_saved_roster_operable(monkeypatch):
    tables = league_live_tables()
    db = FakeSupabase(tables)
    _install_env(monkeypatch, db)
    created = _create(TestClient(app)).json()['session']
    for row in tables['players']:
        row['__starting_review_confirmed__'] = False
    saved = live.update_admin_league_live_session_snapshot(db, club_id='club', session_id=created['id'],
        patch={'expected_updated_at': created['updated_at']}, actor_email='admin@example.com',
        actor_role='club_owner', confirmation_text='SAVE SESSION')
    incoming = {'player_id': 5, 'player_name': 'New arrival', 'rating': 1200}
    tables['players'].append({'id': 5, 'club_id': 'club', 'name': 'New arrival', 'rating': 1200, '__starting_review_confirmed__': False})
    with pytest.raises(StartingRatingReviewRequired) as error:
        live.update_admin_league_live_session_snapshot(db, club_id='club', session_id=created['id'],
            patch={'expected_updated_at': saved['session']['updated_at'], 'roster': [*ROSTER, incoming]},
            actor_email='admin@example.com', actor_role='club_owner', confirmation_text='SAVE SESSION')
    assert [row['player_id'] for row in error.value.detail['players']] == [5]


def test_completed_create_retry_returns_receipt_before_new_review(monkeypatch):
    tables = league_live_tables()
    db = FakeSupabase(tables)
    _install_env(monkeypatch, db)
    client = TestClient(app)
    first = _create(client)
    assert first.status_code == 200, first.text
    for row in tables['players']:
        row['__starting_review_confirmed__'] = False
    replay = _create(client)
    assert replay.status_code == 200, replay.text
    assert replay.json()['session']['id'] == first.json()['session']['id']
    assert len(tables['league_live_sessions']) == 1
