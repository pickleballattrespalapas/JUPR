"""Account-free interclub boundaries and unchanged competition rules."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4
import hashlib

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from jupr_app.domain import managed_interclub as domain
from services.api import managed_interclub_routes as routes


def fixture_data():
    clubs = []
    for n in range(2):
        players = [dict(entry_id=str(uuid4()), name=f'Club {n} Player {i}', gender='female' if i < 2 else 'male', starting_rating=3.6) for i in range(4)]
        clubs.append(dict(id=str(uuid4()), name=f'Club {n}', captain='Private Captain', contact='private@example.invalid', rosters=[dict(division='3.5', roster=players)]))
    details = domain.SeasonDetails(name='BCS', start_date='2099-01-01', end_date='2099-03-31', divisions=['3.5'], clubs=clubs).model_dump(mode='json')
    sid, mid = str(uuid4()), str(uuid4())
    plan = domain.MeetDetails(host_club_id=clubs[1]['id'], club_ids=[c['id'] for c in clubs], starts_at='2099-01-16T17:00:00Z', courts=2,
        teams=[dict(club_id=c['id'], division='3.5', roster=c['rosters'][0]['roster']) for c in clubs]).model_dump(mode='json')
    document = domain.generate(mid, plan)
    season = dict(id=sid, organizer_club_id='tres_palapas', revision=1, details=details, published=False, updated_at='2026-11-14T00:00:00Z', source_season_id=None)
    meet = dict(id=mid, season_id=sid, revision=1, details=plan, state='draft', document=document, approved_document=None,
        host_token_hash=hashlib.sha256(('a'*43).encode()).hexdigest(), host_expires_at=(datetime.now(timezone.utc)+timedelta(days=2)).isoformat(), updated_at=season['updated_at'])
    return season, meet


def completed(doc):
    doc = deepcopy(doc)
    for encounter in doc['encounters']:
        for pairing in encounter['pairings']:
            for game in pairing['games']:
                game.update(status='completed', a=11, b=7, played_at='2099-01-16T18:00:00Z')
    return doc


def test_named_clubs_need_no_accounts_and_ratings_optional():
    season, meet = fixture_data()
    for club in season['details']['clubs']:
        for player in club['rosters'][0]['roster']: player['starting_rating'] = None
    assert domain.SeasonDetails.model_validate(season['details'])
    assert len(meet['document']['encounters']) == 1
    assert sum(len(p['games']) for e in meet['document']['encounters'] for p in e['pairings']) == 6
    assert meet['document']['schedule_mode'] == 'staggered'


def test_roster_ownership_overlap_and_rating_eligibility():
    season, meet = fixture_data()
    plan = domain.MeetDetails.model_validate(meet['details'])
    assert domain.validate_meet(plan, season['details'], [], meet['id'])
    with pytest.raises(ValueError, match='already has a meet'):
        domain.validate_meet(plan, season['details'], [dict(meet, id=str(uuid4()))], meet['id'])
    plan.teams[0].roster[0].entry_id = plan.teams[1].roster[0].entry_id
    with pytest.raises(ValueError, match='saved club roster'):
        domain.validate_meet(plan, season['details'], [], meet['id'])
    plan = domain.MeetDetails.model_validate(meet['details'])
    plan.teams[0].roster[0].starting_rating = 4.0
    season['details']['clubs'][0]['rosters'][0]['roster'][0]['starting_rating'] = 4.0
    with pytest.raises(ValueError, match='too high'):
        domain.validate_meet(plan, season['details'], [], meet['id'])


@pytest.mark.parametrize('field,value', [('meet_id','another-meet'), ('phase','final')])
def test_scoring_cannot_change_meet_identity(field,value):
    _, meet = fixture_data(); doc = deepcopy(meet['document']); doc[field] = value
    with pytest.raises(ValueError): domain.score_document(meet['document'], doc, meet['details']['teams'])


def test_host_cannot_change_schedule_starting_players_or_add_matches():
    _, meet = fixture_data()
    for change in ('court','player','extra'):
        doc = deepcopy(meet['document']); p = doc['encounters'][0]['pairings'][0]
        if change == 'court': p['court'] = 8
        if change == 'player': p['players_a'][0] = str(uuid4())
        if change == 'extra': doc['encounters'] = []
        with pytest.raises(ValueError): domain.score_document(meet['document'], doc, meet['details']['teams'])


def test_partial_save_and_official_standings_are_separate():
    season, meet = fixture_data()
    partial = deepcopy(meet['document']); partial['encounters'][0]['pairings'][0]['games'][0]['a'] = 6
    assert domain.score_document(meet['document'], partial, meet['details']['teams'])
    with pytest.raises(ValueError): domain.score_document(meet['document'], partial, meet['details']['teams'], official=True)
    official = domain.score_document(meet['document'], completed(meet['document']), meet['details']['teams'], official=True)
    meet['document'] = official
    assert all(row['points'] == 0 for row in domain.standings(season,[meet])['club_cup']['standings'])
    meet['approved_document'] = official
    assert max(row['points'] for row in domain.standings(season,[meet])['club_cup']['standings']) > 0
    meet['document'] = partial
    assert max(row['points'] for row in domain.standings(season,[meet])['club_cup']['standings']) > 0


class Query:
    def __init__(self, rows): self.rows, self.where, self.fields = rows, [], '*'
    def select(self, fields): self.fields = fields; return self
    def eq(self, key, value): self.where.append((key,value)); return self
    def in_(self, key, values): self.rows = [r for r in self.rows if r.get(key) in values]; return self
    def execute(self):
        return SimpleNamespace(data=[deepcopy(r) if self.fields == '*' else {k:deepcopy(r.get(k)) for k in self.fields.split(',')} for r in self.rows if all(r.get(k)==v for k,v in self.where)])


@pytest.fixture
def api(monkeypatch):
    season, meet = fixture_data(); calls=[]
    state = dict(season=season,meet=meet,calls=calls,code='',allowed='tres_palapas')
    user = SimpleNamespace(user_id=str(uuid4()),email='joe@example.invalid')
    def authorize(factory, authorization, club_id):
        if club_id != state['allowed']: raise HTTPException(403,'No organizer access')
        return user
    def rpc(name, params):
        calls.append((name,params))
        def execute():
            if state['code']:
                error=RuntimeError('Sensitive database detail');error.code=state['code'];raise error
            return SimpleNamespace(data=deepcopy(season if params['p_action']=='season' else meet))
        return SimpleNamespace(execute=execute)
    tables={routes.SEASONS:[season],routes.MEETS:[meet]}
    db=SimpleNamespace(table=lambda name:Query(tables[name]),rpc=rpc)
    state['db']=db
    monkeypatch.setattr(routes,'site_administrator',authorize)
    app=FastAPI();routes.install_managed_interclub_routes(app,get_supabase_client=lambda:db)
    return TestClient(app),state


def test_host_link_reads_only_its_meet_without_contacts_or_hash(api):
    client,s=api
    response=client.post('/public/interclub-host/read',json={'token':'a'*43})
    assert response.status_code==200, response.text
    assert response.headers['cache-control']=='no-store'
    assert response.headers['referrer-policy']=='no-referrer'
    assert 'private@example' not in response.text and 'Private Captain' not in response.text
    assert s['meet']['host_token_hash'] not in response.text and 'organizer_club_id' not in response.text
    assert set(response.json())=={'season','clubs','meet'}
    assert client.post('/public/interclub-host/read',json={'token':'b'*43}).status_code==404
    s['meet']['host_expires_at']='2000-01-01T00:00:00Z'
    assert client.post('/public/interclub-host/read',json={'token':'a'*43}).status_code==404


def test_host_cannot_submit_unfinished_meet_or_inject_admin_actions(api):
    client,s=api
    for action in ('approve','share','reopen','plan','generate'):
        assert client.post('/public/interclub-host/write',json={'token':'a'*43,'action':action,'expected_revision':1}).status_code==422
    assert client.post('/public/interclub-host/write',json={'token':'a'*43,'action':'submit','expected_revision':1}).status_code==422
    assert not s['calls']


def test_host_save_rechecks_hash_under_lock_and_never_supplies_admin_identity(api):
    client,s=api
    response=client.post('/public/interclub-host/write',json={'token':'a'*43,'action':'scores','expected_revision':1,'document':s['meet']['document']})
    assert response.status_code==200,response.text
    _,params=s['calls'][-1]
    assert params['p_actor_id'] is None and params['p_club_id'] is None
    assert params['p_token_hash']==s['meet']['host_token_hash']
    assert 'a'*43 not in str(params)
    for code,status in [('42501',403),('PT409',409),('unexpected',503)]:
        s['code']=code
        response=client.post('/public/interclub-host/write',json={'token':'a'*43,'action':'scores','expected_revision':1,'document':s['meet']['document']})
        assert response.status_code==status and 'Sensitive' not in response.text
        assert response.headers['cache-control']=='no-store'


def test_admin_is_scoped_to_owner_and_host_does_not_publish(api):
    client,s=api;sid=s['season']['id']
    assert client.get(f'/admin/clubs/other/interclub/managed/{sid}').status_code==403
    assert client.get(f'/public/managed-interclub/{sid}').status_code==404
    s['season']['published']=True
    s['meet']['document']=completed(s['meet']['document'])
    response=client.get(f'/public/managed-interclub/{sid}')
    assert response.status_code==200,response.text
    assert 'private@example' not in response.text and 'Private Captain' not in response.text
    assert not response.json()['document']['competition_results']
    s['meet']['approved_document']=s['meet']['document']
    response=client.get(f'/public/managed-interclub/{sid}')
    assert len(response.json()['document']['competition_results'])==1
    assert len(response.json()['document']['players'])==8


def test_submit_uses_saved_revision_and_rejects_body_scores(api):
    client,s=api;s['meet']['document']=completed(s['meet']['document'])
    response=client.post('/public/interclub-host/write',json={'token':'a'*43,'action':'submit','expected_revision':1})
    assert response.status_code==200,response.text
    assert s['calls'][-1][1]['p_action']=='submit' and s['calls'][-1][1]['p_payload']=={}
    assert client.post('/public/interclub-host/write',json={'token':'a'*43,'action':'submit','expected_revision':1,'document':s['meet']['document']}).status_code==422


def test_existing_season_import_keeps_original_records_and_official_scores():
    season, meet = fixture_data()
    clubs = season['details']['clubs']; sid, mid = season['id'], meet['id']
    official = completed(meet['document'])
    tables = {
        'pcs_interclub_seasons':[dict(id=sid,organizer_club_id='tres_palapas',details={**season['details'],'club_ids':[c['id'] for c in clubs]})],
        'clubs':[dict(id=c['id'],name=c['name']) for c in clubs],
        'pcs_interclub_current_rosters':[dict(id=str(uuid4()),season_id=sid,meet_id=mid,club_id=c['id'],division='3.5',roster=c['rosters'][0]['roster'],withdrawn=False) for c in clubs],
        'pcs_interclub_competition_batches':[dict(season_id=sid,meet_id=mid,phase='regular',state='approved',document=official,approved_document=official)],
        'pcs_interclub_meets':[dict(id=mid,season_id=sid,competition_phase='regular',**meet['details'])],
    }
    db = SimpleNamespace(table=lambda name:Query(tables[name]))
    before = deepcopy(tables)
    imported = routes.import_legacy(db,'tres_palapas',sid)
    assert imported['meets'][0]['approved_document'] == official
    assert imported['meets'][0]['state'] == 'approved'
    assert imported['details']['name'] == season['details']['name']
    assert len(imported['details']['clubs']) == 2
    assert tables == before
    # Keep a historical injury substitute even after the current roster changed.
    substitute = str(uuid4())
    original_club = official['encounters'][0]['club_a']
    pairing = official['encounters'][0]['pairings'][0]
    for game in pairing['games']:
        game['players_a'] = [substitute, pairing['players_a'][1]]
        game['injury_reason'] = 'Injury substitution before the opening game'
    tables['pcs_interclub_entries'] = [dict(id=substitute, season_id=sid, club_id=original_club, player_id=101, starting_rating=3.6)]
    tables['players'] = [dict(id=101, club_id=original_club, name='Historical substitute', gender='female')]
    imported = routes.import_legacy(db,'tres_palapas',sid)
    reserves = next(t for t in imported['meets'][0]['details']['teams'] if t['club_id'] == original_club)['reserves']
    assert reserves[0]['entry_id'] == substitute
    assert any(p['entry_id'] == substitute for c in imported['details']['clubs'] for r in c['rosters'] for p in r['roster'])
    # An incompatible historical meet must not be silently dropped.
    tables['pcs_interclub_competition_batches'].append(dict(tables['pcs_interclub_competition_batches'][0],phase='final'))
    with pytest.raises(HTTPException) as exc: routes.import_legacy(db,'tres_palapas',sid)
    assert exc.value.status_code == 422
