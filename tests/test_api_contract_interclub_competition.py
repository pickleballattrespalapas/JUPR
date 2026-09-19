"""Permission, revision and immutable-score identity contracts for meet operations."""
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api import admin_auth_routes, interclub_competition_routes as routes
from jupr_app.domain import interclub_competition as engine


class Query:
    def __init__(self, rows): self.rows, self.filters, self.columns = rows, [], '*'
    def select(self, fields): self.columns=fields; return self
    def eq(self, key, value): self.filters.append(lambda row: row.get(key)==value); return self
    def in_(self, key, values): self.filters.append(lambda row: row.get(key) in values); return self
    def execute(self):
        return SimpleNamespace(data=[dict(r) if self.columns=='*' else {k:r.get(k) for k in self.columns.split(',')} for r in self.rows if all(f(r) for f in self.filters)])


@pytest.fixture
def setup(monkeypatch):
    sid, mid, uid = map(str,[uuid4(),uuid4(),uuid4()])
    user=SimpleNamespace(user_id=uid,email='qa@example.invalid')
    assignment=dict(club_id='home',email=user.email,user_id=uid,role='administrator')
    season=dict(id=sid,organizer_club_id='home',details=dict(name='BCS QA',club_ids=['home','away'],divisions=['3.5']),rules={})
    meet=dict(id=mid,season_id=sid,plan_index=0,host_club_id='away',club_ids=['home','away'],starts_at='2099-01-10T18:00:00Z',roster_deadline='2099-01-09T18:00:00Z',revision=1,courts=2,duration_minutes=180)
    teams=[]
    for club in ['home','away']:
        teams.append(dict(id=str(uuid4()),season_id=sid,meet_id=mid,club_id=club,division='3.5',name=club,revision=1,withdrawn=False,status='eligible',issues=[],
                          roster=[dict(entry_id=str(uuid4()),name=f'{club} Player {i}',starting_rating=3.6,gender='female' if i<2 else 'male',player_id=i,email='private@example.invalid') for i in range(4)]))
    document=engine.generate_round_robin(mid,teams,format='gender',played_at=meet['starts_at'],courts=2)
    for encounter in document['encounters']:
        for pairing in encounter['pairings']: pairing['eligibility_deadline']=meet['roster_deadline']
    document=engine.validate_document(document)
    saved=dict(id=str(uuid4()),season_id=sid,meet_id=mid,phase='regular',revision=1,state='draft',document=document,roster_sources=routes._sources(teams),ratings_status='not_requested',approved_document=None,approved_revision=None)
    tables={'admin_role_assignments':[assignment],'pcs_interclub_seasons':[season], 'pcs_interclub_meet_workspaces':[meet],
            'pcs_interclub_participations':[dict(season_id=sid,club_id=c,status='accepted') for c in ['home','away']],
            'pcs_interclub_meet_eligibility_snapshots':[], 'pcs_interclub_pool_members':[], 'pcs_interclub_entries':[], 'pcs_interclub_current_rosters':teams,'pcs_interclub_competition_batches':[saved], 'clubs':[dict(id=c,name=c.title(),slug=c) for c in ['home','away','unrelated']]}
    state=dict(season=season,meet=meet,user=user,assignment=assignment,teams=teams,saved=saved,tables=tables,calls=[],reads=[],error='')
    def table(name): state['reads'].append(name); return Query(tables[name])
    def rpc(name, params):
        state['calls'].append((name,params))
        def execute():
            if state['error']:
                error=RuntimeError('private SQL failure');error.code=state['error'];raise error
            return SimpleNamespace(data=deepcopy(saved))
        return SimpleNamespace(execute=execute)
    monkeypatch.setattr(admin_auth_routes,'authenticate_bearer',lambda _:user)
    state['db']=SimpleNamespace(table=table,rpc=rpc)
    app=FastAPI();routes.install_interclub_competition_routes(app,get_supabase_client=lambda:state['db'])
    return TestClient(app),state


def base(s,club='home'):
    return f"/admin/clubs/{club}/interclub/competition/{s['season']['id']}"


def path(s,club='home'):
    return base(s,club)+f"/meets/{s['meet']['id']}/regular"


def complete(s):
    for e in s['saved']['document']['encounters']:
        for p in e['pairings']:
            for g in p['games']: g.update(status='completed',a=11,b=7,winner='a')


def test_read_is_scoped_and_does_not_disclose_contacts_or_local_player_ids(setup):
    client,s=setup
    r=client.get(path(s));assert r.status_code==200
    assert 'private@example' not in r.text and 'player_id' not in r.text
    r=client.get(base(s));assert r.status_code==200
    assert 'unrelated' not in r.text
    s['assignment']['club_id']='unrelated'
    assert client.get(base(s,'unrelated')).status_code==404
    assert not s['calls']


def test_host_can_save_submit_but_cannot_approve_or_reopen(setup):
    client,s=setup;s['assignment']['club_id']='away'
    r=client.put(path(s,'away'),json=dict(expected_revision=1,document=s['saved']['document']))
    assert r.status_code==200
    assert s['calls'][-1][1]['p_actor_id']==s['user'].user_id
    assert s['calls'][-1][1]['p_club_id']=='away'
    complete(s)
    assert client.post(path(s,'away')+'/submit',json={'expected_revision':1}).status_code==200
    for action,body in [('approve',{'expected_revision':1}),('reopen',{'expected_revision':1,'reason':'Correct score'})]:
        assert client.post(path(s,'away')+'/'+action,json=body).status_code==403


def test_nonhost_participant_cannot_edit_other_meet(setup):
    client,s=setup;s['season']['organizer_club_id']='organizer'
    assert client.get(path(s)).status_code==200
    assert client.put(path(s),json=dict(expected_revision=1,document=s['saved']['document'])).status_code==403


@pytest.mark.parametrize('scope,expected', [([{'kind':'club'}],200),([{'kind':'program_type','program_type':'leagues'}],200),([{'kind':'resource','program_type':'leagues','resource_id':'another'}],403),([],403)])
def test_operator_needs_relevant_host_assignment(setup,scope,expected):
    client,s=setup;s['assignment'].update(club_id='away',role='operator',scopes=scope)
    assert client.put(path(s,'away'),json=dict(expected_revision=1,document=s['saved']['document'])).status_code==expected
    assert client.post(path(s,'away')+'/approve',json={'expected_revision':1}).status_code==403


@pytest.mark.parametrize('change',[dict(revoked_at='2020-01-01'),dict(expires_at='2020-01-01T00:00:00Z'),dict(user_id='other')])
def test_stale_or_other_identity_assignments_rejected(setup,change):
    client,s=setup;s['assignment'].update(change)
    assert client.get(path(s)).status_code==403
    assert not s['calls']


def test_every_game_must_be_disposed_before_submit(setup):
    client,s=setup
    assert client.post(path(s)+'/submit',json={'expected_revision':1}).status_code==422
    assert not s['calls']
    complete(s)
    assert client.post(path(s)+'/submit',json={'expected_revision':1}).status_code==200
    assert s['calls'][-1][1]['p_document'] is None # SQL uses its own canonical revision


@pytest.mark.parametrize('mutation', ['meet','phase','opponent','game_id','pairing_id','deadline','encounter_delete','reschedule'])
def test_client_cannot_replace_schedule_or_deadline(setup,mutation):
    client,s=setup;document=deepcopy(s['saved']['document']);e=document['encounters'][0];p=e['pairings'][0]
    if mutation=='meet':document['meet_id']=str(uuid4())
    if mutation=='phase':document['phase']='final'
    if mutation=='opponent':e['club_b']='unrelated'
    if mutation=='game_id':p['games'][0]['id']='invented'
    if mutation=='pairing_id':p['id']='invented'
    if mutation=='deadline':p['eligibility_deadline']='2098-01-01T00:00:00Z'
    if mutation=='encounter_delete':document['encounters']=[]
    if mutation=='reschedule':document['weather']='rescheduled'
    assert client.put(path(s),json=dict(expected_revision=1,document=document)).status_code==422
    assert not s['calls']


def test_old_revision_does_not_write(setup):
    client,s=setup
    assert client.put(path(s),json=dict(expected_revision=0,document=s['saved']['document'])).status_code==409
    assert not s['calls']


def test_generation_preserves_source_roster_revisions_and_no_contacts(setup):
    client,s=setup
    r=client.post(path(s)+'/generate',json=dict(expected_revision=1,format='gender'))
    assert r.status_code==200
    params=s['calls'][-1][1]
    assert params['p_roster_sources']==routes._sources(s['teams'])
    assert 'player_id' not in str(params['p_document']) and 'private@example' not in str(params)


def test_scores_prevent_silent_regeneration(setup):
    client,s=setup;complete(s)
    assert client.post(path(s)+'/generate',json=dict(expected_revision=1,format='mixed')).status_code==409
    assert not s['calls']


def test_public_reads_preserve_prior_approved_version_during_correction(setup):
    _,s=setup;complete(s)
    official=deepcopy(s['saved']['document']);s['saved'].update(approved_document=official,approved_revision=3,state='draft')
    s['saved']['document']['encounters'][0]['pairings'][0]['games'][0]['a']=12
    assert routes.approved_documents(s['db'],s['season']['id'])==[official]


@pytest.mark.parametrize('code,status',[('42501',403),('40001',409),('PT409',409),('22023',422),('P0002',404),('23505',409),('other',503)])
def test_rpc_errors_are_safe_and_actionable(setup,code,status):
    client,s=setup;s['error']=code
    r=client.put(path(s),json=dict(expected_revision=1,document=s['saved']['document']))
    assert r.status_code==status and 'private SQL' not in r.text


def test_finals_cannot_be_selected_before_qualification(setup):
    client,s=setup
    url=path(s).removesuffix('regular')+'final/generate'
    assert client.post(url,json=dict(expected_revision=0,format='mlp',division='3.5',club_a='home',club_b='away')).status_code==422
    assert not s['calls']


def test_fifth_approved_pool_player_is_available_without_private_contact_details(setup):
    client,s=setup
    s['meet']['roster_deadline']='2020-01-01T00:00:00Z'
    entry_id=str(uuid4());member_id=str(uuid4())
    s['tables']['pcs_interclub_entries']=[dict(id=entry_id,club_id='home',season_id=s['season']['id'],player_id=50,pool_member_id=member_id,starting_rating=3.6)]
    s['tables']['pcs_interclub_meet_eligibility_snapshots']=[dict(meet_id=s['meet']['id'],entry_id=entry_id,club_id='home',rating=3.7,gender='female',deadline=s['meet']['roster_deadline'])]
    s['tables']['players']=[dict(id=50,club_id='home',name='Approved substitute',gender='female',email='private@example.invalid')]
    r=client.get(path(s));assert r.status_code==200
    choices=r.json()['eligible_players']['home']
    assert choices[0]['name']=='Approved substitute' and choices[0]['eligibility_rating']==3.7
    assert 'player_id' not in r.text and 'private@example' not in r.text


def test_prepared_mixed_pairs_can_be_rearranged_but_not_replaced_with_nonroster_player(setup):
    client,s=setup
    document=engine.generate_round_robin(s['meet']['id'],s['teams'],format='mixed')
    for e in document['encounters']:
        for p in e['pairings']:p['eligibility_deadline']=s['meet']['roster_deadline']
    document=engine.validate_document(document);s['saved']['document']=deepcopy(document)
    p1,p2=document['encounters'][0]['pairings']
    p1['players_a'][1],p2['players_a'][1]=p2['players_a'][1],p1['players_a'][1]
    assert client.put(path(s),json=dict(expected_revision=1,document=document)).status_code==200
    p1['players_a'][1]=str(uuid4())
    assert client.put(path(s),json=dict(expected_revision=1,document=document)).status_code==422


def test_prepared_pairing_is_immutable_after_score_entry(setup):
    client,s=setup;complete(s)
    document=deepcopy(s['saved']['document']);p1,p2=document['encounters'][0]['pairings']
    p1['players_a'][0],p2['players_a'][0]=p2['players_a'][0],p1['players_a'][0]
    assert client.put(path(s),json=dict(expected_revision=1,document=document)).status_code==422


def test_scoped_operator_gets_only_relevant_competition_seasons(setup):
    client,s=setup;s['assignment'].update(club_id='away',role='operator',scopes=[{'kind':'resource','program_type':'leagues','resource_id':s['meet']['id']}])
    assert len(client.get('/admin/clubs/away/interclub/competition').json()['seasons'])==1
    s['assignment']['scopes'][0]['resource_id']='other'
    assert client.get('/admin/clubs/away/interclub/competition').json()['seasons']==[]


def test_meet_creation_requires_organizer_and_accepted_clubs(setup):
    client,s=setup
    payload=dict(host_club_id='away',club_ids=['home','away'],starts_at='2099-03-01T18:00:00Z',roster_deadline='2099-02-27T18:00:00Z',courts=2,duration_minutes=180,competition_phase='final')
    assert client.post(base(s)+'/meets',json=payload).status_code==200
    assert s['calls'][-1][0]=='pcs_create_interclub_competition_meet'
    payload['club_ids']=['home','unrelated']
    assert client.post(base(s)+'/meets',json=payload).status_code==422
    s['assignment']['club_id']='away';payload['club_ids']=['home','away']
    assert client.post(base(s,'away')+'/meets',json=payload).status_code==403


def test_wrong_meet_competition_phase_is_rejected(setup):
    client,s=setup;s['meet']['competition_phase']='final'
    assert client.get(path(s)).status_code==422
    assert client.post(path(s)+'/generate',json=dict(expected_revision=1,format='gender')).status_code==422


def test_replay_keeps_complete_pairings_and_resets_unfinished_pairing(setup):
    client,s=setup;complete(s)
    e=s['saved']['document']['encounters'][0];e['pairings'][1]['games'][2].update(status='pending',a=None,b=None,winner=None)
    original=deepcopy(e['pairings'][0])
    r=client.post(path(s)+'/reschedule',json=dict(expected_revision=1,reason='Weather cancellation',starts_at='2099-02-01T18:00:00Z',roster_deadline='2099-01-30T18:00:00Z'))
    assert r.status_code==200
    replay=s['calls'][-1][1]['p_document'];pairs=replay['encounters'][0]['pairings']
    assert pairs[0]==original
    assert all(g['status']=='pending' and g['a'] is None and g['played_at'] is None for g in pairs[1]['games'])
    assert pairs[1]['eligibility_deadline']=='2099-01-30T18:00:00Z'


def test_historical_entry_names_are_display_only_not_eligible_choices(setup):
    client,s=setup;player=s['teams'][0]['roster'][0]
    s['tables']['pcs_interclub_entries']=[dict(id=player['entry_id'],club_id='home',season_id=s['season']['id'],player_id=50,pool_member_id=str(uuid4()),starting_rating=3.6)]
    s['tables']['players']=[dict(id=50,club_id='home',name='Prior completed lineup',gender='female',email='private@example.invalid')]
    r=client.get(path(s));assert r.status_code==200
    assert r.json()['display_players']==[dict(entry_id=player['entry_id'],club_id='home',name='Prior completed lineup',gender='female')]
    assert not r.json()['eligible_players']
    assert 'private@example' not in r.text


def test_nonplaying_accepted_host_can_run_championship_meet(setup):
    client,s=setup;s['assignment']['club_id']='away';s['meet']['club_ids']=['home','third']
    assert client.get(path(s,'away')).status_code==200
    assert len(client.get(base(s,'away')).json()['meets'])==1


def test_organizer_operator_scope_can_access_meet_without_organizer_playing(setup):
    client,s=setup;s['assignment'].update(role='operator',scopes=[{'kind':'resource','program_type':'leagues','resource_id':s['meet']['id']}]);s['meet']['club_ids']=['away','third']
    assert client.get(path(s)).status_code==200
    assert len(client.get(base(s)).json()['meets'])==1
    assert not client.get(base(s)).json()['is_organizer']


def test_final_generation_binds_exact_approved_standings_revisions(setup,monkeypatch):
    client,s=setup;s['meet']['competition_phase']='final'
    s['saved']['approved_document']=deepcopy(s['saved']['document']);s['saved']['approved_revision']=4
    monkeypatch.setattr(engine,'league_standings',lambda *_args,**_kwargs:{'qualification':{'3.5':{'status':'ready','qualifiers':['home','away'],'playoff_required':[]}}})
    url=path(s).removesuffix('regular')+'final/generate'
    r=client.post(url,json=dict(expected_revision=0,format='mlp',division='3.5',club_a='home',club_b='away'))
    assert r.status_code==200
    args=s['calls'][-1][1]
    assert args['p_qualification_sources']==[dict(id=s['saved']['id'],revision=4)]
    assert args['p_document']['phase']=='final' and len(args['p_document']['encounters'][0]['pairings'])==4
    assert all(g['played_at'] is None for p in args['p_document']['encounters'][0]['pairings'] for g in p['games'])


def test_club_must_remain_qualified_when_submitting_final(setup,monkeypatch):
    client,s=setup;s['meet']['competition_phase']='final';s['saved']['phase']='final'
    s['saved']['document']=engine.generate_championship(s['meet']['id'],'3.5',s['teams'][0],s['teams'][1],played_at='2026-01-01T18:00:00Z')
    complete(s)
    monkeypatch.setattr(engine,'league_standings',lambda *_args,**_kwargs:{'qualification':{'3.5':{'status':'playoff_required','qualifiers':['home'],'playoff_required':['away','third']}}})
    url=path(s).removesuffix('regular')+'final/submit'
    assert client.post(url,json={'expected_revision':1}).status_code==409
    assert not s['calls']


def test_opposing_lineups_stay_blind_to_participants_until_deadline(setup):
    client,s=setup;s['season']['organizer_club_id']='organizer'
    r=client.get(path(s));assert r.status_code==200
    detail=r.json()
    assert detail['lineups_hidden'] and detail['batch'] is None
    assert [t['club_id'] for t in detail['teams']]==['home']
    assert not detail['display_players']
    assert not client.get(base(s)).json()['batches']
    s['meet']['roster_deadline']='2020-01-01T00:00:00Z'
    detail=client.get(path(s)).json()
    assert not detail['lineups_hidden'] and detail['batch'] is not None
    assert {t['club_id'] for t in detail['teams']}=={'home','away'}


def test_two_player_roster_generates_only_missing_pair_forfeits_without_fake_players(setup):
    client,s=setup;s['teams'][0]['roster']=s['teams'][0]['roster'][:2]
    r=client.post(path(s)+'/generate',json=dict(expected_revision=1,format='gender'))
    assert r.status_code==200
    document=s['calls'][-1][1]['p_document'];encounter=document['encounters'][0]
    home_side='a' if encounter['club_a']=='home' else 'b'
    present=next(p for p in encounter['pairings'] if p['kind']=='women')
    absent=next(p for p in encounter['pairings'] if p['kind']=='men')
    assert len(present['players_'+home_side])==2 and not absent['players_'+home_side]
    assert all(g['status']=='pending' for g in present['games'])
    assert all(g['status']=='forfeit' and g['winner']!=home_side and g['a'] is None and g['b'] is None for g in absent['games'])
    assert not any('player_id' in p for team in document['encounters'] for p in team['pairings'])


def test_automatic_missing_pair_forfeit_does_not_freeze_preplay_arrangement(setup):
    client,s=setup;s['teams'][0]['roster']=s['teams'][0]['roster'][::2] # one woman and one man
    document=engine.generate_round_robin(s['meet']['id'],s['teams'],format='mixed')
    for e in document['encounters']:
        for p in e['pairings']:p['eligibility_deadline']=s['meet']['roster_deadline']
    s['saved']['document']=engine.validate_document(document);document=deepcopy(s['saved']['document'])
    encounter=document['encounters'][0];full_side='a' if encounter['club_a']=='away' else 'b'
    p1,p2=encounter['pairings'];key='players_'+full_side
    p1[key][1],p2[key][1]=p2[key][1],p1[key][1]
    assert client.put(path(s),json=dict(expected_revision=1,document=document)).status_code==200


def test_both_missing_same_pair_generates_explicit_double_forfeit(setup):
    client,s=setup
    for team in s['teams']:team['roster']=team['roster'][:2]
    r=client.post(path(s)+'/generate',json=dict(expected_revision=1,format='gender'))
    assert r.status_code==200
    pairings=s['calls'][-1][1]['p_document']['encounters'][0]['pairings']
    absent=next(p for p in pairings if p['kind']=='men')
    assert not absent['players_a'] and not absent['players_b']
    assert all(g['status']=='double_forfeit' and g['a'] is None and g['b'] is None and g['winner'] is None for g in absent['games'])


def test_partial_team_can_refresh_to_full_lineup_before_play_without_new_game_ids(setup):
    client,s=setup;full_roster=deepcopy(s['teams'][0]['roster']);s['teams'][0]['roster']=full_roster[:2]
    document=engine.generate_round_robin(s['meet']['id'],s['teams'],format='gender')
    for e in document['encounters']:
        for p in e['pairings']:p['eligibility_deadline']=s['meet']['roster_deadline']
    s['saved']['document']=engine.validate_document(document)
    ids=[g['id'] for e in document['encounters'] for p in e['pairings'] for g in p['games']]
    s['teams'][0].update(roster=full_roster,revision=2)
    r=client.post(path(s)+'/refresh-lineups',json={'expected_revision':1})
    assert r.status_code==200
    args=s['calls'][-1][1];refreshed=args['p_document']
    assert all(g['status']=='pending' for e in refreshed['encounters'] for p in e['pairings'] for g in p['games'])
    assert [g['id'] for e in refreshed['encounters'] for p in e['pairings'] for g in p['games']]==ids
    assert dict(team_id=s['teams'][0]['id'],revision=2) in args['p_roster_sources']


def test_lineup_refresh_does_not_clear_an_explicit_forfeit_from_full_rosters(setup):
    client,s=setup
    for game in s['saved']['document']['encounters'][0]['pairings'][0]['games']:
        game.update(status='forfeit',winner='a',a=None,b=None)
    assert client.post(path(s)+'/refresh-lineups',json={'expected_revision':1}).status_code==422
    assert not s['calls']


def test_lineup_refresh_cannot_change_starting_team_after_meet_start(setup):
    client,s=setup;s['meet']['starts_at']='2020-01-01T18:00:00Z'
    assert client.post(path(s)+'/refresh-lineups',json={'expected_revision':1}).status_code==422
    assert not s['calls']
