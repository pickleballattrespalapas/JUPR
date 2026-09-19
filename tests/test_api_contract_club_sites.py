from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from services.api import admin_auth_routes, club_site_routes as routes
from services.api.club_site_models import SiteDocument, SiteBlock
from services.api.interclub_public_routes import ResultsDocument, league_standings, validate_results, public_document

class Query:
    def __init__(self, rows): self.rows=deepcopy(rows); self.filters=[]; self.fields='*'; self.span=None
    def select(self, fields, **kwargs): self.fields=fields; return self
    def eq(self, key, value): self.filters.append(lambda r:r.get(key)==value); return self
    def in_(self, key, values): self.filters.append(lambda r:r.get(key) in values); return self
    def ilike(self,key,value): self.filters.append(lambda r:value.strip('%').casefold() in r.get(key,'').casefold());return self
    def order(self,*args,**kwargs): return self
    def range(self,a,b):self.span=(a,b+1);return self
    def limit(self,n):self.span=(0,n);return self
    def execute(self):
        rows=[r for r in self.rows if all(f(r) for f in self.filters)];count=len(rows)
        if self.span:rows=rows[slice(*self.span)]
        if self.fields!='*':rows=[{k:r.get(k) for k in self.fields.split(',')} for r in rows]
        return SimpleNamespace(data=rows,count=count)

@pytest.fixture
def setup(monkeypatch):
    user=SimpleNamespace(user_id='verified',email='owner@example.test')
    monkeypatch.setattr(admin_auth_routes,'authenticate_bearer',lambda _:user)
    monkeypatch.setattr(routes,'authenticate_bearer',lambda _:user)
    public=SiteDocument(name='Published Alpha').model_dump()
    tables={'clubs':[dict(id='alpha',slug='alpha',name='Operational name',tagline='',logo_url='',is_active=True,public_site_status='published'),dict(id='beta',slug='beta',is_active=True,public_site_status='draft')],
      'pcs_club_sites':[dict(club_id='alpha',revision=5,draft=SiteDocument(name='Private next version').model_dump(),published=public,published_at='2026-09-16T00:00:00Z')],
      'admin_role_assignments':[dict(club_id='alpha',role='administrator',email=user.email,user_id=user.user_id)]}
    state={'tables':tables,'calls':[],'error':None}
    def rpc(name,args):
        state['calls'].append((name,args))
        def execute():
            if state['error']:
                e=RuntimeError('private internal failure');e.code=state['error'];raise e
            return SimpleNamespace(data={'ok':True})
        return SimpleNamespace(execute=execute)
    db=SimpleNamespace(table=lambda n:Query(tables.get(n,[])),rpc=rpc)
    app=FastAPI();routes.install_club_site_routes(app,get_supabase_client=lambda:db)
    return TestClient(app),state,db

def test_public_snapshot_never_returns_draft_or_contact(setup):
    c,_,_=setup;r=c.get('/public/clubs/alpha/site');assert r.status_code==200
    assert r.json()['document']['name']=='Published Alpha'
    assert 'Private next version' not in r.text and 'owner@example.test' not in r.text
    assert r.headers['cache-control']=='no-store' and 'noindex' in r.headers['x-robots-tag']
    assert c.get('/public/clubs/beta/site').status_code==404
    assert c.get('/public/clubs/missing/site').status_code==404

def test_admin_cannot_read_or_publish_another_club(setup):
    c,s,_=setup
    assert c.get('/admin/clubs/beta/site').status_code==403
    assert c.post('/admin/clubs/beta/site/publish',json={'revision':5}).status_code==403
    assert not s['calls']

@pytest.mark.parametrize('change',[{'role':'operator'},{'revoked_at':'2020-01-01'}, {'expires_at':'2020-01-01T00:00:00Z'},{'user_id':'someone-else'}])
def test_operator_revoked_expired_foreign_identity_denied(setup,change):
    c,s,_=setup;s['tables']['admin_role_assignments'][0].update(change)
    assert c.get('/admin/clubs/alpha/site').status_code==403
    assert c.put('/admin/clubs/alpha/site',json={'revision':5,'document':SiteDocument(name='X').model_dump()}).status_code==403

def test_save_publication_identity_and_conflict_handling(setup):
    c,s,_=setup;payload={'revision':5,'document':SiteDocument(name='New draft').model_dump()}
    assert c.put('/admin/clubs/alpha/site',json=payload).status_code==200
    name,args=s['calls'][-1];assert name=='pcs_write_club_site'
    assert args['p_actor_id']=='verified' and args['p_club_id']=='alpha' and args['p_action']=='save'
    assert args['p_revision']==5 and args['p_document']['name']=='New draft'
    s['error']='40001';assert c.post('/admin/clubs/alpha/site/publish',json={'revision':5}).status_code==409
    s['error']='42501';assert c.post('/admin/clubs/alpha/site/publish',json={'revision':5}).status_code==403
    assert c.put('/admin/clubs/alpha/site',json={**payload,'actor_id':'forged'}).status_code==422


def test_leaderboard_settings_save_dates_as_json_without_changing_publication(setup):
    c, state, _ = setup
    settings = {
        'cards': ['most_matches', 'most_improved'], 'show_summary': False,
        'seasons': [{'id': 'winter', 'name': '2026–27', 'start_date': '2026-09-15',
                     'end_date': '2027-09-14', 'timezone': 'America/Mazatlan'}],
        'default_season_id': 'winter', 'min_games': 10,
    }
    doc = SiteDocument(name='Club', leaderboard=settings).model_dump(mode='json')
    response = c.put('/admin/clubs/alpha/site', json={'revision': 5, 'document': doc})
    assert response.status_code == 200
    saved = state['calls'][-1][1]['p_document']['leaderboard']
    assert saved == settings
    assert c.get('/public/clubs/alpha/site').json()['document']['leaderboard']['seasons'] == []


@pytest.mark.parametrize('settings', [
    {'cards': ['most_wins', 'most_wins']},
    {'cards': ['arbitrary_sql']},
    {'default_season_id': 'missing'},
    {'min_games': -1},
    {'seasons': [{'id': 'all', 'name': 'Reserved', 'start_date': '2026-09-15'}]},
    {'seasons': [{'id': 's', 'name': 'Bad dates', 'start_date': '2026-09-15', 'end_date': '2026-09-14'}]},
    {'seasons': [{'id': 's', 'name': 'Bad zone', 'start_date': '2026-09-15', 'timezone': 'not/a/zone'}]},
    {'seasons': [{'id': 's', 'name': 'A', 'start_date': '2026-09-15'}, {'id': 's', 'name': 'B', 'start_date': '2026-09-16'}]},
])
def test_invalid_leaderboard_settings_cannot_be_saved(setup, settings):
    c, state, _ = setup
    doc = SiteDocument(name='Club').model_dump(mode='json')
    doc['leaderboard'] = settings
    assert c.put('/admin/clubs/alpha/site', json={'revision': 5, 'document': doc}).status_code == 422
    assert not state['calls']

def test_page_visibility_round_trip_uses_published_snapshot_and_keeps_shared_access(setup):
    c, state, db = setup
    draft = SiteDocument(name='New draft', page_visibility={'players':'private', 'matches':'private'}).model_dump()
    assert c.put('/admin/clubs/alpha/site',json={'revision':5,'document':draft}).status_code == 200
    _, args = state['calls'][-1]
    assert args['p_document']['page_visibility'] == {'players':'private', 'matches':'private'}
    row = state['tables']['pcs_club_sites'][0]
    row['draft'] = draft
    assert c.get('/public/clubs/alpha/site').json()['document']['page_visibility'] == {}
    row['published'] = deepcopy(draft)
    shared = c.get('/public/clubs/alpha/site')
    assert shared.status_code == 200
    assert shared.json()['document']['page_visibility']['players'] == 'private'
    # A private section is unlisted, not authenticated. Keep direct public access.
    assert routes.published_site(db, 'alpha')['club_id'] == 'alpha'
    assert c.get('/public/clubs/beta/site').status_code == 404

@pytest.mark.parametrize('visibility', [{'admin':'private'}, {'players':'secret'}, {'players':False}, None])
def test_page_visibility_rejects_unknown_sections_or_states(setup,visibility):
    c, state, _ = setup
    document = SiteDocument(name='Alpha').model_dump()
    document['page_visibility'] = visibility
    assert c.put('/admin/clubs/alpha/site',json={'revision':5,'document':document}).status_code == 422
    assert not state['calls']

def test_creation_never_accepts_requested_owner_or_existing_club_access(setup):
    c,s,_=setup
    assert c.post('/clubs/create',json={'name':'Beta','slug':'new-beta','owner_id':'other'}).status_code==422
    assert c.post('/clubs/create',json={'name':'Beta','slug':'new-beta'}).status_code==201
    name,args=s['calls'][-1];assert name=='pcs_create_own_club'
    assert args['p_actor_id']=='verified' and args['p_document']['visibility']=='unlisted'
    s['error']='23505';assert c.post('/clubs/create',json={'name':'Alpha','slug':'alpha'}).status_code==409

@pytest.mark.parametrize('url',['javascript:alert(1)','//evil.test','http://insecure.test','https://safe.test\\@evil.test','data:image/svg+xml;base64,PHN2Zz4='])
def test_custom_content_cannot_execute_or_use_unsafe_links(url):
    with pytest.raises(ValidationError):SiteBlock(id='x',kind='button',url=url)

def test_duplicate_pages_and_image_accessibility_are_validated():
    with pytest.raises(ValidationError):SiteDocument(name='X',pages=[{'slug':'home','title':'Home'},{'slug':'home','title':'Duplicate'}])
    with pytest.raises(ValidationError):SiteBlock(id='x',kind='image',url='https://example.test/pic.png')
    assert SiteBlock(id='x',kind='image',url='https://example.test/pic.png',alt='Courts').alt=='Courts'

def encounter(meet,**kw):return {'id':str(uuid4()),'meet_id':meet,'division':'3.5','club_a':'alpha','club_b':'beta','games':[{'a':11,'b':8},{'a':9,'b':11},{'a':12,'b':10}],**kw}

def test_whole_league_standings_use_only_valid_encounters():
    mid=str(uuid4());season={'details':{'name':'League','start_date':'2026-01-01','end_date':'2026-12-31','timezone':'America/Mazatlan','divisions':['3.5','4.0']}}
    clubs=[{'id':c,'name':c.title()} for c in ['alpha','beta','gamma']]
    meets=[{'id':mid,'club_ids':['alpha','beta','gamma'],'host_club_id':'alpha','starts_at':'2026-09-15T15:00:00Z'}]
    results=ResultsDocument(results=[encounter(mid)])
    validate_results(results,season,clubs,meets)
    doc=public_document(season,clubs,meets,results.model_dump(mode='json'));standing=league_standings(doc)
    alpha=standing[0]['rows'][0];assert alpha['club_id']=='alpha' and alpha['wins']==1 and alpha['point_difference']==3
    beta=next(r for r in standing[0]['rows'] if r['club_id']=='beta');assert beta['losses']==1 and beta['games_won']==1
    assert len(standing[1]['rows'])==3 and all(r['played']==0 for r in standing[1]['rows'])
    for result in [encounter(str(uuid4())),encounter(mid,club_a='foreign'),encounter(mid,division='6.0'),encounter(mid,club_a='beta')]:
        with pytest.raises(HTTPException):validate_results(ResultsDocument(results=[result]),season,clubs,meets)
    with pytest.raises(HTTPException):validate_results(ResultsDocument(results=[encounter(mid),encounter(mid)]),season,clubs,meets)
    with pytest.raises(ValidationError):ResultsDocument(results=[encounter(mid,games=[{'a':11,'b':10}]*3)])

def test_central_public_club_guard_blocks_unpublished_records(setup,monkeypatch):
    from services.api import main
    _,state,db=setup;monkeypatch.setattr(main,'get_supabase_client',lambda:db)
    assert main.get_club('alpha')['name']=='Published Alpha'
    with pytest.raises(HTTPException) as err:main.get_club('beta')
    assert err.value.status_code==404
    state['tables']['pcs_club_sites']=[]
    with pytest.raises(HTTPException) as err:main.get_club('alpha')
    assert err.value.status_code==404
