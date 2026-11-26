"""Loopback browser fixture: real routes/domain + real SQL; synthetic auth only."""
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
import json
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ['JUPR_ENV'] = 'test'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from services.api import managed_interclub_routes as routes
from services.api.club_site_models import default_site

def query(sql, params=()):
    reply = httpx.post('http://127.0.0.1:8144', json={'sql': sql, 'params': params}, timeout=10, trust_env=False).json()
    if 'code' in reply:
        error = RuntimeError(reply['message']); error.code = reply['code']; raise error
    return reply['data']

def identifier(value):
    assert re.fullmatch(r'[a-z_][a-z_0-9]*', value)
    return value

class Read:
    def __init__(self, table): self.table_name, self.fields, self.where = identifier(table), '*', []
    def select(self, fields):
        self.fields = ','.join(identifier(f) for f in fields.split(',')) if fields != '*' else '*'
        return self
    def eq(self, key, value): self.where.append((identifier(key), value)); return self
    def execute(self):
        where = ' and '.join(f'{key}=${i+1}' for i,(key,_) in enumerate(self.where))
        return SimpleNamespace(data=query(f'select {self.fields} from {self.table_name}' + (' where '+where if where else ''), [v for _,v in self.where]))

class DB:
    def table(self, table): return Read(table)
    def rpc(self, name, values):
        arguments = {
            'pcs_managed_interclub_write': ('p_actor_id', 'p_actor_email', 'p_club_id', 'p_season_id', 'p_meet_id', 'p_action', 'p_revision', 'p_payload', 'p_token_hash'),
            'pcs_managed_interclub_club_link': ('p_actor_id', 'p_actor_email', 'p_organizer_club_id', 'p_season_id', 'p_entry_club_id', 'p_token_hash'),
            'pcs_managed_interclub_club_entry': ('p_token_hash', 'p_expected_revision', 'p_registration'),
        }[name]
        params = []
        placeholders = []
        for index, key in enumerate(arguments, start=1):
            value = values.get(key)
            is_json = key in {'p_payload', 'p_registration'}
            params.append(json.dumps(value) if is_json and value is not None else value)
            placeholders.append(f'${index}' + ('::jsonb' if is_json else ''))
        sql = f"select {name}({','.join(placeholders)}) as result"
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=query(sql, params)[0]['result']))

def admin(factory, authorization, club_id):
    if authorization != 'Bearer local-reviewer' or club_id != 'tres_palapas': raise HTTPException(403, 'Fixture organizer only')
    return SimpleNamespace(user_id='10000000-0000-4000-8000-000000000001', email='joe@example.invalid')

routes.site_administrator = admin
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:3044'], allow_methods=['*'], allow_headers=['*'])
routes.install_managed_interclub_routes(app, get_supabase_client=DB)

@app.get('/admin/auth/capabilities')
def capabilities(): return {'authorized': True, 'user': {'email':'joe@example.invalid'}, 'assignments':[{'club_id':'tres_palapas','role':'administrator','permissions':['enter_scores','manage_players']}]}
@app.get('/admin/auth/workspaces')
def workspaces(): return {'workspaces':[{'club_id':'tres_palapas','club_slug':'test','club_name':'Tres Palapas','roles':['administrator']}]}
@app.get('/admin/clubs/{club_id}/interclub/registrations')
def registrations(club_id: str): return {'seasons':[]}
@app.get('/public/clubs/{slug}/site')
@app.get('/public/clubs/{slug}/site/shell')
def site(slug: str): return {'club_id':'tres_palapas','slug':slug,'document':default_site({'name':'Tres Palapas'}),'published_at':'2026-11-14'}
@app.get('/fixture-result')
def result():
    return {name:query('select * from '+name) for name in ('pcs_managed_interclub_seasons','pcs_managed_interclub_meets','pcs_managed_interclub_audit','clubs')}
