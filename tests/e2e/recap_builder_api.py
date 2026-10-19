"""Loopback-only browser rehearsal. Real recap routes/services; synthetic source results and auth."""
import os
import sys
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ['JUPR_ENV'] = 'test'
os.environ['JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP'] = '1'
os.environ['SUPABASE_SERVICE_ROLE_KEY'] = 'loopback-test-only'
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tests.test_admin_match_log_service import FakeSupabase, FakeQuery
from services.api import admin_weekly_recap_routes as routes
from jupr_app.services import admin_weekly_recap_service as service
from jupr_app.services.public_weekly_recap_service import build_public_weekly_recaps, build_weekly_recap_pdf_bytes
from services.api.club_site_models import default_site

class Query(FakeQuery):
    def execute(self):
        if self.update_payload is not None and self.table_name == 'weekly_recaps':
            matched = self._apply_filters(self.storage.get(self.table_name, []))
            if matched:
                self.update_payload['row_version'] = matched[0]['row_version'] + 1
        return super().execute()

db = FakeSupabase({'weekly_recaps': [], 'admin_activity_log': [], 'club_recap_preferences': [], 'leagues_metadata': [{'club_id':'club','league_name':'Fall League'}]})
db.table = lambda name: Query(db.tables, name)
source = json.loads((ROOT/'apps/web/tests/fixtures/weekly-report.json').read_text())
source['report']['club_slug'] = 'test'
source['report']['club_name'] = 'Recap test club'
def generate(ctx, start_date, end_date, **kw):
    value = deepcopy(source)
    value.update(start_date=start_date.isoformat(), end_date=end_date.isoformat(), week_start=start_date.isoformat(), week_end=end_date.isoformat())
    return value
service._recap_context = lambda *a, **kw: SimpleNamespace()
service.compute_weekly_recap = generate
service.get_spotlight_candidates = lambda *a, **kw: {}
def authenticate(value):
    if value != 'Bearer local-reviewer':
        raise HTTPException(401, 'Local test session required')
    return SimpleNamespace(email='reviewer@example.invalid', user_id='local-reviewer')
routes.authenticate_bearer = authenticate
routes.resolve_admin_role = lambda **kw: SimpleNamespace(role='administrator')
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['http://127.0.0.1:3042'], allow_methods=['*'], allow_headers=['*'])
routes.install_admin_weekly_recap_routes(app, get_supabase_client=lambda: db)
@app.get('/admin/auth/capabilities')
def capabilities():
    return {'authorized':True,'user':{'email':'reviewer@example.invalid'}, 'assignments':[{'club_id':'club','role':'administrator','permissions':['enter_scores','manage_players']}]}
@app.get('/admin/auth/workspaces')
def workspaces():
    return {'workspaces':[{'club_id':'club','club_slug':'test','club_name':'Recap test club','roles':['administrator']}]}
@app.get('/public/clubs/{slug}/site')
def site(slug:str):
    return {'club_id':'club','slug':slug,'document':default_site({'name':'Recap test club'}),'published_at':'2026-10-01'}
@app.get('/clubs/{slug}/weekly-recaps')
def recaps(slug:str, week_start:str|None=None):
    return {'club':{'id':'club','slug':'test','name':'Recap test club'},**build_public_weekly_recaps(db,club_id='club',week_start=week_start)}
@app.get('/fixture-result')
def result():
    return db.tables
