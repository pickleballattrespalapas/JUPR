from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from jupr_app.domain.admin.roles import has_permission, resolve_admin_role
from jupr_app.domain.admin.staff_policy import assignment_active, permits, validate_scopes
from services.api.staff_access import install_staff_access


class Query:
    def __init__(self, data): self.data, self.filters = data, []
    def select(self, *_): return self
    def eq(self, key, value): self.filters.append((key, str(value))); return self
    def limit(self, *_): return self
    def execute(self):
        return SimpleNamespace(data=[r for r in self.data if all(str(r.get(k)) == v for k, v in self.filters)])


class DB:
    def __init__(self, assignment):
        self.tables = {
            'admin_role_assignments': [assignment],
            'leagues_metadata': [dict(club_id='tres', league_name='Tuesday', status='active'), dict(club_id='tres', league_name='Friday', status='active')],
        }
    def table(self, name): return Query(self.tables.get(name, []))


@pytest.fixture
def setup(monkeypatch):
    assignment = dict(club_id='tres', email='staff@example.com', user_id='u1', role='operator', scopes=[dict(kind='resource',program_type='leagues',resource_id='Tuesday')])
    db = DB(assignment)
    def authorize(club_id):
        result = resolve_admin_role(supabase=db, club_id=club_id, email='staff@example.com', user_id='u1', allowlist=set())
        if not result.assigned: raise HTTPException(403)
    app = FastAPI()
    @app.patch('/admin/clubs/{club_id}/league-manager/leagues/{league_name}')
    def edit(club_id: str, league_name: str):
        authorize(club_id)
        if not has_permission('operator','manage_matches'): raise HTTPException(403)
        return {'ok': True}
    @app.get('/admin/clubs/{club_id}/league-manager/leagues')
    def listing(club_id: str):
        authorize(club_id)
        return {'leagues': db.tables['leagues_metadata'], 'total_count': 2}
    @app.post('/admin/clubs/{club_id}/unknown-module')
    def unknown(club_id: str):
        authorize(club_id)
        return {'ok': True}
    @app.patch('/admin/clubs/{club_id}/players/editor/players/{player_id}')
    def edit_player(club_id: str, player_id: int):
        authorize(club_id)
        return {'ok': True}
    @app.delete('/admin/clubs/{club_id}/league-manager/leagues/{league_name}')
    def delete(club_id: str, league_name: str):
        authorize(club_id)
        return {'ok': True}
    install_staff_access(app, get_supabase_client=lambda: db)
    return TestClient(app), db, assignment


def call(client, method, path, data=None):
    return client.request(method, '/admin/clubs/tres/' + path, json=data or {}, headers={'Authorization':'Bearer test'})


def test_scoped_operator_can_edit_assigned_league_only(setup):
    client, _, _ = setup
    assert call(client,'PATCH','league-manager/leagues/Tuesday').status_code == 200
    assert call(client,'PATCH','league-manager/leagues/Friday').status_code == 403
    assert not has_permission('operator','manage_matches')  # context reset; legacy UI cannot bypass scopes


def test_scoped_collections_hide_other_programs(setup):
    client, _, _ = setup
    result = call(client,'GET','league-manager/leagues')
    assert result.status_code == 200
    assert [r['league_name'] for r in result.json()['leagues']] == ['Tuesday']
    assert 'total_count' not in result.json()


@pytest.mark.parametrize('change', [{'revoked_at':'2026-01-01T00:00:00Z'}, {'expires_at':'2026-01-01T00:00:00Z'}, {'expires_at':'bad-date'}])
def test_inactive_assignments_deny_reads_and_writes(setup,change):
    client, _, assignment = setup
    assignment.update(change)
    assert call(client,'GET','league-manager/leagues').status_code == 403
    assert call(client,'PATCH','league-manager/leagues/Tuesday').status_code == 403


def test_admin_only_actions_stay_denied_even_with_club_scope(setup):
    client, _, assignment = setup
    assignment['scopes'] = [{'kind':'club'}]
    assert call(client,'DELETE','league-manager/leagues/Tuesday').status_code == 403
    assert call(client,'POST','unknown-module').status_code == 403
    assert call(client,'PATCH','players/editor/players/1', {'rating_jupr':5}).status_code == 403
    assert call(client,'PATCH','players/editor/players/1', {'name':'Updated name'}).status_code == 200


def test_published_league_cannot_be_reopened_by_operator(setup):
    client, db, _ = setup
    db.tables['leagues_metadata'][0]['status'] = 'published'
    assert call(client,'PATCH','league-manager/leagues/Tuesday', {'status':'active'}).status_code == 403


def test_scope_validation_and_expiry():
    with pytest.raises(ValueError): validate_scopes([])
    with pytest.raises(ValueError): validate_scopes([{'kind':'resource','program_type':'leagues'}])
    assert assignment_active({'expires_at':(datetime.now(timezone.utc)+timedelta(hours=1)).isoformat()})
    assert not permits([{'kind':'resource','program_type':'leagues','resource_id':'A'}], 'tournaments', {'A'})
