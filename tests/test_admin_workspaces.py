from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from services.api import admin_auth_routes as routes


class Query:
    def __init__(self, rows, calls):
        self.rows, self.calls = rows, calls
        self.filters = []

    def select(self, columns):
        self.columns = columns
        return self

    def eq(self, key, value):
        self.filters.append(lambda row: row.get(key) == value)
        return self

    def in_(self, key, values):
        self.calls.append((key, values))
        self.filters.append(lambda row: row.get(key) in values)
        return self

    def order(self, _):
        return self

    def execute(self):
        rows = [row for row in self.rows if all(f(row) for f in self.filters)]
        if self.columns != '*':
            rows = [{k: row[k] for k in self.columns.split(',')} for row in rows]
        return SimpleNamespace(data=rows)


def client(monkeypatch, assignments, *, unavailable=False):
    calls = []
    monkeypatch.setattr(routes, 'authenticate_bearer', lambda _: SimpleNamespace(user_id='user-a', email='staff@example.test'))
    def table(name):
        if name == 'admin_role_assignments':
            return Query(assignments, calls)
        assert name == 'clubs'
        if unavailable:
            raise RuntimeError('secret database error')
        return Query([dict(id=id, slug=id, name=id.title(), support_email='private@example.test') for id in ['alpha','beta','private']], calls)
    app = FastAPI()
    routes.install_admin_auth_routes(app, get_supabase_client=lambda: SimpleNamespace(table=table))
    return TestClient(app), calls


def assignment(club, **overrides):
    return dict(club_id=club, email='staff@example.test', role='administrator', user_id='user-a', **overrides)


def test_workspace_list_only_contains_current_users_bound_clubs(monkeypatch):
    c, calls = client(monkeypatch, [assignment('alpha'), assignment('beta'), {**assignment('private'), 'email':'another@example.test'}])
    response = c.get('/admin/auth/workspaces')
    assert response.status_code == 200
    assert response.json()['workspaces'] == [
        dict(club_id=id, club_slug=id, club_name=id.title(), roles=['administrator']) for id in ['alpha','beta']
    ]
    assert calls == [('id', ['alpha','beta'])]
    assert 'private' not in response.text and 'support_email' not in response.text


@pytest.mark.parametrize('invalid', [dict(revoked_at='2026-01-01'), dict(expires_at='2020-01-01T00:00:00Z'), dict(user_id='other-user'), dict(role='invalid')])
def test_revoked_expired_mismatched_and_invalid_assignments_do_not_disclose_club(monkeypatch, invalid):
    c, calls = client(monkeypatch, [{**assignment('private'), **invalid}])
    r = c.get('/admin/auth/workspaces')
    assert r.status_code == 403 and not calls
    assert 'private' not in r.text


def test_operator_can_find_only_their_club_without_acquiring_administrator(monkeypatch):
    c, _ = client(monkeypatch, [{**assignment('beta'), 'role':'operator', 'scopes':[{'kind':'program_type','program_type':'leagues'}]}])
    assert c.get('/admin/auth/workspaces').json()['workspaces'][0]['roles'] == ['operator']


def test_backend_failure_and_bad_jwt_fail_closed(monkeypatch):
    c, _ = client(monkeypatch, [assignment('alpha')], unavailable=True)
    r = c.get('/admin/auth/workspaces')
    assert r.status_code == 503 and 'secret' not in r.text
    monkeypatch.setattr(routes, 'authenticate_bearer', lambda _: (_ for _ in ()).throw(HTTPException(401, 'invalid token')))
    assert c.get('/admin/auth/workspaces').status_code == 401
