"""Navigation must not wait for unused counts or make duplicate publication reads."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api import admin_tournament_routes, admin_tournament_setup_routes


@pytest.mark.parametrize('module,installer,path', [
    (admin_tournament_routes, 'install_admin_tournament_routes', 'admin'),
    (admin_tournament_setup_routes, 'install_admin_tournament_setup_routes', 'setup'),
])
def test_navigation_status_keeps_capabilities_without_database_count(monkeypatch, module, installer, path):
    monkeypatch.setenv('JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS', '1')
    calls = []

    def database():
        calls.append('database')
        raise RuntimeError('The database is busy')

    app = FastAPI()
    getattr(module, installer)(app, get_supabase_client=database)
    client = TestClient(app, raise_server_exceptions=False)
    url = f'/admin/clubs/fixture/tournaments/{path}/status'
    response = client.get(url + '?include_count=false')
    assert response.status_code == 200
    payload = response.json()
    assert payload['enabled'] is True
    assert payload['tournament_count'] is None
    assert 'mutation_runtime' in payload
    assert calls == []

    # Existing count callers retain their read; failure is never reported as 0.
    assert client.get(url).status_code == 500
    assert calls == ['database']

    monkeypatch.setenv('JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS', '0')
    guarded = client.get(url + '?include_count=false').json()
    assert guarded['enabled'] is False
    assert calls == ['database']
