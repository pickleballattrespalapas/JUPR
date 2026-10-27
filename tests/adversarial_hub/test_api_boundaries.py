"""Audited API boundaries reject malformed input without changing state."""
from copy import deepcopy
import pytest
from fastapi.testclient import TestClient
from tests.test_api_contract_round_robin_hub import api, BASE, HEADERS
from tests.test_admin_round_robin_hub_service import create, command, session

def setup_hub(api):
    client,db=api;session(db);hub=command(db,create(db),'add_group',{'session_key':'rr-a'})
    return TestClient(client.app,raise_server_exceptions=False),db,hub

@pytest.mark.parametrize('bad_court',[None,True,1,'Court 1',[]])
def test_malformed_court_entry_is_rejected_without_writes(api,bad_court):
    client,db,hub=setup_hub(api);before=deepcopy(db.storage)
    response=client.post(f"{BASE}/{hub['hub']['id']}/commands",headers=HEADERS,json={
        'action':'configure_courts','payload':{'courts':[bad_court]},'expected_version':hub['hub']['version'],
        'idempotency_key':'malformed-court-probe'})
    assert response.status_code==422,response.text
    assert db.storage==before

@pytest.mark.parametrize('invalid_open',['false','true',0,1,None,[],{}])
def test_nonboolean_court_open_is_rejected_without_writes(api,invalid_open):
    client,db,hub=setup_hub(api);before=deepcopy(db.storage)
    response=client.post(f"{BASE}/{hub['hub']['id']}/commands",headers=HEADERS,json={
        'action':'configure_courts','payload':{'courts':[{'id':'c1','label':'Court 1','open':invalid_open}]},
        'expected_version':hub['hub']['version'],'idempotency_key':'string-false-court-probe'})
    assert response.status_code==422,response.text
    assert db.storage==before

@pytest.mark.parametrize('action,payload',[
    ('add_player',{'session_key':'foreign-session','name':'Audit Guest'}),
    ('remove_player',{'session_key':'foreign-session','participant_id':'p1'}),
    ('assign_next_court',{'session_key':'foreign-session','match_id':'m1'}),
    ('score_and_release',{'court_id':'c1','session_key':'foreign-session','match_id':'m1','score_a':11,'score_b':5}),
])
def test_foreign_session_commands_reject_without_writes(api,action,payload):
    client,db,hub=setup_hub(api);before=deepcopy(db.storage)
    response=client.post(f"{BASE}/{hub['hub']['id']}/commands",headers=HEADERS,json={'action':action,'payload':payload,
        'expected_version':hub['hub']['version'],'idempotency_key':'foreign-session-probe'})
    assert response.status_code in (400,404,409),response.text
    assert db.storage==before

@pytest.mark.parametrize('action,payload',[
    ('configure_courts',{'courts':[{'id':'c1','label':'Court 1','open':False}]}),
    ('rename_hub',{'title':'Should not be written'}),('add_player',{'session_key':'rr-a','name':'Audit Guest'}),
    ('remove_group',{'session_key':'rr-a'}),('finish_hub',{}),
])
def test_stale_command_does_not_partially_mutate(api,action,payload):
    client,db,hub=setup_hub(api);before=deepcopy(db.storage)
    response=client.post(f"{BASE}/{hub['hub']['id']}/commands",headers=HEADERS,json={'action':action,'payload':payload,
        'expected_version':hub['hub']['version']-1,'idempotency_key':'stale-command-probe'})
    assert response.status_code==409,response.text
    assert db.storage==before
