"""Score errors fail at every HTTP boundary before database or lease access."""
from copy import deepcopy
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from services.api import admin_play_generator_routes as admin
from services.api import public_play_generator_routes as public
from services.api import shared_play_generator_routes as shared
from services.api.admin_round_robin_hub_routes import HubCommandRequest
from jupr_app.services.admin_live_ladder_operation_service import LiveLadderUncertainError

@pytest.fixture
def score_api():
    app, calls = FastAPI(), []
    def no_data(*_args, **_kwargs):
        calls.append('data')
        raise AssertionError('Malformed input reached database/auth service execution')
    no_op = lambda: None
    admin.install_admin_play_generator_routes(app, get_supabase_client=no_data)
    public.install_public_play_generator_routes(app, get_club=no_data, get_supabase_client=no_data,
        public_club_payload=no_data, require_public_writes=no_op, require_service_role=no_op,
        requester_hash=no_data, raise_public_error=no_data, public_writes_enabled=lambda: True,
        service_role_configured=lambda: True)
    shared.install_shared_play_generator_routes(app, get_club=no_data, get_supabase_client=no_data,
        require_public_writes=no_op, require_service_role=no_op, require_admin_writes=no_op,
        requester_hash=no_data, raise_public_error=no_data)
    return TestClient(app), calls

def request(surface, scores):
    body = {'expected_version': 'version-1', 'idempotency_key': 'score-validation-0001', 'scores': scores}
    if surface == 'admin': root = '/admin/clubs/club/play-generators'
    else:
        root = f"/clubs/club/{'shared-play-generators' if surface == 'shared' else 'play-generators'}"
        body['edit_token'] = 'organizer-token'
        if surface == 'public': body['expected_version'] = 1
    return f'{root}/sessions/rr/rounds/1/scores', body

@pytest.mark.parametrize('surface', ['admin','public','shared'])
@pytest.mark.parametrize('scores', [
    [{'match_id':'r1-c1'}], [{'match_id':'r1-c1','score_a':None}], [{'match_id':'r1-c1','score_b':None}],
    [{'match_id':'r1-c1','score_a':True,'score_b':False}],
    [{'match_id':'r1-c1','score_a':'11','score_b':'7'}],
    [{'match_id':'r1-c1','score_a':11.0,'score_b':7}],
    [{'match_id':'r1-c1','score_a':11.5,'score_b':7}],
    [{'match_id':'r1-c1','score_a':11,'score_b':None}],
    [{'match_id':'r1-c1','score_a':None,'score_b':7}],
    [{'match_id':'r1-c1','score_a':100,'score_b':7}],
    [{'match_id':'r1-c1','score_a':-1,'score_b':7}],
    [{'match_id':'r1-c1','score_a':11,'score_b':7},{'match_id':'r1-c1','score_a':0,'score_b':11}],
])
def test_malformed_scores_are_422_before_database_or_operation_lease(score_api, surface, scores):
    client,calls = score_api
    path,body = request(surface,scores)
    response = client.patch(path,json=body)
    assert response.status_code == 422, response.text
    assert calls == []

@pytest.mark.parametrize('surface,model', [('admin',admin.GeneratorScoresRequest),('public',public.PublicGeneratorScoresRequest),('shared',shared.SharedScores)])
def test_played_and_explicit_not_played_inputs_are_preserved(surface,model):
    scores = [{'match_id':'played','score_a':11,'score_b':10},{'match_id':'unplayed','score_a':None,'score_b':None}]
    _,body = request(surface,scores)
    assert [r.model_dump() for r in model.model_validate(body).scores] == scores

@pytest.mark.parametrize('model,version,extra', [(admin.GeneratorPlayoffRequest,'v1',{}),(public.PublicGeneratorPlayoffRequest,1,{'edit_token':'token'}),(shared.SharedPlayoff,'v1',{'edit_token':'token'})])
def test_single_championship_is_available_on_every_generator_surface(model,version,extra):
    assert model(expected_version=version,idempotency_key='top-four-0001',playoff_format='top_four',**extra).playoff_format == 'top_four'

def test_uncertain_admin_result_is_structured_for_exact_retry_ui():
    with pytest.raises(HTTPException) as caught: admin._handle(LiveLadderUncertainError('Verify this saved action.'))
    assert caught.value.status_code == 409
    assert caught.value.detail == {'message':'Verify this saved action.','code':'RECOVERY_REQUIRED','recovery_required':True}

@pytest.mark.parametrize('action,payload', [
    ('transfer_player',{'source_session_key':'a','participant_id':'p1','destination_session_key':'b'}),
    ('rejoin_player',{'session_key':'a','participant_id':'p1'}),
    ('resolve_game',{'session_key':'a','match_id':'r1-c1','resolution':'not_played','reason':'Court unavailable'}),
    ('score_and_release',{'session_key':'a','match_id':'r1-c1','court_id':'c1','score_a':None,'score_b':None}),
    ('release_game',{'court_id':'c1'}),
    ('resolve_identities',{'resolution':'same','participants':[{'session_key':'a','participant_id':'p1'},{'session_key':'b','participant_id':'p4'}]}),
    ('set_match_lineup',{'session_key':'a','match_id':'r2-m1','side_a':['p1','p2'],'side_b':['p3','p4']}),
    ('remove_players',{'session_key':'a','participant_ids':['p1','p2']}),
    ('finish_group',{'session_key':'a','finish_after_round':6}),
    ('create_extra_game',{'side_a':[{'session_key':'a','participant_id':'p1'},{'session_key':'b','participant_id':'p1'}],
                         'side_b':[{'session_key':'b','participant_id':'p2'},{'session_key':'b','participant_id':'p3'}],
                         'standings_policy':'exhibition'}),
])
def test_hub_payload_validation_preserves_exact_retry_fingerprint(action,payload):
    before=deepcopy(payload)
    assert HubCommandRequest(action=action,payload=payload,expected_version=1,idempotency_key='hub-valid-contract-0001').payload == before

@pytest.mark.parametrize('action,payload', [
    ('add_group',{'session_key':['a']}), ('add_player',{'session_key':'a','name':'Guest','player_id':True}),
    ('remove_player',{'session_key':'a','participant_id':{'id':'p1'}}),
    ('transfer_player',{'source_session_key':'a','participant_id':'p1'}),
    ('resolve_game',{'session_key':'a','match_id':'m1','resolution':'erase_results'}),
    ('score_and_release',{'session_key':'a','match_id':'m1','court_id':'c1','score_a':True,'score_b':False}),
    ('score_and_release',{'session_key':'a','match_id':'m1','court_id':'c1','score_a':11}),
    ('score_and_release',{'session_key':'a','match_id':'m1','court_id':'c1'}),
    ('score_and_release',{'session_key':'a','match_id':'m1','court_id':'c1','score_a':None}),
    ('score_and_release',{'session_key':'a','match_id':'m1','court_id':'c1','score_b':None}),
    ('set_registration',{'open':'false'}),
    ('resolve_identities',{'resolution':'same','participants':[{'session_key':'a','participant_id':'p1'}]}),
    ('review_checkin',{'checkin_id':'c1','disposition':'existing'}),
    ('edit_match_players',{'session_key':'a','match_id':'m1','participant_id':'p1','replacement_participant_id':'p2','name':'Another'}),
    ('auto_fill_courts',{'excluded_session_keys':['b']}),
    ('set_match_lineup',{'session_key':'a','match_id':'m1','side_a':['p1','p2'],'side_b':['p1','p3']}),
    ('set_match_lineup',{'session_key':'a','match_id':'m1','side_a':['p1'],'side_b':['p2','p3']}),
    ('remove_players',{'session_key':'a','participant_ids':[]}),
    ('remove_players',{'session_key':'a','participant_ids':['p1','p1']}),
    ('remove_players',{'session_key':'a','participant_ids':['p1'],'effective_round':0}),
    ('finish_group',{'session_key':'a','finish_after_round':True}),
    ('finish_group',{'session_key':'a','finish_after_round':-1}),
    ('create_extra_game',{'side_a':[{'session_key':'a','participant_id':'p1'},{'session_key':'a','participant_id':'p1'}],
                         'side_b':[{'session_key':'b','participant_id':'p2'},{'session_key':'b','participant_id':'p3'}],
                         'standings_policy':'exhibition'}),
    ('create_extra_game',{'side_a':[{'session_key':'a','participant_id':'p1'},{'session_key':'b','participant_id':'p1'}],
                         'side_b':[{'session_key':'b','participant_id':'p2'},{'session_key':'b','participant_id':'p3'}],
                         'standings_policy':'home_groups'}),
])
def test_hub_action_payloads_reject_malformed_or_internal_fields(action,payload):
    with pytest.raises(ValidationError): HubCommandRequest(action=action,payload=payload,expected_version=1,idempotency_key='hub-invalid-contract-0001')
