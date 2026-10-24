"""URL round identity is part of the actual durable admin replay contract."""
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from services.api import admin_play_generator_routes as routes
from test_admin_live_ladder_operation_service import FakeSupabase

@pytest.mark.parametrize('suffix,service,extra', [
    ('scores','save_play_generator_round',{'scores':[{'match_id':'r1-c1','score_a':11,'score_b':7}]}),
    ('played','mark_play_generator_round_played',{}), ('skip','skip_play_generator_round',{'reason':'Not played'}),
    ('reopen','reopen_play_generator_round',{}), ('select','select_play_generator_round',{}),
])
def test_same_operation_key_cannot_replay_a_different_url_round(monkeypatch,suffix,service,extra):
    db,calls=FakeSupabase(),[]
    monkeypatch.setattr(routes,'_resolve_role_or_403',lambda **_kwargs: ('staff@example.test','admin'))
    monkeypatch.setattr(routes,'_require_write_gate',lambda: None)
    monkeypatch.setattr(routes,'get_play_generator_session',lambda *_args,**_kwargs:{'session':{'version':'v1'}})
    def mutate(_db,**kwargs):
        calls.append(kwargs['round_number'])
        return {'ok':True,'session':{'version':'v2','round_number':kwargs['round_number']}}
    monkeypatch.setattr(routes,service,mutate)
    app=FastAPI();routes.install_admin_play_generator_routes(app,get_supabase_client=lambda: db)
    client=TestClient(app);body={'expected_version':'v1','idempotency_key':'same-key-different-round',**extra}
    send=client.patch if suffix=='scores' else client.post
    path='/admin/clubs/club/play-generators/sessions/session-a/rounds/{}/'+suffix
    first=send(path.format(1),json=body);assert first.status_code==200,first.text
    replay=send(path.format(1),json=body);assert replay.status_code==200,replay.text
    assert replay.json()['idempotent_replay'] is True
    other=send(path.format(2),json=body);assert other.status_code==409,other.text
    assert calls==[1]
    assert len(db.storage['live_ladder_admin_operations'])==1
    assert db.storage['live_ladder_admin_operations'][0]['request_json']['round_number']==1

def test_public_uncertainty_retains_503_and_exposes_exact_retry_metadata():
    from services.api.main import _raise_public_live_write_error
    from jupr_app.services.public_live_operation_service import PublicLiveRecoveryRequiredError
    with pytest.raises(HTTPException) as caught: _raise_public_live_write_error(PublicLiveRecoveryRequiredError('Retry this same action.'))
    assert caught.value.status_code==503
    assert caught.value.detail=={'message':'Retry this same action.','code':'RECOVERY_REQUIRED','recovery_required':True}
