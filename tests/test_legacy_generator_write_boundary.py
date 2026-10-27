"""Legacy live-session writes cannot bypass the generator or Hub workflow."""
from copy import deepcopy
import pytest
from test_generator_round_controls_service import GeneratorSession
from test_public_play_generator_service import matches
from jupr_app.services import public_live_write_service as legacy

@pytest.mark.parametrize('action',['scores','advance','substitute','complete'])
@pytest.mark.parametrize('marker',['public_mode','admin_mode','kind','event_kind','hub'])
def test_generator_and_hub_sessions_reject_every_legacy_write(action,marker):
    game=GeneratorSession('public');row=game.stored;state=row['state']
    state.pop('generator_kind',None);state['mode']='legacy_public'
    state['page_state']['event'].pop('generatorKind',None)
    if marker.endswith('_mode'): state['mode']=marker.removesuffix('_mode')+'_play_generator'
    elif marker=='kind': state['generator_kind']='round_robin'
    elif marker=='event_kind': state['page_state']['event']['generatorKind']='round_robin'
    else: state['round_robin_hub_id']='hub-id'
    state['official_publish']={'published_match_ids':[123]};before=deepcopy(row)
    args=dict(club_id='club',session_key=row['session_key'],edit_token=game.edit_token,
        expected_version=row['version'],idempotency_key=f'legacy-block-{action}-0001',requester_hash='a'*64)
    call={'scores':legacy.update_public_live_scores,'advance':legacy.advance_public_live_session,
        'substitute':legacy.substitute_public_live_participant,'complete':legacy.complete_public_live_session}[action]
    if action=='scores': args['scores']=[{'match_id':matches(game.session['event']['rounds'][0])[0]['id'],'score_a':0,'score_b':11}]
    elif action=='substitute': args.update(scope='round',round_number=1,original_participant_id='p-1',substitute_name='Other')
    with pytest.raises(legacy.PublicLiveSessionError,match='Generator'): call(game.db,**args)
    assert game.stored==before
    assert game.db.db['public_live_operations'][-1]['status']=='rejected'
    assert not game.db.db.get('matches')
