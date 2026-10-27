"""Owner and handoff retries use immutable proof without repeating an old edit."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
from test_generator_round_controls_service import GeneratorSession
from test_public_play_generator_service import matches
from jupr_app.services import generator_sharing_service as sharing
from jupr_app.services import public_play_generator_service as public
from jupr_app.services.public_live_operation_service import PublicLiveConflictError, PublicLiveRecoveryRequiredError, hash_edit_token

class Runner:
    def __init__(self,audience):
        self.owner=audience=='owner'
        self.game=GeneratorSession('admin' if audience=='shared_admin' else 'public')
        self.db=self.game.db
        self.row.setdefault('version',1)
        if self.owner: self.token=self.game.edit_token
        else:
            self.db.db['clubs']=[{'id':'club','slug':'club'}]
            result=sharing.manage_generator_sharing(self.db,club_id='club',session_key=self.game.session['session_key'],
                action='handoff',expected_version=self.game.session['version'],admin_authorized=self.game.audience=='admin',
                owner_token=getattr(self.game,'edit_token',None))
            self.token=result['sharing']['handoff_token']
        self.version=self.current_version()
        self.body={'round_number':1,'scores':[{'match_id':m['id'],'score_a':11,'score_b':7} for m in matches(self.game.session['event']['rounds'][0])]}
    @property
    def row(self): return self.game.stored
    def current_version(self): return int(self.row.get('version') or 1) if self.owner else self.row['updated_at']
    def save(self,*,key='receipt-save-0001',version=None,body=None):
        args=dict(club_id='club',session_key=self.row['session_key'],edit_token=self.token,
            expected_version=self.version if version is None else version,idempotency_key=key,requester_hash='b'*64)
        if self.owner: return public.save_public_play_generator_round(self.db,**args,**(body or self.body))
        return sharing.mutate_shared_generator(self.db,**args,action='scores',body=body or self.body)

@pytest.fixture(params=['owner','shared_public','shared_admin'])
def runner(request,monkeypatch):
    monkeypatch.setenv('JUPR_PUBLIC_LIVE_TOKEN_SECRET','x'*48)
    return Runner(request.param)

def lose_one_rpc_response(runner,monkeypatch):
    rpc=runner.db.rpc;lost=False
    def request(name,params):
        original=rpc(name,params)
        def execute():
            nonlocal lost
            result=original.execute()
            if not lost:
                lost=True
                raise RuntimeError('simulated transport loss after atomic commit')
            return result
        return SimpleNamespace(execute=execute)
    monkeypatch.setattr(runner.db,'rpc',request)

def test_committed_response_loss_is_recovered_and_commits_exactly_once(runner,monkeypatch):
    version=int(runner.row.get('version') or 1)
    lose_one_rpc_response(runner,monkeypatch)
    first=runner.save();committed=deepcopy(runner.row);replay=runner.save()
    assert first['confirmed_commit'] is True
    assert replay['confirmed_commit'] is replay['idempotent_replay'] is True
    assert runner.row==committed
    assert int(runner.row['version'])==version+1
    assert len(runner.db.db['play_generator_mutation_receipts'])==1

def test_lost_response_retry_survives_a_later_mutation(runner,monkeypatch):
    with monkeypatch.context() as pending:
        lose_one_rpc_response(runner,pending)
        pending.setattr(public,'_recovered_mutation_row',lambda *_args,**_kwargs:None)
        with pytest.raises(PublicLiveRecoveryRequiredError): runner.save()
    original=runner.db.db['public_live_operations'][-1]
    assert original['status']=='recovery_required'
    key=original['operation_key'];body=deepcopy(runner.body)
    for score in body['scores']: score['score_b']=9
    runner.save(key='receipt-later-0001',version=runner.current_version(),body=body)
    assert runner.row['last_operation_key']!=key
    current=deepcopy(runner.row);replay=runner.save()
    assert replay['confirmed_commit'] is replay['idempotent_replay'] is True
    assert runner.row==current
    assert replay['session']['event']==runner.game.reload()['event']
    assert len(runner.db.db['play_generator_mutation_receipts'])==2

@pytest.mark.parametrize('field',['club_id','session_key','request_fingerprint'])
def test_receipt_from_another_session_or_fingerprint_cannot_confirm_retry(runner,field):
    runner.save();operation=runner.db.db['public_live_operations'][-1];operation['status']='recovery_required'
    runner.db.db['play_generator_mutation_receipts'][0][field]='wrong-value';before=deepcopy(runner.row)
    with pytest.raises(PublicLiveConflictError,match='different session or request'): runner.save()
    assert runner.row==before
    assert operation['status']=='recovery_required'

def test_reused_key_with_a_changed_request_cannot_replay_or_overwrite(runner):
    runner.save();body=deepcopy(runner.body);body['scores'][0]['score_b']=8;before=deepcopy(runner.row)
    with pytest.raises(PublicLiveConflictError): runner.save(body=body)
    assert runner.row==before
    assert len(runner.db.db['play_generator_mutation_receipts'])==1

def test_replay_still_requires_the_current_owner_or_handoff_token(runner):
    runner.save()
    if runner.owner: runner.row['edit_token_hash']=hash_edit_token('replacement-owner')
    else: runner.row['state']['generator_sharing'].pop('handoff_nonce')
    before=deepcopy(runner.db.db)
    with pytest.raises(PermissionError): runner.save()
    assert runner.db.db==before

def test_completed_legacy_ledger_entry_still_replays_without_a_receipt(runner):
    runner.save();runner.db.db['play_generator_mutation_receipts']=[];before=deepcopy(runner.row)
    assert runner.save()['idempotent_replay'] is True
    assert runner.row==before

def test_old_unknown_pending_operation_is_never_unlocked_or_repeated(runner):
    runner.row['pending_operation_key']='unknown-old-completion';before=deepcopy(runner.row)
    for _ in range(2):
        with pytest.raises(PublicLiveRecoveryRequiredError): runner.save()
    assert runner.row==before
    assert not runner.db.db.get('play_generator_mutation_receipts')
    assert runner.db.db['public_live_operations'][-1]['status']=='recovery_required'

def test_prewrite_validation_is_terminal_and_does_not_poison_later_saves(runner):
    body={**deepcopy(runner.body),'round_number':999};before=deepcopy(runner.row)
    with pytest.raises(ValueError): runner.save(body=body)
    assert runner.db.db['public_live_operations'][-1]['status']=='rejected'
    assert runner.row==before
    assert not runner.db.db.get('play_generator_mutation_receipts')
    assert runner.save(key='receipt-valid-0001')['confirmed_commit'] is True

def test_confirmed_retry_returns_current_locked_results_without_permitting_a_new_edit(runner):
    runner.save();runner.db.db['public_live_operations'][-1]['status']='recovery_required'
    runner.row['state']['generator_submission']={'status':'pending'}
    runner.row['pending_operation_key']='separate-unknown-operation';before=deepcopy(runner.row)
    replay=runner.save()
    assert replay['idempotent_replay'] is replay['session']['results_locked'] is True
    assert runner.row==before
    with pytest.raises(ValueError,match='locked'): runner.save(key='receipt-new-locked-0001',version=runner.current_version())
    assert runner.row==before


@pytest.mark.parametrize('committed', [False, True])
def test_rpc_response_decode_valueerror_requires_receipt_proof(runner, monkeypatch, committed):
    """A decoder ValueError is ambiguous even when it happens before the write."""
    import json
    before = deepcopy(runner.row)
    original_rpc = runner.db.rpc
    def broken_rpc(name, params):
        request = original_rpc(name, params)
        def execute():
            if committed:
                request.execute()
            raise json.JSONDecodeError('simulated response decoding failure', '', 0)
        return SimpleNamespace(execute=execute)
    monkeypatch.setattr(runner.db, 'rpc', broken_rpc)
    if committed:
        first = runner.save()
        assert first['confirmed_commit'] is True
        assert runner.db.db['public_live_operations'][-1]['status'] == 'completed'
        assert len(runner.db.db['play_generator_mutation_receipts']) == 1
        assert int(runner.row['version']) == int(before['version']) + 1
        after = deepcopy(runner.row)
    else:
        with pytest.raises(PublicLiveRecoveryRequiredError):
            runner.save()
        assert runner.db.db['public_live_operations'][-1]['status'] == 'recovery_required'
        assert runner.row == before
        assert not runner.db.db.get('play_generator_mutation_receipts')
    monkeypatch.setattr(runner.db, 'rpc', original_rpc)
    recovered = runner.save()
    assert recovered['confirmed_commit'] is True
    assert len(runner.db.db['play_generator_mutation_receipts']) == 1
    assert int(runner.row['version']) == int(before['version']) + 1
    if committed:
        assert recovered['idempotent_replay'] is True
        assert runner.row == after
