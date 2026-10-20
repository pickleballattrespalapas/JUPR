from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys

import pytest
from scripts import rehearse_interrupted_match_exclusion as rehearsal
from scripts.prepare_parity_match_exclusion_fixture import FixtureError


def test_crash_exits_process_without_finally():
    code = '''
from scripts.rehearse_interrupted_match_exclusion import crash_after_commit
try:
    crash_after_commit(object(), operation={"operation_status":"pending_replay", "replay_job_id":"job"})
finally:
    print("unexpected cleanup", flush=True)
'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert result.returncode == rehearsal.CRASH_EXIT
    assert result.stdout == ''


@pytest.mark.parametrize('operation', [{}, {'operation_status': 'succeeded', 'replay_job_id': 'job'}, {'operation_status': 'pending_replay'}])
def test_wrong_commit_state_does_not_exit(operation):
    with pytest.raises(FixtureError):
        rehearsal.crash_after_commit(operation=operation)


@pytest.fixture
def recovered():
    operation = {'id':'operation', 'replay_job_id':'replay', 'excluded_match_ids':['3']}
    result = {'ok':True, 'mode':'match_exclusion_recovered', 'operation_status':'succeeded',
        'operation_id':'operation', 'replay_job_id':'replay', 'replay_status':'succeeded',
        'recovery_stage':None, 'excluded_ids':[3]}
    return operation, result


def test_first_and_repeat_recovery_preserve_identity(recovered):
    op, result = recovered
    rehearsal.verify_recovery(result, op, repeated=False)
    rehearsal.verify_recovery({**result, 'mode':'already_recovered', 'idempotent':True}, op, repeated=True)


@pytest.mark.parametrize('key,value', [('operation_id','new'), ('replay_job_id','new'), ('replay_status','pending'), ('operation_status','pending_replay'), ('excluded_ids',[4]), ('recovery_stage','replay'), ('ok',False)])
def test_recovery_rejects_wrong_or_incomplete_result(recovered, key, value):
    op, result = recovered
    with pytest.raises(FixtureError):
        rehearsal.verify_recovery({**result, key:value}, op, repeated=False)


def test_repeat_must_be_idempotent(recovered):
    op, result = recovered
    with pytest.raises(FixtureError):
        rehearsal.verify_recovery({**result, 'mode':'already_recovered'}, op, repeated=True)


def test_production_origin_rejected_before_fixture(monkeypatch, tmp_path):
    monkeypatch.setattr(rehearsal.fixture, '_validate_staging_environment', lambda env: ('origin','secret','sha'))
    monkeypatch.setattr(rehearsal.fixture, 'prepare_fixture', lambda **kw: pytest.fail('must not write'))
    with pytest.raises(FixtureError):
        rehearsal.rehearse(tmp_path, {'GITHUB_REF':'refs/heads/staging',
            'JUPR_PARITY_MUTATION_CONFIRMATION':'RUN DISPOSABLE STAGING WRITES',
            'STAGING_API_BASE_URL':'https://api.juprleagues.com'})


def test_durable_commit_must_exist_before_recovery(monkeypatch):
    monkeypatch.setattr(rehearsal.fixture, '_optional_manifest_ids', lambda m: ([], {'distinct':3}, {'distinct':1}, None))
    rows = {'operations':[{'id':'op','status':'pending_replay','replay_job_id':'job','excluded_match_ids':['3']}],
        'replay_jobs':[{'id':'job','status':'pending'}],
        'matches':[{'id':3,'deleted_at':'2026-10-20','row_version':2}]}
    assert rehearsal.verify_interrupted(rows,{})['id'] == 'op'
    for table, field, value in [('operations','status','succeeded'),('replay_jobs','status','running'),('matches','deleted_at',None),('matches','row_version',1)]:
        bad=deepcopy(rows);bad[table][0][field]=value
        with pytest.raises(FixtureError):
            rehearsal.verify_interrupted(bad,{})
