from types import SimpleNamespace

import pytest
from postgrest.exceptions import APIError

from jupr_app.domain.gamification import program_badge_service as service


class Database:
    def __init__(self, *, club='club', fail_apply=False):
        self.calls = []
        self.club = club
        self.fail_apply = fail_apply
    def rpc(self, name, payload):
        self.calls.append((name, payload))
        def execute():
            if name == 'badge_program_snapshot_v1':
                return SimpleNamespace(data={'club_id': self.club, 'revision': 17, 'players': [], 'as_of': '2026-09-07'})
            if name == 'pending_program_badge_clubs_v1':
                return SimpleNamespace(data=[{'club_id': self.club}])
            if name == 'apply_program_badges_v1' and self.fail_apply:
                raise RuntimeError('Source changed during evaluation')
            return SimpleNamespace(data={'ok': True})
        return SimpleNamespace(execute=execute)


def test_preview_never_writes_and_apply_uses_exact_snapshot_revision(monkeypatch):
    db = Database()
    result = service.reconcile_program_badges(db, 'club', dry_run=True)
    assert result['revision'] == 17
    assert [name for name, _ in db.calls] == ['badge_program_snapshot_v1']
    monkeypatch.setenv('JUPR_ENV', 'staging')
    monkeypatch.setenv('JUPR_STAGING_WRITE_WAVE', 'open')
    service.reconcile_program_badges(db, 'club')
    assert db.calls[-1][0] == 'apply_program_badges_v1'
    assert db.calls[-1][1]['p_revision'] == 17
    assert db.calls[-1][1]['p_awards'] == []


def test_closed_staging_and_production_do_not_run_worker(monkeypatch):
    db = Database()
    for environment, wave in [('production','open'),('staging','none'),('local','open')]:
        monkeypatch.setenv('JUPR_ENV', environment)
        monkeypatch.setenv('JUPR_STAGING_WRITE_WAVE', wave)
        assert service.process_pending_program_badges(db) == {'processed': 0, 'errors': 0}
    assert not db.calls


def test_incomplete_or_wrong_club_history_cannot_apply(monkeypatch):
    monkeypatch.setenv('JUPR_ENV', 'staging')
    monkeypatch.setenv('JUPR_STAGING_WRITE_WAVE', 'open')
    db = Database(club='other-club')
    with pytest.raises(RuntimeError, match='read completely'):
        service.reconcile_program_badges(db, 'club')
    assert not any(name == 'apply_program_badges_v1' for name, _ in db.calls)


def test_failed_worker_does_not_acknowledge_source_revision(monkeypatch):
    monkeypatch.setenv('JUPR_ENV', 'staging')
    monkeypatch.setenv('JUPR_STAGING_WRITE_WAVE', 'open')
    db = Database(fail_apply=True)
    assert service.process_pending_program_badges(db) == {'processed': 0, 'errors': 1}
    assert db.calls[-1] == ('program_badge_failure_v1', {'p_club_id': 'club'})
    db.fail_apply = False
    assert service.process_pending_program_badges(db) == {'processed': 1, 'errors': 0}


def _api_error(code='PT409', message=service.STALE_PROGRAM_REVISION_MESSAGE):
    return APIError({'code': code, 'message': message, 'details': None, 'hint': None})


class RevisionDatabase(Database):
    def __init__(self, errors):
        super().__init__()
        self.errors = list(errors)
        self.snapshots = 0

    def rpc(self, name, payload):
        response = super().rpc(name, payload)

        def execute():
            if name == 'badge_program_snapshot_v1':
                data = response.execute().data
                data['revision'] += self.snapshots
                self.snapshots += 1
                return SimpleNamespace(data=data)
            if name == 'apply_program_badges_v1' and self.errors:
                raise self.errors.pop(0)
            return response.execute()

        return SimpleNamespace(execute=execute)


def _enable_badge_writes(monkeypatch):
    monkeypatch.setenv('JUPR_ENV', 'staging')
    monkeypatch.setenv('JUPR_STAGING_WRITE_WAVE', 'open')


def test_stale_program_revision_refreshes_evidence_once_before_retry(monkeypatch):
    _enable_badge_writes(monkeypatch)
    db = RevisionDatabase([_api_error()])
    # Distinct candidates prove the retry uses recomputed evidence, not merely
    # a bumped revision attached to the original awards.
    monkeypatch.setattr(service, 'evaluate_program_badges', lambda snapshot: {
        'revision': snapshot['revision'], 'awards': [{'evidence_revision': snapshot['revision']}],
        'pending_ties': [], 'review': [],
    })
    assert service.reconcile_program_badges(db, 'club') == {'ok': True}
    assert [name for name, _ in db.calls] == [
        'badge_program_snapshot_v1', 'apply_program_badges_v1',
        'badge_program_snapshot_v1', 'apply_program_badges_v1',
    ]
    applies = [payload for name, payload in db.calls if name == 'apply_program_badges_v1']
    assert [p['p_revision'] for p in applies] == [17, 18]
    assert [p['p_awards'] for p in applies] == [[{'evidence_revision': 17}], [{'evidence_revision': 18}]]


def test_repeated_revision_changes_leave_work_pending_with_durable_backoff(monkeypatch):
    _enable_badge_writes(monkeypatch)
    db = RevisionDatabase([_api_error(), _api_error(), _api_error()])
    assert service.process_pending_program_badges(db) == {'processed': 0, 'errors': 1}
    assert db.snapshots == 2
    assert sum(name == 'apply_program_badges_v1' for name, _ in db.calls) == 2
    assert db.calls[-1] == ('program_badge_failure_v1', {'p_club_id': 'club'})
    assert len(db.errors) == 1


@pytest.mark.parametrize('error', [
    _api_error('40001', 'could not serialize access due to concurrent update'),
    _api_error('PT409', 'A different business conflict'),
    _api_error('42501', 'permission denied'),
    RuntimeError('Response outcome is unknown'),
])
def test_other_database_errors_are_not_treated_as_stale_badge_evidence(monkeypatch, error):
    _enable_badge_writes(monkeypatch)
    db = RevisionDatabase([error])
    with pytest.raises(type(error)) as caught:
        service.reconcile_program_badges(db, 'club')
    assert caught.value is error
    assert db.snapshots == 1
    assert sum(name == 'apply_program_badges_v1' for name, _ in db.calls) == 1


def test_emergency_stop_is_rechecked_before_refreshed_apply(monkeypatch):
    _enable_badge_writes(monkeypatch)
    db = RevisionDatabase([_api_error()])
    enabled = iter([True, False])
    monkeypatch.setattr(service, 'program_badge_writes_enabled', lambda: next(enabled))
    with pytest.raises(PermissionError, match='paused'):
        service.reconcile_program_badges(db, 'club')
    assert db.snapshots == 2
    assert sum(name == 'apply_program_badges_v1' for name, _ in db.calls) == 1
