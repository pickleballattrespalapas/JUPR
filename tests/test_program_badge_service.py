from types import SimpleNamespace

import pytest

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
