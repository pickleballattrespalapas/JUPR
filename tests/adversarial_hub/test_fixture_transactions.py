"""The threaded browser API must not expose half-published fake transactions."""
from pathlib import Path
import subprocess
import sys
import textwrap


def test_browser_fixture_serializes_receipt_publication_and_rebinds_queries():
    # Import the real browser fixture in a child process: its auth/service
    # monkeypatches and synthetic database must not leak into other tests.
    script = textwrap.dedent("""
        import copy
        from concurrent.futures import ThreadPoolExecutor
        from threading import Event
        import sys
        sys.path[:0] = ['tests/e2e', 'tests', '.']
        import round_robin_hub_api as fixture
        from jupr_app.services.admin_play_generator_service import _live_row

        generator = fixture.generator
        cleared, publish, reading = Event(), Event(), Event()
        existing = {'club_id': 'club', 'session_key': 'existing',
                    'state': {'mode': 'admin_play_generator'}}
        class PausedPublication(dict):
            def __deepcopy__(self, memo):
                return copy.deepcopy(dict(self), memo)
            def clear(self):
                super().clear()
                cleared.set()
                assert publish.wait(5), 'Test did not release receipt publication'

        storage = PausedPublication({
            'live_sessions': [existing],
            'live_ladder_admin_operations': [{'operation_key': 'new-operation',
                'club_id': 'club', 'entity_id': 'new', 'request_fingerprint': 'fp'}],
        })
        generator.database.db = storage
        request = {'p_operation_key': 'new-operation', 'p_club_id': 'club',
            'p_request_fingerprint': 'fp', 'p_session_key': 'new-session',
            'p_expected_version': None,
            'p_patch': {'club_id': 'club', 'session_key': 'new-session',
                        'state': {'mode': 'admin_play_generator'}}}
        def read_existing():
            reading.set()
            return _live_row(generator.database, club_id='club', session_key='existing')

        with ThreadPoolExecutor(max_workers=2) as workers:
            writer = workers.submit(generator.database.rpc(
                'commit_admin_play_generator_session', request).execute)
            assert cleared.wait(5), 'Receipt never reached its publication boundary'
            reader = workers.submit(read_existing)
            try:
                assert reading.wait(5)
                # It must wait for the transaction, never turn an existing
                # session into the spurious HTTP 404 seen in the CI browser.
                try:
                    reader.result(timeout=0.05)
                except TimeoutError:
                    pass
                else:
                    raise AssertionError('A read escaped the transaction boundary')
            finally:
                publish.set()
            assert writer.result(timeout=5).data['session_key'] == 'new-session'
            assert reader.result(timeout=5)['session_key'] == 'existing'
        assert len(generator.database.db['live_sessions']) == 2
        assert len(generator.database.db['play_generator_mutation_receipts']) == 1

        # Exercise the real hub RPC wrapper's storage replacement while a
        # previously built query is waiting to execute.
        query = generator.database.table('live_sessions').select('*').eq('session_key', 'after-hub')
        from types import SimpleNamespace
        def replace_storage(_name, _params):
            def execute():
                assert generator.database_lock._is_owned()
                fixture.transactional.storage = {'live_sessions': [
                    {'club_id': 'club', 'session_key': 'after-hub',
                     'state': {'mode': 'admin_play_generator'}}]}
                return SimpleNamespace(data={'ok': True})
            return SimpleNamespace(execute=execute)
        fixture.transactional.rpc = replace_storage
        generator.database.rpc('commit_round_robin_hub', {}).execute()
        assert query.execute().data[0]['session_key'] == 'after-hub', 'Query retained pre-commit storage'
        print('Concurrent receipt reads and post-hub queries see committed storage')
    """)
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
