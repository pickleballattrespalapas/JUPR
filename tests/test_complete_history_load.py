from copy import deepcopy
from types import SimpleNamespace

import pytest

from jupr_app.data.load import load_data
from jupr_app.data.counted_reads import read_counted_rows
from jupr_app.data.paged_reads import DataReadUnavailable
from jupr_app.domain.gamification.match_exclusion_reconcile import _load_reconciliation_context


class Database:
    def __init__(self, cap=137):
        self.cap = cap
        self.tables = {}
        self.calls = []
        self.fail_offset = None
        self.truncate_offset = None
        self.repeat = False
        self.final_count_delta = 0
        self.no_count = False

    def table(self, table):
        db = self
        class Query:
            def __init__(self):
                self.filters = []
                self.bounds = None
                self.max_rows = None
                self.key = 'id'
                self.desc = False
            def select(self, columns, count=None):
                self.count = count
                return self
            def eq(self, key, value):
                self.filters.append(lambda row: row.get(key) == value)
                return self
            def is_(self, key, value):
                self.filters.append(lambda row: row.get(key) is None)
                return self
            def order(self, key, desc=False):
                self.key, self.desc = key, desc
                return self
            def range(self, start, end):
                self.bounds = (start, end)
                return self
            def limit(self, limit):
                self.max_rows = limit
                return self
            def execute(self):
                rows = sorted([r for r in db.tables.get(table, []) if all(f(r) for f in self.filters)],
                              key=lambda r: r[self.key], reverse=self.desc)
                count = len(rows) if not db.no_count else None
                if self.bounds:
                    start, end = self.bounds
                    db.calls.append((table, start, end))
                    if db.fail_offset is not None and start >= db.fail_offset:
                        raise TimeoutError('page unavailable')
                    if db.truncate_offset is not None and start >= db.truncate_offset:
                        rows = []
                    else:
                        if db.repeat: start = 0
                        rows = rows[start:min(end+1, start+db.cap)]
                else:
                    rows = rows[:self.max_rows]
                    if count is not None: count += db.final_count_delta
                return SimpleNamespace(data=deepcopy(rows), count=count)
        return Query()


@pytest.fixture
def large_club(monkeypatch):
    monkeypatch.setenv('JUPR_SKIP_BADGE_SCHEMA_PREFLIGHT', '1')
    monkeypatch.setattr('jupr_app.data.load.time.sleep', lambda _: None)
    db = Database()
    db.tables = {
        'players': [{'id': i, 'club_id': 'club', 'name': f'Player {i}', 'active': True} for i in range(1, 1203)],
        'league_ratings': [{'id': i, 'club_id': 'club', 'player_id': i, 'league_name': 'League'} for i in range(1, 1203)],
        'leagues_metadata': [{'id': i, 'club_id': 'club', 'league_name': f'League {i}'} for i in range(1, 1003)],
        'matches': [{'id': i, 'club_id': 'club', 'deleted_at': None} for i in range(1, 6204)],
        'player_badges': [{'id': i, 'club_id': 'club', 'badge_id': 'first_win', 'player_id': 1} for i in range(1, 1103)],
    }
    db.tables['matches'] += [{'id': 9001, 'club_id': 'club', 'deleted_at': '2026-01-01'},
                             {'id': 9002, 'club_id': 'other', 'deleted_at': None}]
    db.tables['players'].append({'id': 9002, 'club_id': 'other', 'name': 'Other'})
    return db


def test_complete_shared_load_crosses_server_and_old_5000_cap(large_club):
    result = load_data(large_club, 'club')
    assert len(result[0]) == len(result[2]) == 1202
    assert len(result[3]) == 6203
    assert result[3].id.tolist() == list(range(6203, 0, -1))
    assert len(result[4]) == 1002
    assert len(result[6]) == 1102
    assert result[8][1202] == 'Player 1202'
    assert 9002 not in result[8]
    assert not result[9]


def test_large_correction_context_passes_complete_history_guard(large_club):
    ctx = _load_reconciliation_context(large_club, club_id='club', match_limit=None)
    assert len(ctx.df_matches) == 6203
    assert len(ctx.df_players_all) == 1202


def test_explicit_preview_limit_still_rejected_for_correction(large_club):
    assert len(load_data(large_club, 'club', match_limit=501)[3]) == 501
    with pytest.raises(RuntimeError, match='exceeds match_limit'):
        _load_reconciliation_context(large_club, club_id='club', match_limit=501)


@pytest.mark.parametrize('fault,exception', [('fail_offset', TimeoutError), ('truncate_offset', DataReadUnavailable),
                                          ('repeat', DataReadUnavailable), ('final_count_delta', DataReadUnavailable),
                                          ('no_count', DataReadUnavailable)])
def test_incomplete_or_changing_history_never_returns_partial_data(large_club, fault, exception):
    setattr(large_club, fault, 137 if fault.endswith('offset') else 1)
    with pytest.raises(exception):
        load_data(large_club, 'club')


def test_empty_complete_table_and_invalid_bounds():
    db = Database()
    query = lambda count: db.table('matches').select('*', count=count)
    assert read_counted_rows(query) == []
    with pytest.raises(ValueError): read_counted_rows(query, page_size=0)
    with pytest.raises(ValueError): read_counted_rows(query, limit=0)


def test_badge_plan_retains_oldest_win_beyond_previous_cap(large_club):
    from jupr_app.domain.gamification.match_exclusion_reconcile import (
        build_match_exclusion_badge_plan, resolve_match_exclusion_badge_ids,
    )
    for row in large_club.tables['matches']:
        row.update(date='2026-01-01T10:00:00Z', league='League', match_type='League',
                   t1_p1=1, t1_p2=2, t2_p1=3, t2_p2=4,
                   score_t1=11 if row['id'] == 1 else 5, score_t2=7 if row['id'] == 1 else 11)
    large_club.tables['badges'] = [{'badge_id': 'first_win', 'state': 'live', 'is_active': True,
                                   'eval_triggers': ['match_recorded', 'match_updated']}]
    assert resolve_match_exclusion_badge_ids(large_club, club_id='club') == ['first_win']
    ctx = _load_reconciliation_context(large_club, club_id='club', match_limit=None)
    rows = build_match_exclusion_badge_plan(ctx=ctx, club_id='club', player_id=1, badge_ids=['first_win'])
    assert len(rows) == 1
    assert rows[0]['match_id'] == '1'
    # Excluding that oldest win must revoke the award, not manufacture a newer win.
    large_club.tables['matches'][0]['deleted_at'] = '2026-10-20T00:00:00Z'
    ctx = _load_reconciliation_context(large_club, club_id='club', match_limit=None)
    assert build_match_exclusion_badge_plan(ctx=ctx, club_id='club', player_id=1, badge_ids=['first_win']) == []


def test_real_postgrest_client_pages_with_exact_content_range():
    import httpx
    from postgrest import SyncPostgrestClient
    source = [{'id': i, 'club_id': 'club'} for i in range(1007)]
    calls = []
    def respond(request):
        params = request.url.params
        assert params['club_id'] == 'eq.club'
        assert params['order'] == 'id.asc'
        start = int(params.get('offset', 0))
        size = min(int(params.get('limit', 1000)), 137)
        page = source[start:start+size]
        exact = 'count=exact' in request.headers.get('prefer', '')
        calls.append((start, exact))
        total = str(len(source)) if exact else '*'
        return httpx.Response(200, json=page, headers={'Content-Range': f'{start}-{start+len(page)-1}/{total}'})
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        db = SyncPostgrestClient('https://synthetic.invalid', http_client=http)
        rows = read_counted_rows(lambda count: db.table('matches').select('*', count=count).eq('club_id', 'club'))
    assert rows == source
    assert [start for start, _ in calls[:-1]] == list(range(0, 1007, 137))
    assert sum(exact for _, exact in calls) == 2
