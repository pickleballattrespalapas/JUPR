from types import SimpleNamespace
from datetime import datetime, timezone
import pandas as pd
import pytest

from jupr_app.data.paged_reads import DataReadUnavailable
from jupr_app.domain.gamification.badge_types import BadgeEvaluationContext, BadgeCandidate
from jupr_app.domain.gamification.evaluators import (
    evaluate_level_up, evaluate_hall_of_fame_night, evaluate_legendary_upset,
    evaluate_clean_sweep_week, evaluate_most_improved_monthly, _max_consecutive_weeks,
)
from jupr_app.domain.gamification.match_facts import build_canonical_player_match_facts
from jupr_app.domain.gamification.badges_repo import upsert_player_badges
from jupr_app.services.public_badge_codex_service import build_public_badge_codex
from jupr_app.services.leaderboard_service import _badge_map
from jupr_app.services.public_player_service import _fetch_player_badges


class Query:
    """A capped server that checks the real award projection and applies ranges."""
    def __init__(self, db, table):
        self.db, self.table = db, table
        self.filters, self.start, self.end = [], 0, 999
    def select(self, columns):
        if self.table == 'player_badges':
            assert 'created_at' not in columns.split(',')
        return self
    def eq(self, key, value):
        self.filters.append(lambda r: str(r.get(key)) == str(value)); return self
    def in_(self, key, values):
        self.filters.append(lambda r: r.get(key) in values); return self
    def is_(self, key, value):
        self.filters.append(lambda r: r.get(key) is value); return self
    def order(self, key):
        self.key = key; return self
    def range(self, start, end):
        self.start, self.end = start, end; return self
    def execute(self):
        if self.db.fail_table == self.table and self.start >= self.db.fail_at:
            raise RuntimeError('read failed')
        rows = [r for r in self.db.tables.get(self.table, []) if all(f(r) for f in self.filters)]
        rows = sorted(rows, key=lambda r: str(r.get(getattr(self, 'key', 'id')) or ''))
        return SimpleNamespace(data=rows[self.start:min(self.end + 1, self.start + self.db.cap)])
    def upsert(self, *args, **kwargs):
        raise AssertionError('Existing awards must never be rewritten')


class Database:
    def __init__(self, tables, cap=137, fail_table=None, fail_at=0):
        self.tables, self.cap, self.fail_table, self.fail_at = tables, cap, fail_table, fail_at
    def table(self, name):
        return Query(self, name)


def database():
    return Database({
        'badges': [{'badge_id': 'participant', 'name': 'Participant', 'prestige': 10, 'is_active': True, 'state': 'live'}],
        'players': [{'id': i, 'club_id': 'club', 'name': f'Player {i}', 'active': True} for i in range(1, 1104)],
        'player_badges': [{'id': i, 'club_id': 'club', 'player_id': i, 'badge_id': 'participant', 'earned_at': '2026-01-01', 'revoked_at': None} for i in range(1, 1104)],
    })


def test_all_three_public_reads_cross_server_limit_and_exclude_revocations():
    db = database()
    db.tables['player_badges'][-1]['revoked_at'] = '2026-02-01'
    db.tables['player_badges'].append({'id': 2000, 'club_id': 'other', 'player_id': 5000, 'badge_id': 'participant'})
    result = build_public_badge_codex(db, club_id='club')
    assert result['summary']['unique_earner_count'] == 1102
    assert result['sections'][0]['badges'][0]['earners_count'] == 1102
    assert len(_badge_map(db, club_id='club')) == 1102
    assert _fetch_player_badges(db, club_id='club', player_id=1103) == []


def test_failed_second_page_is_unavailable_never_false_zero():
    db = database(); db.fail_table, db.fail_at = 'player_badges', 137
    with pytest.raises(DataReadUnavailable):
        build_public_badge_codex(db, club_id='club')


def test_lifetime_legacy_key_beyond_first_page_preserves_earned_at(monkeypatch):
    from jupr_app.domain.gamification import badges_repo
    monkeypatch.setattr(badges_repo, '_PLAYER_BADGES_CONTRACT_CHECKED', True)
    db = database()
    row = db.tables['player_badges'][-1]
    row.update(context_id=None, context_type='overall')
    candidate = BadgeCandidate('participant', 1103, 'club', 'overall', 'overall', None)
    assert upsert_player_badges(db, 'club', [candidate]) == []
    assert row['earned_at'] == '2026-01-01'


def context(rows=(), **kwargs):
    return BadgeEvaluationContext('club', None, kwargs.pop('as_of', None), SimpleNamespace(**kwargs), pd.DataFrame(rows), pd.DataFrame())


def test_level_up_uses_jupr_units():
    ctx = context(df_leagues=pd.DataFrame([{'player_id': 1, 'league_name': 'Open', 'rating': 1599}, {'player_id': 2, 'league_name': 'Open', 'rating': 4.0}]))
    result = [(c.player_id, c.value_json['milestone']) for c in evaluate_level_up(ctx)]
    assert result == [(1, 3.0), (1, 3.5), (2, 3.0), (2, 3.5), (2, 4.0)]


def test_hall_of_fame_excludes_losing_side_and_legendary_cutoff_is_15_percent():
    rows = [{'player_id': i, 'league': 'Open', 'match_id': str(i), 'win': win, 'abs_elo_delta': 100, 'expected_win_prob': probability} for i,win,probability in [(1,True,.15),(2,False,.1),(3,True,.1501)]]
    assert {c.player_id for c in evaluate_hall_of_fame_night(context(rows))} == {1, 3}
    assert {c.player_id for c in evaluate_legendary_upset(context(rows))} == {1}


def test_iso_week_53_is_not_treated_as_a_gap_or_skipped():
    assert _max_consecutive_weeks(['2026-W51', '2026-W52', '2026-W53', '2027-W01']) == 4
    assert _max_consecutive_weeks(['2026-W52', '2027-W01']) == 1
    assert _max_consecutive_weeks(['2025-W52', '2026-W01']) == 2


def test_mixed_dates_are_order_independent_and_snapshot_delta_is_signed():
    matches = [{'id': i, 'club_id': 'club', 'league': 'Open', 'date': d, 't1_p1': 1, 't2_p1': 2, 'score_t1': 11, 'score_t2': 8, 't1_p1_r': 1400, 't1_p1_r_end': 1399, 't2_p1_r': 1800, 'elo_delta': 9} for i,d in enumerate(['2026-01-01', '2026-01-02T12:00:00+00:00', '2026-01-03T12:00:00.123Z'], 1)]
    ctx = SimpleNamespace(club_id='club', df_matches=pd.DataFrame(matches))
    forward = build_canonical_player_match_facts(ctx)
    ctx.df_matches = ctx.df_matches.iloc[::-1]
    reverse = build_canonical_player_match_facts(ctx)
    assert len(forward) == len(reverse) == 6
    assert set(forward.match_id) == set(reverse.match_id)
    assert set(forward[forward.player_id == 1].elo_delta_signed) == {-1}


def test_final_period_awards_wait_for_boundary():
    rows = [{'player_id': 1, 'league': league, 'date_dt': pd.Timestamp('2026-01-28', tz='UTC'), 'week_key': '2026-W05', 'month_key': '2026-01', 'win': True, 'elo_delta_signed': 10} for league in ['A', 'B']]
    open_ctx = context(rows, as_of=datetime(2026,1,29,tzinfo=timezone.utc))
    assert list(evaluate_clean_sweep_week(open_ctx)) == []
    assert list(evaluate_most_improved_monthly(open_ctx)) == []
    closed = context(rows, as_of=datetime(2026,2,2,tzinfo=timezone.utc))
    assert len(list(evaluate_clean_sweep_week(closed))) == 1
    assert len(list(evaluate_most_improved_monthly(closed))) == 2
