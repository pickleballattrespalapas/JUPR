from __future__ import annotations

import copy
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.domain.recaps import player_weekly_digest as digest
from jupr_app.domain.player_rating_series import build_player_overall_rating_series
from jupr_app.domain.notifications.player_update_email_template import build_player_update_email_html, build_player_update_email_text


class Query:
    def __init__(self, db, name):
        self.db, self.name, self.filters, self.ordering = db, name, [], []
        self.bounds = None
    def select(self, _columns): return self
    def eq(self, key, value): self.filters.append(lambda row: str(row.get(key)) == str(value)); return self
    def in_(self, key, values):
        values = {str(value) for value in values}
        self.filters.append(lambda row: str(row.get(key)) in values); return self
    def or_(self, expression):
        choices = [piece.split('.eq.') for piece in expression.split(',')]
        self.filters.append(lambda row: any(str(row.get(key)) == value for key, value in choices)); return self
    def gte(self, key, value): self.filters.append(lambda row: str(row.get(key) or '') >= value); return self
    def lte(self, key, value): self.filters.append(lambda row: str(row.get(key) or '') <= value); return self
    def order(self, key, desc=False): self.ordering.append((key, desc)); return self
    def range(self, start, end): self.bounds = (start, end + 1); return self
    def limit(self, count): self.bounds = (0, count); return self
    def execute(self):
        if self.db.fail_reads: raise RuntimeError('Unavailable')
        rows = [copy.deepcopy(row) for row in self.db.rows.get(self.name, []) if all(f(row) for f in self.filters)]
        for key, desc in reversed(self.ordering): rows.sort(key=lambda row: row.get(key) or '', reverse=desc)
        if self.bounds: rows = rows[slice(*self.bounds)]
        # Deliberately model a server cap smaller than every session.
        return SimpleNamespace(data=rows[:3])


class DB:
    def __init__(self, matches, badges=()):
        self.rows = {'matches':list(matches), 'player_badges':list(badges), 'badges':[]}
        self.fail_reads = False
    def table(self, name): return Query(self, name)


def game(mid, *, score=(11, 5), partner=2, when='2026-10-11T19:00:00+00:00', before=1600., after=1604., **extra):
    return {'id':mid, 'club_id':'club', 'date':when, 'league':'OVERALL', 'match_type':'Round-Robin Generator',
        'match_format':'doubles', 'rating_scope':'overall_only', 'deleted_at':None,
        't1_p1':1, 't1_p2':partner, 't2_p1':90, 't2_p2':91,
        'score_t1':score[0], 'score_t2':score[1], 't1_p1_r':before, 't1_p1_r_end':after,
        't1_p2_r':1400., 't2_p1_r':1400., 't2_p2_r':1400., 'elo_delta':abs(after-before), **extra}


def badge(bid, *, mid=None, context='match', context_id=None, value=None, revoked=None, earned='2026-10-11T15:24:05+00:00', rid=None):
    return {'id':rid or f'{bid}-{mid}-{context_id}', 'club_id':'club', 'player_id':1, 'badge_id':bid,
        'earned_at':earned, 'context_type':context, 'context_id':context_id or f'{mid}:{bid}',
        'match_id':mid, 'value_json':value or {}, 'revoked_at':revoked}


def context(db):
    return SimpleNamespace(supabase=db, club_id='club', id_to_name={1:'Joe', 2:'Joel', 90:'Ross', 91:'Brant'},
        df_players_all=pd.DataFrame([{'id':pid, 'rating':1400} for pid in [1,2,90,91]]))


@pytest.fixture(autouse=True)
def no_tournament_trophies(monkeypatch):
    monkeypatch.setattr(digest, 'get_player_tournament_trophies', lambda *_args: [])


def compute(db, match_ids, **kwargs):
    return digest.compute_player_weekly_digest(context(db), 1, date(2026,10,11), date(2026,10,11), match_ids=match_ids, **kwargs)


def test_reported_session_excludes_backfilled_badges_and_other_same_day_rr_in_actual_email():
    session = [game(100+i, score=(11,5) if i<4 else (8,11), before=1600+i*4, after=1604+i*4) for i in range(8)]
    other = game(200, score=(11,0), partner=8)
    historical = [game(2023, score=(11,0), when='2026-04-07T19:00:00+00:00'), game(2423, score=(15,0), when='2026-05-03T19:00:00+00:00')]
    db = DB(session+[other]+historical, [badge('pickle_perfection', mid=2023), badge('pickle_perfection', mid=2423),
        badge('pickle_perfection', mid=200), badge('hot_streak', mid=2023, value={'qualifying_match_ids':['100','101','2023']}),
        badge('blowout_artist', mid=100, revoked='2026-10-11T15:30:00Z'), badge('mentor', context='overall')])
    before = copy.deepcopy(db.rows)
    result = compute(db, list(range(100,108)))
    assert result['summary']['matches_played'] == 8
    assert result['summary']['record'] == '4-4'
    assert {row['id'] for row in result['matches_played_rows']} == set(range(100,108))
    assert result['meta']['scope'] == 'session'
    assert result['badges_earned'] == []
    assert result['trophies_earned'] == []
    for rendered in (build_player_update_email_html(result, None), build_player_update_email_text(result)):
        assert 'Games from this session' in rendered
        assert 'Pickle Perfection' not in rendered
        assert 'Hot Streak' not in rendered
        assert '15-0' not in rendered and '11-0' not in rendered
        assert '11-5' in rendered
    assert db.rows == before, 'Email filtering never mutates historic profile awards or matches'


def test_current_shutout_and_streak_trigger_count_even_if_awarded_days_after_play():
    db = DB([game(10,score=(11,0)), game(11), game(12)], [
        badge('pickle_perfection', mid=10, earned='2026-10-20T12:00:00Z'),
        badge('hot_streak', mid=11, context='league', value={'tier':3}),
        badge('clutch_performer', mid=12, context='overall', value={'qualifying_match_ids':['10','11','12']}),
        badge('blowout_artist', mid=10, revoked='2026-10-21T12:00:00Z')])
    result = compute(db, [10,11])
    assert {row['badge_id'] for row in result['badges_earned']} == {'pickle_perfection','hot_streak'}
    for rendered in (build_player_update_email_html(result,None), build_player_update_email_text(result)):
        assert 'Awards from this session' in rendered
        assert 'Pickle Perfection' in rendered
        assert 'Clutch Performer' not in rendered


def test_backdated_exact_ids_ignore_date_window_and_unrelated_interleaved_rating_changes():
    db = DB([game(8,when='2026-01-01T19:00:00Z', before=1000, after=1004),
             game(9,when='2026-01-01T19:00:00Z', before=1004, after=1200),
             game(10,when='2026-01-01T19:00:00Z', before=1200, after=1208),
             game(3000,before=2000,after=2010)])
    result = compute(db, ['8','10'])
    assert result['summary']['overall_jupr_before'] == 2.5
    assert result['summary']['overall_jupr_after'] == 3.02
    assert result['summary']['overall_delta'] == .03
    points = result['chart']['points']
    assert [point['id'] for point in points] == ['anchor',8,10]
    assert [point['overall_after'] for point in points] == [2.5,2.51,3.02]
    assert {row['id'] for row in result['matches_played_rows']} == {8,10}


def test_singles_games_do_not_contaminate_overall_doubles_rating():
    db = DB([game(1),game(2,match_format='singles',t1_p2=None,t2_p2=None,before=800,after=804)])
    result = compute(db,[1,2])
    assert result['summary']['matches_played'] == 2
    assert result['summary']['overall_jupr_before'] == 4.
    assert result['summary']['overall_jupr_after'] == 4.01
    assert result['summary']['overall_delta'] == .01
    assert result['chart']['points'][-1]['overall_after'] is None
    result = compute(db,[2])
    assert result['summary']['overall_jupr_before'] is None
    assert result['summary']['overall_jupr_after'] is None
    assert result['summary']['overall_delta'] is None


@pytest.mark.parametrize('problem',['missing','deleted','other_club','read_failure','malformed','empty'])
def test_scoped_match_read_fails_closed_instead_of_partial_or_zero_game_email(problem):
    db=DB([game(1),game(2)])
    ids=[1,2]
    if problem=='missing': db.rows['matches'].pop()
    elif problem=='deleted': db.rows['matches'][1]['deleted_at']='2026-10-12T00:00:00Z'
    elif problem=='other_club': db.rows['matches'][1]['club_id']='other'
    elif problem=='read_failure': db.fail_reads=True
    elif problem=='malformed': ids=[1,'invalid']
    else: ids=[]
    with pytest.raises((ValueError,RuntimeError)):
        compute(db,ids)


def test_other_players_games_are_verified_but_not_added_to_personal_session_record():
    db=DB([game(1),game(2,t1_p1=2,t1_p2=3,t2_p1=4,t2_p2=5)])
    result=compute(db,[1,2])
    assert result['summary']['matches_played']==1
    assert result['meta']['match_ids']==['1']


def test_receipt_proves_lifetime_milestone_but_not_old_backfilled_threshold():
    db=DB([game(1)], [badge('dedicated_participant_50',context='overall',context_id='overall',value={'games':56}),
         badge('high_roller',context='overall',context_id='lifetime_wins_100',value={'wins':123})])
    proof={'player_updates':[{'player_id':1,'rating_mode':'doubles',
        'expected':{'matches_played':48,'wins':122},'after':{'matches_played':56,'wins':123}}]}
    result=compute(db,[1],session_evidence=proof)
    assert [row['badge_id'] for row in result['badges_earned']]==['dedicated_participant_50']
    assert compute(db,[1])['badges_earned']==[]


def test_draft_master_is_attributed_only_to_fifth_partner_win_not_other_same_day_rr():
    games=[game(i,partner=10+i) for i in range(6,11)]
    award=badge('draft_master',context='week',context_id='2026-W41',value={'week':'2026-W41','partners':5})
    db=DB(games,[award])
    assert compute(db,[9])['badges_earned']==[]
    assert [row['badge_id'] for row in compute(db,[10])['badges_earned']]==['draft_master']


def test_weekly_admin_digest_retains_date_range_badge_selection():
    db=DB([game(1)], [badge('pickle_perfection',mid=2023),badge('pickle_perfection',mid=2423)])
    result=digest.compute_player_weekly_digest(context(db),1,date(2026,10,11),date(2026,10,11))
    assert len(result['badges_earned'])==2
    assert result['badges_grouped'][0]['count']==2
    assert result['meta'].get('scope') is None


def test_legacy_alias_history_cannot_make_an_old_partner_milestone_look_new():
    legacy=[]
    for mid in (1,2):
        row=game(mid,partner=10+mid)
        for source,target in [('t1_p1','team1_player1'),('t1_p2','team1_player2'),('t2_p1','team2_player1'),('t2_p2','team2_player2')]:
            row[target]=row.pop(source)
        legacy.append(row)
    award=badge('draft_master',context='week',context_id='2026-W41',value={'week':'2026-W41','partners':7})
    db=DB(legacy+[game(mid,partner=10+mid) for mid in range(6,11)],[award])
    assert compute(db,[10])['badges_earned']==[]
    assert [row['badge_id'] for row in compute(db,[8])['badges_earned']]==['draft_master']


@pytest.mark.parametrize('patch',[{'date':None},{'date':'invalid'},{'score_t1':None},{'score_t2':float('nan')},{'score_t1':11,'score_t2':11}])
def test_malformed_selected_game_cannot_disappear_from_the_session(patch):
    db=DB([game(1),game(2,**patch)])
    with pytest.raises(ValueError):
        compute(db,[1,2])


def test_receipt_evidence_keeps_new_wins_and_rating_milestones():
    db=DB([game(1)], [badge('high_roller',context='overall',context_id='lifetime_wins_100',value={'wins':100}),
        badge('level_up',context='league',context_id='milestone:4.0',value={'league':'Monday','milestone':4.0}),
        badge('level_up',context='league',context_id='milestone:3.5',value={'league':'Monday','milestone':3.5})])
    proof={'player_updates':[{'player_id':1,'rating_mode':'doubles','expected':{'wins':99},'after':{'wins':100}}],
        'league_rating_updates':[{'player_id':1,'league_name':'Monday','expected':{'rating':1590},'after':{'rating':1610}}]}
    result=compute(db,[1],session_evidence=proof)
    assert {(row['badge_id'],row['context_id']) for row in result['badges_earned']} == {
        ('high_roller','lifetime_wins_100'),('level_up','milestone:4.0')}


def test_session_does_not_reconstruct_missing_overall_snapshots_from_singles():
    db=DB([game(1,t1_p1_r=None,t1_p1_r_end=None),
        game(2,match_format='singles',t1_p2=None,t2_p2=None,before=800,after=804)])
    result=compute(db,[1,2])
    assert result['summary']['overall_jupr_before'] is None
    assert result['summary']['overall_jupr_after'] is None
    assert all(point['overall_after'] is None for point in result['chart']['points'])
