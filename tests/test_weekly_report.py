from copy import deepcopy
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.domain.recaps.weekly_report import generator_result, _new_faces, _achievements, _add_interclub, _add_tournament_draws
from jupr_app.domain.recaps.weekly_report_public import sanitize_report, safe_url
from jupr_app.domain.recaps.weekly_recap import _compute_stats, _build_spotlight_candidates, _event_highlights, _filter_matches, _load_matches
from jupr_app.services.admin_weekly_recap_service import apply_weekly_recap_edits
from jupr_app.services.public_weekly_recap_service import sanitize_public_recap_json, build_weekly_recap_pdf_bytes
from jupr_app.domain.adaptive_play_engine import create_generator_preview, start_generator_event, save_generator_round, advance_generator_event, complete_generator_event, generator_event_standings


class Query:
    def __init__(self, data, cap=2):
        self.data = deepcopy(data)
        self.cap = cap
        self.start, self.end = 0, 10**9
    def select(self, *a): return self
    def eq(self, key, value):
        self.data = [r for r in self.data if r.get(key) == value]; return self
    def in_(self, key, values):
        self.data = [r for r in self.data if r.get(key) in values]; return self
    def gte(self, key, value):
        self.data = [r for r in self.data if str(r.get(key) or '') >= value]; return self
    def lte(self, key, value):
        self.data = [r for r in self.data if str(r.get(key) or '') <= value]; return self
    def lt(self, key, value):
        self.data = [r for r in self.data if str(r.get(key) or '') < value]; return self
    def order(self, key, **kw):
        self.data.sort(key=lambda r: str(r.get(key) or '')); return self
    def range(self, start, end):
        self.start, self.end = start, end; return self
    def execute(self):
        return SimpleNamespace(data=self.data[self.start:min(self.end+1, self.start+self.cap)])


class DB:
    def __init__(self, **tables): self.tables = tables
    def table(self, name): return Query(self.tables.get(name, []))


def finished_event():
    event = start_generator_event(create_generator_preview(generator_kind='round_robin', play_format='doubles', title='Evening social', participant_names=['Alex','Bea','Cam','Dee'], player_ids=[1,2,3,4], total_rounds=3))
    for number in range(1, 4):
        row = next(r for r in event['rounds'] if r['number'] == number)
        event = save_generator_round(event, round_number=number, scores=[{'match_id':m['id'], 'score_a':11, 'score_b':5} for m in row['matches']])
        event = advance_generator_event(event)
    return event


def session(event, **submission):
    return {'session_key':'session', 'title':event['name'], 'state':{'page_state':{'event':event}, 'generator_submission': {'status':'approved', 'match_date':'2026-10-13', **submission}}}


def match(i, **overrides):
    return {'id':i,'club_id':'club','date':f'2026-10-{12 + i:02}T12:00:00+00:00','league':'Fall','match_type':'League','t1_p1':1,'t1_p2':2,'t2_p1':3,'t2_p2':4,'score_t1':11,'score_t2':7,'t1_p1_r':1200,'t1_p2_r':1200,'t2_p1_r':1600,'t2_p2_r':1600, **overrides}


def test_generator_results_include_withdrawn_players_but_not_unapproved_results():
    event = finished_event()
    event['participants'][0]['active'] = False
    result, coverage = generator_result(session(event), '2026-10-12', '2026-10-18')
    assert '4 players · 3 games' in result['details'][0]
    assert result['title'] == 'Evening social'
    assert coverage['status'] == 'approved'
    result, source = generator_result(session(event, status='pending'), '2026-10-12', '2026-10-18')
    assert result is None and source['status'] == 'pending'
    assert generator_result(session(event, match_date='2026-10-20'), '2026-10-12', '2026-10-18') == (None, None)


def test_approval_date_correction_and_direct_publication_use_play_date():
    event = finished_event()
    row = session(event, review={'request': {'match_date':'2026-10-20'}})
    assert generator_result(row, '2026-10-12', '2026-10-18') == (None, None)
    row['state'].pop('generator_submission')
    row['state']['official_publish'] = {'published_match_ids':['r1-c1'], 'match_context_by_live_id':{'r1-c1':'ctx'}}
    result, _ = generator_result(row, '2026-10-12', '2026-10-18', [match(1,context_id='ctx')])
    assert result and result['date'] == '2026-10-13'


def test_moneyball_report_uses_canonical_settlement_not_most_wins(monkeypatch):
    event = finished_event(); event['eventFormat'] = 'moneyball_expectations'
    monkeypatch.setattr('jupr_app.domain.adaptive_play_engine.generator_event_standings', lambda e:[{'name':'Most Wins','rank':0,'matches':3,'wins':3,'expectedWins':2.9,'differential':9,'expectedDifferential':8.7,'moneyballEligible':False}, {'name':'Above Expected','rank':1,'matches':3,'wins':2,'expectedWins':0.7,'differential':4,'expectedDifferential':-6.1}])
    monkeypatch.setattr('jupr_app.domain.moneyball_expectations.settlement', lambda e,r:{'final':True,'winners':[{'name':'Above Expected'}]})
    result, _ = generator_result(session(event), '2026-10-12','2026-10-18')
    assert result['summary'] == 'Winner: Above Expected'
    assert any('ineligible after substitution' in line for line in result['details'])
    assert any('2 wins vs 0.7 expected' in line for line in result['details'])


def test_rating_expectations_exclude_unrated_and_missing_snapshots():
    frame = pd.DataFrame([match(i) for i in range(1,5)] + [match(5,rating_scope='unrated'), match(6,t1_p1_r=None)])
    frame['date_dt'] = pd.to_datetime(frame['date'],utc=True)
    stats, _, _, upsets = _compute_stats(frame, {1:2000})
    assert stats[1]['rated_games'] == 4
    candidates = _build_spotlight_candidates(stats, upsets, {1:'Alex'})
    assert candidates['ABOVE_EXPECTATIONS_WEEK'][0].value_json['games'] == 4
    assert candidates['ABOVE_EXPECTATIONS_WEEK'][0].value_json['above'] > 3


def test_top_performer_is_consistent_with_event_summaries():
    stats = {1:{'games':10,'wins':6,'losses':4}, 2:{'games':5,'wins':5,'losses':0}, 3:{'games':2,'wins':2,'losses':0}}
    candidates = _build_spotlight_candidates(stats, [], {})
    highlights = _event_highlights(stats, {}, count=1,short_labels=False,prefer_jump=False)
    assert candidates['TOP_PERFORMER_WEEK'][0].player_ids == [2]
    assert highlights[0]['player_ids'][0] == 2
    assert 3 not in highlights[0]['player_ids']


def test_full_match_reads_do_not_stop_at_server_cap():
    data = [match(i) for i in range(1,7)]
    db = DB(matches=data)
    frame = _load_matches(pd.DataFrame(), db,'club', date(2026,10,12),date(2026,10,18),datetime(2026,10,12,tzinfo=timezone.utc),datetime(2026,10,19,tzinfo=timezone.utc),include_tournaments=True)
    assert len(frame) == 6
    assert len(_filter_matches(pd.DataFrame([match(1,score_t1=7,score_t2=7)]),include_tournaments=True)) == 0


def test_new_faces_include_social_only_and_require_complete_history():
    old_events = [{'id':'old','club_id':'club','status':'saved','result_mode':'social_unrated','event_date':'2026-09-01'}]
    people = [{'id':'old-a','event_id':'old','club_person_id':'repeat'}, {'id':'old-b','event_id':'old','linked_player_id':1}]
    old_games = [{'id':'old-game','event_id':'old','t1_p1_participant_id':'old-a','t1_p2_participant_id':'old-b','score_t1':11,'score_t2':7}]
    social = {'events':[{'id':'current'}], 'participants':[{'id':'a','club_person_id':'repeat','display_name_snapshot':'Returning'}, {'id':'b','club_person_id':'new','display_name_snapshot':'New Social'}, {'id':'c','linked_player_id':2}, {'id':'d','linked_player_id':3}], 'matches':[{'event_id':'current','t1_p1_participant_id':'a','t1_p2_participant_id':'b','t2_p1_participant_id':'c','t2_p2_participant_id':'d','score_t1':11,'score_t2':7}]}
    result = _new_faces(DB(live_events=old_events,live_event_participants=people,live_event_matches=old_games), 'club',[match(1)],social,'2026-10-12','2026-10-18',{1:'Alex',2:'Bea',3:'Cam',4:'Dee'})
    assert 'New Social' in result and 'Returning' not in result and 'Alex' not in result


def test_achievements_use_trigger_game_not_insertion_time_and_ignore_revocations():
    badges = [{'id':1,'club_id':'club','player_id':1,'badge_id':'old','earned_at':'2026-10-14','match_id':999}, {'id':2,'club_id':'club','player_id':1,'badge_id':'new','earned_at':'2026-10-20','match_id':1}, {'id':3,'club_id':'club','player_id':2,'badge_id':'revoked','match_id':1,'revoked_at':'2026-10-14'}]
    ctx = SimpleNamespace(supabase=DB(player_badges=badges),club_id='club',id_to_name={1:'Alex'},df_badges=pd.DataFrame())
    assert _achievements(ctx,[match(1)]) == ['Alex — new']


def test_public_report_and_email_inputs_never_expose_private_coverage():
    source = {'report':{'headline':'Weekly news','coverage':{'pending':[{'organizer_email':'private'}]},'event_results':[{'title':'RR','state':{'edit_token':'private'},'href':'javascript:alert(1)','details':['Finished']}], 'upcoming':[{'text':'Sign up','url':'javascript:alert(1)'},{'text':'Moneyball','url':'https://app.courtreserve.com/event'}]}}
    result = sanitize_public_recap_json(source)
    assert 'coverage' not in result['report'] and 'state' not in result['report']['event_results'][0]
    assert result['report']['event_results'][0]['href'] == ''
    assert result['report']['upcoming'][0]['url'] == ''
    assert safe_url('https://user:password@example.com') == ''
    assert result['report']['upcoming'][1]['url'].startswith('https://')


def test_editorial_edits_are_snapshot_fields_and_allow_clearing():
    generated = {'report':{'headline':'Old','opening':'Old intro'},'spotlight':[]}
    result = apply_weekly_recap_edits(generated,{'editorial':{'headline':'New','opening':'','event_notes':'Club note','upcoming':[{'text':'Play','url':'https://example.com'},{'text':'Bad','url':'data:text/html,no'}]}},{})
    assert result['report']['headline'] == 'New' and result['report']['opening'] == ''
    assert result['report']['upcoming'][1]['url'] == ''
    assert generated['report']['opening'] == 'Old intro'
    pdf = build_weekly_recap_pdf_bytes(sanitize_public_recap_json(result),week_start='2026-10-12',week_end='2026-10-18')
    assert b'Club note' in pdf


def test_interclub_uses_only_published_results_for_member_club():
    publication = {'season_id':'season','published':{'name':'Baja League','clubs':[{'id':'club','name':'Tres'},{'id':'other','name':'Guests'}],'meets':[{'id':'meet','starts_at':'2026-10-13T12:00:00Z'}],'competition_results':[{'id':'result','meet_id':'meet','division':'4.0','club_a':'club','club_b':'other','outcome':{'points_a':3,'points_b':1,'games_a':7,'games_b':5}}]}}
    db = DB(pcs_interclub_participations=[{'season_id':'season','club_id':'club','status':'accepted'}],pcs_interclub_publications=[publication])
    report = {'event_results':[]}; _add_interclub(db,'club','2026-10-12','2026-10-18',report)
    assert report['event_results'][0]['summary'] == 'Tres vs Guests'
    assert 'Points: 3–1' in report['event_results'][0]['details']
    report = {'event_results':[]}; _add_interclub(db,'unrelated','2026-10-12','2026-10-18',report)
    assert report['event_results'] == []


def test_tournament_podiums_are_division_scoped_and_use_closeout_week(monkeypatch):
    db = DB(tournament_lifecycle_receipts=[{'id':'receipt','club_id':'club','action':'complete','tournament_id':'t','created_at':'2026-10-13T12:00:00Z'}],tournaments=[{'id':'t','club_id':'club','name':'Baja Classic','status':'COMPLETED'}])
    monkeypatch.setattr('jupr_app.services.public_tournament_results_service.build_public_tournament_results', lambda *a,**k:{'draws':[{'name':'Men 4.0','podium':[{'placement':1,'team_name':'Alex + Bea'}]},{'name':'Mixed 4.5','podium':[{'placement':1,'team_name':'Cam + Dee'}]}]})
    report = {'start_date':'2026-10-12','end_date':'2026-10-18','event_results':[]}
    _add_tournament_draws(db,'club',[],report,{})
    assert len(report['event_results']) == 2
    assert report['event_results'][0]['title'] == 'Baja Classic · Men 4.0'
    report = {'start_date':'2026-10-19','end_date':'2026-10-25','event_results':[]}
    _add_tournament_draws(db,'club',[],report,{})
    assert not report['event_results']


def test_season_races_are_snapshots_and_show_progress_before_eligibility():
    from jupr_app.domain.recaps.weekly_report import _season_races
    matches = [match(i) for i in range(1, 5)] + [match(9,date='2026-11-01T12:00:00+00:00',score_t1=2,score_t2=11)]
    ctx = SimpleNamespace(supabase=DB(),club_id='club',df_matches=pd.DataFrame(matches),
        df_players_all=pd.DataFrame([{'id':i,'name':f'P{i}','rating':1200} for i in range(1,5)]),
        df_leagues=pd.DataFrame([{'league':'Fall','player_id':i,'rating':1200,'is_active':True} for i in range(1,5)]),
        df_meta=pd.DataFrame([{'league_name':'Fall','club_id':'club','league_type':'ladder','match_format':'doubles','awards_config':{'categories':{'most_wins':{'enabled':True,'min_games':5}}}}]))
    races = _season_races(ctx,'2026-10-12','2026-10-18')
    assert races and any('4 of 5 games to qualify' in line for line in races[0]['lines'])
    ctx.df_meta.iloc[0]['awards_config']['categories']['most_wins']['min_games'] = 4
    races = _season_races(ctx,'2026-10-12','2026-10-18')
    assert any('P1 — 4' in line for line in races[0]['lines'])


def test_refresh_preserves_editorial_and_omits_unavailable_spotlight_selection(monkeypatch):
    import jupr_app.services.admin_weekly_recap_service as svc
    before = {'id':'r','row_version':1,'status':'draft','edits_json':{'editorial':{'opening':'Keep my note'},'spotlight_overrides':{'TOP_PERFORMER_WEEK':{'players':['gone'],'include':True}}}}
    monkeypatch.setattr(svc,'is_admin_weekly_recap_enabled',lambda:True)
    monkeypatch.setattr(svc,'require_staging_communications_mutations',lambda:None)
    monkeypatch.setattr(svc,'_fetch_recap_row',lambda *a,**kw:before)
    monkeypatch.setattr(svc,'_recap_context',lambda *a,**kw:SimpleNamespace())
    monkeypatch.setattr(svc,'compute_weekly_recap',lambda *a,**kw:{'report':{'opening':''},'end_date':'2026-10-18','spotlight':[]})
    monkeypatch.setattr(svc,'get_spotlight_candidates',lambda *a,**kw:{})
    monkeypatch.setattr(svc,'_required_audit_intent',lambda *a,**kw:None)
    monkeypatch.setattr(svc,'_audit',lambda *a,**kw:[])
    monkeypatch.setattr(svc,'_upsert_recap_row',lambda *a,**kw:kw['payload'])
    result = svc.generate_admin_weekly_recap(DB(),club_id='club',week_start='2026-10-12',week_end='2026-10-18',actor_email='qa@example.invalid',actor_role='club_owner',confirmation_text='GENERATE RECAP',expected_recap_id='r',expected_row_version=1)
    assert result['recap']['final_json']['report']['opening'] == 'Keep my note'
    assert result['recap']['final_json']['spotlight'] == []
