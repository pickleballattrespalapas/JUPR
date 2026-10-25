from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.domain.recaps.recap_evidence import METRIC_KEYS, court_result, generator_courts, league_courts, social_courts
from jupr_app.domain.recaps.weekly_recap import get_spotlight_candidates
from jupr_app.domain.recaps.weekly_report import build_weekly_report, generator_result
from jupr_app.services.admin_weekly_recap_service import apply_weekly_recap_edits, normalize_spotlight_overrides
from jupr_app.services.public_weekly_recap_service import sanitize_public_recap_json
from tests.test_weekly_recap_range_driven import _ctx
from tests.test_weekly_report import DB, match, session


def rated_games():
    return [match(i, score_t2=9, **{f'{slot}_r_end': 1600 + i * 4 for slot in ('t1_p1', 't1_p2', 't2_p1', 't2_p2')}) for i in (1, 2, 3)]


def candidates(games):
    return get_spotlight_candidates(_ctx(pd.DataFrame(games)), start_date=date(2026, 10, 12), end_date=date(2026, 10, 18))


def test_metric_choices_use_selected_dates_actual_scores_and_last_recorded_rating():
    games = rated_games()
    games += [match(4, date='2026-10-20T12:00:00Z', score_t1=0), match(5, rating_scope='unrated', score_t1=0)]
    result = candidates(games)
    wins = result['METRIC_most_wins'][0]
    assert wins['display'] == 'A — 3 wins · 3 games'
    assert result['METRIC_point_differential'][0]['value_json']['value'] == 6
    assert result['METRIC_average_margin'][0]['value_json']['value'] == 2
    assert result['METRIC_longest_win_streak'][0]['value_json']['value'] == 3
    assert result['METRIC_close_game_record'][0]['display'] == 'A — 100.0% (3/3) · 3 games'
    assert result['METRIC_highest_rating'][0]['display'] == 'A — 4.030 · 3 games'
    assert 'with B (3/3)' in result['METRIC_best_partnership'][0]['display']
    assert result['METRIC_playing_days'][0]['value_json']['value'] == 3


def test_metrics_require_qualifying_games_and_complete_rating_evidence():
    assert all(not value for key, value in candidates(rated_games()[:2]).items() if key in METRIC_KEYS)
    games = rated_games()
    games[-1].update(t1_p1_r_end=None, t2_p1_r=None)
    result = candidates(games)
    assert 'player:1' not in [row['candidate_id'] for row in result['METRIC_highest_rating']]
    assert result['METRIC_opponent_strength'] == []
    assert result['METRIC_most_upsets'] == []
    assert result['METRIC_most_wins']  # Score-only evidence is still valid.


def court_games():
    return [
        {'teamA': [1, 2], 'teamB': [3, 4], 'scoreA': 11, 'scoreB': 5},
        {'teamA': [1, 3], 'teamB': [2, 4], 'scoreA': 11, 'scoreB': 5},
        {'teamA': [1, 4], 'teamB': [2, 3], 'scoreA': 11, 'scoreB': 5},
    ]


def result(games, complete=True):
    return court_result(identity='court', title='Fall', day='2026-10-13', round_number=2, court_number=1,
                        games=games, names={'1': 'Alex', '2': 'Bea', '3': 'Cam', '4': 'Dee'}, complete=complete)


def test_court_results_distinguish_winner_tie_and_incomplete_evidence():
    item = result(court_games())
    assert item['summary'] == 'Winner: Alex'
    assert item['details'][0] == 'Alex — 3–0 · point difference +18'
    assert result(court_games()[:1])['summary'] == 'Tied for first: Alex, Bea'
    assert result(court_games()[:1], complete=False)['summary'].startswith('Leader from recorded games:')
    games = court_games(); games[-1]['scoreA'] = None
    assert result(games)['summary'].startswith('Leader from recorded games:')
    assert 'incomplete' in result(games)['details'][-1]
    assert result([{'teamA': [1, 1], 'teamB': [2, 3], 'scoreA': 11, 'scoreB': 5}]) is None


def league_fixture():
    games = [{**match(index, date='2026-10-13T12:00:00Z'),
              **dict(zip(('t1_p1', 't1_p2', 't2_p1', 't2_p2'), game['teamA'] + game['teamB'])),
              'score_t1': game['scoreA'], 'score_t2': game['scoreB']} for index, game in enumerate(court_games(), 1)]
    saved = [{**game, 'court': 1} for game in games]
    round_row = {'id': 'r1', 'club_id': 'club', 'status': 'submitted', 'session_id': 'session', 'round_number': 1,
                 'match_date': '2026-10-13', 'matches_json': saved, 'submitted_match_ids': [1, 2, 3]}
    tables = {'league_live_rounds': [round_row, {**round_row, 'id': 'foreign', 'club_id': 'other'},
                                   {**round_row, 'id': 'draft', 'status': 'draft'}, {**round_row, 'id': 'later', 'match_date': '2026-10-20'}],
              'league_live_sessions': [{'id': 'session', 'club_id': 'club', 'league_name': 'Fall', 'week_tag': 'Week 1'}]}
    ctx = SimpleNamespace(supabase=DB(**tables), club_id='club', id_to_name={1: 'Alex', 2: 'Bea', 3: 'Cam', 4: 'Dee'})
    return ctx, games


def test_league_courts_are_club_date_scope_and_submission_scoped_and_use_corrected_scores():
    ctx, games = league_fixture()
    rows, warnings = league_courts(ctx, games, '2026-10-12', '2026-10-18')
    assert not warnings and len(rows) == 1
    assert rows[0]['summary'] == 'Winner: Alex'
    assert rows[0]['id'] == 'league:session:1:1'
    for game in games:
        game['score_t1'], game['score_t2'] = 5, 11
    corrected, _ = league_courts(ctx, games, '2026-10-12', '2026-10-18')
    assert corrected[0]['summary'] == 'Tied for first: Bea, Cam, Dee'
    ctx.recap_leagues = ['Other']
    assert league_courts(ctx, games, '2026-10-12', '2026-10-18') == ([], [])


@pytest.mark.parametrize('change', ['deleted', 'missing', 'seats', 'date', 'league'])
def test_league_courts_omit_stale_or_unverifiable_records(change):
    ctx, games = league_fixture()
    if change == 'missing': games.pop()
    elif change == 'deleted': games[-1]['deleted_at'] = '2026-10-14'
    elif change == 'seats': games[-1]['t1_p1'] = 9
    elif change == 'date': games[-1]['date'] = '2026-10-20'
    elif change == 'league': games[-1]['league'] = 'Other'
    rows, warnings = league_courts(ctx, games, '2026-10-12', '2026-10-18')
    assert not rows and warnings


def test_generator_courts_use_saved_rounds_and_report_approval_gate():
    from jupr_app.domain.adaptive_play_engine import create_generator_preview, start_generator_event, save_generator_round
    event = start_generator_event(create_generator_preview(generator_kind='ladder', play_format='doubles', title='Morning ladder',
                                                          participant_names=['Alex', 'Bea', 'Cam', 'Dee'], player_ids=[1, 2, 3, 4], total_rounds=2))
    games = event['rounds'][0]['courts'][0]['matches']
    event = save_generator_round(event, round_number=1, scores=[{'match_id': game['id'], 'score_a': 11, 'score_b': 5} for game in games])
    row = session(event)
    verified, _ = generator_result(row, '2026-10-12', '2026-10-18')
    courts = generator_courts(row, verified)
    assert len(courts) == 1 and 'Round 1 · Court 1' in courts[0]['title']
    ctx = SimpleNamespace(supabase=DB(live_sessions=[{**row, 'club_id': 'club', 'state->>mode': 'admin_play_generator'}]), club_id='club', id_to_name={})
    report = build_weekly_report(ctx, {'numbers': {}}, pd.DataFrame(), {}, date(2026, 10, 12), date(2026, 10, 18))
    assert any(item['kind'] == 'Ladder court' for item in report['event_results'])
    ctx.supabase.tables['live_sessions'][0]['state']['generator_submission']['status'] = 'pending'
    report = build_weekly_report(ctx, {'numbers': {}}, pd.DataFrame(), {}, date(2026, 10, 12), date(2026, 10, 18))
    assert not report['event_results'] and report['coverage']['pending']


def test_new_metrics_are_opt_in_and_all_explicit_players_survive_public_and_email_selection():
    key = 'METRIC_most_wins'
    options = {key: [{'candidate_id': str(i), 'label': 'Most wins', 'display': f'Chosen player {i} — {12-i} wins'} for i in range(12)]}
    raw = {'report': {}, 'spotlight': [], 'spotlight_candidates_snapshot': options}
    assert normalize_spotlight_overrides({}, [])[key]['include'] is False
    assert not apply_weekly_recap_edits(raw, {'builder': {}}, options)['spotlight']
    edits = {'spotlight_overrides': {key: {'include': True, 'players': [str(i) for i in range(1, 11)]}},
             'builder': {'sections': {'spotlight': {'detail': 'brief', 'email_item_ids': [f'spotlight-{key}']}}}}
    final = apply_weekly_recap_edits(raw, edits, options)
    public = sanitize_public_recap_json(final, week_start='2026-10-12', week_end='2026-10-18')
    for channel in ('edition', 'email_edition'):
        block = next(b for b in public['report'][channel] if b['key'] == 'spotlight')
        assert block['items'][0]['lines'] == [row['display'] for row in options[key][1:11]]
    assert 'spotlight_candidates_snapshot' not in public
    edits['spotlight_overrides'][key]['players'] = []
    assert not apply_weekly_recap_edits(raw, edits, options)['spotlight']


def test_social_ladder_courts_use_actual_saved_seats_and_disclose_partial_scores():
    source = {'event_type': 'league', 'raw_event_json': {
        'participants': [{'id': str(i), 'name': name} for i, name in enumerate(['Alex', 'Bea', 'Cam', 'Dee'], 1)],
        'rounds': [{'number': 1, 'courts': [{'courtNumber': 2, 'miniRounds': [{'matches': court_games()}]}]}],
    }}
    verified = {'id': 'social', 'title': 'Social ladder', 'date': '2026-10-13'}
    rows = social_courts(source, verified)
    assert rows[0]['id'] == 'social:social:1:2' and rows[0]['summary'] == 'Winner: Alex'
    source['raw_event_json']['rounds'][0]['courts'][0]['miniRounds'][0]['matches'][-1]['scoreA'] = None
    assert social_courts(source, verified)[0]['summary'].startswith('Leader from recorded games:')
    source['event_type'] = 'round_robin'
    assert social_courts(source, verified) == []


def test_league_float_ids_keep_names_complete_courts_and_combined_identity():
    ctx, games = league_fixture()
    for index, game in enumerate(games):
        game["id"] = float(game["id"])
        for slot in ("t1_p1", "t1_p2", "t2_p1", "t2_p2"):
            game[slot] = float(game[slot]) if index % 2 else game[slot]
    rows, warnings = league_courts(ctx, games, '2026-10-12', '2026-10-18')
    assert not warnings
    assert rows[0]['summary'] == 'Winner: Alex'
    assert rows[0]['court_players'] == [{'id': 'league:1', 'name': 'Alex'}]
    assert rows[0]['details'][0] == 'Alex — 3–0 · point difference +18'
    assert rows[0]['details'][-1] == '3 recorded games'


def test_mixed_numeric_ids_cannot_hide_duplicate_seats():
    assert result([{'teamA': [1, 1.0], 'teamB': [2, 3], 'scoreA': 11, 'scoreB': 5}]) is None


def test_opaque_generator_ids_remain_distinct():
    from jupr_app.domain.recaps.recap_evidence import identity_key
    assert identity_key('player-101') == 'player-101'
    assert identity_key('101.5') == '101.5'
    assert identity_key('101.0') == '101'


def test_match_loader_keeps_nullable_ids_as_integers_without_bigint_rounding(monkeypatch):
    from jupr_app.data import load
    large_id = 9007199254740993
    raw = [{'id': 1, 't1_p1': 101, 't1_p2': large_id, 't2_p1': 601, 't2_p2': 2},
           {'id': 2, 't1_p1': None, 't1_p2': None, 't2_p1': 601, 't2_p2': None}]
    monkeypatch.setattr(load, '_read_rows', lambda *args, **kwargs: raw)
    frame, degraded, _ = load._fetch_matches(None, 'club', None)
    assert not degraded
    assert all(str(frame[col].dtype) == 'Int64' for col in raw[0])
    records = frame.to_dict('records')
    assert records[0]['t1_p1'] == 101 and isinstance(records[0]['t1_p1'], int)
    assert records[0]['t1_p2'] == large_id
    assert records[1]['t1_p2'] is None
    assert raw[0]['t1_p2'] == large_id
