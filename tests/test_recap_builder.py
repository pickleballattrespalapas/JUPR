from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from jupr_app.domain.recaps.recap_builder import catalog, apply_builder, normalize_config, build_editions, clean_item
from jupr_app.services import admin_weekly_recap_service as service
from jupr_app.services.public_weekly_recap_service import _public_recap_row, sanitize_public_recap_json, _weekly_recap_pdf_lines
from jupr_app.services.recap_preferences_service import save_preferences
from tests.test_admin_match_log_service import FakeSupabase


def fixture():
    return json.loads(Path('apps/web/tests/fixtures/weekly-report.json').read_text())


def test_selection_order_email_and_public_pdf_omit_hidden_content():
    raw = fixture()
    items = catalog(raw)
    config = normalize_config({'order': ['events', 'numbers'], 'sections': {
        'events': {'item_ids': [items['events'][-1]['id'], items['events'][0]['id']], 'email_item_ids': [items['events'][-1]['id']]},
        'numbers': {'enabled': False}, 'achievements': {'enabled': False},
        'spotlight': {'enabled': False}, 'races': {'enabled': False}, 'welcome': {'enabled': False},
    }})
    final = apply_builder(deepcopy(raw), config)
    public = sanitize_public_recap_json(final, week_start='2026-10-05', week_end='2026-10-11')
    assert public['numbers'] == {} and public['around_club'] == {} and public['spotlight'] == []
    assert 'coverage' not in public['report'] and 'catalog' not in public['report']
    assert 'event_results' not in public['report']
    block = public['report']['edition'][0]
    assert [i['id'] for i in block['items']] == config['sections']['events']['item_ids']
    assert len(public['report']['email_edition'][0]['items']) == 1
    pdf = '\n'.join(_weekly_recap_pdf_lines(public, week_start='2026-10-05', week_end='2026-10-11'))
    assert items['events'][1]['title'] not in pdf
    assert 'Weekly Numbers' not in pdf and 'Spotlight Reel' not in pdf


def test_refresh_preserves_explicit_selection_and_omits_removed_sources():
    raw = fixture(); groups = catalog(raw)
    config = {'sections': {'events': {'item_ids': [groups['events'][0]['id']]}}}
    raw['report']['event_results'].reverse()
    raw['report']['event_results'].append({'id': 'new', 'kind': 'Round robin', 'title': 'Unreviewed new event', 'summary': '', 'details': []})
    final = apply_builder(raw, config)
    event = next(b for b in final['report']['edition'] if b['key'] == 'events')
    assert [i['id'] for i in event['items']] == [groups['events'][0]['id']]
    config['sections']['events']['item_ids'] = []
    assert not any(b['key'] == 'events' for b in apply_builder(raw, config)['report']['edition'])


def test_templates_keep_statistics_and_drop_week_specific_stories():
    config = normalize_config({'sections': {'numbers': {'item_ids': ['matches']}, 'events': {'item_ids': ['event-1']}}, 'custom_blocks': [{'id': 'story', 'text': 'Old story'}], 'step': 4}, template=True)
    assert config['sections']['numbers']['item_ids'] == ['matches']
    assert 'item_ids' not in config['sections']['events']
    assert config['custom_blocks'] == []
    assert config['step'] == 0


def test_shared_client_server_projection_fixture():
    shared = json.loads(Path('apps/web/tests/fixtures/weekly-recap-builder.json').read_text())
    assert catalog(fixture()) == shared['catalog']
    assert build_editions(fixture(), shared['config']) == shared['expected']


def test_tournament_draws_with_same_legacy_title_keep_their_own_podiums():
    shared = json.loads(Path('apps/web/tests/fixtures/weekly-recap-draws.json').read_text())
    raw = shared['recap']
    before = deepcopy(raw)
    groups = catalog(raw)
    assert groups == shared['catalog']
    ids = [item['id'] for item in groups['events']]
    assert len(ids) == len(set(ids)) == 4
    # Old checkboxes selected both draws via one ID. Expand that exact choice.
    for key in ('edition', 'email_edition'):
        items = build_editions(raw, shared['config'])[key][0]['items']
        assert [item['lines'][0] for item in items] == shared['winners']
    reordered = deepcopy(raw)
    reordered['report']['event_results'].reverse()
    assert {item['lines'][0]: item['id'] for item in catalog(reordered)['events']} == {
        item['lines'][0]: item['id'] for item in groups['events']}
    config = {'sections': {'events': {'item_ids': [ids[0], ids[3]], 'email_item_ids': [ids[3]]}}}
    editions = build_editions(raw, config)
    assert [item['lines'][0] for item in editions['edition'][0]['items']] == [shared['winners'][0], shared['winners'][3]]
    assert [item['lines'][0] for item in editions['email_edition'][0]['items']] == [shared['winners'][3]]
    assert raw == before


def test_tournament_refresh_preserves_selections_without_transferring_ambiguous_edits():
    shared = json.loads(Path('apps/web/tests/fixtures/weekly-recap-draws.json').read_text())
    old_catalog = catalog(shared['recap'])
    current = shared['refreshed']
    assert catalog(current) == shared['refreshed_catalog']
    selected = old_catalog['events'][0]['id']
    config = {'sections': {'events': {'item_ids': [selected]}}, 'text_overrides': {
        selected: {'title': 'Featured champions'}, shared['config']['sections']['events']['item_ids'][0]: {'title': 'Ambiguous edit'}}}
    items = build_editions(current, config)['edition'][0]['items']
    assert len(items) == 1 and items[0]['lines'][0] == shared['winners'][0]
    assert items[0]['title'] == 'Featured champions'
    all_config = deepcopy(shared['config'])
    all_config['text_overrides'] = {shared['config']['sections']['events']['item_ids'][0]: {'title': 'Ambiguous edit'}}
    all_items = build_editions(current, all_config)['edition'][0]['items']
    assert [item['lines'][0] for item in all_items] == shared['winners']
    assert all(item['title'] != 'Ambiguous edit' for item in all_items)
    public = sanitize_public_recap_json(apply_builder(deepcopy(current), config), week_start='2026-10-26', week_end='2026-11-01')
    assert 'selection_aliases' not in public['report']['edition'][0]['items'][0]
    before = [item['id'] for item in catalog(current)['events']]
    renamed = deepcopy(current)
    renamed['report']['event_results'][0]['title'] = 'A renamed championship draw'
    renamed['report']['event_results'][0]['details'][0] = '1. Corrected names'
    assert [item['id'] for item in catalog(renamed)['events']] == before


def test_compact_ladder_recognition_uses_selected_courts_and_retains_every_name():
    shared = json.loads(Path('apps/web/tests/fixtures/weekly-recap-courts.json').read_text())
    recap = {'report': {'event_results': [{**item, 'summary': item['text'], 'details': item['lines']} for item in shared['events']]}}
    groups = catalog(recap)
    by_source = dict(zip([i['id'] for i in shared['events']], [i['id'] for i in groups['events']]))
    config = {'sections': {'events': {'detail': 'brief', 'email_item_ids': [by_source[i] for i in shared['email_ids']]}}}
    report = build_editions(recap, config)
    assert report['edition'][0]['items'][0]['lines'] == shared['expected_lines']
    assert report['email_edition'][0]['items'][0]['lines'] == shared['expected_email_lines']
    assert len(report['edition'][0]['items']) == 2
    assert report['edition'][0]['items'][1]['title'] == 'Social winners'
    config['sections']['events']['item_ids'] = [by_source['a1'], by_source['a3']]
    selected = build_editions(recap, config)
    assert selected['edition'][0]['items'][0]['lines'] == ['Alex ×2']
    assert selected['email_edition'][0]['items'][0]['lines'] == ['Alex']
    config['text_overrides'] = {'ladder-court-winners': {'title': 'Our court champions', 'lines': {'Alex ×2': 'Alex — two court wins!'}}}
    assert build_editions(recap, config)['edition'][0]['items'][0]['lines'] == ['Alex — two court wins!']
    # A whole-club week can have more players than the old per-card 30-line cap.
    recap['report']['event_results'] = [
        {'id': str(i), 'kind': 'Ladder court', 'title': f'Court {i}', 'summary': f'Winner: Player {i}',
         'details': [f'Player {i} — 3–0 · point difference +9']} for i in range(40)]
    final = apply_builder(recap, {})
    public = sanitize_public_recap_json(final, week_start='2026-10-12', week_end='2026-10-18')
    assert len(public['report']['edition'][0]['items'][0]['lines']) == 40
    assert 'Player 39' in '\n'.join(_weekly_recap_pdf_lines(public, week_start='2026-10-12', week_end='2026-10-18'))


def test_scope_cannot_change_without_regenerating_results():
    with pytest.raises(ValueError, match='scope changed'):
        service.apply_weekly_recap_edits(fixture(), {'scope': {'league_names': ['Other league']}}, {})


def test_draft_revision_and_refresh_leave_published_snapshot_live(monkeypatch):
    monkeypatch.setenv('JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP', '1')
    monkeypatch.setattr(service, 'require_staging_communications_mutations', lambda: None)
    raw = fixture(); raw['spotlight_candidates_snapshot'] = {}
    old = apply_builder(deepcopy(raw), {})
    row = {'id': 'r', 'club_id': 'club', 'week_start': '2026-10-05', 'week_end': '2026-10-11', 'status': 'published', 'row_version': 1, 'generated_json': raw, 'final_json': old, 'published_json': old, 'published_at': '2026-10-12T00:00:00Z', 'published_by': 'admin'}
    db = FakeSupabase({'weekly_recaps': [row], 'admin_activity_log': []})
    identity = dict(club_id='club', week_start='2026-10-05', actor_email='admin', actor_role='club_owner', expected_recap_id='r', expected_row_version=1)
    result = service.save_admin_weekly_recap(db, **identity, confirmation_text='SAVE RECAP', edits_json={'builder': {}, 'editorial': {'headline': 'Private revision'}})
    saved = db.tables['weekly_recaps'][0]
    assert saved['status'] == 'published' and saved['published_json'] == old
    assert _public_recap_row(saved, include_recap=True)['recap']['report']['headline'] != 'Private revision'
    assert result['recap']['has_unpublished_changes']
    identity['expected_row_version'] = result['recap']['row_version']
    service.publish_admin_weekly_recap(db, **identity, action='publish', edits_json=None, confirmation_text='PUBLISH RECAP')
    assert _public_recap_row(db.tables['weekly_recaps'][0], include_recap=True)['recap']['report']['headline'] == 'Private revision'


def test_scoped_stats_filter_before_calculation(monkeypatch):
    from jupr_app.domain.recaps import weekly_recap as recap
    raw = pd.DataFrame([{'league': 'Chosen', 'date': '2026-10-10', 't1_p1': 1, 't1_p2': 2, 't2_p1': 3, 't2_p2': 4, 'score_t1': 11, 'score_t2': 8}, {'league': 'Other', 'date': '2026-10-10', 't1_p1': 5, 't1_p2': 6, 't2_p1': 7, 't2_p2': 8, 'score_t1': 11, 'score_t2': 2}])
    raw['match_type'] = 'League'
    raw['id'] = [1, 2]
    ctx = SimpleNamespace(club_id='club', supabase=None, df_matches=raw, recap_leagues=['Chosen'])
    monkeypatch.setattr(recap, '_load_matches', lambda *a, **kw: raw)
    monkeypatch.setattr(recap, '_load_social_live_data', lambda *a: pytest.fail('Unrelated social data must not be loaded'))
    from datetime import date
    result = recap.compute_weekly_recap(ctx, date(2026,10,5), date(2026,10,11))
    assert result['numbers']['matches'] == 1
    assert result['numbers']['players'] == 4
    assert result['report']['scope_label'] == 'Chosen'
    assert all(e['title'] != 'Other' for e in result['report']['event_results'])


def test_template_conflict_and_cross_club_isolation():
    db = FakeSupabase({'club_recap_preferences': [{'club_id':'a','revision':2,'settings':{}}], 'leagues_metadata':[]})
    with pytest.raises(service.StaleWeeklyRecapStateError):
        save_preferences(db,'a',1,{'templates':[]})
    with pytest.raises(service.StaleWeeklyRecapStateError):
        save_preferences(db,'b',2,{'templates':[]})
    result = save_preferences(db,'a',2,{'templates':[{'id':'default','name':'Member email','builder':{}}], 'default_template_id':'default'})
    assert result['revision'] == 3


def test_custom_links_and_images_are_sanitized():
    final = apply_builder(fixture(), {'custom_blocks':[{'id':'note','title':'Coach tip','text':'Stay patient','url':'javascript:alert(1)','image_url':'data:image/svg+xml,<svg onload=evil />'}]})
    block = next(b for b in final['report']['edition'] if b['key']=='custom')
    assert block['items'][0]['url'] == '' and block['items'][0]['image_url'] == ''


def test_rich_stories_keep_selection_email_order_and_photo_details_in_public_projection():
    config = json.loads(Path('apps/web/tests/fixtures/weekly-recap-stories.json').read_text())['config']
    final = apply_builder(fixture(), config)
    public = sanitize_public_recap_json(final, week_start='2026-10-05', week_end='2026-10-11')
    report = public['report']
    assert [b['key'] for b in report['edition']][:2] == ['custom', 'events']
    assert [b['key'] for b in report['email_edition']][:2] == ['events', 'custom']
    full = report['edition'][0]['items']; email = report['email_edition'][1]['items']
    assert [i['id'] for i in full] == ['photo', 'register']
    assert [i['id'] for i in email] == ['register', 'photo']
    assert full[0]['caption'] == 'Saturday at the club'
    assert full[0]['credit'] == 'Club volunteer'
    assert full[0]['layout'] == 'photo' and full[0]['image_fit'] == 'cover'
    assert full[1]['layout'] == 'promotion'
    assert 'Next week draft' not in json.dumps(public)
    pdf = '\n'.join(_weekly_recap_pdf_lines(public, week_start='2026-10-05', week_end='2026-10-11'))
    assert 'Saturday at the club' in pdf and 'Photo: Club volunteer' in pdf
    template = normalize_config(config, template=True)
    assert template['email_order'][:2] == ['events', 'custom']
    assert template['custom_blocks'] == [] and 'item_ids' not in template['sections']['custom']


@pytest.mark.parametrize('value', ['url(evil)', [], {}, 42, None])
def test_unsupported_card_styles_cannot_inject_styles_or_break_normalization(value):
    item = clean_item({'layout': value, 'image_fit': value, 'caption': 'x' * 800, 'credit': 'y' * 300})
    assert 'layout' not in item and 'image_fit' not in item
    assert len(item['caption']) == 600 and len(item['credit']) == 160


def test_layout_management_preserves_identity_and_rejects_stale_or_duplicate_changes():
    settings = {'templates': [{'id': 'a', 'name': 'Weekly', 'builder': {}}, {'id': 'b', 'name': 'Social', 'builder': {}}], 'default_template_id': 'a'}
    db = FakeSupabase({'club_recap_preferences': [{'club_id': 'club', 'revision': 1, 'settings': settings}], 'leagues_metadata': []})
    renamed = deepcopy(settings); renamed['templates'][0]['name'] = 'Member update'
    result = save_preferences(db, 'club', 1, renamed)
    assert result['revision'] == 2 and result['settings']['default_template_id'] == 'a'
    with pytest.raises(service.StaleWeeklyRecapStateError):
        save_preferences(db, 'club', 1, settings)
    duplicate = deepcopy(renamed); duplicate['templates'][1]['name'] = ' MEMBER UPDATE '
    with pytest.raises(ValueError, match='already exists'):
        save_preferences(db, 'club', 2, duplicate)
    with pytest.raises(ValueError, match='saved default'):
        save_preferences(db, 'club', 2, {'templates': renamed['templates'][1:], 'default_template_id': 'a'})
    result = save_preferences(db, 'club', 2, {'templates': renamed['templates'][1:], 'default_template_id': ''})
    assert result['revision'] == 3 and [t['id'] for t in result['settings']['templates']] == ['b']


def test_preview_text_edits_are_shared_by_public_email_pdf_and_preserve_sources():
    config = json.loads(Path('apps/web/tests/fixtures/weekly-recap-text.json').read_text())
    source = fixture(); original = deepcopy(source)
    final = service.apply_weekly_recap_edits(source, {'builder': config}, {})
    assert source == original
    public = sanitize_public_recap_json(final, week_start='2026-10-05', week_end='2026-10-11')
    for channel in ('edition', 'email_edition'):
        blocks = {b['key']: b for b in public['report'][channel]}
        assert blocks['events']['title'] == 'This week’s results'
        event = next(i for i in blocks['events']['items'] if i['title'] == 'Thursday champions')
        assert event['kind'] == 'Social play' and event['text'] == 'Congratulations, Cam!'
        assert event['lines'][:2] == ['A lively day on the courts', '1. Cam — first place']
        assert event['results_label'] == 'See the final standings'
        league = next(i for i in blocks['events']['items'] if i['links'])
        assert league['links'][0]['label'] == 'See our recorded league week'
        assert blocks['numbers']['items'][0]['text'] == 'Ninety-six'
        assert blocks['upcoming']['items'][0]['text'] == 'Next week’s Moneyball'
        photo = next(i for i in blocks['custom']['items'] if i['id'] == 'photo')
        assert photo['caption'] == 'Friends on court' and photo['credit_label'] == 'Courtesy of'
        assert photo['credit'] == 'Morgan' and photo['alt'] == 'Members celebrating'
    assert public['report']['display_text'] == config['display_text']
    serialized = json.dumps(public)
    assert 'text_overrides' not in serialized and 'Must never appear' not in serialized
    pdf = '\n'.join(_weekly_recap_pdf_lines(public, week_start='2026-10-05', week_end='2026-10-11'))
    assert 'Courtesy of Morgan' in pdf and 'October 5–11 · A week to remember' in pdf and 'Around our club' in pdf
    template = normalize_config(config, template=True)
    assert 'text_overrides' not in template and 'display_text' not in template


def test_text_overrides_follow_source_ids_and_do_not_move_player_edits_to_a_new_winner():
    source = fixture(); config = json.loads(Path('apps/web/tests/fixtures/weekly-recap-text.json').read_text())
    source['report']['event_results'].reverse()
    projected = build_editions(source, config)
    assert 'Thursday champions' in json.dumps(projected)
    player = source['spotlight'][0]
    player['players'] = ['New winner — new result']
    assert 'Alex — a standout performance' not in json.dumps(build_editions(source, config))
    source['report']['event_results'] = []
    assert 'Thursday champions' not in json.dumps(build_editions(source, config))
    event = catalog(fixture())['events'][1]
    config['text_overrides'].pop(event['id'])
    assert event['title'] in json.dumps(build_editions(fixture(), config))
    source['report']['upcoming'].insert(0, {'text': 'A different event', 'url': 'https://example.invalid/new'})
    upcoming = next(b for b in build_editions(source, config)['edition'] if b['key'] == 'upcoming')['items']
    assert [i['text'] for i in upcoming][:2] == ['A different event', 'Next week’s Moneyball']
    source['report']['upcoming'].pop(1)
    assert 'Next week’s Moneyball' not in json.dumps(build_editions(source, config))


def test_text_overrides_allow_blank_text_but_cannot_change_identity_or_urls():
    source = fixture(); event = catalog(source)['events'][1]
    config = {'sections': {'events': {'title': ''}}, 'text_overrides': {event['id']: {
        'title': '', 'text': '<b>Editorial wording</b>', 'results_label': '',
        'lines': {event['lines'][0]: ''}, 'id': 'other', 'href': 'javascript:evil', 'image_url': 'https://untrusted.invalid/tracker',
    }}, 'display_text': {'date': '', 'scope': '', 'footer_link': '', 'unknown': 'secret'}}
    public = sanitize_public_recap_json(apply_builder(source, config), week_start='2026-10-05', week_end='2026-10-11')
    block = next(b for b in public['report']['edition'] if b['key'] == 'events')
    item = next(i for i in block['items'] if i['id'] == event['id'])
    assert block['title'] == '' and item['title'] == '' and item['results_label'] == ''
    assert item['text'] == 'bEditorial wording/b' and event['lines'][0] not in item['lines']
    assert item['href'] == event['href'] and item['image_url'] == event['image_url']
    assert public['report']['display_text'] == {'date': '', 'scope': '', 'footer_link': ''}
    assert normalize_config({'text_overrides': {'item': {'lines': ['bad'], 'title': {}, 'link_labels': 5}}})['text_overrides'] == {'item': {}}
