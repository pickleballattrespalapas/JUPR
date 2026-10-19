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
