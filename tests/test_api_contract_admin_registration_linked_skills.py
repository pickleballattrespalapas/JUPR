from copy import deepcopy
import pytest
from tests.test_api_contract_admin_tournament import FakeSupabase, tournament_tables
from jupr_app.services.admin_tournament_service import get_admin_tournament_detail, update_admin_tournament_registration


def setup_case(monkeypatch, *, singles_games=0, player_club='club', name='Linked Player'):
    monkeypatch.setenv('JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS', '1')
    tables = tournament_tables()
    tables['players'] = [dict(id=200, club_id=player_club, name=name, rating=2046.3415,
                              singles_rating=1800, singles_matches_played=singles_games)]
    row = tables['tournament_registrations'][0]
    row.update(player_id=200, doubles_skill=4.8, singles_skill=4.0)
    return tables, FakeSupabase(tables)


def save(client, tables, patch, **kwargs):
    return update_admin_tournament_registration(client, club_id='club', tournament_id='tour_1',
        registration_id='registration_1', patch=patch,
        expected_updated_at=tables['tournament_registrations'][0]['updated_at'],
        actor_email='staff@example.com', actor_role='administrator', confirmation_text='SAVE REGISTRATION', **kwargs)


def test_editor_exposes_current_skills_without_rewriting_snapshot(monkeypatch):
    tables, client = setup_case(monkeypatch)
    before = deepcopy(tables['tournament_registrations'])
    row = get_admin_tournament_detail(client, club_id='club', tournament_id='tour_1')['registrations'][0]
    assert row['doubles_skill'] == 4.8
    assert row['linked_profile_skills']['doubles_skill'] == pytest.approx(5.11585375)
    assert row['linked_profile_skills']['singles_skill'] is None
    assert tables['tournament_registrations'] == before


@pytest.mark.parametrize('singles_games,expected', [(0, 4.0), (3, 4.5)])
def test_save_uses_profile_not_stale_or_forged_skills(monkeypatch, singles_games, expected):
    tables, client = setup_case(monkeypatch, singles_games=singles_games)
    result = save(client, tables, {'doubles_skill': 2.0, 'singles_skill': 4.0})
    assert result['registration']['doubles_skill'] == pytest.approx(5.11585375)
    assert result['registration']['singles_skill'] == expected
    assert tables['tournament_registrations'][0]['doubles_skill'] == pytest.approx(5.11585375)


def test_new_link_refreshes_skills_during_preflight_without_writes(monkeypatch):
    tables, client = setup_case(monkeypatch)
    tables['tournament_registrations'][0]['player_id'] = None
    before = deepcopy(tables['tournament_registrations'])
    result = save(client, tables, {'player_id': 200}, dry_run=True)
    assert result['patch']['doubles_skill'] == pytest.approx(5.11585375)
    assert 'singles_skill' not in result['patch']
    assert tables['tournament_registrations'] == before


def test_unlink_allows_manual_skills(monkeypatch):
    tables, client = setup_case(monkeypatch)
    result = save(client, tables, {'player_id': None, 'doubles_skill': 4.2}, dry_run=True)
    assert result['patch']['doubles_skill'] == 4.2


@pytest.mark.parametrize('player_club,name', [('other', 'Linked Player'), ('club', 'Old (MERGED into New #201)')])
def test_unavailable_link_cannot_supply_skills(monkeypatch, player_club, name):
    tables, client = setup_case(monkeypatch, player_club=player_club, name=name)
    row = get_admin_tournament_detail(client, club_id='club', tournament_id='tour_1')['registrations'][0]
    assert row['linked_profile_skills'] is None
    with pytest.raises(ValueError):
        save(client, tables, {'doubles_skill': 4.8}, dry_run=True)
