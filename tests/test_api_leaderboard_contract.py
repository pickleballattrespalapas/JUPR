import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(
        "services.api.main.get_club",
        lambda club_slug: {
            "club_id": "club-1",
            "club_slug": club_slug,
            "club_name": "Test Club",
        },
    )
    monkeypatch.setattr("services.api.main.get_supabase_client", lambda: object())

    def fake_build_public_leaderboard(
        supabase,
        *,
        club_id,
        league_name=None,
        league_view="active",
        status="active",
        search=None,
        sort="rank",
        player_id=None,
        limit=50,
        offset=0,
    ):
        assert club_id == "club-1"
        rows = [
            {
                "rank_position": 7,
                "club_id": "club-1",
                "league_name": league_name or "Open",
                "player_id": "p1",
                "player_name": "Alex",
                "rating": 1600,
                "rating_jupr": 1600,
                "wins": 10,
                "losses": 2,
                "matches_played": 12,
                "is_active": True,
                "updated_at": "2026-05-04T00:00:00Z",
                "email": "private@example.com",
            },
            {
                "club_id": "club-1",
                "league_name": league_name or "Open",
                "player_id": "p2",
                "player_name": "Blair",
                "rating": 1500,
                "wins": 8,
                "losses": 4,
                "matches_played": 12,
                "is_active": True,
                "internal_notes": "hidden",
            },
        ]
        return {
            "scopes": [{"name": "OVERALL", "label": "Overall", "min_games": 0}, {"name": league_name or "Open", "label": league_name or "Open", "min_games": 0}],
            "selected_scope": league_name or "Open",
            "scope": {"name": league_name or "Open", "label": league_name or "Open", "min_games": 0},
            "filters": {"league_view": league_view, "status": status, "search": search or "", "sort": sort},
            "summary": {"ranked_players": 2, "active_players": 2, "inactive_players": 0, "leaderboard_scopes": 2, "filtered_players": 2},
            "leaderboard": rows,
            "snapshot": None,
            "highlights": {"highest_rating": rows[:1], "most_improved": [], "best_win_pct": rows[:1], "most_wins": rows[:1]},
            "pagination": {"total": 2, "offset": offset, "limit": limit, "has_more": False},
        }

    monkeypatch.setattr("services.api.main.build_public_leaderboard", fake_build_public_leaderboard)
    return TestClient(app)


def test_primary_and_compat_routes_return_same_normalized_contract(client):
    primary = client.get("/clubs/test-club/leaderboards?league_name=Pro")
    compat = client.get("/clubs/test-club/leaderboards/public?league_name=Pro")

    assert primary.status_code == 200
    assert compat.status_code == 200
    assert primary.json() == compat.json()

    payload = primary.json()
    assert payload["club"] == {"id": "club-1", "slug": "test-club", "name": "Test Club"}
    assert isinstance(payload["leaderboard"], list)

    first, second = payload["leaderboard"]
    assert first["rank"] == 7
    assert first["rank_position"] == 7
    assert "email" not in first

    assert second["rank"] == 2
    assert "rank_position" not in second or second["rank_position"] is None
    assert "internal_notes" not in second


@pytest.mark.parametrize('suffix', ['', '/public'])
def test_season_parameter_and_public_settings_contract(client, monkeypatch, suffix):
    from services.api import main
    from services.api.club_site_models import LeaderboardSettings
    previous = main.build_public_leaderboard
    settings = LeaderboardSettings(cards=['most_matches'], show_summary=False, seasons=[{
        'id': 'winter', 'name': 'Winter', 'start_date': '2026-09-15',
        'end_date': None, 'timezone': 'America/Mazatlan',
    }], default_season_id='winter').model_dump(mode='json')

    def build(db, *, season=None, **kwargs):
        assert season == 'winter'
        payload = previous(db, **kwargs)
        payload['leaderboard_settings'] = settings
        payload['period'] = {**settings['seasons'][0], 'secret': 'hidden'}
        payload['highlights']['most_matches'] = payload['leaderboard']
        return payload

    monkeypatch.setattr(main, 'build_public_leaderboard', build)
    response = client.get(f'/clubs/test-club/leaderboards{suffix}?season=winter')
    assert response.status_code == 200
    data = response.json()
    assert data['leaderboard_settings'] == settings
    assert data['period'] == settings['seasons'][0]
    assert len(data['highlights']['most_matches']) == 2
    assert 'email' not in data['highlights']['most_matches'][0]


def test_unknown_season_returns_actionable_client_error(client, monkeypatch):
    from services.api import main
    from jupr_app.services.leaderboard_service import LeaderboardPeriodInvalid

    def build(*args, **kwargs):
        raise LeaderboardPeriodInvalid('That leaderboard season is unavailable. Choose another season or All time.')

    monkeypatch.setattr(main, 'build_public_leaderboard', build)
    response = client.get('/clubs/test-club/leaderboards?season=deleted')
    assert response.status_code == 422
    assert 'All time' in response.json()['detail']


@pytest.mark.parametrize('suffix', ['', '/public'])
def test_all_card_metrics_are_public_but_internal_calculations_are_not(client, monkeypatch, suffix):
    from jupr_app.domain.leaderboard_metrics import LEADERBOARD_CARD_KEYS
    from services.api import main
    from services.api.club_site_models import LeaderboardSettings
    previous = main.build_public_leaderboard
    settings = LeaderboardSettings(
        cards=list(LEADERBOARD_CARD_KEYS),
        card_options={'best_partnership': {'minimum': 8, 'depth': 3}},
        timezone='Pacific/Auckland',
    ).model_dump(mode='json')

    def build(db, **kwargs):
        payload = previous(db, **kwargs)
        payload['leaderboard_settings'] = settings
        payload['highlights'] = {key: [{
            **payload['leaderboard'][0], 'metric_value': 75.0,
            'metric_display': '75.0% with Blair · 8 games', 'metric_sample': 8,
            '_partners': {'private': 'internal calculation'},
        }] for key in LEADERBOARD_CARD_KEYS}
        payload['highlights']['private_metric'] = [{'email': 'hidden'}]
        return payload

    monkeypatch.setattr(main, 'build_public_leaderboard', build)
    response = client.get(f'/clubs/test-club/leaderboards{suffix}')
    assert response.status_code == 200
    data = response.json()
    assert data['leaderboard_settings'] == settings
    assert data['period']['timezone'] == 'Pacific/Auckland'
    assert set(data['highlights']) == set(LEADERBOARD_CARD_KEYS)
    for rows in data['highlights'].values():
        assert rows[0]['metric_value'] == 75.0
        assert rows[0]['metric_display'] == '75.0% with Blair · 8 games'
        assert rows[0]['metric_sample'] == 8
        assert 'email' not in rows[0]
        assert '_partners' not in rows[0]


@pytest.mark.parametrize('suffix', ['', '/public'])
def test_team_upset_projection_keeps_both_teammates_without_private_fields(client, monkeypatch, suffix):
    from services.api import main
    previous = main.build_public_leaderboard

    def build(db, **kwargs):
        payload = previous(db, **kwargs)
        payload['highlights']['biggest_upset'] = [{
            'rank': 1, 'team_key': 'p1:p2', 'player_id': None,
            'player_name': 'Alex & Blair',
            'team_members': [
                {'player_id': 'p1', 'player_name': 'Alex', 'email': 'private@example.test'},
                {'player_id': 'p2', 'player_name': 'Blair', 'internal_notes': 'hidden'},
            ],
            'metric_value': 0.474, 'metric_display': '+0.474 JUPR', 'metric_sample': 2,
            'matches_played': 8, '_matches': [{'secret': 'hidden'}],
        }]
        return payload

    monkeypatch.setattr(main, 'build_public_leaderboard', build)
    response = client.get(f'/clubs/test-club/leaderboards{suffix}')
    assert response.status_code == 200
    teams = response.json()['highlights']['biggest_upset']
    assert len(teams) == 1
    assert teams[0]['team_key'] == 'p1:p2'
    assert teams[0]['player_id'] is None
    assert teams[0]['team_members'] == [
        {'player_id': 'p1', 'player_name': 'Alex'},
        {'player_id': 'p2', 'player_name': 'Blair'},
    ]
    assert teams[0]['rank'] == 1
    assert teams[0]['metric_value'] == 0.474
    assert teams[0]['metric_sample'] == 2
    assert '_matches' not in teams[0]
