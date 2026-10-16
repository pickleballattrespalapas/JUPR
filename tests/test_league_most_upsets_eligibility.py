from __future__ import annotations

from copy import deepcopy

import pytest

from jupr_app.domain.league_analytics import award_category_catalog
from jupr_app.services.admin_league_awards_service import (
    _computed_configured_awards,
    _public_award_races,
    freeze_admin_league_awards,
    get_admin_league_awards_wizard,
    get_public_league_award_progress,
    persist_admin_league_awards_preview,
)
from tests.test_admin_match_log_service import FakeSupabase


LEAGUE = "Fall Ladder League 2026"
CATEGORIES = ("most_improved", "best_win_pct", "most_upsets", "partner_variety")


def _storage(games: int) -> dict[str, list[dict]]:
    match = {
        "club_id": "club", "league": LEAGUE, "date": "2026-10-16",
        "t1_p1": 1, "t1_p2": 2, "t2_p1": 3, "t2_p2": 4,
        "t1_p1_r": 1400, "t1_p2_r": 1400,
        "t2_p1_r": 1600, "t2_p2_r": 1600,
        "score_t1": 9, "score_t2": 11,
    }
    return {
        "leagues_metadata": [{
            "club_id": "club", "league_name": LEAGUE,
            "status": "active", "is_active": True,
            "league_type": "Ladder", "match_format": "doubles",
            "min_games": 32, "awards_config_version": 7, "end_awards": {},
            "awards_config": {"categories": {
                key: {"enabled": True, "depth": 3, "minimum": 32}
                for key in CATEGORIES
            }},
        }],
        "matches": [
            {**match, "id": i, "score_t1": 11 if i == 1 else 9,
             "score_t2": 9 if i == 1 else 11}
            for i in range(1, games + 1)
        ] + [
            {**match, "id": 1000, "league": "Another league"},
            {**match, "id": 1001, "club_id": "another-club"},
        ],
        "players": [
            {"club_id": "club", "id": i, "name": f"Player {i}"}
            for i in range(1, 5)
        ],
        "league_ratings": [
            {"club_id": "club", "league_name": LEAGUE, "player_id": i,
             "rating": 1500, "starting_rating": 1400, "is_active": True}
            for i in range(1, 5)
        ],
        "team_league_teams": [], "team_league_fixtures": [],
        "admin_activity_log": [], "badges": [], "player_badges": [],
    }


@pytest.mark.parametrize("games,eligible", [(31, False), (32, True)])
def test_active_league_uses_only_its_games_and_keeps_saved_setup(
    monkeypatch, games, eligible,
) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    tables = _storage(games)
    before = deepcopy(tables)
    supabase = FakeSupabase(tables)

    admin = get_admin_league_awards_wizard(
        supabase, club_id="club", league_name=LEAGUE,
    )
    category = next(row for row in admin["award_catalog"] if row["key"] == "most_upsets")
    assert category["minimum_metric"] == "games"
    assert category["metric"] == "upset_wins"
    assert admin["league"]["status"] == "active"
    assert admin["wizard"]["status"] == "not_started"
    assert {row["games"] for row in admin["player_analytics"]} == {games}

    public = get_public_league_award_progress(
        supabase, club_id="club", league_name=LEAGUE,
    )
    races = {row["category_key"]: row for row in public.get("races", [])}
    assert bool(races) is eligible
    if eligible:
        assert set(races) == set(CATEGORIES)
        race = races["most_upsets"]
        assert race["minimum_metric"] == "games"
        assert race["min_games"] == 32
        assert [row["metric_value"] for row in race["entries"]] == [1, 1, 0, 0]
        assert {row["category_key"] for row in admin["award_progress"]} == set(CATEGORIES)
    else:
        assert admin["award_progress"] == []

    assert tables == before
    assert supabase.operations == []


def test_upset_ranking_preserves_top_three_without_requiring_32_upsets() -> None:
    analytics = {
        "catalog": award_category_catalog(),
        "player_analytics": [
            {"player_id": 1, "player_name": "Below minimum", "games": 31, "upset_wins": 20},
            {"player_id": 2, "player_name": "Third", "games": 32, "upset_wins": 3},
            {"player_id": 3, "player_name": "Second", "games": 40, "upset_wins": 5},
            {"player_id": 4, "player_name": "First", "games": 33, "upset_wins": 7},
            {"player_id": 5, "player_name": "Fourth", "games": 60, "upset_wins": 1},
        ],
    }
    config = {"categories": {"most_upsets": {"enabled": True, "depth": 3, "minimum": 32}}}
    awards = _computed_configured_awards(analytics, config)
    races = _public_award_races(
        {"min_games": 32}, analytics, config, explicit_categories=True,
    )

    assert [(row["player_id"], row["rank"]) for row in awards] == [(4, 1), (3, 2), (2, 3)]
    assert [row["player_id"] for row in races[0]["entries"]] == [4, 3, 2, 5]
    assert all(row["minimum_metric"] == "games" for row in awards)


def test_final_awards_keep_corrected_eligibility_and_recorded_games(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test")
    tables = _storage(32)
    matches = deepcopy(tables["matches"])
    config = deepcopy(tables["leagues_metadata"][0]["awards_config"])
    supabase = FakeSupabase(tables)

    frozen = freeze_admin_league_awards(
        supabase, club_id="club", league_name=LEAGUE,
        actor_email="owner@example.com", actor_role="club_owner",
        confirmation_text="FREEZE LEAGUE AWARDS",
        idempotency_key="freeze:most-upsets-eligibility",
    )
    preview = persist_admin_league_awards_preview(
        supabase, club_id="club", league_name=LEAGUE,
        actor_email="owner@example.com", actor_role="club_owner",
        idempotency_key="preview:most-upsets-eligibility",
    )

    awards = preview["wizard"]["preview"]["awards"]
    upsets = [row for row in awards if row["category_key"] == "most_upsets"]
    assert {row["player_id"] for row in upsets if row["rank"] == 1} == {1, 2}
    assert all(row["metric_value"] < 32 and row["minimum_metric"] == "games" for row in upsets)
    assert {row["category_key"] for row in awards} == set(CATEGORIES)
    assert frozen["wizard"]["frozen_snapshot"]["awards"] == awards
    assert tables["matches"] == matches
    assert tables["leagues_metadata"][0]["awards_config"] == config
    assert tables["leagues_metadata"][0]["awards_config_version"] == 7
