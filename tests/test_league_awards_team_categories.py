from __future__ import annotations

import pytest

from jupr_app.services.admin_league_awards_service import (
    freeze_admin_league_awards,
    get_public_league_award_progress,
    persist_admin_league_awards_preview,
    save_admin_league_award_overrides,
    save_admin_league_awards_config,
)
from tests.test_admin_match_log_service import FakeSupabase


class CountingSupabase(FakeSupabase):
    def __init__(self, tables) -> None:
        super().__init__(tables)
        self.table_calls: list[str] = []

    def table(self, name):
        self.table_calls.append(str(name))
        return super().table(name)


def _storage() -> dict[str, list[dict]]:
    return {
        "leagues_metadata": [
            {
                "club_id": "club",
                "league_name": "Open",
                "status": "active",
                "is_active": True,
                "league_type": "Team",
                "match_format": "doubles",
                "min_games": 1,
                "awards_config": {},
                "awards_config_version": 0,
                "end_awards": {},
            }
        ],
        "league_ratings": [],
        "matches": [],
        "players": [
            {"club_id": "club", "id": 1, "name": "Alex"},
            {"club_id": "club", "id": 2, "name": "Blair"},
            {"club_id": "club", "id": 3, "name": "Casey"},
            {"club_id": "club", "id": 4, "name": "Devon"},
        ],
        "team_league_teams": [
            {
                "id": "team-a",
                "club_id": "club",
                "league_name": "Open",
                "team_name": "Aces",
                "status": "confirmed",
                "captain_player_id": 1,
                "partner_player_id": 2,
            },
            {
                "id": "team-b",
                "club_id": "club",
                "league_name": "Open",
                "team_name": "Dinkers",
                "status": "confirmed",
                "captain_player_id": 3,
                "partner_player_id": 4,
            },
        ],
        "team_league_fixtures": [
            {
                "id": "fixture-1",
                "club_id": "club",
                "league_name": "Open",
                "phase": "regular",
                "status": "complete",
                "week_number": 1,
                "team_a_id": "team-a",
                "team_b_id": "team-b",
                "team_a_score": 11,
                "team_b_score": 8,
                "winner_team_id": "team-a",
            }
        ],
        "admin_activity_log": [],
        "badges": [],
        "player_badges": [],
    }


def _enable(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test")


def test_team_award_category_survives_freeze_preview_and_confirmation(
    monkeypatch,
) -> None:
    _enable(monkeypatch)
    supabase = FakeSupabase(_storage())

    configured = save_admin_league_awards_config(
        supabase,
        club_id="club",
        league_name="Open",
        awards_config={
            "categories": {
                "team_champion": {
                    "enabled": True,
                    "depth": 1,
                    "minimum": 1,
                }
            }
        },
        expected_config_version=0,
        actor_email="owner@example.com",
        actor_role="club_owner",
    )
    assert configured["awards_config_version"] == 1
    assert configured["team_analytics"][0]["team_name"] == "Aces"

    frozen = freeze_admin_league_awards(
        supabase,
        club_id="club",
        league_name="Open",
        actor_email="owner@example.com",
        actor_role="club_owner",
        confirmation_text="FREEZE LEAGUE AWARDS",
        idempotency_key="freeze:team-category",
    )
    assert frozen["wizard"]["frozen_snapshot"]["awards"][0]["team_id"] == "team-a"

    previewed = persist_admin_league_awards_preview(
        supabase,
        club_id="club",
        league_name="Open",
        actor_email="owner@example.com",
        actor_role="club_owner",
        idempotency_key="preview:team-category",
    )
    confirmed = save_admin_league_award_overrides(
        supabase,
        club_id="club",
        league_name="Open",
        overrides=[],
        preview_fingerprint=previewed["wizard"]["preview"]["fingerprint"],
        actor_email="owner@example.com",
        actor_role="club_owner",
        idempotency_key="override:team-category",
    )

    award = confirmed["wizard"]["final_awards"][0]
    assert award["recipient_type"] == "team"
    assert award["team_id"] == "team-a"
    assert award["recipient_name"] == "Aces"


def test_award_catalog_only_exposes_measures_supported_by_league_format(
    monkeypatch,
) -> None:
    _enable(monkeypatch)
    storage = _storage()
    supabase = FakeSupabase(storage)

    team = save_admin_league_awards_config(
        supabase,
        club_id="club",
        league_name="Open",
        awards_config={"categories": {}},
        expected_config_version=0,
        actor_email="owner@example.com",
        actor_role="club_owner",
    )
    assert "team_champion" in {row["key"] for row in team["award_catalog"]}

    storage["leagues_metadata"][0].update(
        league_type="Individual", match_format="singles"
    )
    individual = save_admin_league_awards_config(
        supabase,
        club_id="club",
        league_name="Open",
        awards_config={"categories": {}},
        expected_config_version=1,
        actor_email="owner@example.com",
        actor_role="club_owner",
    )
    keys = {row["key"] for row in individual["award_catalog"]}
    assert "team_champion" not in keys
    assert "best_partnership" not in keys
    assert "partner_variety" not in keys


def test_individual_league_rejects_team_only_award_configuration(
    monkeypatch,
) -> None:
    _enable(monkeypatch)
    storage = _storage()
    storage["leagues_metadata"][0].update(
        league_type="Individual", match_format="doubles"
    )

    with pytest.raises(ValueError, match="Unknown award categories: team_champion"):
        save_admin_league_awards_config(
            FakeSupabase(storage),
            club_id="club",
            league_name="Open",
            awards_config={
                "categories": {
                    "team_champion": {"enabled": True, "depth": 1, "minimum": 1}
                }
            },
            expected_config_version=0,
            actor_email="owner@example.com",
            actor_role="club_owner",
        )


def test_singles_match_drives_live_and_public_award_progress(monkeypatch) -> None:
    _enable(monkeypatch)
    storage = _storage()
    storage["leagues_metadata"][0].update(
        league_type="Individual",
        match_format="singles",
    )
    storage["league_ratings"] = [
        {
            "club_id": "club",
            "league_name": "Open",
            "player_id": player_id,
            "rating": rating,
            "starting_rating": rating,
            "is_active": True,
        }
        for player_id, rating in ((1, 1600), (3, 1800))
    ]
    storage["matches"] = [
        {
            "id": 99,
            "club_id": "club",
            "league": "Open",
            "match_format": "singles",
            "date": "2026-08-15",
            "week_tag": "Week 1",
            "t1_p1": 1,
            "t1_p2": None,
            "t2_p1": 3,
            "t2_p2": None,
            "score_t1": 11,
            "score_t2": 8,
            "t1_p1_r": 1600,
            "t2_p1_r": 1800,
        }
    ]
    supabase = CountingSupabase(storage)

    configured = save_admin_league_awards_config(
        supabase,
        club_id="club",
        league_name="Open",
        awards_config={
            "categories": {
                "most_wins": {"enabled": True, "depth": 1, "minimum": 1}
            }
        },
        expected_config_version=0,
        actor_email="owner@example.com",
        actor_role="club_owner",
    )

    assert configured["provenance"]["included_count"] == 1
    assert configured["provenance"]["exclusion_counts"] == {}
    assert [row["player_name"] for row in configured["player_analytics"]] == [
        "Alex",
        "Casey",
    ]
    assert configured["award_progress"][0]["recipient_name"] == "Alex"
    public = get_public_league_award_progress(
        supabase, club_id="club", league_name="Open"
    )
    assert public["award_count"] == 1
    assert public["awards"][0]["category_key"] == "most_wins"
    assert public["awards"][0]["recipient_name"] == "Alex"
    assert supabase.table_calls[-4:] == [
        "leagues_metadata",
        "league_ratings",
        "matches",
        "players",
    ]

    supabase.table_calls.clear()
    preloaded = get_public_league_award_progress(
        supabase,
        club_id="club",
        league_name="Open",
        metadata=storage["leagues_metadata"][0],
        league_rows=storage["league_ratings"],
        match_rows=storage["matches"],
        player_rows=storage["players"],
        team_rows=(),
        fixture_rows=(),
    )
    assert preloaded == public
    assert supabase.table_calls == []


def test_public_awards_skip_analytics_reads_without_enabled_categories() -> None:
    supabase = CountingSupabase(_storage())

    public = get_public_league_award_progress(
        supabase, club_id="club", league_name="Open"
    )

    assert public == {"awards": [], "award_count": 0}
    assert supabase.table_calls == ["leagues_metadata"]


def test_public_awards_wait_for_configured_minimum(monkeypatch) -> None:
    _enable(monkeypatch)
    storage = _storage()
    storage["leagues_metadata"][0].update(
        league_type="Individual",
        match_format="singles",
        awards_config={
            "categories": {
                "most_wins": {"enabled": True, "depth": 1, "minimum": 2}
            }
        },
    )
    storage["league_ratings"] = [
        {
            "club_id": "club",
            "league_name": "Open",
            "player_id": player_id,
            "rating": 1600,
            "starting_rating": 1600,
            "is_active": True,
        }
        for player_id in (1, 3)
    ]
    storage["matches"] = [
        {
            "id": 100,
            "club_id": "club",
            "league": "Open",
            "date": "2026-08-16",
            "t1_p1": 1,
            "t1_p2": None,
            "t2_p1": 3,
            "t2_p2": None,
            "score_t1": 11,
            "score_t2": 8,
        }
    ]

    public = get_public_league_award_progress(
        FakeSupabase(storage), club_id="club", league_name="Open"
    )

    assert public == {"awards": [], "award_count": 0}


def test_player_co_winners_survive_freeze_preview_and_confirmation(
    monkeypatch,
) -> None:
    _enable(monkeypatch)
    storage = _storage()
    storage["league_ratings"] = [
        {
            "club_id": "club",
            "league_name": "Open",
            "player_id": player_id,
            "rating": 1600,
            "starting_rating": 1600,
        }
        for player_id in (1, 2, 3, 4)
    ]
    storage["matches"] = [
        {
            "id": 1,
            "club_id": "club",
            "league": "Open",
            "date": "2026-08-01",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "score_t1": 11,
            "score_t2": 8,
            "t1_p1_r": 1600,
            "t1_p2_r": 1600,
            "t2_p1_r": 1600,
            "t2_p2_r": 1600,
        }
    ]
    supabase = FakeSupabase(storage)
    save_admin_league_awards_config(
        supabase,
        club_id="club",
        league_name="Open",
        awards_config={
            "categories": {
                "most_wins": {
                    "enabled": True,
                    "depth": 1,
                    "minimum": 1,
                }
            }
        },
        expected_config_version=0,
        actor_email="owner@example.com",
        actor_role="club_owner",
    )
    frozen = freeze_admin_league_awards(
        supabase,
        club_id="club",
        league_name="Open",
        actor_email="owner@example.com",
        actor_role="club_owner",
        confirmation_text="FREEZE LEAGUE AWARDS",
        idempotency_key="freeze:co-winners",
    )
    frozen_awards = frozen["wizard"]["frozen_snapshot"]["awards"]
    assert [row["player_name"] for row in frozen_awards] == [
        "Alex",
        "Blair",
    ]
    assert len({row["award_key"] for row in frozen_awards}) == 2
    assert all(row["rank"] == 1 for row in frozen_awards)
    assert all(row["is_co_winner"] for row in frozen_awards)

    previewed = persist_admin_league_awards_preview(
        supabase,
        club_id="club",
        league_name="Open",
        actor_email="owner@example.com",
        actor_role="club_owner",
        idempotency_key="preview:co-winners",
    )
    confirmed = save_admin_league_award_overrides(
        supabase,
        club_id="club",
        league_name="Open",
        overrides=[],
        preview_fingerprint=previewed["wizard"]["preview"]["fingerprint"],
        actor_email="owner@example.com",
        actor_role="club_owner",
        idempotency_key="override:co-winners",
    )

    final_awards = confirmed["wizard"]["final_awards"]
    assert [row["player_name"] for row in final_awards] == [
        "Alex",
        "Blair",
    ]
    assert len({row["award_key"] for row in final_awards}) == 2

    alex_award = next(
        row for row in frozen_awards if row["player_name"] == "Alex"
    )
    corrected = save_admin_league_award_overrides(
        supabase,
        club_id="club",
        league_name="Open",
        overrides=[
            {
                "award_key": alex_award["award_key"],
                "category_key": alex_award["category_key"],
                "rank": alex_award["rank"],
                "player_id": 3,
                "reason": "Committee correction after score review",
            }
        ],
        preview_fingerprint=previewed["wizard"]["preview"]["fingerprint"],
        actor_email="owner@example.com",
        actor_role="club_owner",
        idempotency_key="override:one-co-winner",
    )
    replacement = next(
        row
        for row in corrected["wizard"]["final_awards"]
        if row["award_key"] == alex_award["award_key"]
    )
    assert replacement["player_id"] == 3
    assert replacement["recipient_name"] == "Casey"
    assert replacement["metric_value"] == 0
    assert replacement["computed_player_id"] == 1
    assert replacement["computed_recipient_name"] == "Alex"
    assert replacement["computed_metric_value"] == 1
    assert (
        corrected["wizard"]["override_notes"][alex_award["award_key"]]
        == "Committee correction after score review"
    )
