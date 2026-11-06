from copy import deepcopy

from jupr_app.domain.league_analytics import compute_league_player_analytics
from jupr_app.services.admin_league_awards_service import (
    get_public_league_award_progress,
    freeze_admin_league_awards,
)
from tests.test_admin_match_log_service import FakeSupabase


def storage():
    return {
        "leagues_metadata": [{
            "club_id": "club", "league_name": "Ladder", "status": "active",
            "is_active": True, "league_type": "Ladder", "match_format": "doubles",
            "min_games": 1, "end_awards": {}, "awards_config_version": 1,
            "awards_config": {"categories": {
                "best_partnership": {"enabled": True, "minimum": 1, "depth": 1},
                "most_wins": {"enabled": True, "minimum": 1, "depth": 1},
            }},
        }],
        "players": [{"id": i, "club_id": "club", "name": f"Player {i}"} for i in range(1, 5)],
        "league_ratings": [{
            "club_id": "club", "league_name": "Ladder", "player_id": i,
            "rating": 1500, "starting_rating": 1400, "is_active": True,
        } for i in range(1, 5)],
        "matches": [{
            "id": i, "club_id": "club", "league": "Ladder", "date": "2026-11-01",
            "t1_p1": 1, "t1_p2": partner, "t2_p1": opp1, "t2_p2": opp2,
            "score_t1": 11, "score_t2": 8,
        } for i, (partner, opp1, opp2) in enumerate([(2, 3, 4), (3, 2, 4), (4, 2, 3)], 1)],
        "team_league_teams": [], "team_league_fixtures": [],
        "admin_activity_log": [], "badges": [], "player_badges": [],
    }


def test_public_partnership_ranking_has_every_pair_once_and_all_members():
    tables = storage()
    before = deepcopy(tables)
    public = get_public_league_award_progress(FakeSupabase(tables), club_id="club", league_name="Ladder")
    race = next(row for row in public["races"] if row["category_key"] == "best_partnership")
    assert race["recipient_type"] == "team"
    assert race["eligible_count"] == 6
    assert len({row["team_id"] for row in race["entries"]}) == 6
    assert [row["rank"] for row in race["entries"]] == [1, 1, 1, 4, 4, 4]
    assert race["entries"][0]["metric_display"] == "100.0% (1/1)"
    assert all(len(row["team_members"]) == 2 for row in race["entries"])
    assert tables == before


def test_pair_minimum_applies_before_ranking_and_inactive_member_excludes_pair():
    tables = storage()
    tables["league_ratings"][1]["is_active"] = False
    tables["matches"].append({**tables["matches"][1], "id": 4})
    tables["leagues_metadata"][0]["awards_config"]["categories"]["best_partnership"]["minimum"] = 2
    result = get_public_league_award_progress(FakeSupabase(tables), club_id="club", league_name="Ladder")
    race = next(row for row in result["races"] if row["category_key"] == "best_partnership")
    assert [row["team_id"] for row in race["entries"]] == ["pair:1:3"]
    assert race["entries"][0]["games"] == 2


def test_frozen_rankings_use_saved_pairs_players_and_qualification_without_live_reads():
    tables = storage()
    analytics = compute_league_player_analytics(tables["matches"], club_id="club", league_name="Ladder", players=tables["players"])
    meta = deepcopy(tables["leagues_metadata"][0])
    snapshot = {
        "frozen_at": "2026-11-01T19:00:00Z", "min_games": 1,
        "awards_config": deepcopy(meta["awards_config"]), "awards": [],
        "player_analytics": analytics["players"], "partnership_analytics": analytics["partnerships"],
        "team_analytics": [],
    }
    meta.update(status="ended", is_active=False, min_games=999, awards_config={})
    meta["end_awards"] = {"workflow": {"version": 3, "status": "overrides_confirmed", "frozen_snapshot": snapshot}}
    class NoReads:
        def table(self, name):
            raise AssertionError(f"Unexpected current data read: {name}")
    result = get_public_league_award_progress(NoReads(), club_id="club", league_name="Ladder", metadata=meta)
    assert result["finalized"] is True
    assert result["frozen_at"] == snapshot["frozen_at"]
    assert {row["category_key"] for row in result["races"]} == {"most_wins", "best_partnership"}
    assert all(row["min_games"] == 1 for row in result["races"])
    assert len(next(row for row in result["races"] if row["category_key"] == "best_partnership")["entries"]) == 6
    del snapshot["partnership_analytics"]
    result = get_public_league_award_progress(NoReads(), club_id="club", league_name="Ladder", metadata=meta)
    assert result["unavailable_categories"] == ["best_partnership"]
    assert [row["category_key"] for row in result["races"]] == ["most_wins"]


def test_new_freeze_retains_full_pair_field_for_future_rankings(monkeypatch):
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_MANAGER", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_LEAGUE_AWARDS_WRITE", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test")
    result = freeze_admin_league_awards(
        FakeSupabase(storage()), club_id="club", league_name="Ladder",
        actor_email="owner@example.com", actor_role="club_owner",
        confirmation_text="FREEZE LEAGUE AWARDS", idempotency_key="freeze:stat-rankings",
    )
    snapshot = result["wizard"]["frozen_snapshot"]
    assert len(snapshot["partnership_analytics"]) == 6
    assert snapshot["min_games"] == 1
    assert snapshot["catalog"]
