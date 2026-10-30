from datetime import datetime, timezone

import pytest

from jupr_app.data.public_read_cache import PublicReadCache
from jupr_app.services import leaderboard_service as leaderboards
from jupr_app.services.public_player_service import (
    get_public_match_detail, get_public_player_profile, get_public_player_share_metadata,
)
from tests.test_public_leaderboard_projection import _fixture, _Query
from tests.test_public_player_service import FakeSupabase, FakeQuery


def test_badges_only_load_response_players_and_reuse_across_clients(monkeypatch):
    monkeypatch.setattr(leaderboards, "public_read_cache", PublicReadCache())
    sb = _fixture()
    sb.supabase_url = "https://database.example"
    # This player cannot appear on the page or any top card; its badges should
    # never be transferred from Postgres even though it belongs to this club.
    sb.store["players"].append({"id": 77, "club_id": "club-1", "name": "Unrelated",
                                "rating": 1000, "wins": 0, "losses": 1, "matches_played": 1,
                                "active": False, "last_game_at": datetime.now(timezone.utc).isoformat()})
    sb.store["player_badges"].extend({"club_id": "club-1", "player_id": 77,
                                      "badge_id": "champ"} for _ in range(2000))
    reads = []

    class Query(_Query):
        def execute(self):
            reads.append((self.table_name, dict(self.filters), dict(self.in_filters)))
            return super().execute()

    sb.table = lambda name: Query(sb.store, name)
    first = leaderboards.build_public_leaderboard(sb, club_id="club-1", limit=1)
    badge_reads = [read for read in reads if read[0] == "player_badges"]
    assert len(badge_reads) == 2  # scoped data page plus exhaustion check
    assert all(read[1]["club_id"] == "club-1" and read[2]["player_id"] == {"1", "2"} for read in badge_reads)
    assert not any(read[0] == "league_ratings" for read in reads)
    assert first["leaderboard"][0]["badge_count"] == 4
    second = _fixture()
    second.supabase_url = sb.supabase_url
    second.table = lambda name: Query(sb.store, name)
    leaderboards.build_public_leaderboard(second, club_id="club-1", limit=1)
    assert len([read for read in reads if read[0] == "player_badges"]) == 2


def test_share_metadata_loads_only_one_player_and_strips_private_fields():
    sb = FakeSupabase()
    reads = []

    class Query(FakeQuery):
        def execute(self):
            reads.append((self.table_name, dict(self.filters), self.row_limit))
            return super().execute()

    sb.table = lambda name: Query(name, sb.rows_by_table)
    result = get_public_player_share_metadata(sb, club_id="club-1", player_id=1)
    assert result["player"]["name"] == "Alex"
    assert "email" not in result["player"] and "legal_name" not in result["player"]
    assert reads == [("players", {"club_id": "club-1", "id": 1}, 1)]
    assert get_public_player_share_metadata(sb, club_id="other-club", player_id=1) is None
    # Inactive players remain publicly discoverable.
    assert get_public_player_share_metadata(sb, club_id="club-1", player_id=5)["player"]["name"] == "Inactive Izzy"


def test_profile_and_match_names_are_scoped_to_participants():
    sb = FakeSupabase()
    reads = []

    class Query(FakeQuery):
        def execute(self):
            if self.table_name == "players":
                reads.append((dict(self.filters), dict(self.in_filters), self.selected_columns))
            return super().execute()

    sb.table = lambda name: Query(name, sb.rows_by_table)
    profile = get_public_player_profile(sb, club_id="club-1", player_id=1)
    assert profile["match_history"]
    assert all(filters.get("id") == 1 or "id" in in_filters for filters, in_filters, _ in reads)
    assert all(columns == ["id", "name"] for _, in_filters, columns in reads if in_filters)
    reads.clear()
    assert get_public_match_detail(sb, club_id="club-1", match_id=99)
    assert all(filters == {"club_id": "club-1"} and in_filters == {"id": {"1", "2", "3", "4"}}
               for filters, in_filters, _ in reads)


@pytest.mark.parametrize("failed_table", ["player_badges", "badges"])
def test_badge_outage_preserves_ranking_and_recovers_without_cached_failure(monkeypatch, failed_table):
    monkeypatch.setattr(leaderboards, "public_read_cache", PublicReadCache())
    sb = _fixture()
    unavailable = [True]

    class Query(_Query):
        def execute(self):
            if unavailable[0] and self.table_name == failed_table:
                raise RuntimeError("private database failure")
            return super().execute()

    sb.table = lambda name: Query(sb.store, name)
    partial = leaderboards.build_public_leaderboard(sb, club_id="club-1", player_id=1)
    assert partial["badges_available"] is False
    assert [row["player_id"] for row in partial["leaderboard"]] == [1, 2]
    assert partial["leaderboard"][0]["rating_jupr"] == 4.0
    assert partial["snapshot"]["player_id"] == 1
    assert partial["highlights"]["highest_rating"][0]["player_id"] == 1
    assert all(row["badges"] == [] for row in partial["leaderboard"])
    assert "private database failure" not in str(partial)
    unavailable[0] = False
    recovered = leaderboards.build_public_leaderboard(sb, club_id="club-1", player_id=1)
    assert recovered["badges_available"] is True
    assert recovered["leaderboard"][0]["badge_count"] == 4
    assert recovered["snapshot"]["badges"]


def test_optional_badge_recovery_does_not_swallow_core_player_failure():
    sb = _fixture()

    class Query(_Query):
        def execute(self):
            if self.table_name == "players":
                raise RuntimeError("core data unavailable")
            return super().execute()

    sb.table = lambda name: Query(sb.store, name)
    with pytest.raises(leaderboards.LeaderboardDataUnavailable, match="players"):
        leaderboards.build_public_leaderboard(sb, club_id="club-1")
