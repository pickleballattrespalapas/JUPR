from copy import deepcopy

import pytest

from jupr_app.domain.public_rating import has_rated_history, public_rating
from jupr_app.services.public_player_service import (
    build_public_player_directory,
    get_public_match_detail,
    get_public_player_profile,
    get_public_player_share_metadata,
)
from tests.test_public_player_service import FakeSupabase


@pytest.mark.parametrize("record,expected", [
    ({}, False), ({"matches_played": 0}, False),
    ({"matches_played": None, "wins": 1}, True),
    ({"losses": 1}, True), ({"matches_played": "1"}, True),
    ({"matches_played": "invalid"}, False),
    ({"matches_played": float("nan")}, False),
    ({"matches_played": -1}, False),
    ({"matches_played": 0, "wins": 10}, False),
])
def test_public_rating_requires_rated_history(record, expected):
    row = {"rating": 1537.2, **record}
    assert has_rated_history(row) is expected
    assert public_rating(row) == (1537.2 if expected else None)


def provisional_player_db():
    db = FakeSupabase()
    player = db.rows_by_table["players"][0]
    player.update(rating=1937.2, starting_rating=1937.2, wins=0, losses=0, matches_played=0,
                  singles_rating=1811.6, singles_wins=0, singles_losses=0, singles_matches_played=0)
    db.rows_by_table["matches"] = []
    for league in db.rows_by_table["league_ratings"]:
        league.update(rating=1937.2, starting_rating=1937.2, wins=0, losses=0, matches_played=0)
    return db


def test_unrated_only_profile_directory_search_and_share_hide_all_estimates_without_writes():
    db = provisional_player_db()
    before = deepcopy(db.rows_by_table)
    directory = build_public_player_directory(db, club_id="club-1", search="Alex")
    profile = get_public_player_profile(db, club_id="club-1", player_id=1)
    share = get_public_player_share_metadata(db, club_id="club-1", player_id=1)
    for player in [directory["players"][0], profile["player"], share["player"]]:
        for field in ["rating", "rating_jupr", "starting_rating", "starting_rating_jupr", "singles_rating", "singles_rating_jupr"]:
            assert player[field] is None
    for field in ["current_rating_jupr", "current_singles_rating_jupr", "starting_rating_jupr", "highest_rating_jupr", "lowest_rating_jupr", "last_10_delta_jupr"]:
        assert profile["rating_summary"][field] is None
    assert profile["rating_history"] == []
    assert profile["league_ratings"][0]["rating_jupr"] is None
    assert profile["league_ratings"][0]["rating_gain_jupr"] is None
    assert profile["social"]  # Unrated participation and the profile remain available.
    ranked = build_public_player_directory(db, club_id="club-1")["players"]
    assert ranked[-1]["id"] == 1  # The hidden high estimate does not determine public order.
    assert db.rows_by_table == before


@pytest.mark.parametrize("singles", [False, True])
def test_first_rated_game_publishes_only_its_own_discipline(singles):
    db = provisional_player_db()
    row = db.rows_by_table["players"][0]
    row["singles_matches_played" if singles else "matches_played"] = 1
    row["singles_wins" if singles else "wins"] = 1
    player = build_public_player_directory(db, club_id="club-1", search="Alex")["players"][0]
    assert player["rating_jupr"] == (None if singles else pytest.approx(4.843))
    assert player["singles_rating_jupr"] == (pytest.approx(4.529) if singles else None)


def test_unrated_match_snapshots_never_reintroduce_estimates_into_profile():
    db = provisional_player_db()
    match = deepcopy(FakeSupabase().rows_by_table["matches"][0])
    match["rating_scope"] = "unrated"
    db.rows_by_table["matches"] = [match]
    profile = get_public_player_profile(db, club_id="club-1", player_id=1)
    assert len(profile["rating_history"]) == 1
    assert all(profile["rating_history"][0][key] is None for key in [
        "rating_before_jupr", "rating_after_jupr", "rating_delta_jupr"
    ])
    assert profile["rating_breakdowns"][0]["matches"] == 1
    assert profile["rating_breakdowns"][0]["wins"] == 1
    assert profile["rating_summary"]["highest_rating_jupr"] is None
    assert profile["match_history"][0]["player_rating_before_jupr"] is None
    assert profile["match_history"][0]["elo_delta"] is None
    assert profile["match_history"][0]["score_t1"] == 11
    detail = get_public_match_detail(db, club_id="club-1", match_id=99)
    assert "rating_snapshot" not in detail


def test_match_explorer_does_not_replace_a_hidden_estimate_with_a_fake_rating():
    from jupr_app.services.public_match_explorer_service import build_public_match_explorer_preview
    from tests.test_public_match_explorer_service import fake_supabase
    db = fake_supabase()
    db._tables["players"][0].update(matches_played=0, wins=0, losses=0)
    with pytest.raises(ValueError, match="Not rated"):
        build_public_match_explorer_preview(db, club_id="club", me=1, partner=2, opp1=3, opp2=4)


def test_challenge_ladder_retains_unrated_player_without_publishing_seed():
    from jupr_app.services.public_challenge_ladder_service import build_public_challenge_ladder
    from tests.test_public_challenge_ladder_service import fake_supabase
    db = fake_supabase()
    db._tables["players"][1].update(matches_played=0, wins=0, losses=0)
    payload = build_public_challenge_ladder(db, club_id="club")
    player = next(p for tier in payload["tiers"] for p in tier["players"] if p["player_id"] == 2)
    assert player["rating_jupr"] is None
    pending = next(section for section in payload["challenge_sections"] if section["name"] == "Pending Acceptance")
    assert pending["challenges"][0]["challenger"]["current_rating_jupr"] is None
