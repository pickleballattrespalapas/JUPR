from __future__ import annotations

import pytest

from jupr_app.domain.team_league_roster import (
    default_mixed_counts,
    normalize_roster_settings,
    validate_playing_lineup,
    validate_team_members,
)
from jupr_app.services.team_league_service import confirmed_roster_fingerprint


@pytest.mark.parametrize(
    ("team_size", "expected_counts"),
    [(2, (1, 1)), (3, (1, 2)), (4, (2, 2))],
)
def test_roster_policy_supports_two_to_four_primary_players(
    team_size: int,
    expected_counts: tuple[int, int],
) -> None:
    assert default_mixed_counts(team_size) == expected_counts
    policy = normalize_roster_settings(
        {
            "team_size": team_size,
            "team_category": "open",
            "max_alternates": 2,
        }
    )
    assert policy["team_size"] == team_size
    assert policy["max_alternates"] == 2
    assert (policy["mixed_required_men"], policy["mixed_required_women"]) == (
        expected_counts
    )


@pytest.mark.parametrize("team_size", [0, 1, 5, 20])
def test_roster_policy_rejects_sizes_outside_supported_range(team_size: int) -> None:
    with pytest.raises(ValueError, match="2, 3, or 4"):
        normalize_roster_settings({"team_size": team_size})


def test_mixed_primary_roster_requires_configured_complete_composition() -> None:
    settings = {
        "team_size": 3,
        "team_category": "mixed",
        "max_alternates": 1,
        "mixed_required_men": 1,
        "mixed_required_women": 2,
    }
    eligible = [
        {"player_id": 1, "role": "captain", "status": "active", "gender": "man"},
        {"player_id": 2, "role": "primary", "status": "active", "gender": "woman"},
        {"player_id": 3, "role": "primary", "status": "active", "gender": "female"},
        {"player_id": 4, "role": "alternate", "status": "active", "gender": "male"},
    ]

    validate_team_members(settings=settings, members=eligible, require_complete=True)
    ineligible = [dict(row) for row in eligible]
    ineligible[2]["gender"] = "man"
    with pytest.raises(ValueError, match="1 men and 2 women"):
        validate_team_members(
            settings=settings,
            members=ineligible,
            require_complete=True,
        )


def test_roster_capacity_separates_primaries_from_assigned_alternates() -> None:
    settings = {
        "team_size": 2,
        "team_category": "open",
        "max_alternates": 1,
    }
    with pytest.raises(ValueError, match="more primary"):
        validate_team_members(
            settings=settings,
            members=[
                {"player_id": 1, "role": "captain", "status": "active"},
                {"player_id": 2, "role": "primary", "status": "active"},
                {"player_id": 3, "role": "primary", "status": "active"},
            ],
            require_complete=False,
        )
    with pytest.raises(ValueError, match="more alternates"):
        validate_team_members(
            settings=settings,
            members=[
                {"player_id": 1, "role": "captain", "status": "active"},
                {"player_id": 2, "role": "primary", "status": "active"},
                {"player_id": 3, "role": "alternate", "status": "active"},
                {"player_id": 4, "role": "alternate", "status": "active"},
            ],
            require_complete=True,
        )


def test_match_lineup_category_is_checked_after_substitution() -> None:
    validate_playing_lineup(
        category="mixed",
        player_rows=[{"gender": "Man"}, {"gender": "Woman"}],
    )
    with pytest.raises(ValueError, match="one man and one woman"):
        validate_playing_lineup(
            category="mixed",
            player_rows=[{"gender": "Man"}, {"gender": "Male"}],
        )


def test_normalized_fingerprint_includes_alternates_and_is_order_stable() -> None:
    teams = [{"id": "team-a", "status": "confirmed"}]
    members = [
        {"team_id": "team-a", "player_id": 3, "role": "alternate", "status": "active"},
        {"team_id": "team-a", "player_id": 2, "role": "primary", "status": "active"},
        {"team_id": "team-a", "player_id": 1, "role": "captain", "status": "active"},
    ]

    assert confirmed_roster_fingerprint(teams, members) == confirmed_roster_fingerprint(
        teams, list(reversed(members))
    )
    assert confirmed_roster_fingerprint(teams, members) != confirmed_roster_fingerprint(
        teams, members[:-1]
    )
