from __future__ import annotations

from copy import deepcopy

import pytest

from jupr_app.domain.tournaments.bracket_builder import build_round_robin_games
from jupr_app.services.admin_tournament_game_service import (
    rebuild_admin_tournament_round_robin_games,
    reconcile_admin_tournament_round_robin_games,
)
from tests.test_admin_match_log_service import FakeSupabase


UPDATED = "2026-08-25T12:00:00Z"


def _tables(*, team_count: int = 9, existing_count: int = 21) -> dict[str, list[dict]]:
    teams = [
        {
            "id": f"team-{number}",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "team_number": number,
            "player1_id": number * 2 - 1,
            "player2_id": number * 2,
            "updated_at": UPDATED,
        }
        for number in range(1, team_count + 1)
    ]
    generated = build_round_robin_games(
        tournament_id="tour-1",
        team_ids_by_number={row["team_number"]: row["id"] for row in teams},
    )
    games = [
        {
            **row,
            "id": f"existing-{index}",
            "draw_id": "draw-1",
            "registration_day_id": "day-1",
            "event_option_id": "event-1",
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
            "updated_at": UPDATED,
        }
        for index, row in enumerate(generated[:existing_count], start=1)
    ]
    return {
        "tournaments": [
            {
                "id": "tour-1",
                "club_id": "club",
                "name": "Summer Classic",
                "status": "PUBLISHED",
                "updated_at": UPDATED,
            }
        ],
        "tournament_event_draws": [
            {
                "id": "draw-1",
                "tournament_id": "tour-1",
                "registration_day_id": "day-1",
                "event_option_id": "event-1",
                "name": "Men's 3.5",
                "status": "active",
                "updated_at": UPDATED,
            }
        ],
        "tournament_teams": teams,
        "tournament_games": games,
        "tournament_podium": [],
        "matches": [],
        "player_badges": [],
        "tournament_day_live_queue": [],
        "tournament_day_live_draws": [],
        "admin_activity_log": [],
    }


def _team_versions(tables: dict[str, list[dict]]) -> list[dict[str, str]]:
    return [
        {"id": str(row["id"]), "updated_at": str(row["updated_at"])}
        for row in tables["tournament_teams"]
    ]


def test_reconcile_nine_team_partial_draw_preserves_finalized_game_and_adds_missing_pairs(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "test")
    tables = _tables()
    first_before = deepcopy(tables["tournament_games"][0])
    tables["tournament_games"][0].update(
        {
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": tables["tournament_games"][0]["team_a_id"],
            "loser_team_id": tables["tournament_games"][0]["team_b_id"],
            "finalized_at": UPDATED,
        }
    )
    finalized_before = deepcopy(tables["tournament_games"][0])

    result = reconcile_admin_tournament_round_robin_games(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="RECONCILE GAMES",
        expected_draw_updated_at=UPDATED,
        expected_team_versions=_team_versions(tables),
        allow_non_atomic_test_adapter=True,
    )

    assert result["preserved_game_count"] == 21
    assert result["preserved_finalized_game_count"] == 1
    assert result["inserted_game_count"] == 15
    assert result["game_count"] == 36
    assert tables["tournament_games"][0] == finalized_before
    assert tables["tournament_games"][0] != first_before
    pairs = {
        tuple(sorted((row["team_a_id"], row["team_b_id"])))
        for row in tables["tournament_games"]
    }
    assert len(pairs) == 36


def test_reconcile_counts_and_returns_parent_matchups_only_when_series_children_exist(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "test")
    tables = _tables()
    parent = tables["tournament_games"][0]
    parent.update(
        {
            "scoring_format": "BEST_2_OF_3",
            "parent_result_only": True,
            "score_a": 2,
            "score_b": 0,
            "winner_team_id": parent["team_a_id"],
            "loser_team_id": parent["team_b_id"],
            "finalized_at": UPDATED,
        }
    )
    child_ids = {"series-game-1", "series-game-2"}
    tables["tournament_games"].extend(
        [
            {
                **parent,
                "id": child_id,
                "stage": "SERIES_GAME",
                "series_parent_game_id": parent["id"],
                "series_game_number": game_number,
                "scoring_format": "GAME_TO_11",
                "parent_result_only": False,
                "score_a": 11,
                "score_b": 7 + game_number,
                "updated_at": f"2026-08-25T12:00:0{game_number}Z",
            }
            for game_number, child_id in enumerate(sorted(child_ids), start=1)
        ]
    )

    result = reconcile_admin_tournament_round_robin_games(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="RECONCILE GAMES",
        expected_draw_updated_at=UPDATED,
        expected_team_versions=_team_versions(tables),
        allow_non_atomic_test_adapter=True,
    )

    assert result["preserved_game_count"] == 21
    assert result["preserved_finalized_game_count"] == 1
    assert result["inserted_game_count"] == 15
    assert result["game_count"] == 36
    assert len(result["games"]) == 36
    assert not any(row.get("stage") == "SERIES_GAME" for row in result["games"])
    assert child_ids.issubset({str(row["id"]) for row in tables["tournament_games"]})


def test_reconcile_refuses_official_match_dependency(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "test")
    tables = _tables()
    tables["matches"] = [
        {
            "id": 1,
            "club_id": "club",
            "tournament_id": "tour-1",
            "tournament_game_id": tables["tournament_games"][0]["id"],
        }
    ]

    with pytest.raises(ValueError, match="official Match Log"):
        reconcile_admin_tournament_round_robin_games(
            FakeSupabase(tables),
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            actor_email="director@example.com",
            actor_role="club_owner",
            confirmation_text="RECONCILE GAMES",
            expected_draw_updated_at=UPDATED,
            expected_team_versions=_team_versions(tables),
            allow_non_atomic_test_adapter=True,
        )


def test_explicit_rebuild_replaces_only_unstarted_partial_schedule(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "test")
    tables = _tables(existing_count=10)

    result = rebuild_admin_tournament_round_robin_games(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="REBUILD GAMES",
        expected_draw_updated_at=UPDATED,
        expected_team_versions=_team_versions(tables),
        allow_non_atomic_test_adapter=True,
    )

    assert result["replaced_game_count"] == 10
    assert result["game_count"] == 36
    assert len(tables["tournament_games"]) == 36


def test_deployed_recovery_never_falls_back_to_non_atomic_writes(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "staging")
    tables = _tables()

    with pytest.raises(PermissionError, match="atomic database RPC"):
        reconcile_admin_tournament_round_robin_games(
            FakeSupabase(tables),
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            actor_email="director@example.com",
            actor_role="club_owner",
            confirmation_text="RECONCILE GAMES",
            expected_draw_updated_at=UPDATED,
            expected_team_versions=_team_versions(tables),
            atomic=False,
        )
