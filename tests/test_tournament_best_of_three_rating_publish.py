from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from jupr_app.domain.match_processing import process_matches
from jupr_app.domain.player_activity import coerce_utc_datetime
from jupr_app.domain.replay_history import FULL_RESET_LABEL, replay_history
from jupr_app.domain.tournament_admin_operations import (
    stable_tournament_admin_fingerprint,
)
from jupr_app.services.admin_tournament_lifecycle_service import (
    build_admin_tournament_lifecycle,
    build_tournament_rating_game_plan,
)
from jupr_app.services.admin_tournament_match_publish_service import (
    _build_official_match_payloads,
    build_admin_tournament_official_publish_plan,
    reconcile_admin_tournament_official_publish,
)
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_admin_tournament_podium_review_service import podium_review_tables
from tests.test_rating_integrity_clashes import _Supabase, _seed_players


TOURNAMENT = {
    "id": "tour-1",
    "name": "Summer Classic",
    "start_date": "2026-09-01",
}
DRAW = {"id": "draw-1", "name": "Open Doubles"}
EVENT = {"id": "event-1", "division_name": "Open Doubles"}
TEAMS = [
    {"id": "team-a", "player1_id": 1, "player2_id": 2},
    {"id": "team-b", "player1_id": 3, "player2_id": 4},
]


def _series_parent(
    parent_id: str,
    *,
    stage: str,
    score_a: int,
    score_b: int,
    rr_round_number: int | None = None,
    playoff_game_code: str | None = None,
    playoff_round: str | None = None,
) -> dict:
    winner = "team-a" if score_a > score_b else "team-b"
    loser = "team-b" if winner == "team-a" else "team-a"
    return {
        "id": parent_id,
        "tournament_id": "tour-1",
        "draw_id": "draw-1",
        "registration_day_id": "day-1",
        "event_option_id": "event-1",
        "stage": stage,
        "rr_round_number": rr_round_number,
        "rr_slot_number": 1 if rr_round_number else None,
        "playoff_game_code": playoff_game_code,
        "playoff_round": playoff_round,
        "team_a_id": "team-a",
        "team_b_id": "team-b",
        "score_a": score_a,
        "score_b": score_b,
        "winner_team_id": winner,
        "loser_team_id": loser,
        "finalized_at": "2026-09-01T12:00:00Z",
        "scoring_format": "BEST_2_OF_3",
        "result_type": "PLAYED",
        "parent_result_only": True,
        "updated_at": "2026-09-01T12:00:00Z",
    }


def _series_child(
    parent: dict,
    game_number: int,
    score_a: int,
    score_b: int,
) -> dict:
    winner = "team-a" if score_a > score_b else "team-b"
    loser = "team-b" if winner == "team-a" else "team-a"
    return {
        "id": f"{parent['id']}-g{game_number}",
        "tournament_id": parent["tournament_id"],
        "draw_id": parent["draw_id"],
        "registration_day_id": parent["registration_day_id"],
        "event_option_id": parent["event_option_id"],
        "stage": "SERIES_GAME",
        "team_a_id": parent["team_a_id"],
        "team_b_id": parent["team_b_id"],
        "score_a": score_a,
        "score_b": score_b,
        "winner_team_id": winner,
        "loser_team_id": loser,
        "finalized_at": f"2026-09-01T12:00:0{game_number}Z",
        "scoring_format": "GAME_TO_11",
        "score_review_json": {
            "accepted": True,
            "scoring_format": "GAME_TO_11",
            "score_a": score_a,
            "score_b": score_b,
        },
        "result_type": "PLAYED",
        "parent_result_only": False,
        "series_parent_game_id": parent["id"],
        "series_game_number": game_number,
        "updated_at": f"2026-09-01T12:00:0{game_number}Z",
    }


def _attach_parent_review(parent: dict, children: list[dict]) -> None:
    parent["score_review_json"] = {
        "accepted": True,
        "scoring_format": "BEST_2_OF_3",
        "score_a": parent["score_a"],
        "score_b": parent["score_b"],
        "game_scores": [
            {
                "game_number": child["series_game_number"],
                "score_a": child["score_a"],
                "score_b": child["score_b"],
                "score_review": child["score_review_json"],
            }
            for child in sorted(
                children, key=lambda row: int(row["series_game_number"])
            )
        ],
    }


def test_rating_payloads_use_series_children_in_finalized_chronology_and_bonus_only_clinch() -> None:
    round_robin = _series_parent(
        "rr-parent",
        stage="ROUND_ROBIN",
        rr_round_number=2,
        score_a=2,
        score_b=0,
    )
    final = _series_parent(
        "final-parent",
        stage="PLAYOFF",
        playoff_game_code="P4",
        playoff_round="FINAL",
        score_a=2,
        score_b=1,
    )
    round_robin_children = [
        _series_child(round_robin, 1, 11, 4),
        _series_child(round_robin, 2, 11, 6),
    ]
    final_children = [
        _series_child(final, 1, 11, 7),
        _series_child(final, 2, 8, 11),
        _series_child(final, 3, 11, 9),
    ]
    _attach_parent_review(round_robin, round_robin_children)
    _attach_parent_review(final, final_children)
    games = [
        final_children[2],
        final,
        round_robin_children[1],
        final_children[0],
        round_robin,
        final_children[1],
        round_robin_children[0],
    ]

    payloads = _build_official_match_payloads(
        tournament=TOURNAMENT,
        draw=DRAW,
        event_option=EVENT,
        teams=TEAMS,
        games=games,
        playoff_winner_bonus_elo=20,
    )

    assert [row["tournament_game_id"] for row in payloads] == [
        "final-parent-g1",
        "rr-parent-g1",
        "final-parent-g2",
        "rr-parent-g2",
        "final-parent-g3",
    ]
    assert [(row["score_t1"], row["score_t2"]) for row in payloads] == [
        (11, 7),
        (11, 4),
        (8, 11),
        (11, 6),
        (11, 9),
    ]
    assert [
        row["tournament_game_id"]
        for row in payloads
        if row.get("winner_bonus_elo")
    ] == ["final-parent-g3"]
    assert all(
        row["tournament_game_id"] not in {"rr-parent", "final-parent"}
        for row in payloads
    )


def test_initial_publish_and_replay_apply_best_of_three_games_in_rating_parity() -> None:
    round_robin = _series_parent(
        "rr-parent",
        stage="ROUND_ROBIN",
        rr_round_number=1,
        score_a=2,
        score_b=1,
    )
    final = _series_parent(
        "final-parent",
        stage="PLAYOFF",
        playoff_game_code="P4",
        playoff_round="FINAL",
        score_a=1,
        score_b=2,
    )
    round_robin_children = [
        _series_child(round_robin, 1, 11, 4),
        _series_child(round_robin, 2, 6, 11),
        _series_child(round_robin, 3, 11, 9),
    ]
    final_children = [
        _series_child(final, 1, 3, 11),
        _series_child(final, 2, 11, 5),
        _series_child(final, 3, 7, 11),
    ]
    for child, finalized_at in zip(
        final_children,
        (
            "2026-09-01T12:00:00Z",
            "2026-09-01T13:00:00Z",
            "2026-09-01T14:00:00Z",
        ),
        strict=True,
    ):
        child["finalized_at"] = finalized_at
    for child, finalized_at in zip(
        round_robin_children,
        (
            # Exact tie with final game 2 exercises the deterministic source-id
            # tie-break and microsecond spacing used before atomic insertion.
            "2026-09-01T13:00:00Z",
            "2026-09-01T15:00:00Z",
            "2026-09-01T16:00:00Z",
        ),
        strict=True,
    ):
        child["finalized_at"] = finalized_at
    _attach_parent_review(round_robin, round_robin_children)
    _attach_parent_review(final, final_children)

    payloads = _build_official_match_payloads(
        tournament=TOURNAMENT,
        draw=DRAW,
        event_option=EVENT,
        teams=TEAMS,
        games=[
            round_robin,
            *round_robin_children,
            final,
            *final_children,
        ],
        playoff_winner_bonus_elo=20,
    )

    assert [row["tournament_game_id"] for row in payloads] == [
        "final-parent-g1",
        "final-parent-g2",
        "rr-parent-g1",
        "final-parent-g3",
        "rr-parent-g2",
        "rr-parent-g3",
    ]
    application_dates = [coerce_utc_datetime(row["date"]) for row in payloads]
    assert all(value is not None for value in application_dates)
    assert all(
        earlier < later
        for earlier, later in zip(
            application_dates,
            application_dates[1:],
        )
    )
    assert payloads[2]["date"] == "2026-09-01T13:00:00.000001+00:00"

    seed_players = _seed_players()
    initial_supabase = _Supabase()
    initial_supabase.tables["players"] = deepcopy(seed_players)
    initial = process_matches(
        payloads,
        supabase=initial_supabase,
        club_id="club",
        name_to_id={},
        df_players_all=pd.DataFrame(seed_players),
        df_leagues=pd.DataFrame(),
        df_meta=pd.DataFrame(),
        build_write_plan_only=True,
    )
    initial_plan = initial["write_plan"]
    expected_players = {
        int(row["player_id"]): row["after"]
        for row in initial_plan["player_updates"]
    }

    replay_supabase = _Supabase()
    replay_supabase.tables["players"] = deepcopy(seed_players)
    replay_supabase.tables["matches"] = [
        {**deepcopy(row), "id": index, "deleted_at": None}
        for index, row in enumerate(initial_plan["match_rows"], start=1)
    ]
    replay_history(
        supabase=replay_supabase,
        club_id="club",
        df_meta=pd.DataFrame(),
        target_reset=FULL_RESET_LABEL,
    )

    for player in replay_supabase.tables["players"]:
        expected = expected_players[int(player["id"])]
        assert float(player["rating"]) == pytest.approx(float(expected["rating"]))
        assert int(player["wins"]) == int(expected["wins"])
        assert int(player["losses"]) == int(expected["losses"])
        assert int(player["matches_played"]) == int(expected["matches_played"])
    for replayed, initial_row in zip(
        replay_supabase.tables["matches"],
        initial_plan["match_rows"],
        strict=True,
    ):
        for field in (
            "t1_p1_r",
            "t1_p2_r",
            "t2_p1_r",
            "t2_p2_r",
            "t1_p1_r_end",
            "t1_p2_r_end",
            "t2_p1_r_end",
            "t2_p2_r_end",
        ):
            assert float(replayed[field]) == pytest.approx(float(initial_row[field]))


def test_finalized_best_of_three_parent_without_children_fails_closed() -> None:
    parent = _series_parent(
        "orphan-parent",
        stage="PLAYOFF",
        playoff_game_code="P4",
        playoff_round="FINAL",
        score_a=2,
        score_b=0,
    )

    plan = build_tournament_rating_game_plan([parent])

    assert plan["rating_games"] == []
    assert [row["code"] for row in plan["errors"]] == [
        "BEST_OF_THREE_SERIES_GAMES_MISSING"
    ]
    with pytest.raises(ValueError, match="rating evidence is incomplete"):
        _build_official_match_payloads(
            tournament=TOURNAMENT,
            draw=DRAW,
            event_option=EVENT,
            teams=TEAMS,
            games=[parent],
        )


def test_mid_series_retirement_preserves_played_rating_games_without_clinch_bonus() -> None:
    parent = _series_parent(
        "retirement-parent",
        stage="PLAYOFF",
        playoff_game_code="P4",
        playoff_round="FINAL",
        score_a=0,
        score_b=2,
    )
    parent["result_type"] = "RETIREMENT"
    parent["parent_result_only"] = False
    child = _series_child(parent, 1, 11, 7)
    _attach_parent_review(parent, [child])
    parent["score_review_json"].update(
        {
            "retirement_completed_games_preserved": True,
            "synthetic_progression_score": True,
            "rating_publish_eligible": False,
            "non_playing_team_id": "team-a",
        }
    )

    plan = build_tournament_rating_game_plan([parent, child])

    assert plan["errors"] == []
    assert [row["id"] for row in plan["rating_games"]] == [child["id"]]
    assert plan["rating_games"][0]["_series_clinching"] is False
    payloads = _build_official_match_payloads(
        tournament=TOURNAMENT,
        draw=DRAW,
        event_option=EVENT,
        teams=TEAMS,
        games=[parent, child],
        playoff_winner_bonus_elo=20,
    )
    assert [row["tournament_game_id"] for row in payloads] == [child["id"]]
    assert payloads[0].get("winner_bonus_elo") in (None, 0)


def test_retirement_cannot_preserve_a_completed_best_of_three_series() -> None:
    parent = _series_parent(
        "invalid-retirement-parent",
        stage="ROUND_ROBIN",
        rr_round_number=1,
        score_a=0,
        score_b=2,
    )
    parent["result_type"] = "RETIREMENT"
    parent["parent_result_only"] = False
    children = [
        _series_child(parent, 1, 11, 7),
        _series_child(parent, 2, 11, 8),
    ]
    _attach_parent_review(parent, children)
    parent["score_review_json"].update(
        {
            "retirement_completed_games_preserved": True,
            "synthetic_progression_score": True,
            "rating_publish_eligible": False,
            "non_playing_team_id": "team-a",
        }
    )

    plan = build_tournament_rating_game_plan([parent, *children])

    assert plan["rating_games"] == []
    assert "RETIREMENT_SERIES_EVIDENCE_INVALID" in {
        row["code"] for row in plan["errors"]
    }


def test_legacy_aggregate_best_of_three_requires_original_game_detail() -> None:
    legacy_parent = _series_parent(
        "legacy-parent",
        stage="ROUND_ROBIN",
        rr_round_number=1,
        score_a=2,
        score_b=1,
    )
    legacy_parent.pop("parent_result_only")

    plan = build_tournament_rating_game_plan([legacy_parent])

    assert plan["rating_games"] == []
    assert [row["code"] for row in plan["errors"]] == [
        "BEST_OF_THREE_INDIVIDUAL_GAME_DETAIL_REQUIRED"
    ]
    assert "cannot be inferred" in plan["errors"][0]["message"]


def test_immutable_publish_plan_binds_child_matches_and_all_source_versions() -> None:
    parent = _series_parent(
        "final-parent",
        stage="PLAYOFF",
        playoff_game_code="P4",
        playoff_round="FINAL",
        score_a=2,
        score_b=1,
    )
    children = [
        _series_child(parent, 1, 11, 7),
        _series_child(parent, 2, 8, 11),
        _series_child(parent, 3, 11, 9),
    ]
    _attach_parent_review(parent, children)
    tables = {
        "tournaments": [{**TOURNAMENT, "club_id": "club", "updated_at": "tour-v1"}],
        "tournament_event_draws": [
            {
                **DRAW,
                "tournament_id": "tour-1",
                "event_option_id": "event-1",
                "updated_at": "draw-v1",
            }
        ],
        "tournament_event_options": [
            {**EVENT, "tournament_id": "tour-1"}
        ],
        "tournament_teams": [
            {**team, "tournament_id": "tour-1", "draw_id": "draw-1", "updated_at": f"{team['id']}-v1"}
            for team in TEAMS
        ],
        "tournament_games": [parent, *reversed(children)],
    }

    plan = build_admin_tournament_official_publish_plan(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        playoff_winner_bonus_elo=20,
    )

    assert plan["tournament_game_ids"] == [
        "final-parent-g1",
        "final-parent-g2",
        "final-parent-g3",
    ]
    assert plan["rating_application_order"] == [
        "final-parent-g1",
        "final-parent-g2",
        "final-parent-g3",
    ]
    assert plan["competition_game_count"] == 1
    assert plan["bonus_tournament_game_ids"] == ["final-parent-g3"]
    assert {row["id"] for row in plan["game_versions"]} == {
        "final-parent",
        "final-parent-g1",
        "final-parent-g2",
        "final-parent-g3",
    }


def test_best_of_three_publish_recovery_keeps_competition_and_rating_counts_distinct() -> None:
    parent = _series_parent(
        "final-parent",
        stage="PLAYOFF",
        playoff_game_code="P4",
        playoff_round="FINAL",
        score_a=2,
        score_b=1,
    )
    children = [
        _series_child(parent, 1, 11, 7),
        _series_child(parent, 2, 8, 11),
        _series_child(parent, 3, 11, 9),
    ]
    _attach_parent_review(parent, children)
    tables = {
        "tournaments": [
            {**TOURNAMENT, "club_id": "club", "updated_at": "tour-v1"}
        ],
        "tournament_event_draws": [
            {
                **DRAW,
                "tournament_id": "tour-1",
                "event_option_id": "event-1",
                "updated_at": "draw-v1",
            }
        ],
        "tournament_event_options": [{**EVENT, "tournament_id": "tour-1"}],
        "tournament_teams": [
            {
                **team,
                "tournament_id": "tour-1",
                "draw_id": "draw-1",
                "updated_at": f"{team['id']}-v1",
            }
            for team in TEAMS
        ],
        "tournament_games": [parent, *children],
        "matches": [],
        "admin_activity_log": [],
    }
    plan = build_admin_tournament_official_publish_plan(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        playoff_winner_bonus_elo=20,
    )
    tables["matches"] = [
        {**deepcopy(row), "id": index}
        for index, row in enumerate(
            plan["match_payload_projections"], start=1
        )
    ]
    operation_key = "operation-key"
    request_fingerprint = "request-fingerprint"
    client_idempotency_key = "client-idempotency-key"
    tables["admin_activity_log"] = [
        {
            "club_id": "club",
            "entity_id": "draw-1",
            "action_type": "publish_tournament_games_to_matches_admin",
            "after_json": {
                "publish_plan_fingerprint": stable_tournament_admin_fingerprint(
                    plan
                ),
                "guarded_operation_key": operation_key,
                "guarded_request_fingerprint": request_fingerprint,
                "client_idempotency_key": client_idempotency_key,
            },
        }
    ]

    recovered = reconcile_admin_tournament_official_publish(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        expected_plan=plan,
        guarded_operation_key=operation_key,
        guarded_request_fingerprint=request_fingerprint,
        client_idempotency_key=client_idempotency_key,
    )

    assert recovered["game_count"] == 1
    assert recovered["match_count"] == 3


def test_lifecycle_counts_and_standings_use_parent_but_rating_uses_children(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    parent = tables["tournament_games"][0]
    parent.update(
        {
            "score_a": 2,
            "score_b": 1,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "scoring_format": "BEST_2_OF_3",
            "result_type": "PLAYED",
            "parent_result_only": True,
        }
    )
    parent_for_children = {
        **parent,
        "registration_day_id": None,
        "event_option_id": None,
        "team_a_id": "team-a",
        "team_b_id": "team-b",
    }
    # Reuse the child builder, then restore the fixture's actual team ids.
    children = [
        _series_child(parent_for_children, 1, 11, 7),
        _series_child(parent_for_children, 2, 8, 11),
        _series_child(parent_for_children, 3, 11, 9),
    ]
    for child in children:
        child.update(
            {
                "team_a_id": "team-1",
                "team_b_id": "team-2",
                "winner_team_id": (
                    "team-1" if int(child["score_a"]) > int(child["score_b"]) else "team-2"
                ),
                "loser_team_id": (
                    "team-2" if int(child["score_a"]) > int(child["score_b"]) else "team-1"
                ),
            }
        )
    _attach_parent_review(parent, children)
    tables["tournament_games"].extend(children)

    lifecycle = build_admin_tournament_lifecycle(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )

    assert lifecycle["counts"]["games"] == 3
    assert lifecycle["counts"]["finalized_games"] == 3
    assert lifecycle["counts"]["rating_publish_eligible_games"] == 5
    standings = {
        row["team_id"]: row for row in lifecycle["draws"][0]["standings"]
    }
    assert standings["team-1"]["wins"] == 2
    assert standings["team-1"]["losses"] == 0
    blocker_codes = {
        row["code"]
        for row in lifecycle["domain_readiness"]["official_publish"]["blockers"]
    }
    assert "BEST_OF_THREE_RATING_SOURCE_INVALID" not in blocker_codes


def test_lifecycle_surfaces_legacy_best_of_three_game_detail_blocker(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    tables["tournament_games"][0].update(
        {
            "score_a": 2,
            "score_b": 0,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "scoring_format": "BEST_2_OF_3",
            "result_type": "PLAYED",
        }
    )
    tables["tournament_games"][0].pop("parent_result_only", None)

    lifecycle = build_admin_tournament_lifecycle(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        selected_draw_id="draw-1",
    )

    assert lifecycle["counts"]["games"] == 3
    assert lifecycle["counts"]["rating_publish_eligible_games"] == 2
    blockers = lifecycle["domain_readiness"]["official_publish"]["blockers"]
    missing_detail = next(
        row
        for row in blockers
        if row["code"] == "BEST_OF_THREE_INDIVIDUAL_GAME_DETAIL_REQUIRED"
    )
    assert missing_detail["count"] == 1
    assert "cannot be reconstructed" in missing_detail["message"]
    assert "BEST_OF_THREE_RATING_SOURCE_INVALID" not in {
        row["code"] for row in blockers
    }
