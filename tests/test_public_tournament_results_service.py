from __future__ import annotations

from copy import deepcopy
import json

import pytest

from tests.conftest import require_api_dependency
from tests.test_public_tournament_registration_service import (
    FakeSupabase,
    fake_storage,
)

from jupr_app.domain.tournament_registration_repo import get_public_tournament_bundle
from jupr_app.services.public_tournament_results_service import (
    build_public_tournament_index,
    build_public_tournament_results,
)


def _results_storage() -> dict:
    storage = fake_storage()
    storage["tournament_registration_days"].append(
        {
            "id": "day2",
            "tournament_id": "t1",
            "sort_order": 2,
            "label": "Sunday",
            "event_date": "2026-09-02",
            "enabled": True,
        }
    )
    storage["players"] = [
        {
            "id": 10,
            "club_id": "club-1",
            "name": "Alex Ace",
            "email": "alex-private@example.com",
            "admin_notes": "never public",
        },
        {
            "id": 11,
            "club_id": "club-1",
            "name": "Blair Backhand",
            "email": "blair-private@example.com",
        },
    ]
    storage["tournament_event_draws"] = [
        {
            "id": "draw-private-id",
            "tournament_id": "t1",
            "event_option_id": "event1",
            "registration_day_id": "day1",
            "draw_kind": "STANDARD",
            "name": "Open Doubles",
            "status": "published",
            "admin_notes": "private draw note",
        }
    ]
    storage["tournament_teams"] = [
        {
            "id": "team-private-a",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "team_number": 1,
            "player1_id": 10,
        },
        {
            "id": "team-private-b",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "team_number": 2,
            "player1_id": 11,
        },
    ]
    storage["tournament_games"] = [
        {
            "id": "game-private-rr",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "stage": "ROUND_ROBIN",
            "rr_round_number": 1,
            "rr_slot_number": 1,
            "team_a_id": "team-private-a",
            "team_b_id": "team-private-b",
            "score_a": 15,
            "score_b": 11,
            "winner_team_id": "team-private-a",
            "finalized_at": "2026-09-01T16:00:00Z",
            "admin_notes": "private game note",
        },
        {
            "id": "game-private-final",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "stage": "PLAYOFF",
            "playoff_game_code": "F1",
            "playoff_round": "Final",
            "team_a_id": "team-private-a",
            "team_b_id": "team-private-b",
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "outcome_type": "WALKOVER",
            "finalized_at": "2026-09-02T16:00:00Z",
        },
    ]
    storage["tournament_podium"] = [
        {
            "id": "podium-private-id",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "placement": 1,
            "team_id": "team-private-a",
        },
        {
            "id": "podium-private-id-2",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "placement": 2,
            "team_id": "team-private-b",
        },
    ]
    return storage


def _three_way_tie_storage() -> dict:
    storage = _results_storage()
    storage["players"].append(
        {
            "id": 12,
            "club_id": "club-1",
            "name": "Casey Counter",
            "email": "casey-private@example.com",
        }
    )
    storage["tournament_teams"].append(
        {
            "id": "team-private-c",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "team_number": 3,
            "player1_id": 12,
        }
    )
    storage["tournament_games"] = [
        {
            "id": "game-private-a-b",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "stage": "ROUND_ROBIN",
            "rr_round_number": 1,
            "rr_slot_number": 1,
            "team_a_id": "team-private-a",
            "team_b_id": "team-private-b",
            "score_a": 11,
            "score_b": 9,
            "winner_team_id": "team-private-a",
            "finalized_at": "2026-09-01T16:00:00Z",
        },
        {
            "id": "game-private-b-c",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "stage": "ROUND_ROBIN",
            "rr_round_number": 2,
            "rr_slot_number": 1,
            "team_a_id": "team-private-b",
            "team_b_id": "team-private-c",
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-private-b",
            "finalized_at": "2026-09-01T17:00:00Z",
        },
        {
            "id": "game-private-c-a",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "stage": "ROUND_ROBIN",
            "rr_round_number": 3,
            "rr_slot_number": 1,
            "team_a_id": "team-private-c",
            "team_b_id": "team-private-a",
            "score_a": 11,
            "score_b": 5,
            "winner_team_id": "team-private-c",
            "finalized_at": "2026-09-01T18:00:00Z",
        },
    ]
    storage["tournament_podium"] = []
    return storage


def _add_tournament(
    storage: dict,
    *,
    tournament_id: str,
    status: str,
    published: bool = True,
    completed_receipt: bool = False,
) -> None:
    storage["tournaments"].append(
        {
            "id": tournament_id,
            "club_id": "club-1",
            "name": tournament_id,
            "status": status,
            "start_date": "2026-08-01",
            "end_date": "2026-08-02",
        }
    )
    storage["tournament_registration_settings"].append(
        {
            "id": f"settings-{tournament_id}",
            "tournament_id": tournament_id,
            "registration_slug": f"slug-{tournament_id}",
            "registration_status": "open",
            "builder_draft_json": (
                {"published_at": "2026-07-01T00:00:00Z"} if published else {}
            ),
        }
    )
    if completed_receipt:
        storage.setdefault("tournament_lifecycle_receipts", []).append(
            {
                "id": f"receipt-{tournament_id}",
                "tournament_id": tournament_id,
                "action": "complete",
            }
        )


def test_current_and_past_tournament_discovery_is_fail_closed() -> None:
    storage = _results_storage()
    _add_tournament(storage, tournament_id="draft", status="DRAFT")
    _add_tournament(storage, tournament_id="paused", status="PAUSED")
    _add_tournament(storage, tournament_id="inactive", status="INACTIVE")
    _add_tournament(storage, tournament_id="archived", status="ARCHIVED")
    _add_tournament(storage, tournament_id="unpublished", status="ACTIVE", published=False)
    _add_tournament(storage, tournament_id="finished-no-receipt", status="COMPLETED")
    _add_tournament(
        storage,
        tournament_id="finished",
        status="COMPLETED",
        completed_receipt=True,
    )
    supabase = FakeSupabase(storage)

    current = build_public_tournament_index(supabase, club_id="club-1", view="current")
    past = build_public_tournament_index(supabase, club_id="club-1", view="past")

    assert [row["id"] for row in current["tournaments"]] == ["t1"]
    assert [row["id"] for row in past["tournaments"]] == ["finished"]


@pytest.mark.parametrize("status", ["DRAFT", "PAUSED", "INACTIVE", "ARCHIVED"])
def test_direct_results_do_not_expose_hidden_lifecycle_states(status: str) -> None:
    storage = _results_storage()
    storage["tournaments"][0]["status"] = status

    with pytest.raises(ValueError, match="not found"):
        build_public_tournament_results(
            FakeSupabase(storage), club_id="club-1", tournament_id="t1"
        )

    by_id = get_public_tournament_bundle(
        FakeSupabase(storage), club_id="club-1", tournament_id="t1"
    )
    by_slug = get_public_tournament_bundle(
        FakeSupabase(storage), club_id="club-1", registration_slug="tres-open"
    )
    assert by_id == (None, None, [], [])
    assert by_slug == (None, None, [], [])


def test_standard_results_include_multi_day_scores_bracket_standings_and_medals() -> None:
    payload = build_public_tournament_results(
        FakeSupabase(_results_storage()), club_id="club-1", tournament_id="t1"
    )

    assert payload["tournament"]["name"] == "Tres Palapas Open"
    assert len(payload["draws"]) == 1
    draw = payload["draws"][0]
    assert [day["label"] for day in draw["scheduled_days"]] == ["Saturday", "Sunday"]
    assert draw["state"] == "COMPLETE"
    assert draw["standings"][0]["team_name"] == "Alex Ace"
    assert draw["round_robin_complete"] is True
    assert draw["tiebreak_explanations"] == []
    assert draw["ranking_policy"]["criteria"] == [
        "WINS",
        "HEAD_TO_HEAD",
        "POINT_DIFFERENTIAL",
        "POINTS_FOR",
        "TEAM_NUMBER",
    ]
    assert draw["scores"][0]["score_a"] == 15
    assert draw["bracket"][0]["outcome_label"] == "Walkover"
    assert draw["bracket"][0]["state"] == "FINAL"
    assert len(draw["scores"]) == 1
    assert len(draw["bracket"]) == 1
    assert {
        row["public_game_key"] for row in draw["scores"]
    }.isdisjoint({row["public_game_key"] for row in draw["bracket"]})
    assert all(row["state"] == "FINAL" for row in draw["scores"])
    assert [row["medal"] for row in draw["podium"]] == ["Gold", "Silver"]

    serialized = json.dumps(payload)
    for private_value in (
        "alex-private@example.com",
        "blair-private@example.com",
        "private draw note",
        "private game note",
        "draw-private-id",
        "team-private-a",
        "game-private-rr",
        "podium-private-id",
    ):
        assert private_value not in serialized


def test_best_of_three_children_are_nested_in_parent_results_without_leaking() -> None:
    storage = _results_storage()
    round_robin = storage["tournament_games"][0]
    playoff = storage["tournament_games"][1]
    round_robin.update(
        {
            "score_a": 2,
            "score_b": 0,
            "scoring_format": "BEST_2_OF_3",
        }
    )
    playoff.update(
        {
            "score_a": 2,
            "score_b": 1,
            "winner_team_id": "team-private-a",
            "outcome_type": None,
            "scoring_format": "BEST_2_OF_3",
        }
    )

    def series_game(
        child_id: str,
        parent_id: str,
        game_number: int,
        score_a: int,
        score_b: int,
        *,
        finalized: bool = True,
    ) -> dict:
        return {
            "id": child_id,
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "stage": "SERIES_GAME",
            "series_parent_game_id": parent_id,
            "series_game_number": game_number,
            "team_a_id": "team-private-a",
            "team_b_id": "team-private-b",
            "score_a": score_a,
            "score_b": score_b,
            "winner_team_id": (
                "team-private-a" if score_a > score_b else "team-private-b"
            ),
            "finalized_at": "2026-09-02T16:00:00Z" if finalized else None,
            "admin_notes": f"private child note {child_id}",
        }

    # Deliberately store children out of order. The unfinished third RR game
    # must remain private just like every other draft score.
    storage["tournament_games"].extend(
        [
            series_game("private-playoff-game-3", playoff["id"], 3, 11, 8),
            series_game("private-rr-game-2", round_robin["id"], 2, 11, 5),
            series_game("private-playoff-game-1", playoff["id"], 1, 11, 7),
            series_game("private-rr-game-1", round_robin["id"], 1, 11, 9),
            series_game("private-playoff-game-2", playoff["id"], 2, 8, 11),
            series_game(
                "private-rr-draft-game-3",
                round_robin["id"],
                3,
                987,
                0,
                finalized=False,
            ),
        ]
    )

    draw = build_public_tournament_results(
        FakeSupabase(storage), club_id="club-1", tournament_id="t1"
    )["draws"][0]

    assert draw["state"] == "COMPLETE"
    assert draw["round_robin_complete"] is True
    assert len(draw["scores"]) == 1
    assert len(draw["bracket"]) == 1
    assert draw["scores"][0]["stage"] == "ROUND_ROBIN"
    assert draw["scores"][0]["score_a"] == 2
    assert draw["scores"][0]["score_b"] == 0
    assert draw["scores"][0]["game_scores"] == [
        {"game_number": 1, "score_a": 11, "score_b": 9},
        {"game_number": 2, "score_a": 11, "score_b": 5},
    ]
    assert draw["bracket"][0]["stage"] == "PLAYOFF"
    assert draw["bracket"][0]["score_a"] == 2
    assert draw["bracket"][0]["score_b"] == 1
    assert draw["bracket"][0]["game_scores"] == [
        {"game_number": 1, "score_a": 11, "score_b": 7},
        {"game_number": 2, "score_a": 8, "score_b": 11},
        {"game_number": 3, "score_a": 11, "score_b": 8},
    ]

    serialized = json.dumps(draw)
    assert "SERIES_GAME" not in serialized
    assert "private-playoff-game-1" not in serialized
    assert "private child note" not in serialized
    assert "987" not in serialized


def test_public_results_explain_three_way_cycle_through_points_scored() -> None:
    payload = build_public_tournament_results(
        FakeSupabase(_three_way_tie_storage()),
        club_id="club-1",
        tournament_id="t1",
    )

    draw = payload["draws"][0]
    assert draw["round_robin_complete"] is True
    assert [row["team_name"] for row in draw["standings"]] == [
        "Blair Backhand",
        "Casey Counter",
        "Alex Ace",
    ]
    assert draw["tiebreak_explanations"] == [
        {
            "title": "Three-way tie at 1\u20131",
            "summary": (
                "Head-to-head did not fully separate these teams. Total points "
                "scored completed the order: Blair Backhand \u2192 Casey Counter "
                "\u2192 Alex Ace."
            ),
            "steps": [
                {
                    "criterion": "HEAD_TO_HEAD",
                    "outcome": "UNRESOLVED",
                    "detail": (
                        "Head-to-head mini-table: Alex Ace 1\u20131; Blair Backhand "
                        "1\u20131; Casey Counter 1\u20131. Head-to-head did not "
                        "separate these teams."
                    ),
                },
                {
                    "criterion": "POINT_DIFFERENTIAL",
                    "outcome": "PARTIALLY_RESOLVED",
                    "detail": (
                        "Point differential for the remaining tied teams: Blair "
                        "Backhand +2; Casey Counter +2; Alex Ace -4. Point "
                        "differential separated some teams, but Blair Backhand and "
                        "Casey Counter remained tied."
                    ),
                },
                {
                    "criterion": "POINTS_FOR",
                    "outcome": "RESOLVED",
                    "detail": (
                        "Total points scored for the remaining tied teams: Blair "
                        "Backhand 20; Casey Counter 18. Total points scored resolved "
                        "the remaining tie."
                    ),
                },
            ],
        }
    ]

    serialized = json.dumps(payload)
    for private_value in (
        "alex-private@example.com",
        "blair-private@example.com",
        "casey-private@example.com",
        "draw-private-id",
        "team-private-a",
        "team-private-b",
        "team-private-c",
        "game-private-a-b",
    ):
        assert private_value not in serialized


def test_incomplete_tie_ignores_unfinalized_draft_scores_and_names_missing_games() -> None:
    storage = _three_way_tie_storage()
    storage["players"].append(
        {"id": 13, "club_id": "club-1", "name": "Delta Dink"}
    )
    storage["tournament_teams"].append(
        {
            "id": "team-private-d",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "team_number": 4,
            "player1_id": 13,
        }
    )
    storage["tournament_games"] = [
        storage["tournament_games"][0],
        {
            "id": "game-private-b-d",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team-private-b",
            "team_b_id": "team-private-d",
            "score_a": 11,
            "score_b": 0,
            "winner_team_id": "team-private-b",
            "finalized_at": "2026-09-01T17:00:00Z",
        },
        {
            "id": "game-private-c-d",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team-private-c",
            "team_b_id": "team-private-d",
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-private-c",
            "finalized_at": "2026-09-01T18:00:00Z",
        },
        {
            "id": "game-private-a-c-draft",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team-private-a",
            "team_b_id": "team-private-c",
            "score_a": 76,
            "score_b": 0,
            "winner_team_id": "team-private-a",
            "finalized_at": None,
        },
        {
            "id": "game-private-b-c-draft",
            "tournament_id": "t1",
            "draw_id": "draw-private-id",
            "stage": "ROUND_ROBIN",
            "team_a_id": "team-private-b",
            "team_b_id": "team-private-c",
            "score_a": 75,
            "score_b": 0,
            "winner_team_id": "team-private-b",
            "finalized_at": None,
        },
    ]

    draw = build_public_tournament_results(
        FakeSupabase(storage), club_id="club-1", tournament_id="t1"
    )["draws"][0]

    assert draw["round_robin_complete"] is False
    assert [row["team_name"] for row in draw["standings"][:3]] == [
        "Blair Backhand",
        "Casey Counter",
        "Alex Ace",
    ]
    explanation = draw["tiebreak_explanations"][0]
    assert explanation["title"] == "Three-way tie at 1 wins"
    assert explanation["summary"] == (
        "A complete head-to-head comparison was unavailable, so point "
        "differential completed the order: Blair Backhand \u2192 Casey Counter "
        "\u2192 Alex Ace."
    )
    assert explanation["steps"][0]["detail"] == (
        "Available head-to-head records: Alex Ace 1\u20130; Blair Backhand 0\u20131; "
        "Casey Counter 0\u20130. The complete comparison was unavailable because "
        "these matchups had no scored result: Alex Ace vs Casey Counter and "
        "Blair Backhand vs Casey Counter. Head-to-head was not applied."
    )
    serialized = json.dumps(draw)
    standings_by_name = {row["team_name"]: row for row in draw["standings"]}
    assert standings_by_name["Alex Ace"]["points_for"] == 11
    assert standings_by_name["Alex Ace"]["points_against"] == 9
    assert standings_by_name["Blair Backhand"]["points_for"] == 20
    assert standings_by_name["Blair Backhand"]["points_against"] == 11
    assert standings_by_name["Casey Counter"]["points_for"] == 11
    assert standings_by_name["Casey Counter"]["points_against"] == 7
    assert len(draw["scores"]) == 3
    assert "game-private-a-c-draft" not in serialized
    assert "game-private-b-c-draft" not in serialized
    assert "team-private-d" not in serialized


def test_round_robin_completion_suppresses_unstarted_ties_and_resolves_retirement() -> None:
    unstarted = _results_storage()
    round_robin = unstarted["tournament_games"][0]
    round_robin.update(
        {
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "finalized_at": None,
        }
    )

    unstarted_draw = build_public_tournament_results(
        FakeSupabase(unstarted), club_id="club-1", tournament_id="t1"
    )["draws"][0]
    assert unstarted_draw["round_robin_complete"] is False
    assert unstarted_draw["tiebreak_explanations"] == []

    retired = deepcopy(unstarted)
    retired["tournament_teams"][1].update(
        {
            "competition_status": "RETIRED",
            "retirement_max_score": 15,
        }
    )
    retired_draw = build_public_tournament_results(
        FakeSupabase(retired), club_id="club-1", tournament_id="t1"
    )["draws"][0]
    assert retired_draw["round_robin_complete"] is True
    assert retired_draw["standings"][0]["wins"] == 1
    assert retired_draw["standings"][0]["points_for"] == 15
    assert retired_draw["standings"][1]["retired"] is True
    assert retired_draw["standings"][1]["points_against"] == 15
    assert retired_draw["tiebreak_explanations"] == []


require_api_dependency("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from services.api.public_tournament_results_routes import (  # noqa: E402
    install_public_tournament_results_routes,
)


def test_public_results_route_requires_no_patron_auth_but_still_filters_drafts() -> None:
    storage = _results_storage()
    app = FastAPI()
    install_public_tournament_results_routes(
        app,
        get_club=lambda slug: {"id": "club-1", "slug": slug, "name": "Club"},
        get_supabase_client=lambda: FakeSupabase(storage),
        public_club_payload=lambda club, slug: {
            "id": club["id"],
            "slug": slug,
            "name": club["name"],
        },
    )
    client = TestClient(app)

    visible = client.get(
        "/clubs/test/tournament-results", params={"tournament_id": "t1"}
    )
    assert visible.status_code == 200
    assert "email" not in json.dumps(visible.json())

    storage["tournaments"][0]["status"] = "DRAFT"
    hidden = client.get(
        "/clubs/test/tournament-results", params={"tournament_id": "t1"}
    )
    assert hidden.status_code == 404
