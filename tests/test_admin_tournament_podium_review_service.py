from __future__ import annotations

from copy import deepcopy

import pytest

from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.services.admin_tournament_ops_service import get_admin_tournament_ops_state_fingerprint
from jupr_app.services.admin_tournament_podium_review_service import (
    build_admin_tournament_podium_review_fingerprint,
    find_current_admin_tournament_podium_review,
    review_admin_tournament_draw_podium,
)
from tests.test_admin_match_log_service import FakeSupabase


def podium_review_tables() -> dict[str, list[dict]]:
    updated = "2026-08-15T12:00:00Z"
    teams = [
        {
            "id": f"team-{number}",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "team_number": number,
            "player1_id": number * 2 - 1,
            "player2_id": number * 2,
            "updated_at": updated,
        }
        for number in (1, 2, 3)
    ]
    games = [
        {
            "id": "game-1",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "stage": "ROUND_ROBIN",
            "rr_round_number": 1,
            "rr_slot_number": 1,
            "team_a_id": "team-1",
            "team_b_id": "team-2",
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "finalized_at": updated,
            "updated_at": updated,
        },
        {
            "id": "game-2",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "stage": "ROUND_ROBIN",
            "rr_round_number": 2,
            "rr_slot_number": 1,
            "team_a_id": "team-1",
            "team_b_id": "team-3",
            "score_a": 11,
            "score_b": 8,
            "winner_team_id": "team-1",
            "loser_team_id": "team-3",
            "finalized_at": updated,
            "updated_at": updated,
        },
        {
            "id": "game-3",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "stage": "ROUND_ROBIN",
            "rr_round_number": 3,
            "rr_slot_number": 1,
            "team_a_id": "team-2",
            "team_b_id": "team-3",
            "score_a": 11,
            "score_b": 9,
            "winner_team_id": "team-2",
            "loser_team_id": "team-3",
            "finalized_at": updated,
            "updated_at": updated,
        },
    ]
    return {
        "tournaments": [
            {
                "id": "tour-1",
                "club_id": "club",
                "name": "Summer Classic",
                "status": "PUBLISHED",
                "start_date": "2026-09-01",
                "end_date": "2026-09-02",
                "updated_at": updated,
            }
        ],
        "tournament_event_draws": [
            {
                "id": "draw-1",
                "tournament_id": "tour-1",
                "event_option_id": "event-1",
                "name": "Open Doubles",
                "status": "draft",
                "updated_at": updated,
            }
        ],
        "tournament_teams": teams,
        "tournament_games": games,
        "tournament_podium": [
            {
                "id": f"podium-{placement}",
                "tournament_id": "tour-1",
                "draw_id": "draw-1",
                "placement": placement,
                "team_id": f"team-{placement}",
                "source": "ROUND_ROBIN",
                "updated_at": updated,
            }
            for placement in (1, 2, 3)
        ],
        "tournament_registration_days": [],
        "tournament_event_options": [],
        "tournament_registrations": [],
        "tournament_registration_selections": [],
        "players": [
            {"club_id": "club", "id": player_id, "name": f"Player {player_id}"}
            for player_id in range(1, 7)
        ],
        "matches": [],
        "player_badges": [],
        "admin_activity_log": [],
        "tournament_admin_operations": [],
    }


def _versions(rows: list[dict]) -> list[dict[str, str]]:
    return sorted(
        [{"id": str(row["id"]), "updated_at": str(row["updated_at"])} for row in rows],
        key=lambda row: row["id"],
    )


def _review(supabase: FakeSupabase, tables: dict[str, list[dict]]) -> dict:
    return review_admin_tournament_draw_podium(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        expected_state_fingerprint=get_admin_tournament_ops_state_fingerprint(
            supabase,
            club_id="club",
            tournament_id="tour-1",
        ),
        expected_draw_updated_at=tables["tournament_event_draws"][0]["updated_at"],
        expected_team_versions=_versions(tables["tournament_teams"]),
        expected_source_game_versions=_versions(tables["tournament_games"]),
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="REVIEW PODIUM",
    )


def test_podium_review_writes_current_immutable_audit_evidence(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    supabase = FakeSupabase(tables)

    result = _review(supabase, tables)

    assert result["reviewed"] is True
    assert len(result["review_fingerprint"]) == 64
    assert len(tables["admin_activity_log"]) == 1
    audit = tables["admin_activity_log"][0]
    assert audit["action_type"] == "review_tournament_draw_podium_admin"
    assert audit["after_json"]["podium_review_evidence"]["review_fingerprint"] == result["review_fingerprint"]
    current = find_current_admin_tournament_podium_review(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        review_fingerprint=result["review_fingerprint"],
    )
    assert current["current"] is True


def test_team_game_or_podium_drift_invalidates_review(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    supabase = FakeSupabase(tables)
    result = _review(supabase, tables)

    tables["tournament_games"][0].update(
        {
            "score_a": 7,
            "score_b": 11,
            "winner_team_id": "team-2",
            "loser_team_id": "team-1",
            "updated_at": "2026-08-15T12:01:00Z",
        }
    )
    changed = build_admin_tournament_podium_review_fingerprint(
        draw=tables["tournament_event_draws"][0],
        teams=tables["tournament_teams"],
        games=tables["tournament_games"],
        podium=tables["tournament_podium"],
    )
    assert changed != result["review_fingerprint"]
    current = find_current_admin_tournament_podium_review(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        review_fingerprint=changed,
    )
    assert current["reviewed"] is True
    assert current["current"] is False


def test_podium_drift_then_semantic_revert_cannot_resurrect_stale_review(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    supabase = FakeSupabase(tables)
    result = _review(supabase, tables)

    podium_row = tables["tournament_podium"][0]
    podium_row.update(
        {
            "team_id": "team-2",
            "updated_at": "2026-08-15T12:01:00Z",
        }
    )
    podium_row.update(
        {
            "team_id": "team-1",
            "updated_at": "2026-08-15T12:02:00Z",
        }
    )
    reverted = build_admin_tournament_podium_review_fingerprint(
        draw=tables["tournament_event_draws"][0],
        teams=tables["tournament_teams"],
        games=tables["tournament_games"],
        podium=tables["tournament_podium"],
    )

    assert reverted != result["review_fingerprint"]
    current = find_current_admin_tournament_podium_review(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        review_fingerprint=reverted,
    )
    assert current["reviewed"] is True
    assert current["current"] is False


def test_unrelated_review_audit_does_not_churn_draw_fingerprint(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    state = deepcopy(tables)
    before = build_admin_tournament_podium_review_fingerprint(
        draw=state["tournament_event_draws"][0],
        teams=state["tournament_teams"],
        games=state["tournament_games"],
        podium=state["tournament_podium"],
    )
    state["admin_activity_log"].append(
        {
            "club_id": "club",
            "entity_type": "tournament_event_draw",
            "entity_id": "unrelated-draw",
            "action_type": "review_tournament_draw_podium_admin",
            "after_json": {"podium_review_evidence": {"review_fingerprint": "unrelated"}},
        }
    )
    after = build_admin_tournament_podium_review_fingerprint(
        draw=state["tournament_event_draws"][0],
        teams=state["tournament_teams"],
        games=state["tournament_games"],
        podium=state["tournament_podium"],
    )
    assert before == after


def test_podium_review_fails_closed_on_stale_state_and_missing_audit(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    tables = podium_review_tables()
    supabase = FakeSupabase(tables)
    with pytest.raises(StaleTournamentAdminStateError, match="state changed"):
        review_admin_tournament_draw_podium(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            expected_state_fingerprint="0" * 64,
            expected_draw_updated_at=tables["tournament_event_draws"][0]["updated_at"],
            expected_team_versions=_versions(tables["tournament_teams"]),
            expected_source_game_versions=_versions(tables["tournament_games"]),
            actor_email="director@example.com",
            actor_role="club_owner",
            confirmation_text="REVIEW PODIUM",
        )

    tables["__failed_insert_tables__"] = {"admin_activity_log"}
    with pytest.raises(RuntimeError, match="Awards and official publishing remain blocked"):
        _review(supabase, tables)
