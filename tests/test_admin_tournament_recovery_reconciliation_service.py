from __future__ import annotations

from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
)
from jupr_app.services import admin_tournament_recovery_reconciliation_service as recovery
from jupr_app.services.admin_tournament_draw_service import (
    cancel_admin_tournament_empty_draw,
    cancel_admin_tournament_empty_event,
)
from jupr_app.services.admin_tournament_game_service import (
    rebuild_admin_tournament_round_robin_games,
    reconcile_admin_tournament_round_robin_games,
)
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_admin_tournament_empty_event_recovery_service import (
    _tables as _empty_tables,
)
from tests.test_admin_tournament_schedule_recovery_service import (
    UPDATED,
    _tables as _schedule_tables,
    _team_versions,
)


def _operation(
    *,
    action: str,
    entity_type: str,
    entity_id: str,
    payload: dict,
    created_at: str = "2026-08-25T11:00:00Z",
) -> dict:
    request = build_tournament_admin_operation_request(
        club_id="club",
        surface="operations",
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        lock_scope="tour-1",
        expected_state="before",
        payload=payload,
    )
    return {
        **request,
        "status": "recovery_required",
        "request_json": request,
        "result_json": {},
        "created_at": created_at,
    }


def _enable(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "test")
    monkeypatch.setattr(
        recovery,
        "get_admin_tournament_ops_state_fingerprint",
        lambda *_args, **_kwargs: "after",
    )


def test_response_lost_round_robin_reconcile_is_proven_from_exact_schedule(
    monkeypatch,
) -> None:
    _enable(monkeypatch)
    tables = _schedule_tables()
    payload = {
        "draw_id": "draw-1",
        "expected_draw_updated_at": UPDATED,
        "expected_team_versions": _team_versions(tables),
        "preserve_existing_games": True,
    }
    reconcile_admin_tournament_round_robin_games(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="RECONCILE GAMES",
        expected_draw_updated_at=UPDATED,
        expected_team_versions=payload["expected_team_versions"],
        allow_non_atomic_test_adapter=True,
    )

    result = recovery.reconcile_admin_tournament_ops_recovery(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        action="ops_round_robin_reconcile",
        entity_id="draw-1",
        operation=_operation(
            action="ops_round_robin_reconcile",
            entity_type="tournament_event_draw",
            entity_id="draw-1",
            payload=payload,
        ),
    )

    assert result is not None
    assert result["response_loss_reconciled"] is True
    assert result["game_count"] == 36


def test_response_lost_round_robin_reconcile_ignores_series_rating_children(
    monkeypatch,
) -> None:
    _enable(monkeypatch)
    tables = _schedule_tables()
    payload = {
        "draw_id": "draw-1",
        "expected_draw_updated_at": UPDATED,
        "expected_team_versions": _team_versions(tables),
        "preserve_existing_games": True,
    }
    reconcile_admin_tournament_round_robin_games(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="RECONCILE GAMES",
        expected_draw_updated_at=UPDATED,
        expected_team_versions=payload["expected_team_versions"],
        allow_non_atomic_test_adapter=True,
    )
    parent = tables["tournament_games"][0]
    tables["tournament_games"].extend(
        [
            {
                **parent,
                "id": "series-child-linked",
                "stage": "SERIES_GAME",
                "series_parent_game_id": parent["id"],
                "series_game_number": 1,
                "updated_at": "2026-08-25T12:00:01Z",
            },
            {
                **parent,
                "id": "series-child-stage-compatible",
                "stage": "SERIES_GAME",
                "series_parent_game_id": None,
                "series_game_number": 2,
                "updated_at": "2026-08-25T12:00:02Z",
            },
        ]
    )

    result = recovery.reconcile_admin_tournament_ops_recovery(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        action="ops_round_robin_reconcile",
        entity_id="draw-1",
        operation=_operation(
            action="ops_round_robin_reconcile",
            entity_type="tournament_event_draw",
            entity_id="draw-1",
            payload=payload,
        ),
    )

    assert result is not None
    assert result["game_count"] == 36
    assert len(result["games"]) == 36
    assert not any(row.get("stage") == "SERIES_GAME" for row in result["games"])


def test_response_lost_round_robin_rebuild_requires_new_unstarted_rows(
    monkeypatch,
) -> None:
    _enable(monkeypatch)
    tables = _schedule_tables(existing_count=10)
    payload = {
        "draw_id": "draw-1",
        "expected_draw_updated_at": UPDATED,
        "expected_team_versions": _team_versions(tables),
        "replace_unstarted_games": True,
    }
    operation = _operation(
        action="ops_round_robin_rebuild",
        entity_type="tournament_event_draw",
        entity_id="draw-1",
        payload=payload,
    )
    rebuild_admin_tournament_round_robin_games(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="REBUILD GAMES",
        expected_draw_updated_at=UPDATED,
        expected_team_versions=payload["expected_team_versions"],
        allow_non_atomic_test_adapter=True,
    )

    result = recovery.reconcile_admin_tournament_ops_recovery(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        action="ops_round_robin_rebuild",
        entity_id="draw-1",
        operation=operation,
    )
    assert result is not None
    assert result["game_count"] == 36

    preexisting = _schedule_tables(existing_count=36)
    assert recovery.reconcile_admin_tournament_ops_recovery(
        FakeSupabase(preexisting),
        club_id="club",
        tournament_id="tour-1",
        action="ops_round_robin_rebuild",
        entity_id="draw-1",
        operation=operation,
    ) is None


def test_response_lost_empty_draw_and_event_cancellations_are_proven(
    monkeypatch,
) -> None:
    _enable(monkeypatch)

    draw_tables = _empty_tables()
    draw_tables["matches"] = []
    draw_tables["tournament_day_live_queue"] = []
    cancel_admin_tournament_empty_draw(
        FakeSupabase(draw_tables),
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-empty",
        expected_draw_updated_at=UPDATED,
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="CANCEL EMPTY DRAW",
        allow_non_atomic_test_adapter=True,
    )
    draw_result = recovery.reconcile_admin_tournament_ops_recovery(
        FakeSupabase(draw_tables),
        club_id="club",
        tournament_id="tour-1",
        action="ops_empty_draw_cancel",
        entity_id="draw-empty",
        operation=_operation(
            action="ops_empty_draw_cancel",
            entity_type="tournament_event_draw",
            entity_id="draw-empty",
            payload={
                "draw_id": "draw-empty",
                "expected_draw_updated_at": UPDATED,
                "status": "cancelled",
            },
        ),
    )
    assert draw_result is not None
    assert draw_result["draw"]["status"] == "cancelled"

    event_tables = _empty_tables()
    event_tables["tournament_event_draws"] = []
    cancel_admin_tournament_empty_event(
        FakeSupabase(event_tables),
        club_id="club",
        tournament_id="tour-1",
        event_option_id="event-empty",
        actor_email="director@example.com",
        actor_role="club_owner",
        confirmation_text="CANCEL EMPTY EVENT",
        allow_non_atomic_test_adapter=True,
    )
    event_result = recovery.reconcile_admin_tournament_ops_recovery(
        FakeSupabase(event_tables),
        club_id="club",
        tournament_id="tour-1",
        action="ops_empty_event_cancel",
        entity_id="event-empty",
        operation=_operation(
            action="ops_empty_event_cancel",
            entity_type="tournament_event_option",
            entity_id="event-empty",
            payload={
                "event_option_id": "event-empty",
                "enabled": False,
                "status": "cancelled",
            },
        ),
    )
    assert event_result is not None
    assert event_result["event_option"]["enabled"] is False


def test_recovery_evidence_fails_closed_for_unchanged_or_cross_scope_state(
    monkeypatch,
) -> None:
    _enable(monkeypatch)
    tables = _schedule_tables(existing_count=36)
    payload = {
        "draw_id": "draw-1",
        "expected_draw_updated_at": UPDATED,
        "expected_team_versions": _team_versions(tables),
        "preserve_existing_games": True,
    }
    operation = _operation(
        action="ops_round_robin_reconcile",
        entity_type="tournament_event_draw",
        entity_id="draw-1",
        payload=payload,
    )
    monkeypatch.setattr(
        recovery,
        "get_admin_tournament_ops_state_fingerprint",
        lambda *_args, **_kwargs: "before",
    )
    assert recovery.reconcile_admin_tournament_ops_recovery(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        action="ops_round_robin_reconcile",
        entity_id="draw-1",
        operation=operation,
    ) is None

    operation["club_id"] = "other-club"
    assert recovery.reconcile_admin_tournament_ops_recovery(
        FakeSupabase(tables),
        club_id="club",
        tournament_id="tour-1",
        action="ops_round_robin_reconcile",
        entity_id="draw-1",
        operation=operation,
    ) is None
