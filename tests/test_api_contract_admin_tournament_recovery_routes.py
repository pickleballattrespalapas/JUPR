from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from services.api import admin_tournament_routes as routes


def test_local_recovery_routes_keep_atomic_rpc_when_guarded_ledger_is_off(
    monkeypatch,
) -> None:
    calls: list[tuple[str, dict]] = []

    def capture(action):
        def invoke(_supabase, **kwargs):
            calls.append((action, kwargs))
            return {"ok": True, "action": action, "atomic": kwargs["atomic"]}

        return invoke

    monkeypatch.setattr(routes, "is_admin_tournament_admin_enabled", lambda: True)
    monkeypatch.setattr(
        routes,
        "_resolve_tournament_role_or_403",
        lambda **_kwargs: ("owner@example.com", "club_owner"),
    )
    monkeypatch.setattr(
        routes, "require_tournament_admin_mutation_runtime", lambda _surface: None
    )
    monkeypatch.setattr(
        routes, "tournament_admin_guarded_runtime_enabled", lambda _surface: False
    )
    monkeypatch.setattr(
        routes, "reconcile_admin_tournament_round_robin_games", capture("reconcile")
    )
    monkeypatch.setattr(
        routes, "rebuild_admin_tournament_round_robin_games", capture("rebuild")
    )
    monkeypatch.setattr(
        routes, "cancel_admin_tournament_empty_draw", capture("cancel_draw")
    )
    monkeypatch.setattr(
        routes, "cancel_admin_tournament_empty_event", capture("cancel_event")
    )

    app = FastAPI()
    routes.install_admin_tournament_routes(
        app,
        get_supabase_client=lambda: object(),
    )
    client = TestClient(app)
    requests = [
        (
            "/admin/clubs/club/tournaments/admin/tournaments/tour-1/draws/draw-1/games/round-robin/reconcile",
            {
                "expected_draw_updated_at": "2026-08-25T12:00:00Z",
                "confirmation_text": "RECONCILE GAMES",
            },
        ),
        (
            "/admin/clubs/club/tournaments/admin/tournaments/tour-1/draws/draw-1/games/round-robin/rebuild",
            {
                "expected_draw_updated_at": "2026-08-25T12:00:00Z",
                "confirmation_text": "REBUILD GAMES",
            },
        ),
        (
            "/admin/clubs/club/tournaments/admin/tournaments/tour-1/draws/draw-1/cancel-empty",
            {
                "expected_draw_updated_at": "2026-08-25T12:00:00Z",
                "confirmation_text": "CANCEL EMPTY DRAW",
            },
        ),
        (
            "/admin/clubs/club/tournaments/admin/tournaments/tour-1/events/event-1/cancel-empty",
            {"confirmation_text": "CANCEL EMPTY EVENT"},
        ),
    ]

    for path, body in requests:
        response = client.post(
            path,
            headers={"Authorization": "Bearer local"},
            json=body,
        )
        assert response.status_code == 200, response.text
        assert response.json()["atomic"] is True

    assert [action for action, _kwargs in calls] == [
        "reconcile",
        "rebuild",
        "cancel_draw",
        "cancel_event",
    ]
    assert all(kwargs["atomic"] is True for _action, kwargs in calls)


def test_guarded_recovery_routes_supply_exact_read_only_reconcilers(monkeypatch) -> None:
    reconciled: list[tuple[str, str, str, str]] = []

    monkeypatch.setattr(routes, "is_admin_tournament_admin_enabled", lambda: True)
    monkeypatch.setattr(
        routes,
        "_resolve_tournament_role_or_403",
        lambda **_kwargs: ("owner@example.com", "club_owner"),
    )

    def verify(_supabase, **kwargs):
        reconciled.append(
            (
                kwargs["club_id"],
                kwargs["tournament_id"],
                kwargs["action"],
                kwargs["entity_id"],
            )
        )
        assert kwargs["operation"] == {"operation_key": "op-1"}
        return {"ok": True, "response_loss_reconciled": True}

    def guarded(_supabase, **kwargs):
        assert callable(kwargs.get("reconcile"))
        return kwargs["reconcile"]({"operation_key": "op-1"})

    monkeypatch.setattr(routes, "reconcile_admin_tournament_ops_recovery", verify)
    monkeypatch.setattr(routes, "_guarded_ops_mutation", guarded)

    app = FastAPI()
    routes.install_admin_tournament_routes(app, get_supabase_client=lambda: object())
    client = TestClient(app)
    requests = [
        (
            "/admin/clubs/club/tournaments/admin/tournaments/tour-1/draws/draw-1/games/round-robin/reconcile",
            {
                "expected_draw_updated_at": "2026-08-25T12:00:00Z",
                "confirmation_text": "RECONCILE GAMES",
            },
        ),
        (
            "/admin/clubs/club/tournaments/admin/tournaments/tour-1/draws/draw-1/games/round-robin/rebuild",
            {
                "expected_draw_updated_at": "2026-08-25T12:00:00Z",
                "confirmation_text": "REBUILD GAMES",
            },
        ),
        (
            "/admin/clubs/club/tournaments/admin/tournaments/tour-1/draws/draw-1/cancel-empty",
            {
                "expected_draw_updated_at": "2026-08-25T12:00:00Z",
                "confirmation_text": "CANCEL EMPTY DRAW",
            },
        ),
        (
            "/admin/clubs/club/tournaments/admin/tournaments/tour-1/events/event-1/cancel-empty",
            {"confirmation_text": "CANCEL EMPTY EVENT"},
        ),
    ]

    for path, body in requests:
        response = client.post(
            path,
            headers={"Authorization": "Bearer local"},
            json=body,
        )
        assert response.status_code == 200, response.text
        assert response.json()["response_loss_reconciled"] is True

    assert reconciled == [
        ("club", "tour-1", "ops_round_robin_reconcile", "draw-1"),
        ("club", "tour-1", "ops_round_robin_rebuild", "draw-1"),
        ("club", "tour-1", "ops_empty_draw_cancel", "draw-1"),
        ("club", "tour-1", "ops_empty_event_cancel", "event-1"),
    ]
