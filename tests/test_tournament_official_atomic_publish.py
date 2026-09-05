from __future__ import annotations

from types import SimpleNamespace

import pytest

from jupr_app.services.admin_tournament_guarded_operation import StaleTournamentAdminStateError
from jupr_app.domain.admin_activity_log import ActivityLogWriteResult
from jupr_app.services.admin_tournament_match_publish_service import (
    _apply_official_rating_plan_atomic,
    build_admin_tournament_official_publish_plan,
    publish_admin_tournament_draw_matches,
)
from tests.test_admin_match_log_service import FakeSupabase
from tests.test_api_contract_admin_tournament_match_publish import (
    install_official_publish_prerequisites,
    match_publish_tables,
)
from tests.test_api_contract_admin_tournament_singles_publish import (
    singles_tournament_tables,
)


class _RpcCall:
    def __init__(self, owner: "AtomicFakeSupabase", name: str, params: dict) -> None:
        self.owner = owner
        self.name = name
        self.params = params

    def execute(self):
        self.owner.rpc_calls.append((self.name, dict(self.params)))
        if self.owner.rpc_error is not None:
            raise self.owner.rpc_error
        if self.name == "admin_apply_tournament_official_rating_plan_cas":
            for update in self.params.get("p_player_updates") or []:
                player = next(
                    (
                        row
                        for row in self.owner.tables.get("players", [])
                        if str(row.get("club_id"))
                        == str(self.params.get("p_club_id"))
                        and int(row.get("id")) == int(update["player_id"])
                    ),
                    None,
                )
                if player is None:
                    continue
                for key, expected in update["expected"].items():
                    actual = player.get(key)
                    if actual != expected:
                        raise RuntimeError(
                            "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAYER_STALE: "
                            f"{key} expected {expected!r}, found {actual!r}"
                        )
        return SimpleNamespace(data=dict(self.owner.rpc_result))


class AtomicFakeSupabase(FakeSupabase):
    def __init__(self, tables: dict[str, list[dict]]) -> None:
        super().__init__(tables)
        self.rpc_calls: list[tuple[str, dict]] = []
        self.rpc_error: Exception | None = None
        self.rpc_result: dict = {"ok": True, "inserted": 1}

    def rpc(self, name: str, params: dict) -> _RpcCall:
        return _RpcCall(self, name, params)


def _minimal_write_plan() -> dict:
    return {
        "publish_plan": {"draw_id": "draw-1"},
        "match_rows": [{"tournament_game_id": "game-1"}],
        "player_updates": [{"player_id": 1}],
        "league_rating_updates": [],
        "league_metadata_expectations": [],
    }


def test_atomic_rpc_payload_binds_operation_request_and_exact_plan() -> None:
    supabase = AtomicFakeSupabase({})
    result = _apply_official_rating_plan_atomic(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        guarded_operation_key="operation-key",
        guarded_request_fingerprint="request-fingerprint",
        publish_plan_fingerprint="plan-fingerprint",
        write_plan=_minimal_write_plan(),
    )

    assert result["inserted"] == 1
    name, payload = supabase.rpc_calls[0]
    assert name == "admin_apply_tournament_official_rating_plan_cas"
    assert payload["p_operation_key"] == "operation-key"
    assert payload["p_request_fingerprint"] == "request-fingerprint"
    assert payload["p_publish_plan_fingerprint"] == "plan-fingerprint"
    assert payload["p_publish_plan"] == {"draw_id": "draw-1"}
    assert payload["p_match_rows"] == [{"tournament_game_id": "game-1"}]


@pytest.mark.parametrize(
    "marker",
    [
        "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAYER_STALE",
        "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LEAGUE_RATING_STALE",
        "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_EVENT_OPTION_STALE",
        "JUPR_TOURNAMENT_OFFICIAL_PUBLISH_TOURNAMENT_STALE",
    ],
)
def test_atomic_rpc_explicit_dependency_cas_rejection_is_stale(marker: str) -> None:
    supabase = AtomicFakeSupabase({})
    supabase.rpc_error = RuntimeError(marker)
    with pytest.raises(StaleTournamentAdminStateError, match="dependencies changed"):
        _apply_official_rating_plan_atomic(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            guarded_operation_key="operation-key",
            guarded_request_fingerprint="request-fingerprint",
            publish_plan_fingerprint="plan-fingerprint",
            write_plan=_minimal_write_plan(),
        )



def test_atomic_rpc_transport_loss_is_ambiguous_not_stale() -> None:
    supabase = AtomicFakeSupabase({})
    supabase.rpc_error = TimeoutError("connection closed after commit")
    with pytest.raises(RuntimeError, match="response is ambiguous"):
        _apply_official_rating_plan_atomic(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            guarded_operation_key="operation-key",
            guarded_request_fingerprint="request-fingerprint",
            publish_plan_fingerprint="plan-fingerprint",
            write_plan=_minimal_write_plan(),
        )


def _enable_atomic_publish(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_OFFICIAL_PUBLISH", "1")


def test_atomic_publish_receipt_binds_identity_and_failed_queue_blocks_receipt(monkeypatch) -> None:
    _enable_atomic_publish(monkeypatch)
    tables = match_publish_tables()
    install_official_publish_prerequisites(tables)
    supabase = AtomicFakeSupabase(tables)
    plan = build_admin_tournament_official_publish_plan(
        supabase,
        club_id="club",
        tournament_id="tour_1",
        draw_id="draw_1",
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.run_badge_side_effects",
        lambda **_kwargs: {"mode": "queue", "processed": 1, "errored": 0},
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.queue_player_updates",
        lambda **_kwargs: {"mode": "queued", "failed": 0},
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.auto_send_player_updates_for_match_payloads",
        lambda *_args, **_kwargs: {"mode": "disabled"},
    )

    result = publish_admin_tournament_draw_matches(
        supabase,
        club_id="club",
        tournament_id="tour_1",
        draw_id="draw_1",
        actor_email="owner@example.com",
        actor_role="club_owner",
        confirmation_text="PUBLISH MATCHES",
        expected_plan=plan,
        guarded_operation_key="operation-key",
        guarded_request_fingerprint="request-fingerprint",
        client_idempotency_key="client-uuid",
    )

    assert result["match_count"] == 1
    receipt = tables["admin_activity_log"][-1]["after_json"]
    assert receipt["guarded_operation_key"] == "operation-key"
    assert receipt["guarded_request_fingerprint"] == "request-fingerprint"
    assert receipt["client_idempotency_key"] == "client-uuid"
    rpc_payload = supabase.rpc_calls[0][1]
    assert rpc_payload["p_publish_plan"] == plan
    assert rpc_payload["p_match_rows"][0]["tournament_game_id"] == "game_1"
    assert len(rpc_payload["p_player_updates"]) == 4

    failed_tables = match_publish_tables()
    install_official_publish_prerequisites(failed_tables)
    failed_supabase = AtomicFakeSupabase(failed_tables)
    failed_plan = build_admin_tournament_official_publish_plan(
        failed_supabase,
        club_id="club",
        tournament_id="tour_1",
        draw_id="draw_1",
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.queue_player_updates",
        lambda **_kwargs: {"mode": "queued", "failed": 1},
    )
    with pytest.raises(RuntimeError, match="post-processor failed"):
        publish_admin_tournament_draw_matches(
            failed_supabase,
            club_id="club",
            tournament_id="tour_1",
            draw_id="draw_1",
            actor_email="owner@example.com",
            actor_role="club_owner",
            confirmation_text="PUBLISH MATCHES",
            expected_plan=failed_plan,
            guarded_operation_key="failed-operation-key",
            guarded_request_fingerprint="failed-request-fingerprint",
        )
    assert not any(
        row.get("action_type") == "publish_tournament_games_to_matches_admin"
        for row in failed_tables["admin_activity_log"]
    )


def test_atomic_singles_publish_compares_null_rating_while_using_preserved_seed(
    monkeypatch,
) -> None:
    _enable_atomic_publish(monkeypatch)
    tables = singles_tournament_tables()
    for player, seed in zip(tables["players"], (1400, 1300), strict=True):
        player.update(
            {
                "singles_rating": None,
                "singles_wins": 0,
                "singles_losses": 0,
                "singles_matches_played": 0,
                "singles_last_game_at": None,
                "singles_replay_baseline": {
                    "rating": seed,
                    "wins": 0,
                    "losses": 0,
                    "matches_played": 0,
                    "last_game_at": None,
                },
            }
        )
    install_official_publish_prerequisites(tables)
    supabase = AtomicFakeSupabase(tables)
    plan = build_admin_tournament_official_publish_plan(
        supabase,
        club_id="club",
        tournament_id="tour_1",
        draw_id="draw_1",
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.run_badge_side_effects",
        lambda **_kwargs: {"mode": "queue", "processed": 1, "errored": 0},
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.queue_player_updates",
        lambda **_kwargs: {"mode": "queued", "failed": 0},
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.auto_send_player_updates_for_match_payloads",
        lambda *_args, **_kwargs: {"mode": "disabled"},
    )

    result = publish_admin_tournament_draw_matches(
        supabase,
        club_id="club",
        tournament_id="tour_1",
        draw_id="draw_1",
        actor_email="owner@example.com",
        actor_role="club_owner",
        confirmation_text="PUBLISH MATCHES",
        expected_plan=plan,
        guarded_operation_key="singles-operation-key",
        guarded_request_fingerprint="singles-request-fingerprint",
        client_idempotency_key="singles-client-uuid",
    )

    assert result["match_count"] == 1
    rpc_payload = supabase.rpc_calls[0][1]
    assert rpc_payload["p_match_rows"][0]["t1_p1_r"] == 1400
    assert rpc_payload["p_match_rows"][0]["t2_p1_r"] == 1300
    assert all(
        update["expected"]["singles_rating"] is None
        for update in rpc_payload["p_player_updates"]
    )


def test_atomic_publish_missing_bound_receipt_is_ambiguous_after_cas(monkeypatch) -> None:
    _enable_atomic_publish(monkeypatch)
    tables = match_publish_tables()
    install_official_publish_prerequisites(tables)
    supabase = AtomicFakeSupabase(tables)
    plan = build_admin_tournament_official_publish_plan(
        supabase,
        club_id="club",
        tournament_id="tour_1",
        draw_id="draw_1",
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.run_badge_side_effects",
        lambda **_kwargs: {"mode": "queue", "processed": 1, "errored": 0},
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.queue_player_updates",
        lambda **_kwargs: {"mode": "queued", "failed": 0},
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.auto_send_player_updates_for_match_payloads",
        lambda *_args, **_kwargs: {"mode": "disabled"},
    )
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.write_admin_activity_log",
        lambda *_args, **_kwargs: ActivityLogWriteResult(ok=False, warning="receipt unavailable"),
    )

    with pytest.raises(RuntimeError, match="operation-bound post-processor receipt did not persist"):
        publish_admin_tournament_draw_matches(
            supabase,
            club_id="club",
            tournament_id="tour_1",
            draw_id="draw_1",
            actor_email="owner@example.com",
            actor_role="club_owner",
            confirmation_text="PUBLISH MATCHES",
            expected_plan=plan,
            guarded_operation_key="operation-key",
            guarded_request_fingerprint="request-fingerprint",
        )
    assert len(supabase.rpc_calls) == 1
    assert not any(
        row.get("action_type") == "publish_tournament_games_to_matches_admin"
        for row in tables["admin_activity_log"]
    )
