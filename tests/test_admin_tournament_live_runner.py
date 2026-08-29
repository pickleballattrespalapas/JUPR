from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import uuid

import pytest

from jupr_app.domain.tournament_podium import PODIUM_BADGE_MAP
from jupr_app.domain.tournament_admin_operations import (
    build_tournament_admin_operation_request,
    stable_tournament_admin_fingerprint,
)
from jupr_app.services.admin_tournament_podium_review_service import (
    PODIUM_REVIEW_ACTION,
    PODIUM_REVIEW_CONTRACT,
    build_admin_tournament_podium_review_fingerprint,
)
from jupr_app.services.admin_tournament_guarded_operation import (
    StaleTournamentAdminStateError,
    TournamentAdminRecoveryRequiredError,
)
from jupr_app.services.admin_tournament_match_publish_service import (
    build_admin_tournament_official_publish_plan,
    publish_admin_tournament_draw_matches,
    reconcile_admin_tournament_official_publish,
)
from jupr_app.services.admin_tournament_live_service import (
    TOURNAMENT_LIVE_RECONCILE_CONFIRMATION,
    _require_live_command_permission,
    _verified_recovery_outcome,
    build_admin_tournament_live_snapshot,
    build_admin_tournament_live_status,
    execute_admin_tournament_live_command,
    reconcile_admin_tournament_live_operation,
)
from tests.conftest import require_api_dependency
from tests.test_admin_match_log_service import FakeSupabase as BaseFakeSupabase

require_api_dependency("fastapi")
require_api_dependency("supabase")

from fastapi.testclient import TestClient

from services.api.main import app


def test_tournament_live_command_permission_matrix_matches_underlying_routes() -> None:
    _require_live_command_permission("scorekeeper", "save_score")
    _require_live_command_permission("organizer", "generate_round_robin")
    _require_live_command_permission("club_owner", "publish_official_matches")
    with pytest.raises(PermissionError):
        _require_live_command_permission("scorekeeper", "generate_round_robin")
    with pytest.raises(PermissionError):
        _require_live_command_permission("organizer", "publish_official_matches")


class _FakeRpc:
    def __init__(self, tables: dict[str, list[dict]], name: str, params: dict) -> None:
        self.tables = tables
        self.name = name
        self.params = params

    @staticmethod
    def _versions(rows: list[dict]) -> list[dict[str, str]]:
        return sorted(
            [{"id": str(row.get("id") or ""), "updated_at": str(row.get("updated_at") or "")} for row in rows],
            key=lambda row: row["id"],
        )

    def execute(self):
        if self.name == "admin_score_tournament_game_result_cas":
            game = next(row for row in self.tables["tournament_games"] if row["id"] == self.params["p_game_id"])
            draw = next(row for row in self.tables["tournament_event_draws"] if row["id"] == game["draw_id"])
            if str(game.get("updated_at") or "") != str(self.params["p_expected_updated_at"]):
                raise RuntimeError("JUPR_TOURNAMENT_GAME_STALE")
            if str(draw.get("updated_at") or "") != str(self.params["p_expected_draw_updated_at"]):
                raise RuntimeError("JUPR_TOURNAMENT_DRAW_STALE")
            game.update(dict(self.params["p_game_patch"]))
            dependencies: list[dict] = []
            for patch in self.params.get("p_dependency_updates") or []:
                target = next(row for row in self.tables["tournament_games"] if row["id"] == patch["id"])
                if str(target.get("updated_at") or "") != str(patch.get("expected_updated_at") or ""):
                    raise RuntimeError("JUPR_TOURNAMENT_DEPENDENCY_STALE")
                target.update({key: value for key, value in patch.items() if key not in {"id", "expected_updated_at"}})
                dependencies.append(dict(target))
            return SimpleNamespace(data={"game": dict(game), "dependency_updates": dependencies})
        if self.name == "admin_insert_tournament_draw_games_cas":
            draw = next(row for row in self.tables["tournament_event_draws"] if row["id"] == self.params["p_draw_id"])
            teams = [row for row in self.tables["tournament_teams"] if row.get("draw_id") == self.params["p_draw_id"]]
            games = [row for row in self.tables["tournament_games"] if row.get("draw_id") == self.params["p_draw_id"]]
            if str(draw.get("updated_at") or "") != str(self.params["p_expected_draw_updated_at"]):
                raise RuntimeError("JUPR_TOURNAMENT_DRAW_STALE")
            if self._versions(teams) != sorted(self.params.get("p_expected_teams") or [], key=lambda row: row["id"]):
                raise RuntimeError("JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE")
            if self._versions(games) != sorted(self.params.get("p_expected_source_games") or [], key=lambda row: row["id"]):
                raise RuntimeError("JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE")
            inserted = [dict(row) for row in self.params.get("p_games") or []]
            self.tables["tournament_games"].extend(inserted)
            return SimpleNamespace(data={"games": inserted})
        if self.name == "admin_replace_tournament_draw_podium_cas":
            draw = next(row for row in self.tables["tournament_event_draws"] if row["id"] == self.params["p_draw_id"])
            teams = [row for row in self.tables["tournament_teams"] if row.get("draw_id") == self.params["p_draw_id"]]
            games = [row for row in self.tables["tournament_games"] if row.get("draw_id") == self.params["p_draw_id"]]
            if str(draw.get("updated_at") or "") != str(self.params["p_expected_draw_updated_at"]):
                raise RuntimeError("JUPR_TOURNAMENT_DRAW_STALE")
            if self._versions(teams) != sorted(self.params.get("p_expected_teams") or [], key=lambda row: row["id"]):
                raise RuntimeError("JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE")
            if self._versions(games) != sorted(self.params.get("p_expected_source_games") or [], key=lambda row: row["id"]):
                raise RuntimeError("JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE")
            saved = [dict(row) for row in self.params.get("p_podium") or []]
            self.tables["tournament_podium"] = [
                row for row in self.tables["tournament_podium"] if row.get("draw_id") != self.params["p_draw_id"]
            ] + saved
            return SimpleNamespace(data={"podium": saved})
        if self.name == "admin_award_tournament_draw_podium_cas":
            draw = next(row for row in self.tables["tournament_event_draws"] if row["id"] == self.params["p_draw_id"])
            teams = [row for row in self.tables["tournament_teams"] if row.get("draw_id") == self.params["p_draw_id"]]
            podium = sorted(
                [
                    {
                        "placement": int(row.get("placement") or 0),
                        "team_id": str(row.get("team_id") or ""),
                        "source": str(row.get("source") or "").upper(),
                    }
                    for row in self.tables["tournament_podium"]
                    if row.get("draw_id") == self.params["p_draw_id"]
                ],
                key=lambda row: row["placement"],
            )
            if str(draw.get("updated_at") or "") != str(self.params["p_expected_draw_updated_at"]):
                raise RuntimeError("JUPR_TOURNAMENT_DRAW_STALE")
            if self._versions(teams) != sorted(self.params.get("p_expected_teams") or [], key=lambda row: row["id"]):
                raise RuntimeError("JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE")
            if podium != self.params.get("p_expected_podium"):
                raise RuntimeError("JUPR_TOURNAMENT_PODIUM_SNAPSHOT_STALE")
            if any(
                str(row.get("context_id") or "").startswith("tour-1:draw:draw-1:podium:")
                for row in self.tables["player_badges"]
            ):
                raise RuntimeError("JUPR_TOURNAMENT_AWARD_ALREADY_EXISTS")
            saved = [{**dict(row), "revoked_at": None} for row in self.params.get("p_badges") or []]
            self.tables["player_badges"].extend(saved)
            return SimpleNamespace(data={"badges": saved})
        raise AssertionError(f"Unexpected RPC: {self.name}")


class FakeSupabase(BaseFakeSupabase):
    def __init__(self, tables, *, strict_select_tables=None):
        super().__init__(tables, strict_select_tables=strict_select_tables)
        self.rpc_calls: list[tuple[str, dict]] = []

    def rpc(self, name, params):
        self.rpc_calls.append((name, deepcopy(params)))
        return _FakeRpc(self.tables, name, params)


def live_tables() -> dict[str, list[dict]]:
    return {
        "tournaments": [
            {
                "id": "tour-1",
                "club_id": "club",
                "name": "Summer Draw",
                "status": "PUBLISHED",
                "start_date": "2026-07-20",
                "end_date": "2026-07-20",
                "created_at": "2026-07-01T00:00:00Z",
                "updated_at": "2026-07-19T10:00:00Z",
            }
        ],
        "tournament_event_draws": [
            {
                "id": "draw-1",
                "tournament_id": "tour-1",
                "registration_day_id": "day-1",
                "event_option_id": "event-1",
                "name": "Open Draw",
                "status": "ACTIVE",
                "created_at": "2026-07-19T10:00:00Z",
                "updated_at": "2026-07-19T10:00:00Z",
            }
        ],
        "tournament_event_options": [
            {
                "id": "event-1",
                "tournament_id": "tour-1",
                "registration_day_id": "day-1",
                "scoring_default": "GAME_TO_11",
                "updated_at": "2026-07-19T10:00:00Z",
            }
        ],
        "tournament_teams": [
            {
                "id": f"team-{number}",
                "tournament_id": "tour-1",
                "draw_id": "draw-1",
                "team_number": number,
                "player1_id": number * 2 - 1,
                "player2_id": number * 2,
                "source": "REGISTRATION",
                "updated_at": "2026-07-19T10:00:00Z",
            }
            for number in range(1, 5)
        ],
        "tournament_games": [
            {
                "id": "game-1",
                "tournament_id": "tour-1",
                "draw_id": "draw-1",
                "stage": "ROUND_ROBIN",
                "rr_round_number": 1,
                "rr_slot_number": 1,
                "team_a_id": "team-1",
                "team_b_id": "team-2",
                "score_a": None,
                "score_b": None,
                "winner_team_id": None,
                "loser_team_id": None,
                "finalized_at": None,
                "created_at": "2026-07-19T10:00:00Z",
                "updated_at": "2026-07-19T10:00:00Z",
            }
        ],
        "tournament_podium": [],
        "players": [
            {"id": player_id, "club_id": "club", "name": f"Player {player_id}", "active": True, "is_active": True}
            for player_id in range(1, 9)
        ],
        "matches": [],
        "player_badges": [],
        "tournament_admin_operations": [],
        "admin_activity_log": [],
    }


def install_current_live_podium_review(
    tables: dict[str, list[dict]],
    *,
    install_awards: bool = False,
) -> None:
    draw = tables["tournament_event_draws"][0]
    draw_id = str(draw["id"])
    teams = [row for row in tables["tournament_teams"] if row.get("draw_id") == draw_id]
    games = [row for row in tables["tournament_games"] if row.get("draw_id") == draw_id]
    podium = [row for row in tables["tournament_podium"] if row.get("draw_id") == draw_id]
    fingerprint = build_admin_tournament_podium_review_fingerprint(
        draw=draw,
        teams=teams,
        games=games,
        podium=podium,
    )
    tables["admin_activity_log"] = [
        row
        for row in tables.get("admin_activity_log", [])
        if row.get("action_type") != PODIUM_REVIEW_ACTION
    ]
    tables["admin_activity_log"].append(
        {
            "club_id": "club",
            "actor_email": "reviewer@example.com",
            "action_type": PODIUM_REVIEW_ACTION,
            "entity_type": "tournament_event_draw",
            "entity_id": draw_id,
            "after_json": {
                "podium_review_evidence": {
                    "contract": PODIUM_REVIEW_CONTRACT,
                    "tournament_id": "tour-1",
                    "draw_id": draw_id,
                    "review_fingerprint": fingerprint,
                }
            },
        }
    )
    if not install_awards:
        return
    teams_by_id = {str(row["id"]): row for row in teams}
    tables["player_badges"] = []
    for podium_row in podium:
        placement = int(podium_row["placement"])
        team = teams_by_id[str(podium_row["team_id"])]
        for player_id in (team.get("player1_id"), team.get("player2_id")):
            if player_id is None:
                continue
            tables["player_badges"].append(
                {
                    "id": f"award-{placement}-{player_id}",
                    "club_id": "club",
                    "player_id": player_id,
                    "badge_id": PODIUM_BADGE_MAP[placement],
                    "context_type": "tournament",
                    "context_id": f"tour-1:draw:{draw_id}:podium:{placement}",
                    "revoked_at": None,
                }
            )


def _enable_live(monkeypatch) -> None:
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "staging")
    monkeypatch.setenv("JUPR_STAGING_WRITE_WAVE", "tournament-live")
    monkeypatch.setenv("JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")


def _command(snapshot: dict, *, idempotency_key: str | None = None, **overrides) -> dict:
    draw = next(iter(snapshot.get("draws") or []), {})
    game = next((row for row in snapshot.get("games") or [] if row.get("id") == "game-1"), {})
    return {
        "command": "save_score",
        "expected_state_fingerprint": snapshot["state_fingerprint"],
        "idempotency_key": idempotency_key or str(uuid.uuid4()),
        "confirmation_text": "SAVE SCORE",
        "expected_draw_updated_at": draw.get("updated_at") or "2026-07-19T10:00:00Z",
        "expected_game_updated_at": game.get("updated_at") or "2026-07-19T10:00:00Z",
        "game_id": "game-1",
        "score_a": 11,
        "score_b": 7,
        **overrides,
    }


def _execute_score(supabase, request: dict) -> dict:
    return execute_admin_tournament_live_command(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        request=request,
        actor_email="admin@example.com",
        actor_role="club_owner",
    )


def _recovery_operation(action: str, payload: dict, *, expected_state: str = "f" * 64) -> dict:
    return {
        "action": action,
        "expected_state": expected_state,
        "request_json": {"payload": payload},
    }


def test_live_status_is_staging_only_and_checks_private_operation_store(monkeypatch) -> None:
    tables = live_tables()
    supabase = FakeSupabase(tables)
    monkeypatch.setenv("JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS", "1")
    monkeypatch.setenv("JUPR_ENV", "local")
    monkeypatch.setenv("JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES", "1")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "server-only-test-key")

    local = build_admin_tournament_live_status(supabase, club_id="club")
    assert local["writes_enabled"] is False
    assert local["staging_only"] is True
    assert local["product_boundary"] == "draw_scoped_tournament_runner_not_jupr_live"

    monkeypatch.setenv("JUPR_ENV", "staging")
    staging = build_admin_tournament_live_status(supabase, club_id="club")
    assert staging["writes_enabled"] is True
    assert staging["official_publish_writes_enabled"] is False
    assert staging["operation_store_ready"] is True
    assert staging["audit_store_ready"] is True
    assert staging["write_flag"]["name"] == "JUPR_ENABLE_STAGING_NEXT_ADMIN_TOURNAMENT_LIVE_WRITES"


def test_live_status_fails_closed_when_required_audit_store_is_unavailable(monkeypatch) -> None:
    class AuditUnavailableSupabase(FakeSupabase):
        def table(self, name):
            if name == "admin_activity_log":
                raise RuntimeError("audit unavailable")
            return super().table(name)

    _enable_live(monkeypatch)
    status = build_admin_tournament_live_status(AuditUnavailableSupabase(live_tables()), club_id="club")

    assert status["operation_store_ready"] is True
    assert status["audit_store_ready"] is False
    assert status["writes_enabled"] is False
    assert any("audit storage is unavailable" in warning for warning in status["warnings"])


def test_closed_official_publish_gate_blocks_before_live_intent(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    game = tables["tournament_games"][0]
    game.update(
        {
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "finalized_at": "2026-07-19T11:00:00Z",
            "updated_at": "2026-07-19T11:00:00Z",
        }
    )
    tables["tournament_podium"] = [
        {
            "id": f"podium-{placement}",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "placement": placement,
            "team_id": f"team-{placement}",
            "source": "ROUND_ROBIN",
            "created_at": "2026-07-19T11:00:00Z",
            "updated_at": "2026-07-19T11:00:00Z",
        }
        for placement in (1, 2, 3)
    ]
    install_current_live_podium_review(tables, install_awards=True)
    supabase = FakeSupabase(tables)
    snapshot = build_admin_tournament_live_snapshot(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
    )
    assert snapshot["runtime"]["writes_enabled"] is True
    assert snapshot["runtime"]["official_publish_writes_enabled"] is False
    assert snapshot["readiness"]["publish_official_matches"]["ready"] is False
    before_audit = deepcopy(tables["admin_activity_log"])
    request = _command(
        snapshot,
        command="publish_official_matches",
        confirmation_text="PUBLISH MATCHES",
        expected_team_versions=_FakeRpc._versions(tables["tournament_teams"]),
        expected_source_game_versions=_FakeRpc._versions(tables["tournament_games"]),
    )

    with pytest.raises(PermissionError, match="official publishing is disabled"):
        execute_admin_tournament_live_command(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            request=request,
            actor_email="admin@example.com",
            actor_role="club_owner",
        )

    assert tables["tournament_admin_operations"] == []
    assert tables["admin_activity_log"] == before_audit


def test_draw_snapshot_is_stable_python_authority_with_progression_and_readiness(monkeypatch) -> None:
    _enable_live(monkeypatch)
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_live_service.get_admin_tournament_ops_state_fingerprint",
        lambda *_args, **_kwargs: "c" * 64,
    )
    supabase = FakeSupabase(live_tables())

    first = build_admin_tournament_live_snapshot(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
    )
    second = build_admin_tournament_live_snapshot(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
    )

    assert first["authority"] == "python_fastapi"
    assert first["scope"] == "draw"
    assert first["product_boundary"] == "draw_scoped_tournament_runner_not_jupr_live"
    assert first["state_fingerprint"] == second["state_fingerprint"]
    assert len(first["state_fingerprint"]) == 64
    assert first["ops_state_fingerprint"] == "c" * 64
    assert first["ops_state_fingerprint"] != first["state_fingerprint"]
    assert first["readiness"]["save_score"]["ready"] is True
    assert first["readiness"]["generate_playoffs"]["ready"] is False
    assert first["progression"] == {
        "phase": "round_robin",
        "open_games": 1,
        "completed_games": 0,
        "published_games": 0,
        "expected_awards": 0,
        "verified_awards": 0,
    }


def test_orphan_draw_award_evidence_locks_score_and_progression(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    tables["player_badges"].append(
        {
            "club_id": "club",
            "player_id": 1,
            "badge_id": "tournament_champion",
            "context_type": "tournament",
            "context_id": "tour-1:draw:draw-1:podium:1",
            "earned_at": "2026-07-19T12:00:00Z",
            "revoked_at": None,
        }
    )
    supabase = FakeSupabase(tables)
    snapshot = build_admin_tournament_live_snapshot(supabase, club_id="club", tournament_id="tour-1", draw_id="draw-1")

    assert snapshot["award_evidence"]["unexpected_count"] == 1
    assert snapshot["readiness"]["save_score"]["ready"] is False
    assert snapshot["readiness"]["generate_playoffs"]["ready"] is False
    with pytest.raises(ValueError, match="Scores are locked after draw-scoped podium awards"):
        _execute_score(supabase, _command(snapshot))
    assert tables["tournament_admin_operations"] == []


def test_score_command_is_stale_safe_audited_and_exactly_idempotent(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    supabase = FakeSupabase(tables)
    snapshot = build_admin_tournament_live_snapshot(supabase, club_id="club", tournament_id="tour-1", draw_id="draw-1")
    request = _command(snapshot)

    first = _execute_score(supabase, request)
    replay = _execute_score(supabase, request)

    assert first["idempotent_replay"] is False
    assert replay["idempotent_replay"] is True
    assert replay["operation_key"] == first["operation_key"]
    assert replay["client_idempotency_key"] == request["idempotency_key"]
    assert tables["tournament_games"][0]["winner_team_id"] == "team-1"
    assert tables["tournament_games"][0]["result_type"] == "PLAYED"
    assert tables["tournament_games"][0]["result_note"] is None
    assert tables["tournament_games"][0]["result_recorded_by"] == "admin@example.com"
    assert tables["tournament_games"][0]["score_review_json"]["accepted"] is True
    assert supabase.rpc_calls[0][0] == "admin_score_tournament_game_result_cas"
    assert supabase.rpc_calls[0][1]["p_expected_updated_at"] == "2026-07-19T10:00:00Z"
    assert supabase.rpc_calls[0][1]["p_expected_draw_updated_at"] == "2026-07-19T10:00:00Z"
    assert len(tables["tournament_admin_operations"]) == 1
    assert tables["tournament_admin_operations"][0]["status"] == "completed"
    assert [row["action_type"] for row in tables["admin_activity_log"]] == [
        "tournament_live_score_intent",
        "score_tournament_game_admin",
        "tournament_live_score_completion",
    ]

    current = build_admin_tournament_live_snapshot(supabase, club_id="club", tournament_id="tour-1", draw_id="draw-1")
    operation = current["operations"][0]
    assert operation["status"] == "completed"
    assert operation["audit_evidence"]["intent_present"] is True
    assert operation["audit_evidence"]["completion_present"] is True


def test_score_command_binds_all_post_lock_game_versions_before_rpc(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    supabase = FakeSupabase(tables)
    snapshot = build_admin_tournament_live_snapshot(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
    )
    changed_games = [{**row, "updated_at": "2026-07-19T10:01:00Z"} for row in tables["tournament_games"]]
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_score_service._games_for_draw",
        lambda *_args, **_kwargs: changed_games,
    )

    with pytest.raises(StaleTournamentAdminStateError, match="source game set changed after review"):
        _execute_score(supabase, _command(snapshot))

    assert supabase.rpc_calls == []
    assert tables["tournament_admin_operations"][0]["status"] == "failed"


def test_stale_or_inexact_command_has_no_intent_audit_or_domain_write(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    supabase = FakeSupabase(tables)
    before = deepcopy(tables["tournament_games"])

    stale = _command({"state_fingerprint": "0" * 64})
    with pytest.raises(ValueError, match="changed after it was loaded"):
        _execute_score(supabase, stale)
    inexact = {**stale, "expected_state_fingerprint": build_admin_tournament_live_snapshot(supabase, club_id="club", tournament_id="tour-1", draw_id="draw-1")["state_fingerprint"], "confirmation_text": "save score"}
    with pytest.raises(ValueError, match="SAVE SCORE exactly"):
        _execute_score(supabase, inexact)

    assert tables["tournament_games"] == before
    assert tables["tournament_admin_operations"] == []
    assert tables["admin_activity_log"] == []


def test_client_idempotency_key_cannot_be_reused_for_changed_payload(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    supabase = FakeSupabase(tables)
    snapshot = build_admin_tournament_live_snapshot(supabase, club_id="club", tournament_id="tour-1", draw_id="draw-1")
    key = str(uuid.uuid4())
    _execute_score(supabase, _command(snapshot, idempotency_key=key))
    current = build_admin_tournament_live_snapshot(supabase, club_id="club", tournament_id="tour-1", draw_id="draw-1")

    with pytest.raises(ValueError, match="already used for a different"):
        _execute_score(supabase, _command(current, idempotency_key=key, score_a=9, score_b=11))

    assert len(tables["tournament_admin_operations"]) == 1


def test_reconcile_proves_completed_score_without_repeating_domain_write(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    supabase = FakeSupabase(tables)
    before = build_admin_tournament_live_snapshot(supabase, club_id="club", tournament_id="tour-1", draw_id="draw-1")
    client_key = str(uuid.uuid4())
    request = build_tournament_admin_operation_request(
        club_id="club",
        surface="tournament_live",
        action="tournament_live_score",
        entity_type="tournament_event_draw",
        entity_id="draw-1",
        lock_scope="tournament:tour-1:draw:draw-1",
        expected_state=before["state_fingerprint"],
        payload={
            "command": "save_score",
            "game_id": "game-1",
            "score_a": 11,
            "score_b": 7,
            "expected_draw_updated_at": "2026-07-19T10:00:00Z",
            "expected_game_updated_at": "2026-07-19T10:00:00Z",
            "score_plan": {
                "game": {
                    "id": "game-1",
                    "stage": "ROUND_ROBIN",
                    "playoff_game_code": None,
                    "playoff_round": None,
                    "team_a_id": "team-1",
                    "team_b_id": "team-2",
                    "team_a_source": None,
                    "team_b_source": None,
                    "score_a": 11,
                    "score_b": 7,
                    "winner_team_id": "team-1",
                    "loser_team_id": "team-2",
                    "finalized": True,
                },
                "downstream_games": [],
            },
        },
        idempotency_key=client_key,
    )
    tables["tournament_games"][0].update(
        {
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "finalized_at": "2026-07-19T12:00:00Z",
        }
    )
    tables["tournament_admin_operations"].append(
        {
            **{key: request[key] for key in ("operation_key", "request_fingerprint", "club_id", "surface", "action", "entity_type", "entity_id", "lock_scope", "expected_state")},
            "client_idempotency_key": client_key,
            "status": "recovery_required",
            "request_json": request,
            "result_json": {},
            "error_text": "response lost",
            "attempt_count": 1,
            "created_by": "admin@example.com",
            "updated_by": "admin@example.com",
            "created_at": "2026-07-19T12:00:00Z",
            "updated_at": "2026-07-19T12:00:00Z",
        }
    )

    result = reconcile_admin_tournament_live_operation(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        operation_key=request["operation_key"],
        confirmation_text=TOURNAMENT_LIVE_RECONCILE_CONFIRMATION,
        actor_email="admin@example.com",
        actor_role="club_owner",
    )

    assert result["recovery_disposition"] == "completed"
    assert result["reconciled"] is True
    assert tables["tournament_admin_operations"][0]["status"] == "completed"
    assert tables["tournament_admin_operations"][0]["attempt_count"] == 2
    assert tables["admin_activity_log"][-1]["action_type"] == "tournament_live_score_reconciliation"
    assert len(tables["tournament_games"]) == 1


def test_round_robin_recovery_requires_exact_reviewed_structure_not_count(monkeypatch) -> None:
    _enable_live(monkeypatch)
    supabase = FakeSupabase(live_tables())
    exact = {
        "stage": "ROUND_ROBIN",
        "rr_round_number": 1,
        "rr_slot_number": 1,
        "team_a_id": "team-1",
        "team_b_id": "team-2",
    }

    completed = _verified_recovery_outcome(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        operation=_recovery_operation(
            "tournament_live_round_robin",
            {"command": "generate_round_robin", "round_robin_plan": [exact]},
        ),
    )
    wrong_pair = {**exact, "team_b_id": "team-3"}
    uncertain = _verified_recovery_outcome(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        operation=_recovery_operation(
            "tournament_live_round_robin",
            {"command": "generate_round_robin", "round_robin_plan": [wrong_pair]},
        ),
    )

    assert completed["status"] == "completed"
    assert uncertain["status"] == "uncertain"
    assert uncertain["evidence"]["exact_round_robin_set_match"] is False


def test_playoff_and_podium_recovery_require_exact_source_and_team_sets(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    playoff = {
        "id": "playoff-1",
        "tournament_id": "tour-1",
        "draw_id": "draw-1",
        "stage": "PLAYOFF",
        "playoff_game_code": "SF1",
        "playoff_round": "Semifinal",
        "team_a_id": "team-1",
        "team_b_id": "team-4",
        "team_a_source": {"seed": 1},
        "team_b_source": {"seed": 4},
        "score_a": None,
        "score_b": None,
        "winner_team_id": None,
        "loser_team_id": None,
        "finalized_at": None,
        "updated_at": "2026-07-19T11:00:00Z",
    }
    tables["tournament_games"].append(playoff)
    tables["tournament_podium"] = [
        {"id": f"podium-{placement}", "tournament_id": "tour-1", "draw_id": "draw-1", "placement": placement, "team_id": team_id, "source": "PLAYOFF"}
        for placement, team_id in ((1, "team-1"), (2, "team-4"), (3, "team-2"))
    ]
    supabase = FakeSupabase(tables)
    playoff_plan = {
        "stage": "PLAYOFF",
        "playoff_game_code": "SF1",
        "playoff_round": "Semifinal",
        "team_a_id": "team-1",
        "team_b_id": "team-4",
        "team_a_source": {"seed": 1},
        "team_b_source": {"seed": 4},
    }
    podium_plan = [
        {"placement": placement, "team_id": team_id, "source": "PLAYOFF"}
        for placement, team_id in ((1, "team-1"), (2, "team-4"), (3, "team-2"))
    ]

    exact_playoff = _verified_recovery_outcome(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        operation=_recovery_operation(
            "tournament_live_playoffs",
            {"command": "generate_playoffs", "advance_count": 4, "playoff_plan": [playoff_plan]},
        ),
    )
    source_mismatch = _verified_recovery_outcome(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        operation=_recovery_operation(
            "tournament_live_playoffs",
            {
                "command": "generate_playoffs",
                "advance_count": 4,
                "playoff_plan": [{**playoff_plan, "team_a_source": {"seed": 2}}],
            },
        ),
    )
    exact_podium = _verified_recovery_outcome(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        operation=_recovery_operation(
            "tournament_live_podium",
            {"command": "generate_podium", "podium_plan": podium_plan},
        ),
    )
    team_mismatch = _verified_recovery_outcome(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        operation=_recovery_operation(
            "tournament_live_podium",
            {"command": "generate_podium", "podium_plan": [{**podium_plan[0], "team_id": "team-3"}, *podium_plan[1:]]},
        ),
    )

    assert exact_playoff["status"] == "completed"
    assert source_mismatch["status"] == "uncertain"
    assert exact_podium["status"] == "completed"
    assert team_mismatch["status"] == "uncertain"


def test_score_recovery_requires_exact_downstream_dependency_projection(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    tables["tournament_games"][0].update(
        {
            "stage": "PLAYOFF",
            "playoff_game_code": "SF1",
            "playoff_round": "Semifinal",
            "team_a_source": {"seed": 1},
            "team_b_source": {"seed": 2},
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "finalized_at": "2026-07-19T12:00:00Z",
        }
    )
    tables["tournament_games"].append(
        {
            "id": "final-1",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "stage": "PLAYOFF",
            "playoff_game_code": "F",
            "playoff_round": "Final",
            "team_a_id": "team-4",
            "team_b_id": "team-3",
            "team_a_source": {"winnerOf": "SF1"},
            "team_b_source": {"seed": 3},
            "score_a": None,
            "score_b": None,
            "winner_team_id": None,
            "loser_team_id": None,
            "finalized_at": None,
            "updated_at": "2026-07-19T10:00:00Z",
        }
    )
    supabase = FakeSupabase(tables)
    primary = {
        "id": "game-1",
        "stage": "PLAYOFF",
        "playoff_game_code": "SF1",
        "playoff_round": "Semifinal",
        "team_a_id": "team-1",
        "team_b_id": "team-2",
        "team_a_source": {"seed": 1},
        "team_b_source": {"seed": 2},
        "score_a": 11,
        "score_b": 7,
        "winner_team_id": "team-1",
        "loser_team_id": "team-2",
        "finalized": True,
    }
    downstream = {
        "id": "final-1",
        "stage": "PLAYOFF",
        "playoff_game_code": "F",
        "playoff_round": "Final",
        "team_a_id": "team-1",
        "team_b_id": "team-3",
        "team_a_source": {"winnerOf": "SF1"},
        "team_b_source": {"seed": 3},
        "score_a": None,
        "score_b": None,
        "winner_team_id": None,
        "loser_team_id": None,
        "finalized": False,
    }
    operation = _recovery_operation(
        "tournament_live_score",
        {
            "command": "save_score",
            "game_id": "game-1",
            "score_a": 11,
            "score_b": 7,
            "score_plan": {"game": primary, "downstream_games": [downstream]},
        },
    )

    mismatch = _verified_recovery_outcome(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        operation=operation,
    )
    tables["tournament_games"][1]["team_a_id"] = "team-1"
    exact = _verified_recovery_outcome(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        operation=operation,
    )

    assert mismatch["status"] == "uncertain"
    assert mismatch["evidence"]["score_and_identity_match"] is True
    assert mismatch["evidence"]["downstream_dependency_set_match"] is False
    assert exact["status"] == "completed"


def test_award_recovery_uses_stored_recipient_set_and_rejects_duplicates(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    expected = {
        "player_id": 1,
        "badge_id": "tournament_champion",
        "context_id": "tour-1:draw:draw-1:podium:1",
    }
    tables["player_badges"].append(
        {**expected, "id": "badge-row-1", "club_id": "club", "context_type": "tournament", "revoked_at": None}
    )
    supabase = FakeSupabase(tables)
    operation = _recovery_operation(
        "tournament_live_awards",
        {"command": "award_podium", "award_plan": [expected]},
    )

    exact = _verified_recovery_outcome(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        operation=operation,
    )
    tables["player_badges"].append(
        {**expected, "id": "badge-row-duplicate", "club_id": "club", "context_type": "tournament", "revoked_at": None}
    )
    duplicate = _verified_recovery_outcome(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        operation=operation,
    )

    assert exact["status"] == "completed"
    assert duplicate["status"] == "uncertain"
    assert duplicate["evidence"]["exact_award_set_match"] is False


def _award_command(snapshot: dict, *, idempotency_key: str | None = None) -> dict:
    return {
        "command": "award_podium",
        "expected_state_fingerprint": snapshot["state_fingerprint"],
        "idempotency_key": idempotency_key or str(uuid.uuid4()),
        "confirmation_text": "AWARD PODIUM",
        "expected_draw_updated_at": snapshot["draws"][0]["updated_at"],
        "expected_team_versions": [
            {"id": row["id"], "updated_at": row["updated_at"]}
            for row in snapshot["teams"]
        ],
        "expected_source_game_versions": [
            {"id": row["id"], "updated_at": row["updated_at"]}
            for row in snapshot["games"]
        ],
    }


def _install_completed_draw_podium(
    tables: dict[str, list[dict]],
    *,
    include_versions: bool,
) -> None:
    tables["tournament_games"][0].update(
        {
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "finalized_at": "2026-07-19T11:00:00Z",
            "updated_at": "2026-07-19T11:00:00Z",
        }
    )
    tables["tournament_podium"] = [
        {
            "id": f"podium-{placement}",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "placement": placement,
            "team_id": f"team-{placement}",
            "source": "ROUND_ROBIN",
            "created_at": "2026-07-19T11:00:00Z",
            **(
                {"updated_at": "2026-07-19T11:00:00Z"}
                if include_versions
                else {}
            ),
        }
        for placement in (1, 2, 3)
    ]
    install_current_live_podium_review(tables)


def test_missing_podium_versions_fail_before_durable_live_intent(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    _install_completed_draw_podium(tables, include_versions=False)
    supabase = FakeSupabase(tables)
    snapshot = build_admin_tournament_live_snapshot(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
    )

    with pytest.raises(ValueError, match="reviewed podium version set is incomplete"):
        execute_admin_tournament_live_command(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            request=_award_command(snapshot),
            actor_email="admin@example.com",
            actor_role="club_owner",
        )

    assert tables["player_badges"] == []
    assert tables["tournament_admin_operations"] == []
    assert not any(
        row.get("action_type") == "tournament_live_awards_intent"
        for row in tables["admin_activity_log"]
    )


def test_exact_retry_closes_proven_pre_mutation_podium_version_failure(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    _install_completed_draw_podium(tables, include_versions=True)
    supabase = FakeSupabase(tables)
    snapshot = build_admin_tournament_live_snapshot(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
    )
    client_key = str(uuid.uuid4())
    command = _award_command(snapshot, idempotency_key=client_key)
    award_podium_plan = [
        {
            "placement": placement,
            "team_id": f"team-{placement}",
            "source": "ROUND_ROBIN",
        }
        for placement in (1, 2, 3)
    ]
    award_plan = sorted(
        [
            {
                "player_id": player_id,
                "badge_id": PODIUM_BADGE_MAP[placement],
                "context_id": f"tour-1:draw:draw-1:podium:{placement}",
            }
            for placement, player_ids in (
                (1, (1, 2)),
                (2, (3, 4)),
                (3, (5, 6)),
            )
            for player_id in player_ids
        ],
        key=lambda row: (row["context_id"], row["badge_id"], row["player_id"]),
    )
    stored_payload = {
        "command": "award_podium",
        "expected_draw_updated_at": command["expected_draw_updated_at"],
        "expected_team_versions": command["expected_team_versions"],
        "expected_source_game_versions": command["expected_source_game_versions"],
        "award_podium_plan": award_podium_plan,
        "award_plan": award_plan,
    }
    operation = build_tournament_admin_operation_request(
        club_id="club",
        surface="tournament_live",
        action="tournament_live_awards",
        entity_type="tournament_event_draw",
        entity_id="draw-1",
        lock_scope="tournament:tour-1:draw:draw-1",
        expected_state=snapshot["state_fingerprint"],
        payload=stored_payload,
        idempotency_key=client_key,
    )
    tables["tournament_admin_operations"].append(
        {
            **{
                key: operation[key]
                for key in (
                    "operation_key",
                    "request_fingerprint",
                    "club_id",
                    "surface",
                    "action",
                    "entity_type",
                    "entity_id",
                    "lock_scope",
                    "expected_state",
                )
            },
            "client_idempotency_key": client_key,
            "status": "recovery_required",
            "request_json": operation,
            "result_json": {},
            "error_text": (
                "The reviewed podium version set is incomplete or duplicated. "
                "Reload the live board."
            ),
            "attempt_count": 1,
            "created_by": "admin@example.com",
            "updated_by": "admin@example.com",
            "created_at": "2026-07-19T12:00:00Z",
            "updated_at": "2026-07-19T12:00:00Z",
        }
    )

    result = execute_admin_tournament_live_command(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        request=command,
        actor_email="admin@example.com",
        actor_role="club_owner",
    )

    assert result["recovery_disposition"] == "not_applied"
    assert result["reconciled"] is True
    assert tables["player_badges"] == []
    assert tables["tournament_admin_operations"][0]["status"] == "failed"
    assert tables["tournament_admin_operations"][0]["attempt_count"] == 2
    assert tables["admin_activity_log"][-1]["action_type"] == (
        "tournament_live_awards_recovery_not_applied"
    )


def test_atomic_award_refuses_podium_drift_after_live_lock_with_zero_badges(monkeypatch) -> None:
    class DriftingAwardSupabase(FakeSupabase):
        def rpc(self, name, params):
            if name == "admin_award_tournament_draw_podium_cas":
                self.tables["tournament_podium"][0]["team_id"] = "team-4"
                self.tables["tournament_event_draws"][0]["updated_at"] = "2026-07-19T12:00:00Z"
            return super().rpc(name, params)

    _enable_live(monkeypatch)
    tables = live_tables()
    tables["tournament_podium"] = [
        {
            "id": f"podium-{placement}",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "placement": placement,
            "team_id": team_id,
            "source": "ROUND_ROBIN",
            "updated_at": "2026-07-19T10:00:00Z",
        }
        for placement, team_id in ((1, "team-1"), (2, "team-2"), (3, "team-3"))
    ]
    tables["tournament_games"][0].update(
        {
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "finalized_at": "2026-07-19T11:00:00Z",
        }
    )
    install_current_live_podium_review(tables)
    supabase = DriftingAwardSupabase(tables)
    snapshot = build_admin_tournament_live_snapshot(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
    )
    request = {
        "command": "award_podium",
        "expected_state_fingerprint": snapshot["state_fingerprint"],
        "idempotency_key": str(uuid.uuid4()),
        "confirmation_text": "AWARD PODIUM",
        "expected_draw_updated_at": snapshot["draws"][0]["updated_at"],
        "expected_team_versions": [
            {"id": row["id"], "updated_at": row["updated_at"]} for row in snapshot["teams"]
        ],
        "expected_source_game_versions": [
            {"id": row["id"], "updated_at": row["updated_at"]} for row in snapshot["games"]
        ],
    }

    with pytest.raises(StaleTournamentAdminStateError, match="changed while trophies were being awarded"):
        execute_admin_tournament_live_command(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            request=request,
            actor_email="admin@example.com",
            actor_role="club_owner",
        )

    assert tables["player_badges"] == []
    assert tables["tournament_admin_operations"][0]["status"] == "failed"


def test_publish_expected_plan_recheck_blocks_changed_score_before_processors(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    tables["tournament_games"][0].update(
        {
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "finalized_at": "2026-07-19T11:00:00Z",
        }
    )
    supabase = FakeSupabase(tables)
    plan = build_admin_tournament_official_publish_plan(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
    )
    tables["tournament_games"][0].update(
        {
            "score_a": 7,
            "score_b": 11,
            "winner_team_id": "team-2",
            "loser_team_id": "team-1",
            "updated_at": "2026-07-19T12:00:00Z",
        }
    )
    tables["tournament_podium"] = [
        {
            "id": f"podium-{placement}",
            "tournament_id": "tour-1",
            "draw_id": "draw-1",
            "placement": placement,
            "team_id": team_id,
            "source": "ROUND_ROBIN",
        }
        for placement, team_id in ((1, "team-2"), (2, "team-1"), (3, "team-3"))
    ]
    install_current_live_podium_review(tables, install_awards=True)
    processor_calls: list[list[dict]] = []
    monkeypatch.setattr(
        "jupr_app.services.admin_tournament_match_publish_service.process_matches",
        lambda payloads, **_kwargs: processor_calls.append(payloads) or {"inserted": len(payloads)},
    )

    with pytest.raises(StaleTournamentAdminStateError, match="official match payload changed after review"):
        publish_admin_tournament_draw_matches(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            actor_email="admin@example.com",
            actor_role="club_owner",
            confirmation_text="PUBLISH MATCHES",
            expected_plan=plan,
        )

    assert processor_calls == []
    assert tables["matches"] == []


def test_publish_recovery_requires_exact_match_content_fingerprint(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    tables["tournament_games"][0].update(
        {
            "score_a": 11,
            "score_b": 7,
            "winner_team_id": "team-1",
            "loser_team_id": "team-2",
            "finalized_at": "2026-07-19T11:00:00Z",
        }
    )
    supabase = FakeSupabase(tables)
    plan = build_admin_tournament_official_publish_plan(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
    )
    reconcile_identity = {
        "guarded_operation_key": "current-operation-key",
        "guarded_request_fingerprint": "current-request-fingerprint",
        "client_idempotency_key": "current-client-idempotency-key",
    }
    tables["matches"].append(
        {
            "id": "match-1",
            "club_id": "club",
            "date": "2026-07-19T11:00:00+00:00",
            "league": "Tournament · Summer Draw · Open Draw",
            "week_tag": "Open Draw",
            "match_type": "Tournament",
            "match_format": "doubles",
            "t1_p1": 1,
            "t1_p2": 2,
            "t2_p1": 3,
            "t2_p2": 4,
            "score_t1": 11,
            "score_t2": 7,
            "context_type": "tournament_game",
            "context_id": "game-1",
            "tournament_id": "tour-1",
            "tournament_game_id": "game-1",
            "rating_scope": "",
        }
    )
    with pytest.raises(TournamentAdminRecoveryRequiredError, match="Rating/player updates are not proven"):
        reconcile_admin_tournament_official_publish(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            expected_plan=plan,
            **reconcile_identity,
        )
    tables["admin_activity_log"].append(
        {
            "club_id": "club",
            "entity_id": "draw-1",
            "action_type": "publish_tournament_games_to_matches_admin",
            "after_json": {
                "publish_plan_fingerprint": stable_tournament_admin_fingerprint(plan),
                "guarded_operation_key": "stale-operation-key",
                "guarded_request_fingerprint": "stale-request-fingerprint",
                "client_idempotency_key": "stale-client-idempotency-key",
            },
        }
    )
    with pytest.raises(TournamentAdminRecoveryRequiredError, match="one exact post-processor completion receipt"):
        reconcile_admin_tournament_official_publish(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            expected_plan=plan,
            **reconcile_identity,
        )
    tables["admin_activity_log"][0]["after_json"].update(reconcile_identity)

    exact = reconcile_admin_tournament_official_publish(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
        expected_plan=plan,
        **reconcile_identity,
    )
    tables["matches"][0]["date"] = "2026-07-19T11:01:00+00:00"
    with pytest.raises(TournamentAdminRecoveryRequiredError, match="changed official-match content"):
        reconcile_admin_tournament_official_publish(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            expected_plan=plan,
            **reconcile_identity,
        )
    tables["matches"][0]["date"] = "2026-07-19T11:00:00+00:00"
    tables["matches"][0]["score_t1"] = 10
    with pytest.raises(TournamentAdminRecoveryRequiredError, match="changed official-match content"):
        reconcile_admin_tournament_official_publish(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            expected_plan=plan,
            **reconcile_identity,
        )

    assert exact["match_count"] == 1


def test_duplicate_official_publish_evidence_cannot_release_recovery_lock(monkeypatch) -> None:
    _enable_live(monkeypatch)
    tables = live_tables()
    tables["tournament_games"][0].update(
        {"score_a": 11, "score_b": 7, "winner_team_id": "team-1", "loser_team_id": "team-2", "finalized_at": "now"}
    )
    supabase = FakeSupabase(tables)
    snapshot = build_admin_tournament_live_snapshot(supabase, club_id="club", tournament_id="tour-1", draw_id="draw-1")
    request = build_tournament_admin_operation_request(
        club_id="club",
        surface="tournament_live",
        action="tournament_live_official_publish",
        entity_type="tournament_event_draw",
        entity_id="draw-1",
        lock_scope="tournament:tour-1:draw:draw-1",
        expected_state=snapshot["state_fingerprint"],
        payload={
            "command": "publish_official_matches",
            "playoff_winner_bonus_elo": 0.0,
            "expected_draw_updated_at": "2026-07-19T10:00:00Z",
            "expected_team_versions": [
                {"id": f"team-{number}", "updated_at": "2026-07-19T10:00:00Z"}
                for number in range(1, 5)
            ],
            "expected_source_game_versions": [
                {"id": "game-1", "updated_at": "2026-07-19T10:00:00Z"}
            ],
            "publish_plan": {
                "draw_id": "draw-1",
                "tournament_game_ids": ["game-1"],
                "match_count": 1,
                "singles_match_count": 0,
                "doubles_match_count": 1,
                "playoff_winner_bonus_elo": 0.0,
                "bonus_tournament_game_ids": [],
            },
        },
        idempotency_key=str(uuid.uuid4()),
    )
    tables["matches"].extend(
        [
            {
                "id": match_id,
                "club_id": "club",
                "tournament_id": "tour-1",
                "tournament_game_id": "game-1",
                "context_type": "tournament_game",
                "context_id": "game-1",
            }
            for match_id in ("match-1", "match-duplicate")
        ]
    )
    tables["tournament_admin_operations"].append(
        {
            **{key: request[key] for key in ("operation_key", "request_fingerprint", "club_id", "surface", "action", "entity_type", "entity_id", "lock_scope", "expected_state")},
            "client_idempotency_key": request["idempotency_key"],
            "status": "recovery_required",
            "request_json": request,
            "result_json": {},
            "attempt_count": 1,
            "created_by": "admin@example.com",
            "updated_by": "admin@example.com",
        }
    )

    with pytest.raises(TournamentAdminRecoveryRequiredError, match="cannot prove"):
        reconcile_admin_tournament_live_operation(
            supabase,
            club_id="club",
            tournament_id="tour-1",
            draw_id="draw-1",
            operation_key=request["operation_key"],
            confirmation_text=TOURNAMENT_LIVE_RECONCILE_CONFIRMATION,
            actor_email="admin@example.com",
            actor_role="club_owner",
        )
    assert tables["tournament_admin_operations"][0]["status"] == "recovery_required"
    evidence = build_admin_tournament_live_snapshot(supabase, club_id="club", tournament_id="tour-1", draw_id="draw-1")
    assert evidence["publication_evidence"]["complete"] is False
    assert evidence["publication_evidence"]["duplicate_game_ids"] == ["game-1"]


def _install_api(monkeypatch, supabase) -> None:
    _enable_live(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "http://example.local")
    monkeypatch.setattr("services.api.main.create_client", lambda _url, _credential: supabase)
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.authenticate_bearer",
        lambda _authorization: SimpleNamespace(email="admin@example.com", user_id="user-1"),
    )
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="club_owner"),
    )


def test_fastapi_live_surface_is_draw_scoped_and_replays_exact_command(monkeypatch) -> None:
    tables = live_tables()
    supabase = FakeSupabase(tables)
    _install_api(monkeypatch, supabase)
    client = TestClient(app)
    headers = {"Authorization": "Bearer local"}
    snapshot_response = client.get(
        "/admin/clubs/club/tournament-live/tournaments/tour-1/snapshot?draw_id=draw-1",
        headers=headers,
    )
    assert snapshot_response.status_code == 200, snapshot_response.text
    snapshot = snapshot_response.json()
    request = _command(snapshot)
    endpoint = "/admin/clubs/club/tournament-live/tournaments/tour-1/draws/draw-1/commands"

    first = client.post(endpoint, headers=headers, json=request)
    replay = client.post(endpoint, headers=headers, json=request)

    assert first.status_code == 200, first.text
    assert replay.status_code == 200, replay.text
    assert first.json()["authority"] == "python_fastapi"
    assert replay.json()["idempotent_replay"] is True
    assert replay.json()["operation_key"] == first.json()["operation_key"]


def test_fastapi_live_role_matrix_allows_scorekeeper_reads_but_denies_bracket_writes(monkeypatch) -> None:
    tables = live_tables()
    supabase = FakeSupabase(tables)
    _install_api(monkeypatch, supabase)
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="scorekeeper"),
    )
    client = TestClient(app)
    headers = {"Authorization": "Bearer local"}
    snapshot_response = client.get(
        "/admin/clubs/club/tournament-live/tournaments/tour-1/snapshot?draw_id=draw-1",
        headers=headers,
    )
    assert snapshot_response.status_code == 200
    list_response = client.get(
        "/admin/clubs/club/tournaments/admin/ops/tournaments",
        headers=headers,
    )
    assert list_response.status_code == 200
    request = _command(snapshot_response.json())
    request.update({"command": "generate_round_robin", "confirmation_text": "GENERATE GAMES"})
    denied = client.post(
        "/admin/clubs/club/tournament-live/tournaments/tour-1/draws/draw-1/commands",
        headers=headers,
        json=request,
    )
    assert denied.status_code == 403


def test_fastapi_live_organizer_cannot_publish_rated_matches(monkeypatch) -> None:
    tables = live_tables()
    supabase = FakeSupabase(tables)
    _install_api(monkeypatch, supabase)
    monkeypatch.setattr(
        "services.api.admin_tournament_routes.resolve_admin_role",
        lambda **_kwargs: SimpleNamespace(role="organizer"),
    )
    snapshot = build_admin_tournament_live_snapshot(
        supabase,
        club_id="club",
        tournament_id="tour-1",
        draw_id="draw-1",
    )
    request = _command(snapshot)
    request.update({"command": "publish_official_matches", "confirmation_text": "PUBLISH MATCHES"})
    response = TestClient(app).post(
        "/admin/clubs/club/tournament-live/tournaments/tour-1/draws/draw-1/commands",
        headers={"Authorization": "Bearer local"},
        json=request,
    )
    assert response.status_code == 403


def test_fastapi_stale_live_command_returns_409(monkeypatch) -> None:
    tables = live_tables()
    supabase = FakeSupabase(tables)
    _install_api(monkeypatch, supabase)
    response = TestClient(app).post(
        "/admin/clubs/club/tournament-live/tournaments/tour-1/draws/draw-1/commands",
        headers={"Authorization": "Bearer local"},
        json=_command({"state_fingerprint": "0" * 64}),
    )
    assert response.status_code == 409
    assert "changed after it was loaded" in response.json()["detail"]
    assert tables["tournament_admin_operations"] == []
